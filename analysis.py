import re
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORY_RULES = {
    "Data & Analytics": ["data analyst", "data analysis", "data scientist", "data science", "data engineer", "analytics", "business intelligence"],
    "AI & Machine Learning": ["machine learning", "artificial intelligence", "deep learning", " llm", "generative ai", " ai "],
    "Software Development": ["software developer", "software engineer", "programmer", "python", "java developer", "full stack", "backend developer"],
    "Web Development": ["web developer", "wordpress", "shopify", "frontend", "front-end", "back-end", "webflow"],
    "Marketing": ["marketing", "seo", "social media", "advertising", "media buyer"],
    "Design": ["graphic design", "designer", "ui/ux", "ux design", "ui design", "logo design"],
    "Writing & Content": ["writer", "writing", "content", "copywriter", "copywriting"],
    "Finance & Accounting": ["accountant", "accounting", "finance", "financial", "bookkeeping"],
    "Sales & Business Development": ["sales", "business development", "lead generation"],
    "Customer Support & Admin": ["customer service", "customer support", "virtual assistant", "administrative"],
    "Engineering": ["mechanical engineer", "electrical engineer", "civil engineer", "engineering"],
}

def categorize_job(title):
    t = f" {str(title).lower()} "
    for category, terms in CATEGORY_RULES.items():
        if any(term in t for term in terms):
            return category
    return "Other"

def clean_dataset(raw_path, cleaned_path):
    df = pd.read_csv(raw_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    if "published_date" in df:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
        # Use timezone-naive timestamps for easier grouping/serialization.
        df["published_date"] = df["published_date"].dt.tz_convert(None)
    for col in ["hourly_low", "hourly_high", "budget"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"hourly_low", "hourly_high"}.issubset(df.columns):
        df["hourly_rate"] = df[["hourly_low", "hourly_high"]].mean(axis=1)
    else:
        df["hourly_rate"] = np.nan
    df["job_category"] = df["title"].map(categorize_job) if "title" in df else "Other"
    df["month"] = df["published_date"].dt.to_period("M").astype("string") if "published_date" in df else pd.NA
    Path(cleaned_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    return df

def generate_reports(df, reports_dir):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Dataset quality report
    quality = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    quality.to_csv(reports_dir / "data_quality_report.csv", index=False)

    # Task 1: title keywords vs hourly rate (hourly jobs only)
    hourly = df[df["hourly_rate"].notna() & (df["hourly_rate"] > 0)].copy()
    if not hourly.empty:
        hourly["title_keyword"] = hourly["title"].fillna("").str.lower().str.findall(r"[a-z][a-z+#.-]{2,}")
        kw = hourly.explode("title_keyword")
        kw = kw[kw["title_keyword"].notna() & (kw["title_keyword"].str.len() >= 3)]
        salary = kw.groupby("title_keyword")["hourly_rate"].agg(
            Job_Postings="count", Average_Hourly_Rate="mean", Median_Hourly_Rate="median",
            Minimum_Hourly_Rate="min", Maximum_Hourly_Rate="max"
        ).reset_index()
        salary = salary[salary["Job_Postings"] >= 5].sort_values("Average_Hourly_Rate", ascending=False)
    else:
        salary = pd.DataFrame(columns=["title_keyword","Job_Postings","Average_Hourly_Rate","Median_Hourly_Rate","Minimum_Hourly_Rate","Maximum_Hourly_Rate"])
    salary.to_csv(reports_dir / "task1_keyword_salary_analysis.csv", index=False)

    # Task 2: category frequency and first/last month comparison
    cat_counts = df.groupby("job_category").size().rename("Total_Postings").reset_index()
    if "month" in df:
        monthly_cat = df.dropna(subset=["month"]).groupby(["month", "job_category"]).size().rename("Postings").reset_index()
        monthly_cat.to_csv(reports_dir / "monthly_job_category_trends.csv", index=False)
        months = sorted(monthly_cat["month"].dropna().unique())
        rows = []
        for cat in df["job_category"].dropna().unique():
            s = monthly_cat[monthly_cat["job_category"] == cat].set_index("month")["Postings"]
            first = int(s.get(months[0], 0)) if months else 0
            last = int(s.get(months[-1], 0)) if months else 0
            rows.append({"job_category":cat, "first_month_postings":first, "last_month_postings":last,
                         "absolute_growth":last-first,
                         "growth_percent":((last-first)/first*100 if first else (100.0 if last else 0.0)),
                         "total_postings":int(s.sum())})
        growth = pd.DataFrame(rows).sort_values(["absolute_growth","total_postings"], ascending=False)
    else:
        growth = cat_counts.rename(columns={"Total_Postings":"total_postings"})
    growth.to_csv(reports_dir / "task2_emerging_job_categories.csv", index=False)

    # Task 3: baseline demand classifier (avoid pretending a tiny time range is robust)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = []
    if "published_date" in df and df["published_date"].notna().any():
        daily = (df.dropna(subset=["published_date"])
                   .assign(day=lambda x: x["published_date"].dt.floor("D"))
                   .groupby(["day","job_category"]).size().rename("postings").reset_index())
        if not daily.empty:
            all_days = pd.date_range(daily["day"].min(), daily["day"].max(), freq="D")
            cats = sorted(df["job_category"].dropna().unique())
            idx = pd.MultiIndex.from_product([all_days, cats], names=["day","job_category"])
            daily = daily.set_index(["day","job_category"]).reindex(idx, fill_value=0).reset_index()
            daily = daily.sort_values(["job_category","day"])
            g = daily.groupby("job_category")["postings"]
            for lag in [1,3,7]:
                daily[f"lag_{lag}"] = g.shift(lag)
            daily["rolling_mean_7"] = daily.groupby("job_category")["postings"].transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
            daily["target_count"] = daily.groupby("job_category")["postings"].shift(-1)
            daily = daily.dropna(subset=["lag_1","lag_3","target_count","rolling_mean_7"])
            if len(daily) >= 30 and daily["target_count"].nunique() > 1:
                features = ["lag_1","lag_3","lag_7","rolling_mean_7"]
                daily["high_demand"] = (daily["target_count"] > daily["target_count"].median()).astype(int)
                split = max(1, int(len(daily)*0.8))
                train, test = daily.iloc[:split], daily.iloc[split:]
                if len(test) and train["high_demand"].nunique() > 1:
                    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
                    model.fit(train[features].fillna(0), train["high_demand"])
                    pred = model.predict(test[features].fillna(0))
                    metrics.append({"model":"RandomForestClassifier","accuracy":accuracy_score(test["high_demand"],pred),
                                    "precision":precision_score(test["high_demand"],pred,zero_division=0),
                                    "recall":recall_score(test["high_demand"],pred,zero_division=0),
                                    "f1":f1_score(test["high_demand"],pred,zero_division=0),
                                    "note":"Prototype only; limited dataset time span."})
                    import joblib
                    joblib.dump(model, reports_dir.parent / "models" / "high_demand_job_model.pkl")
                    pd.DataFrame({"feature":features,"importance":model.feature_importances_}).to_csv(reports_dir / "task3_feature_importance.csv", index=False)
    pd.DataFrame(metrics, columns=["model","accuracy","precision","recall","f1","note"]).to_csv(reports_dir / "task3_model_metrics.csv", index=False)

    # Task 4: hourly rates by country
    country = df[df["hourly_rate"].notna() & (df["hourly_rate"] > 0) & df["country"].notna()].copy() if "country" in df else pd.DataFrame()
    if not country.empty:
        country_report = country.groupby("country")["hourly_rate"].agg(
            Job_Postings="count", Average_Hourly_Rate="mean", Median_Hourly_Rate="median",
            Minimum_Hourly_Rate="min", Maximum_Hourly_Rate="max"
        ).reset_index().sort_values("Average_Hourly_Rate", ascending=False)
    else:
        country_report = pd.DataFrame(columns=["country","Job_Postings","Average_Hourly_Rate","Median_Hourly_Rate","Minimum_Hourly_Rate","Maximum_Hourly_Rate"])
    country_report.to_csv(reports_dir / "task4_country_hourly_rate_analysis.csv", index=False)

    # Task 6: monthly trend
    if "month" in df:
        monthly = df.dropna(subset=["month"]).groupby("month").agg(
            job_postings=("title","size"),
            average_hourly_rate=("hourly_rate","mean"),
            countries=("country","nunique")
        ).reset_index()
    else:
        monthly = pd.DataFrame(columns=["month","job_postings","average_hourly_rate","countries"])
    monthly.to_csv(reports_dir / "task6_monthly_trends.csv", index=False)
    cat_counts.sort_values("Total_Postings", ascending=False).to_csv(reports_dir / "job_category_counts.csv", index=False)
