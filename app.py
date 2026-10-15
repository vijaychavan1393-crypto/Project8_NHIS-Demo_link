from pathlib import Path
import sys
import joblib
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))

app = Flask(__name__, template_folder=str(ROOT / "app" / "templates"), static_folder=str(ROOT / "app" / "static"))

def load_artifacts():
    required = [MODELS / "job_tfidf_vectorizer.pkl", MODELS / "job_vectors.pkl", MODELS / "recommendation_jobs.pkl"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Model files are missing. Run `python run_setup.py` first. Missing: " + ", ".join(missing))
    return joblib.load(required[0]), joblib.load(required[1]), pd.read_pickle(required[2])

try:
    tfidf, job_vectors, jobs = load_artifacts()
    print("Models loaded successfully!")
    print(f"Number of jobs: {len(jobs):,}")
except Exception as exc:
    tfidf = job_vectors = jobs = None
    print(f"Setup required: {exc}")

def make_dashboard():
    charts = []
    df = pd.read_csv(ROOT / "data" / "cleaned_job_market_data.csv", low_memory=False)
    if "published_date" in df:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    total_jobs = len(df)
    categories = int(df["job_category"].nunique()) if "job_category" in df else 0
    countries = int(df["country"].nunique()) if "country" in df else 0
    avg_rate = df["hourly_rate"].dropna().mean() if "hourly_rate" in df else float("nan")
    if "published_date" in df and df["published_date"].notna().any():
        monthly = df.dropna(subset=["published_date"]).assign(month=lambda x: x["published_date"].dt.to_period("M").astype(str)).groupby("month").size().reset_index(name="Job postings")
        fig = px.line(monthly, x="month", y="Job postings", markers=True, title="Monthly Job Postings")
        charts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    if "job_category" in df:
        top = df["job_category"].value_counts().head(15).rename_axis("Category").reset_index(name="Postings")
        fig = px.bar(top, x="Category", y="Postings", title="Most Common Job Categories")
        charts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    if {"country","hourly_rate"}.issubset(df.columns):
        rates = df.dropna(subset=["country","hourly_rate"])
        rates = rates[rates["hourly_rate"] > 0].groupby("country")["hourly_rate"].agg(["mean","count"]).reset_index()
        rates = rates[rates["count"] >= 5].sort_values("mean", ascending=False).head(15)
        if not rates.empty:
            fig = px.bar(rates, x="country", y="mean", title="Average Hourly Rate by Country (min. 5 jobs)", labels={"mean":"Average hourly rate"})
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    if "job_category" in df and "published_date" in df:
        trend = df.dropna(subset=["published_date"]).assign(month=lambda x: x["published_date"].dt.to_period("M").astype(str)).groupby(["month","job_category"]).size().reset_index(name="Postings")
        if not trend.empty:
            fig = px.line(trend, x="month", y="Postings", color="job_category", markers=True, title="Category Trends Over Time")
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    return {"total_jobs":f"{total_jobs:,}", "categories":categories, "countries":countries,
            "avg_rate":(f"{avg_rate:.2f}" if pd.notna(avg_rate) else "N/A"), "charts":charts}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", results=None, query="", error=None)

@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.form.get("profile", "").strip()
    if not query:
        return render_template("index.html", results=None, query=query, error="Please enter skills or a job profile.")
    if tfidf is None:
        return render_template("index.html", results=None, query=query, error="Run `python run_setup.py` before starting the app.")
    query_vector = tfidf.transform([query.lower()])
    scores = cosine_similarity(query_vector, job_vectors).ravel()
    count = min(10, len(scores))
    indices = scores.argsort()[-count:][::-1]
    results = jobs.iloc[indices].copy()
    results["match"] = (scores[indices] * 100).round(1)
    display_cols = [c for c in ["title","country","is_hourly","hourly_low","hourly_high","budget","published_date","link","match"] if c in results.columns]
    results = results[display_cols].fillna("")
    return render_template("index.html", results=results.to_dict(orient="records"), query=query, error=None)

@app.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        data = make_dashboard()
        return render_template("dashboard.html", **data)
    except Exception as exc:
        return render_template("error.html", message=str(exc)), 500

@app.route("/health", methods=["GET"])
def health():
    return {"status":"ok", "models_loaded":tfidf is not None, "jobs":len(jobs) if jobs is not None else 0}

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
