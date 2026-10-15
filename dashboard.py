import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re


# ============================================
# PATH
# ============================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_job_market_data_v2.csv"
)


# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================
# FIND DATE COLUMN
# ============================================

possible_date_columns = [
    col for col in df.columns
    if any(
        keyword in col.lower()
        for keyword in [
            "date",
            "posted",
            "created",
            "time"
        ]
    )
]

print(
    "Possible date columns:",
    possible_date_columns
)


# IMPORTANT:
# Change this according to your dataset

DATE_COLUMN = possible_date_columns[0]


df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce"
)


# ============================================
# CREATE MONTH COLUMN
# ============================================

df["month"] = (
    df[DATE_COLUMN]
    .dt.to_period("M")
    .astype(str)
)

print(
    df[
        ["month"]
    ].head()
)
monthly_jobs = (
    df.dropna(subset=[DATE_COLUMN])
      .groupby("month")
      .size()
      .reset_index(
          name="Job_Postings"
      )
)

print(monthly_jobs)
fig_monthly_jobs = px.line(
    monthly_jobs,
    x="month",
    y="Job_Postings",
    markers=True,
    title="Monthly Job Posting Trend"
)

fig_monthly_jobs.update_layout(
    xaxis_title="Month",
    yaxis_title="Number of Job Postings"
)
if "job_category" in df.columns:

    monthly_category = (
        df.dropna(
            subset=[
                "month",
                "job_category"
            ]
        )
        .groupby(
            [
                "month",
                "job_category"
            ]
        )
        .size()
        .reset_index(
            name="Job_Postings"
        )
    )

    print(
        monthly_category.head()
    )
    fig_category_trend = px.line(
    monthly_category,
    x="month",
    y="Job_Postings",
    color="job_category",
    markers=True,
    title="Monthly Job Category Trends"
)

fig_category_trend.update_layout(
    xaxis_title="Month",
    yaxis_title="Job Postings"
)
category_summary = (
    df["job_category"]
    .value_counts()
    .reset_index()
)

category_summary.columns = [
    "job_category",
    "Job_Postings"
]

print(category_summary.head(10))
fig_categories = px.bar(
    category_summary.head(10),
    x="Job_Postings",
    y="job_category",
    orientation="h",
    title="Top Job Categories"
)

fig_categories.update_layout(
    yaxis={
        "categoryorder": "total ascending"
    }
)
possible_rate_columns = [
    col for col in df.columns
    if any(
        keyword in col.lower()
        for keyword in [
            "hourly",
            "rate",
            "salary",
            "budget"
        ]
    )
]

print(
    "Possible rate columns:",
    possible_rate_columns
)

RATE_COLUMN = possible_rate_columns[0]
def extract_numeric_rate(value):

    if pd.isna(value):
        return np.nan

    value = str(value)

    value = value.replace(
        ",",
        ""
    )

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        value
    )

    if not numbers:
        return np.nan

    numbers = [
        float(x)
        for x in numbers
    ]

    if len(numbers) >= 2:
        return np.mean(
            numbers[:2]
        )

    return numbers[0]


df["hourly_rate_numeric"] = (
    df[RATE_COLUMN]
    .apply(
        extract_numeric_rate
    )
)
monthly_rate = (
    df.dropna(
        subset=[
            "month",
            "hourly_rate_numeric"
        ]
    )
    .groupby("month")
    ["hourly_rate_numeric"]
    .mean()
    .reset_index(
        name="Average_Hourly_Rate"
    )
)

print(monthly_rate)
fig_rate_trend = px.line(
    monthly_rate,
    x="month",
    y="Average_Hourly_Rate",
    markers=True,
    title="Monthly Average Hourly Rate"
)

fig_rate_trend.update_layout(
    xaxis_title="Month",
    yaxis_title="Average Hourly Rate"
)
possible_country_columns = [
    col for col in df.columns
    if any(
        keyword in col.lower()
        for keyword in [
            "country",
            "client_country",
            "location"
        ]
    )
]

print(
    "Possible country columns:",
    possible_country_columns
)
COUNTRY_COLUMN = possible_country_columns[0]
country_summary = (
    df[COUNTRY_COLUMN]
    .value_counts()
    .reset_index()
)

country_summary.columns = [
    COUNTRY_COLUMN,
    "Job_Postings"
]

print(
    country_summary.head(10)
)
fig_country = px.bar(
    country_summary.head(15),
    x="Job_Postings",
    y=COUNTRY_COLUMN,
    orientation="h",
    title="Top Countries by Job Postings"
)

fig_country.update_layout(
    yaxis={
        "categoryorder":
        "total ascending"
    }
)
total_jobs = len(df)

total_categories = (
    df["job_category"]
    .nunique()
    if "job_category" in df.columns
    else 0
)

total_countries = (
    df[COUNTRY_COLUMN]
    .nunique()
)

average_rate = (
    df["hourly_rate_numeric"]
    .mean()
)
print(
    "Total Jobs:",
    total_jobs
)

print(
    "Total Categories:",
    total_categories
)

print(
    "Total Countries:",
    total_countries
)

print(
    "Average Hourly Rate:",
    round(
        average_rate,
        2
    )
)
DASHBOARD_DIR = os.path.join(
    BASE_DIR,
    "dashboard"
)

os.makedirs(
    DASHBOARD_DIR,
    exist_ok=True
)
fig_monthly_jobs.write_html(
    os.path.join(
        DASHBOARD_DIR,
        "monthly_jobs.html"
    )
)
fig_category_trend.write_html(
    os.path.join(
        DASHBOARD_DIR,
        "category_trend.html"
    )
)
fig_categories.write_html(
    os.path.join(
        DASHBOARD_DIR,
        "top_categories.html"
    )
)
fig_rate_trend.write_html(
    os.path.join(
        DASHBOARD_DIR,
        "rate_trend.html"
    )
)
fig_country.write_html(
    os.path.join(
        DASHBOARD_DIR,
        "country_jobs.html"
    )
)
