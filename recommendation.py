import pandas as pd
import numpy as np
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def recommend_jobs(
    user_profile,
    df,
    tfidf,
    job_vectors,
    top_n=10
):

    user_profile = clean_text(
        user_profile
    )

    user_vector = tfidf.transform(
        [user_profile]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        job_vectors
    ).flatten()

    top_indices = (
        similarity_scores
        .argsort()[-top_n:][::-1]
    )

    recommendations = df.iloc[
        top_indices
    ].copy()

    recommendations[
        "similarity_score"
    ] = similarity_scores[top_indices]

    recommendations[
        "match_percentage"
    ] = (
        recommendations[
            "similarity_score"
        ] * 100
    ).round(2)

    return recommendations