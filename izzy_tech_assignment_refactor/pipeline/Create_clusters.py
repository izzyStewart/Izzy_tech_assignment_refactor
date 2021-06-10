# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Import libraries needed
import izzy_tech_assignment_refactor
from izzy_tech_assignment_refactor.utils import skills_utils
from izzy_tech_assignment_refactor import config
import pandas as pd
import numpy as np
import json
import re
import nltk
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
import itertools
import math
from textblob import TextBlob as tb

# %%
tfidf = skills_utils.tfidf
stopwords = skills_utils.stopwords
stemmer = skills_utils.stemmer


# %%
def new_cols_clean(df, cols):
    for col in cols:
        df[col + "_without_stopwords"] = (
            df[col]
            .astype(str)
            .apply(
                lambda x: " ".join(
                    [word for word in x.split() if word not in (stopwords)]
                )
            )
        )
        df[col + "_without_stopwords"] = df[
            col + "_without_stopwords"
        ].str.lower()  # Lowercase column
        df[col + "_without_stopwords"] = df[col + "_without_stopwords"].replace(
            "\n", " ", regex=True
        )
    df["text_col"] = (
        df[cols[0] + "_without_stopwords"]
        + " "
        + df[cols[1] + "_without_stopwords"]
        + " "
        + df[cols[2] + "_without_stopwords"]
    ).astype(str)


# %%
# Function to tokenise and stem text column also removing characters and numbers
def token_stem(text):
    tokens = [
        word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)
    ]
    filtered_tokens = []
    for token in tokens:
        if re.search("[a-zA-Z]", token):
            filtered_tokens.append(token)
    stemmed = [stemmer.stem(t) for t in filtered_tokens]
    return stemmed


# %%
# Adding tf-idf vectorizer with custom parameters
tfidf_vectorizer = TfidfVectorizer(
    max_df=config["tfidf_parameters"]["max_df"],
    max_features=config["tfidf_parameters"]["max_features"],
    min_df=config["tfidf_parameters"]["min_df"],
    use_idf=config["tfidf_parameters"]["use_idf"],
    tokenizer=token_stem,
    ngram_range=(
        config["tfidf_parameters"]["ngram_range_start"],
        config["tfidf_parameters"]["ngram_range_end"],
    ),
    stop_words="english",
)


# %%
def assign_cluster(tfidf_matrix, df):
    # Computing the distance between skills using cosine similarity
    dist = 1 - cosine_similarity(tfidf_matrix)
    linkage_matrix = ward(
        dist
    )  # Define the linkage_matrix using ward clustering pre-computed distances
    # Using fcluster, add cluster ID for varying heights in hierarcial cluster
    for height in range(1, config["number_of_clusters"] + 1):
        df["C height " + str(height)] = pd.Series(
            fcluster(linkage_matrix, height, criterion="distance")
        )
    return linkage_matrix, df
