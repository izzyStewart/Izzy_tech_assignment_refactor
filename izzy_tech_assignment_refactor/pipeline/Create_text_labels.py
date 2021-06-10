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
def clean_text(df, col):
    df[col] = df[col].fillna("")
    df[col] = df[col].astype(str)


# %%
# Create a corpus from all clusters in specified height column
def get_corpus(df, col, text_col):
    doc_list = []
    for i in df[col].unique():
        document = " ".join(df.loc[df[col] == i][text_col].tolist())
        doc_list.append(document)
    X = tfidf.fit_transform(doc_list)
    feature_names = np.array(tfidf.get_feature_names())
    return feature_names


# %%
# Get top tf-idf words using feature names and words from specified cluster
def get_top_tf_idf_words(feature_names, response, top_n=1):
    sorted_nzs = np.argsort(response.data)[: -(top_n + 1) : -1]
    return feature_names[response.indices[sorted_nzs]]


# %%
# Using the above fucntions with tfidf create a new column with text labels for each cluster in the specified height
def label_clusters(df, feature_names, col, new_col, text_col):
    df[new_col] = ""
    for i in df[col].unique():
        new_doc = " ".join(df.loc[df[col] == i][text_col].tolist())
        new_doc = [new_doc]
        words = tfidf.transform(new_doc)
        df.loc[df[col] == i, new_col] = [
            get_top_tf_idf_words(feature_names, word, 1) for word in words
        ][0][0]


# %%
def apply_labels(df):
    # Loop through height columns applying above functions
    for i in range(
        1, config["number_of_clusters"] + 1
    ):  # Set for number of desired clusters / labels
        feature_names = get_corpus(df, "C height " + str(i), "text_col")
        label_clusters(
            df, feature_names, "C height " + str(i), "C label " + str(i), "text_col"
        )
