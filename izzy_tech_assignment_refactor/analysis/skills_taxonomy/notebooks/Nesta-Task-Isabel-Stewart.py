# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
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
# Import libraries needed and nltk downloads
import izzy_tech_assignment_refactor
from izzy_tech_assignment_refactor import config
import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
import re
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
import itertools
import math
from textblob import TextBlob as tb
from wordcloud import WordCloud

# %%
nltk.download("stopwords")
nltk.download("punkt")

# %% [markdown]
# ### Getters

# %%
# Open and read in file
skills = pd.read_csv(
    f"{izzy_tech_assignment_refactor.PROJECT_DIR}/inputs/data/skills_en.csv",
    usecols=["skillType", "reuseLevel", "preferredLabel", "altLabels", "description"],
)


# %% [markdown]
# ### Pipeline

# %%
def new_cols_clean(df, cols):
    for col in cols:
        df[col + "_without_stopwords"] = (
            df[col]
            .astype(str)
            .apply(
                lambda x: " ".join([word for word in x.split() if word not in (stop)])
            )
        )
        df[col + "_without_stopwords"] = df[
            col + "_without_stopwords"
        ].str.lower()  # Lowercase column
        df[col + "_without_stopwords"] = df[col + "_without_stopwords"].replace(
            "\n", " ", regex=True
        )


# %%
new_cols_clean(skills, ["preferredLabel", "description", "altLabels"])

# %%
skills["text_col"] = (
    skills["preferredLabel_without_stopwords"]
    + " "
    + skills["description_without_stopwords"]
    + " "
    + skills["altLabels_without_stopwords"]
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


# %% [markdown]
# Note: add below hyper-paramters to config

# %%
# Adding tf-idf vectorizer with custom parameters
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.008,
    max_features=200000,
    min_df=0.002,
    use_idf=True,
    tokenizer=token_stem,
    ngram_range=(1, 3),
)

# Fitting above vectorizer on text column for skills datafram
tfidf_matrix = tfidf_vectorizer.fit_transform(skills.text_col)

# %%
# Computing the distance between skills using cosine similarity
dist = 1 - cosine_similarity(tfidf_matrix)
linkage_matrix = ward(
    dist
)  # Define the linkage_matrix using ward clustering pre-computed distances

# %%
# Using fcluster, add cluster ID for varying heights in hierarcial cluster
for height in range(1, 21):
    skills["C height " + str(height)] = pd.Series(
        fcluster(linkage_matrix, height, criterion="distance")
    )

# %%
skills.drop(
    [
        "preferredLabel_without_stopwords",
        "description_without_stopwords",
        "altLabels_without_stopwords",
    ],
    axis=1,
    inplace=True,
)

# %% [markdown]
# Rethink name and change to outputs file

# %%
skills.to_excel("skills-processed.xlsx", index=False)

# %%
skills["text_col"] = skills["text_col"].fillna("")
skills["text_col"] = skills["text_col"].astype(str)  # Change column type to string


# %%
# Create a corpus from all clusters in specified height column
def get_corpus(col):
    doc_list = []
    for i in skills[col].unique():
        document = " ".join(skills.loc[skills[col] == i]["text_col"].tolist())
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
def label_clusters(feature_names, col, new_col):
    skills[new_col] = ""
    for i in skills[col].unique():
        new_doc = " ".join(skills.loc[skills[col] == i]["text_col"].tolist())
        new_doc = [new_doc]
        words = tfidf.transform(new_doc)
        skills.loc[skills[col] == i, new_col] = [
            get_top_tf_idf_words(feature_names, word, 1) for word in words
        ][0][0]


# %% [markdown]
# Change below to be a function

# %%
# Loop through height columns applying above functions
for i in range(1, 21):
    feature_names = get_corpus("C height " + str(i))
    label_clusters(feature_names, "C height " + str(i), "C label " + str(i))

# %%
skills.drop(["text_col"], axis=1, inplace=True)  # Remove un-needed columns

# %% [markdown]
# Rethink name and change to outputs file

# %%
skills.to_excel("Skills_taxonomy.xlsx", index=False)  # Save results to excel

# %% [markdown]
# ### Utils

# %%
tfidf = TfidfVectorizer(stop_words="english")

# %%
# Adding stopwords and stemmer
stopwords = nltk.corpus.stopwords.words("english")
stemmer = SnowballStemmer("english")

# %% [markdown]
# ### Analysis

# %%
skills.agg(["count", "nunique"])  # Count and number of unique values per column

# %%
skills.skillType.value_counts().sort_values().plot(kind="barh")  # Plot bar chart
plt.xlabel("Count")  # x-axis title
plt.ylabel("Skill type")  # y-axis title
plt.title("Count of skill type")  # Title
plt.show()  # Show

# %%
skills.reuseLevel.value_counts().sort_values().plot(kind="barh")  # Plot bar chart
plt.xlabel("Count")  # x-axis title
plt.ylabel("Type")  # y-axis title
plt.title("Count of reuse level")  # Title
plt.show()  # Show

# %%
# Get word count from cleaned columns
word_val_count = (
    skills["Label_without_stopwords"]
    .str.split(expand=True)
    .stack()
    .value_counts()
    .rename_axis("Word")
    .reset_index(name="Count")
)
word_val_count_desc = (
    skills["Desc_without_stopwords"]
    .str.split(expand=True)
    .stack()
    .value_counts()
    .rename_axis("Word")
    .reset_index(name="Count")
)

# %%
word_val_count.head(3)  # Looking at top 3

# %%
word_val_count_desc.head(3)  # Top 3

# %%
# Plot as a bar chart
word_val_count.head(20).plot("Word", "Count", kind="bar")
plt.title("Top 20 most common words in skills label")  # Title
plt.show()

# %%
# Word cloud of cleaned skill label column
text = " ".join(skills["Label_without_stopwords"])
wordcloud = WordCloud(width=1600, height=800).generate(text)
plt.figure(figsize=(20, 10), facecolor="k")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# %%
# Word cloud of cleaned skill description column
text = " ".join(skills["Desc_without_stopwords"])
wordcloud = WordCloud(width=1600, height=800).generate(text)
plt.figure(figsize=(20, 10), facecolor="k")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# %%
# Produce and plot dendrogram
plt.figure(figsize=(25, 10))  # Figure size
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("distance")
dendrogram(
    linkage_matrix,
    leaf_rotation=90.0,  # Rotate x axis labels
    leaf_font_size=8.0,  # Font size
)
plt.show()  # Show plot

# %%
# Plotting last 4 clusters in dendogram
dendrogram(linkage_matrix, truncate_mode="lastp", p=4)
plt.show()

# method 2: level is equal to two
dendrogram(linkage_matrix, truncate_mode="level", p=2)
plt.show()

# %%
# Looking at the text label for programming example - cluster height 2
skills.loc[skills["C height 2"] == 946]["C label 2"].head(1)

# %%
# Height 2 see skills for programming example
skills.loc[skills["C height 2"] == 946]["preferredLabel"]

# %%
# Looking at the text label for programming example - cluster height 4
skills.loc[skills["C height 4"] == 387]["C label 4"].head(1)

# %%
# Height 4 see skills for programming example
skills.loc[skills["C height 4"] == 387]["preferredLabel"]

# %%
# Looking at the text label for programming example - cluster height 9
skills.loc[skills["C height 9"] == 72]["C label 9"].head(1)

# %%
# Height 9 see skills for programming example
skills.loc[skills["C height 9"] == 72]["preferredLabel"]
