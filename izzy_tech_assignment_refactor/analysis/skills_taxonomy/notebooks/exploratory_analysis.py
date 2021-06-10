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
from izzy_tech_assignment_refactor.getters import get_files as gf
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram, fcluster

# %%
project_directory = izzy_tech_assignment_refactor.PROJECT_DIR

# %%
skills = gf.read_output_files(project_directory, gf.skills_clusters)

# %%
skills.iloc[:, :6].agg(["count", "nunique"])

# %%
skills.skillType.value_counts().sort_values().plot(kind="barh")  # Plot bar chart
plt.xlabel("Count")  # x-axis title
plt.ylabel("Skill type")  # y-axis title
plt.title("Count of skill type")  # Title
plt.show()
plt.savefig(f"{project_directory}/outputs/figures/Count-of-skill-type.png")

# %%
skills.reuseLevel.value_counts().sort_values().plot(kind="barh")  # Plot bar chart
plt.xlabel("Count")  # x-axis title
plt.ylabel("Type")  # y-axis title
plt.title("Count of reuse level")  # Title
plt.show()  # Show
plt.savefig(f"{project_directory}/outputs/figures/Count-of-reuse-level.png")

# %%
# Get word count from cleaned columns
word_val_count = (
    skills["preferredLabel_without_stopwords"]
    .str.split(expand=True)
    .stack()
    .value_counts()
    .rename_axis("Word")
    .reset_index(name="Count")
)
word_val_count_desc = (
    skills["description_without_stopwords"]
    .str.split(expand=True)
    .stack()
    .value_counts()
    .rename_axis("Word")
    .reset_index(name="Count")
)

# %%
# Plot as a bar chart
word_val_count.head(20).plot("Word", "Count", kind="bar")
plt.title("Top 20 most common words in skills label")  # Title
plt.show()
plt.savefig(
    f"{project_directory}/outputs/figures/Top-20-most-common-words-in-skills-label.png"
)

# %%
# Plot as a bar chart
word_val_count_desc.head(20).plot("Word", "Count", kind="bar")
plt.title("Top 20 most common words in description label")  # Title
plt.show()
plt.savefig(
    f"{project_directory}/outputs/figures/Top-20-most-common-words-in-description-label.png"
)

# %%
# Word cloud of cleaned skill label column
text = " ".join(skills["preferredLabel_without_stopwords"])
wordcloud = WordCloud(width=1600, height=800).generate(text)
plt.figure(figsize=(20, 10), facecolor="k")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig(f"{project_directory}/outputs/figures/Wordcloud-skills-label.png")

# %%
# Word cloud of cleaned skill description column
text = " ".join(skills["description_without_stopwords"])
wordcloud = WordCloud(width=1600, height=800).generate(text)
plt.figure(figsize=(20, 10), facecolor="k")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig(f"{project_directory}/outputs/figures/Wordcloud-descriptions-label.png")

# %%
linkage_matrix = gf.load_array(project_directory, "linkage_matrix.csv")

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
plt.savefig(
    f"{project_directory}/outputs/figures/Hierarhical-clustering-dendrogram.png"
)

# %%
# Plotting last 4 clusters in dendogram
dendrogram(linkage_matrix, truncate_mode="lastp", p=4)
plt.show()
plt.savefig(f"{project_directory}/outputs/figures/dendrogram-last-4-clusters.png")

# method 2: level is equal to two
dendrogram(linkage_matrix, truncate_mode="level", p=2)
plt.show()
plt.savefig(f"{project_directory}/outputs/figures/dendrogram-level-2.png")
