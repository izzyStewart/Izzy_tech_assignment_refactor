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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# %%
project_directory = izzy_tech_assignment_refactor.PROJECT_DIR

# %%
skills = gf.read_output_files(project_directory, gf.skills_text_labels)

# %%
# Word cloud height 2 - skills labels: algorithms
text = " ".join(skills.loc[skills["C height 2"] == 204]["preferredLabel"])
wordcloud = WordCloud(width=1600, height=800).generate(text)
plt.figure(figsize=(10, 5), facecolor="k")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig(
    f"{project_directory}/outputs/figures/Wordcloud-cluster-label-algorthms.png"
)

# %%
# Word cloud height 4 - skills labels: programming
text = " ".join(skills.loc[skills["C height 4"] == 81]["preferredLabel"])
wordcloud = WordCloud(width=1600, height=800).generate(text)
plt.figure(figsize=(10, 5), facecolor="k")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig(
    f"{project_directory}/outputs/figures/Wordcloud-cluster-label-programming.png"
)
