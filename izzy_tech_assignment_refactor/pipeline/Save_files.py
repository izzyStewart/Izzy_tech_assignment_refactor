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
# Import libraries needed
import izzy_tech_assignment_refactor
from izzy_tech_assignment_refactor.getters import get_files as gf
from izzy_tech_assignment_refactor.utils import skills_utils
from izzy_tech_assignment_refactor.pipeline import Create_clusters as cc
from izzy_tech_assignment_refactor.pipeline import Create_text_labels as ctl
from izzy_tech_assignment_refactor import config
from numpy import savetxt
import pandas as pd

# %%
project_directory = izzy_tech_assignment_refactor.PROJECT_DIR

# %%
skills = gf.read_input_file(project_directory)

# %%
# Calling create clusters module functions
cc.new_cols_clean(skills, ["preferredLabel", "altLabels", "description"])
tfidf_matrix = cc.tfidf_vectorizer.fit_transform(skills.text_col)
linkage_matrix, skills = cc.assign_cluster(tfidf_matrix, skills)

# %%
# save array to csv file
savetxt(
    f"{project_directory}/outputs/data/linkage_matrix.csv",
    linkage_matrix,
    delimiter=",",
)

# %%
skills_utils.excel_save(
    skills, f"{project_directory}/outputs/data/skills_taxonomy_clusters.xlsx", []
)

# %%
# Calling create text labels module functions
ctl.clean_text(skills, "text_col")
ctl.apply_labels(skills)

# %%
skills_utils.excel_save(
    skills,
    f"{project_directory}/outputs/data/skills_taxonomy_text_labels.xlsx",
    [
        "preferredLabel_without_stopwords",
        "description_without_stopwords",
        "altLabels_without_stopwords",
    ],
)
