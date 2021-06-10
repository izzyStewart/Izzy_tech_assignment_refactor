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
import pandas as pd
from numpy import loadtxt


# %%
# Open and read in input csv file
def read_input_file(project_directory):
    file_path = f"{project_directory}/inputs/data/skills_en.csv"
    cols_list = [
        "skillType",
        "reuseLevel",
        "preferredLabel",
        "altLabels",
        "description",
    ]
    df = pd.read_csv(file_path, usecols=cols_list)
    return df


# %%
# Open and read in output excel file
def read_output_files(project_directory, file):
    file_path = f"{project_directory}/outputs/data/"
    df = pd.read_excel(file_path + file)
    return df


# %%
skills_clusters = "skills_taxonomy_clusters.xlsx"
skills_text_labels = "skills_taxonomy_text_labels.xlsx"


# %%
# load numpy array from csv file
def load_array(project_directory, file):
    array = loadtxt(f"{project_directory}/outputs/data/" + file, delimiter=",")
    return array
