# %%
# Import libraries and nltk downloads
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")


# %%
# Setting tfidf vectorizer
tfidf = TfidfVectorizer(stop_words="english")
# Adding stopwords and stemmer
stopwords = stopwords.words("english")
stemmer = SnowballStemmer("english")


# %%
def excel_save(df, filename, cols_to_drop):
    df.drop(cols_to_drop, axis=1, inplace=True)
    df.to_excel(filename, index=False)
