import numpy as np
import pandas as pd

data = pd.read_csv("ted_talks.csv")
print(data.head())

from sklearn.feature_extraction import text

ted_talks = data["transcript"].tolist()
bi_tfidf = text.TfidfVectorizer(input=ted_talks, stop_words="english", ngram_range=(1, 2))
bi_matrix = bi_tfidf.fit_transform(ted_talks)

uni_tfidf = text.TfidfVectorizer(input=ted_talks, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(ted_talks)

from sklearn.metrics.pairwise import cosine_similarity

bi_sim = cosine_similarity(bi_matrix)
uni_sim = cosine_similarity(uni_matrix)


def recommend_ted_talks(x):
    return ".".join(data["title"].loc[x.argsort()[-5:-1]])


data["ted_talks_uni"] = [recommend_ted_talks(x) for x in uni_sim]
data["ted_talks_bi"] = [recommend_ted_talks(x) for x in bi_sim]
print(data['ted_talks_uni'].str.replace("_", " ").str.upper().str.strip().str.split("\n")[1])
