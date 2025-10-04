
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def extract_features(texts):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    features = vectorizer.fit_transform(texts)
    with open("../models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    return features
