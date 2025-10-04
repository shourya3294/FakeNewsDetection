
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

def load_data(path):
    return pd.read_csv(path)

def preprocess(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_data(df):
    df['text'] = df['text'].apply(preprocess)
    return df

if __name__ == "__main__":
    df = load_data("../data/train.csv")
    df = preprocess_data(df)
    df.to_csv("../data/processed_train.csv", index=False)
