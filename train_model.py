
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from feature_extraction import extract_features

if __name__ == "__main__":
    df = pd.read_csv("../data/processed_train.csv")
    X = extract_features(df['text'])
    y = df['label']

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X, y)
    pickle.dump(lr_model, open("../models/logistic_regression.pkl", "wb"))

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)
    pickle.dump(rf_model, open("../models/random_forest.pkl", "wb"))
