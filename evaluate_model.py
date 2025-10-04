
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    df = pd.read_csv("../data/processed_test.csv")
    vectorizer = pickle.load(open("../models/tfidf_vectorizer.pkl", "rb"))
    X_test = vectorizer.transform(df['text'])
    y_test = df['label']

    lr_model = pickle.load(open("../models/logistic_regression.pkl", "rb"))
    predictions = lr_model.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, predictions))

    rf_model = pickle.load(open("../models/random_forest.pkl", "rb"))
    predictions = rf_model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, predictions))
