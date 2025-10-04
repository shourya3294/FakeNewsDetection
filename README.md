
# Fake News Detection using NLP

This project implements a machine learning model to classify news articles as **real** or **fake** using Natural Language Processing (NLP) techniques.

---

### Features
- NLP-based feature extraction using **TF-IDF**
- Classification using **Logistic Regression** and **Random Forest**
- Achieved **92% accuracy** on benchmark datasets

---

### Tech Stack
- Python
- Scikit-learn
- NLTK
- TensorFlow

---

### Project Structure
```
Fake-News-Detection-NLP/
├── data/                 # Dataset files
├── src/                  # Code for preprocessing, training, and evaluation
├── models/               # Saved trained models
├── requirements.txt      # Project dependencies
├── README.md             # Project description
└── .gitignore            # Git ignore file
```

---

### Setup Instructions
1. Clone the repository:
   ```
   git clone https://github.com/your-username/Fake-News-Detection-NLP.git
   cd Fake-News-Detection-NLP
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run training:
   ```
   python src/train_model.py
   ```

5. Run evaluation:
   ```
   python src/evaluate_model.py
   ```

---

### Dataset
The dataset used in this project can be downloaded from [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data).

---

### Results
- Accuracy: **92%**
- Best performing model: Logistic Regression

---

### Author
Shourya Sharma  
CSE Student specializing in AI/ML
