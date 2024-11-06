from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import joblib
import re

# Feature engineering for URLs
class URLFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({
            'url_length': X.apply(lambda x: len(x)),
            'special_char_count': X.apply(lambda x: len(re.findall(r'[-?&=]', x))),
            'num_subdomains': X.apply(lambda x: len(re.findall(r'\.', x))),
            'has_ip': X.apply(lambda x: 1 if re.search(r'\d+\.\d+\.\d+\.\d+', x) else 0),
            'suspicious_words': X.apply(lambda x: 1 if re.search(r'free|login|offer|click', x.lower()) else 0)
        })

# Load and preprocess dataset
data = pd.read_csv('malicious_phish.csv')
X = data['url']
y = data['type'].apply(lambda x: 1 if x == 'malicious' else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Update pipeline
pipeline = Pipeline([
    ('url_features', URLFeatures()),
    ('tfidf', TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Test the model
y_pred = pipeline.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the improved model
joblib.dump(pipeline, 'malicious_url_model.pkl')
