"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib
import re
import numpy as np


# Feature engineering for URLs
class URLFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).fillna('')  # Handle missing values (NaNs)

        return pd.DataFrame({
            'url_length': X.apply(lambda x: len(x)),
            'special_char_count': X.apply(lambda x: len(re.findall(r'[-?&=]', x))),
            'num_subdomains': X.apply(lambda x: len(re.findall(r'\.', x))),
            'has_ip': X.apply(lambda x: 1 if re.search(r'\d+\.\d+\.\d+\.\d+', x) else 0),
            'suspicious_words': X.apply(lambda x: 1 if re.search(r'free|login|offer|click', x.lower()) else 0)
        })


# Load and preprocess dataset
data = pd.read_csv('balanced_malicious_phish.csv')
data['type'] = data['type'].apply(lambda x: 1 if x == 'malicious' else 0)  # Mapping 'malicious' to 1 and 'benign' to 0
X = data['url']
y = data['type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure X_train and X_test are DataFrames with a column named 'url'
X_train_df = pd.DataFrame(X_train, columns=['url'])
X_test_df = pd.DataFrame(X_test, columns=['url'])

# Define the transformation for URL features and TF-IDF
url_features = URLFeatures()

# Create the pipeline
pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('url_features', url_features, 'url'),  # Apply URLFeatures to 'url'
        ('tfidf', TfidfVectorizer(max_features=5000), 'url')  # Apply TF-IDF on 'url'
    ])),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest Classifier
])

# Train the model
pipeline.fit(X_train_df, y_train)

# Test the model
y_pred = pipeline.predict(X_test_df)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the improved model
joblib.dump(pipeline, 'malicious_url_model.pkl')
"""



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib
import re
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Feature engineering for URLs
class URLFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).fillna('')
        return pd.DataFrame({
            'url_length': X.apply(lambda x: len(x)),
            'special_char_count': X.apply(lambda x: len(re.findall(r'[-?&=]', x))),
            'num_subdomains': X.apply(lambda x: len(re.findall(r'\.', x))),
            'has_ip': X.apply(lambda x: 1 if re.search(r'\d+\.\d+\.\d+\.\d+', x) else 0),
            'suspicious_words': X.apply(lambda x: 1 if re.search(r'free|login|offer|click', x.lower()) else 0)
        })

# Load dataset
data = pd.read_csv('balanced_malicious_phish.csv')
data['type'] = data['type'].apply(lambda x: 0 if x == 'benign' else 1)
X = data['url']
y = data['type']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create the pipeline for feature extraction
url_features = URLFeatures()
pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('url_features', url_features, 'url'),
        ('tfidf', TfidfVectorizer(max_features=5000), 'url')
    ]))
])

# Transform the data
X_train_transformed = pipeline.fit_transform(pd.DataFrame(X_train, columns=['url']))
X_test_transformed = pipeline.transform(pd.DataFrame(X_test, columns=['url']))

# Train the model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train_transformed, y_train)

# Save the pipeline and model
joblib.dump(pipeline, 'feature_transform_pipeline.pkl')
joblib.dump(xgb_model, 'malicious_url_model.pkl')
print("Model and pipeline saved successfully.")


