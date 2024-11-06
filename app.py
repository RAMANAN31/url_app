from flask import Flask, request, render_template
import joblib
import re

# Load the improved model
model = joblib.load('malicious_url_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Helper function to preprocess URL for consistency
def preprocess_url(url):
    # Strip scheme (http/https) for better consistency in input
    url = re.sub(r'^https?:\/\/', '', url)
    return url

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the URL input from the form
    url = request.form['url']
    # Preprocess the URL for consistency
    url_processed = preprocess_url(url)
    # Predict if the URL is malicious or safe
    prediction = model.predict([url_processed])[0]
    result = "Malicious" if prediction == 1 else "Safe"
    return render_template('index.html', prediction_text=f"The URL is: {result}")

if __name__ == "__main__":
    app.run(debug=True)
