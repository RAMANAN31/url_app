"""
from flask import Flask, request, render_template
import joblib
import re
import pandas as pd
from url_features import URLFeatures  # Import URLFeatures from the external file

# Load the trained model
model = joblib.load('malicious_url_model.pkl')

app = Flask(__name__)


def preprocess_url(url):
    "Preprocess the input URL by removing the protocol."
    return re.sub(r'^https?:\/\/', '', url)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        if url:
            url_processed = preprocess_url(url)

            # Convert the input URL to a pandas DataFrame with column 'url'
            url_df = pd.DataFrame([url_processed], columns=['url'])

            # Make prediction using the model
            prediction = model.predict(url_df)[0]
            result = "Malicious" if prediction == 1 else "Safe"
            return render_template('index.html', prediction_text=f"The URL is: {result}")
        else:
            return render_template('index.html', prediction_text="Please enter a valid URL.")
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
"""

from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

from url_features import URLFeatures
# Load the pipeline and model
pipeline = joblib.load('feature_transform_pipeline.pkl')
model = joblib.load('malicious_url_model.pkl')

app = Flask(__name__)


# Home route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Prepare the input for prediction
        input_df = pd.DataFrame([url], columns=['url'])
        transformed_input = pipeline.transform(input_df)
        prediction = model.predict(transformed_input)

        # Map the prediction to the class name
        result = 'Benign' if prediction[0] == 0 else 'Malicious'
        return render_template('index.html', url=url, result=result)


if __name__ == '__main__':
    app.run(debug=True)
