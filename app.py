from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize global variables
vectorizer = None
lr_model = None
rf_model = None
lr_accuracy = 0
rf_accuracy = 0

def init_models():
    global vectorizer, lr_model, rf_model, lr_accuracy, rf_accuracy
    
    try:
        # Check if required files exist
        required_files = [
            '100k Human vs AI text.csv',
            'logistic_regression_model.pkl',
            'random_forest_model.pkl'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"Required file {file} not found")
                return False
        
        # Load and prepare the dataset
        logger.info("Loading dataset...")
        try:
            # Read only first 1000 rows for initial testing
            df = pd.read_csv('100k Human vs AI text.csv', encoding='utf-8', nrows=1000)
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            
            if 'text' not in df.columns or 'generated' not in df.columns:
                logger.error("Required columns 'text' and 'generated' not found in dataset")
                return False
            
            # Prepare the data
            X = df['text']
            y = df['generated'].astype(int)
            
            # Initialize and fit the TF-IDF vectorizer
            logger.info("Initializing TF-IDF vectorizer...")
            vectorizer = TfidfVectorizer(max_features=1000)
            X_vectorized = vectorizer.fit_transform(X)
            
            # Initialize and train models
            logger.info("Training models...")
            lr_model = LogisticRegression(max_iter=1000)
            rf_model = RandomForestClassifier(n_estimators=10)
            
            lr_model.fit(X_vectorized, y)
            rf_model.fit(X_vectorized, y)
            
            # Save the trained models
            logger.info("Saving trained models...")
            with open('logistic_regression_model.pkl', 'wb') as f:
                pickle.dump(lr_model, f)
            with open('random_forest_model.pkl', 'wb') as f:
                pickle.dump(rf_model, f)
            
            # Calculate accuracies
            logger.info("Calculating model accuracies...")
            lr_accuracy = round(lr_model.score(X_vectorized, y) * 100, 2)
            rf_accuracy = round(rf_model.score(X_vectorized, y) * 100, 2)
            
            logger.info("Models initialized successfully")
            logger.info(f"Logistic Regression Accuracy: {lr_accuracy}%")
            logger.info(f"Random Forest Accuracy: {rf_accuracy}%")
            return True
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if vectorizer is None or lr_model is None or rf_model is None:
        return jsonify({'error': 'Models not properly initialized'}), 500
        
    try:
        data = request.get_json()
        text = data['text']
        
        text_vectorized = vectorizer.transform([text])
        
        lr_prediction = 'AI-generated' if lr_model.predict(text_vectorized)[0] == 1 else 'Human-written'
        rf_prediction = 'AI-generated' if rf_model.predict(text_vectorized)[0] == 1 else 'Human-written'
        
        return jsonify({
            'lr_prediction': lr_prediction,
            'lr_accuracy': lr_accuracy,
            'rf_prediction': rf_prediction,
            'rf_accuracy': rf_accuracy
        })
        
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if init_models():
        logger.info("Starting Flask server...")
        app.run(debug=True)
    else:
        logger.error("Failed to initialize models. Server not started.")