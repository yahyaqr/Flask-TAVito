from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved AdaBoost model from the file
model_filename = 'adb_model.pkl'
try:
    with open(model_filename, 'rb') as file:
        loaded_adb = pickle.load(file)
    app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")

# Function to preprocess user input
def preprocess_input(json_data):
    try:
        app.logger.debug(f"Received input data: {json_data}")
        
        # Convert yes/no to 1/0
        for field in ['is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising',
                      'is_games_video', 'is_ecommerce', 'is_biotech', 'is_consulting',
                      'is_other_category', 'has_VC', 'has_angel', 'has_round_A', 'has_round_B',
                      'has_round_C', 'has_round_D', 'is_top_500', 'is_in_big_city', 'has_seed',
                      'is_trend_industry']:
            json_data[field] = 1 if json_data[field].lower() == 'yes' else 0

        # Convert numerical fields to float or int
        numerical_fields = ['age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year',
                            'age_last_milestone_year', 'funding_total_usd', 'avg_participants', 'age']
        for field in numerical_fields:
            json_data[field] = float(json_data[field])

        # Convert other numerical fields to int
        int_fields = ['relationships', 'funding_rounds', 'milestones']
        for field in int_fields:
            json_data[field] = int(json_data[field])

        app.logger.debug(f"Preprocessed input data: {json_data}")
        
        return np.array([list(json_data.values())]).astype(np.float64)
    except Exception as e:
        app.logger.error(f"Error in preprocessing input: {e}")
        raise

# Endpoint to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        app.logger.debug(f"JSON data received: {json_data}")
        
        user_data = preprocess_input(json_data)
        app.logger.debug(f"User data for prediction: {user_data}")
        
        prediction = loaded_adb.predict(user_data)
        prediction_label = 'success' if prediction[0] == 1 else 'fail'
        
        app.logger.info(f"Prediction: {prediction_label}")
        return jsonify({'prediction': prediction_label}), 200
    except Exception as e:
        app.logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
