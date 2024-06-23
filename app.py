from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved AdaBoost model from the file
model_filename = 'adb_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_adb = pickle.load(file)

# Function to preprocess user input
def preprocess_input(json_data):
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

    return np.array([list(json_data.values())]).astype(np.float64)

# Endpoint to render index.html
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        print("User Input:", json_data)  # Print user input

        user_data = preprocess_input(json_data)
        prediction = loaded_adb.predict(user_data)
        prediction_label = 'success' if prediction[0] == 1 else 'fail'

        print("Prediction Result:", prediction_label)  # Print prediction result

        return jsonify({'prediction': prediction_label}), 200
    except Exception as e:
        print("Error:", e)  # Print error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
