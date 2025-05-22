from flask import Flask, render_template, request, jsonify
import firebase_admin
from firebase_admin import db, credentials
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'bcdv'  # Secret key for session management

# Initialize Firebase Admin SDK
if not firebase_admin._apps:  # Check if Firebase is not already initialized
    cred = credentials.Certificate("credentials.json")  # Path to Firebase Admin SDK credentials
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://ipdemoday-default-rtdb.asiasoutheast1.firebasedatabase.app/'
    })
else:
    default_app = firebase_admin.get_app()

# Load the trained RandomForestClassifier model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the StandardScaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to preprocess sensor data using the scaler
def preprocess_data(a1, a2, a3, temp):
    input_data = np.array([[a1, a2, a3, temp]])  # Create 2D array for scaler
    input_data_scaled = scaler.transform(input_data)  # Scale the input data
    return input_data_scaled

# Route for the main web interface
@app.route('/')
def index():
    return render_template('index.html')

# Route to fetch the latest sensor data from Firebase
@app.route('/latest_data', methods=['GET'])
def get_latest_data():
    try:
        ref = db.reference('/')
        latest_data = ref.get()
        if latest_data:
            # Extract the latest value for X, Y, Z, and temp
            a1 = list(latest_data['X']['float'].values())[-1]  # X-axis acceleration
            a2 = list(latest_data['Y']['float'].values())[-1]  # Y-axis acceleration
            a3 = list(latest_data['Z']['float'].values())[-1]  # Z-axis acceleration
            temp = list(latest_data['temp']['float'].values())[-1]  # Temperature
            return jsonify({
                'a1': a1,
                'a2': a2,
                'a3': a3,
                'temp': temp
            })
        else:
            return jsonify({'error': 'Acceleration data not found'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Route to predict compressor state based on sensor data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get sensor data from the request
        a1 = float(request.form['a1'])
        a2 = float(request.form['a2'])
        a3 = float(request.form['a3'])
        temp = float(request.form['temp'])
        print("Request Form Data:", request.form)  # Debug print

        # Preprocess the input data
        input_data_scaled = preprocess_data(a1, a2, a3, temp)

        # Make prediction
        prediction_proba = model.predict_proba(input_data_scaled)
        predicted_class = model.predict(input_data_scaled)[0]  # Get predicted class (0, 1, 2, 3)

        # Map class numbers to labels
        class_label = {0: 'Normal', 1: 'Imbalance', 2: 'Horizontal Misalignment', 3: 'Vertical Misalignment'}[predicted_class]
        class_proba = prediction_proba[0][predicted_class]  # Probability of predicted class

        # Generate error signal for non-normal states
        error_signal = None
        if class_label != 'Normal':
            error_signal = f"Possible {class_label} detected. Take action!"

        return jsonify({
            'class': class_label,
            'probability': float(class_proba),  # Convert to float for JSON serialization
            'error_signal': error_signal
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)