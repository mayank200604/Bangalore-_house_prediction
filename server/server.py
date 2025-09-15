from flask import Flask, request, jsonify, send_from_directory
import sys
import os

# Add model folder to Python path so we can import util
sys.path.append(os.path.abspath('../model'))

from util import predict_price, get_all_locations  # Import the helper function from util.py

app = Flask(__name__, static_folder='../client', static_url_path='')

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        total_sqft = float(data.get('total_squareft', 0))
        bhk = int(data.get('bhk', 0))
        bath = int(data.get('bath', 0))
        location = data.get('location', '').strip()

        predicted_price = predict_price(total_sqft, bhk, bath, location)

        return jsonify({'predicted_price_lakhs': predicted_price})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    


@app.route('/locations')
def get_locations():
    return jsonify({'locations': get_all_locations()})


if __name__ == '__main__':
    print("🚀 Starting Flask server at http://127.0.0.1:5000/")
    app.run(debug=True)
