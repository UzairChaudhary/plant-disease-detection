import os
import io
import json
import logging
import traceback

from flask import Flask, request, jsonify
from PIL import Image
from predict import initialize, predict_image, predict_url

# Set the environment variable for Flask
os.environ['FLASK_ENV'] = 'production'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

@app.route('/')
def index():
    return 'Plant Disease Detection Homepage'

@app.route('/image', methods=['POST'])
def predict_image_handler(project=None, publishedName=None):
    try:
        imageData = None
        if 'imageData' in request.files:
            imageData = request.files['imageData']
        elif 'imageData' in request.form:
            imageData = request.form['imageData']
        else:
            imageData = io.BytesIO(request.get_data())
        img = Image.open(imageData)
        results = predict_image(img)
        
        # Add logging to debug the content of results
        logging.info("Prediction results: %s", results)
        
        # Check if predictions exist and are not None
        if results and 'predictions' in results and results['predictions']:
            highest_prediction = max(results['predictions'], key=lambda x: x['probability'])
            highest_tag = highest_prediction['tagName']
            return jsonify({"Detected_disease": highest_tag}), 200
        else:
            return 'No predictions found', 500

    except Exception as e:
        logging.error('EXCEPTION: %s', str(e))
        traceback.print_exc()
        return 'Error processing image', 500

@app.route('/url', methods=['POST'])
def predict_url_handler(project=None, publishedName=None):
    try:
        image_url = json.loads(request.get_data().decode('utf-8'))['url']
        results = predict_url(image_url)
        
        # Add logging to debug the content of results
        logging.info("Prediction results: %s", results)
        
        # Check if predictions exist and are not None
        if results and 'predictions' in results and results['predictions']:
            highest_prediction = max(results['predictions'], key=lambda x: x['probability'])
            highest_tag = highest_prediction['tagName']
            return jsonify({"Detected_disease": highest_tag}), 200
        else:
            return 'No predictions found', 500

    except Exception as e:
        logging.error('Main EXCEPTION: %s', str(e))
        traceback.print_exc()
        return 'Error processing image', 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    initialize()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
