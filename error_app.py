from flask import Flask, request, jsonify
from predict import initialize, predict_url, predict_image
from PIL import Image
import base64
import io
import json

app = Flask(__name__)

# Azure ML model loader
def init():
    initialize()

# Helper to predict an image encoded as base64
def predict_image_base64(encoded_image):
    if encoded_image.startswith('b\''):
        encoded_image = encoded_image[2:-1]

    decoded_img = base64.b64decode(encoded_image.encode('utf-8'))
    img_buffer  = io.BytesIO(decoded_img)

    image = Image.open(io.BytesIO(decoded_img))
    return predict_image(image)

# Azure ML entry point
def run(json_input):
    try:
        results = None
        input_data = json.loads(json_input)
        url = input_data.get("url", None)
        image = input_data.get("image", None)

        if url:
            results = predict_url(url)
        elif image:
            results = predict_image_base64(image)
        else:
            raise Exception("Invalid input. Expected url or image")
        return results
    except Exception as e:
        return str(e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # print(request)
        # data = request.json
        #result = run(json.dumps(data))
        input_url = '{"url": "https://raw.githubusercontent.com/Microsoft/Cognitive-CustomVision-Windows/master/Samples/Images/Test/test_image.jpg" }'
        result = run(input_url)
        print(result)
        return jsonify(result),200
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    init()
    app.run(debug=True)
