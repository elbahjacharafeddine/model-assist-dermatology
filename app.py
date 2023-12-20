from flask import Flask, jsonify, request
from model import get_model
from keras.preprocessing.image import img_to_array
from keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/disease/predict": {"origins": "*"}})


efficient = get_model()
efficient.load_weights("efficient.h5")

classe = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}

# Make a dummy prediction on startup for model warmup
dummy_image = np.random.rand(224, 224, 3)
dummy_image = preprocess_input(np.expand_dims(dummy_image, axis=0))
_ = efficient.predict(dummy_image)

@app.route("/disease/predict", methods=["POST"])
@cross_origin()
def diagnostic_maladie():
    print("file:")
    print(request.files)
    # Check if the request contains a file
    if 'image' not in request.files:
        return jsonify({"message": "No image provided in the request"}), 400

    file = request.files['image']

    # Open the image using PIL
    img = Image.open(file)
    # Resize the image to (224, 224)
    img = img.resize((224, 224))
    # Convert the image to an array
    test_image = img_to_array(img)
    test_image = preprocess_input(np.expand_dims(test_image, axis=0))

    # Predict the image
    preds = efficient.predict(test_image)
    preds = preds.tolist()

    max_value = max(preds[0])
    max_index = preds[0].index(max_value)

    # Return the prediction directly
    result = {
        "predicted_disease": classe[max_index],
        "probability": float("{:.2f}".format(max_value * 100)),
        "probabilities": [float("{:.2f}".format(value * 100)) for value in preds[0]]
    }

    return jsonify(result), 200

@app.route("/hello", methods=["GET"])
@cross_origin()
def hello():
    return "Hello world from dermatology app ...."
if __name__ == "__main__":
    app.run(debug=True)
