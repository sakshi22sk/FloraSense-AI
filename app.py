from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load model
model = load_model("model.h5")

# Class labels (IMPORTANT: same order as training)
class_names = ['daisy','dandelion','rose','sunflower','tulip']

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence,2),
        "image": filepath
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
