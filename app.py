import os
import subprocess
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

def download_keras_model():
    url = "https://github.com/studyingnhan/GAG_WebApp/releases/download/GAG/resnet50_gender_age_fine_tune_best.keras"
    output = "resnet50_gender_age_fine_tune_best.keras"

    print("Downloading model from GitHub Releases...")
    exit_code = subprocess.call(f"curl -L '{url}' -o {output}", shell=True)

    if exit_code != 0 or not os.path.exists(output) or os.path.getsize(output) < 1000000:
        raise RuntimeError("Failed to download model from GitHub Releases.")
    print("Model downloaded successfully!")

model_path = "resnet50_gender_age_fine_tune_best.keras"

if not os.path.exists(model_path):
    download_keras_model()

if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000: 
    model = load_model(model_path)
else:
    raise RuntimeError("Model file is invalid or incomplete.")

app = Flask(__name__)

gender_labels = ['Female', 'Male']
age_labels = ['Adult', 'Child', 'Elderly', 'Teen']

def predict(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img).astype("float32")
    img_array = resnet50.preprocess_input(img_array.copy())
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    gender_probs = predictions[0][0]
    age_probs = predictions[1][0]

    gender_idx = np.argmax(gender_probs)
    age_idx = np.argmax(age_probs)

    gender_result = f"{gender_labels[gender_idx]} – {gender_probs[gender_idx]*100:.1f}%"
    age_result = f"{age_labels[age_idx]} – {age_probs[age_idx]*100:.1f}%"

    return f"Giới tính: {gender_result} | Nhóm tuổi: {age_result}"

os.makedirs("static/uploaded", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    if request.method == "POST":
        image = request.files["image"]
        if image:
            filename = secure_filename(image.filename)
            image_path = os.path.join("static/uploaded", filename)
            image.save(image_path)
            result = predict(image_path)
    return render_template("index.html", result=result, image_path=image_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
