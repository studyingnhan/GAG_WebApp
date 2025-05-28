import os
import gdown
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

model_path = "resnet50_gender_age_fine_tune_best.keras"
if not os.path.exists(model_path):
    gdown.download(
        "https://drive.google.com/uc?id=1eMc6HrgiPn59pFX9N6R79yKtzzj5CbSu",
        model_path,
        quiet=False
    )
os.makedirs("static/uploaded", exist_ok=True)
app = Flask(__name__)
model = load_model(model_path)

gender_labels = ['Female', 'Male']
age_labels = ['Child', 'Teen', 'Adult', 'Elderly']

def predict(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img).astype("float32")
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    gender_probs = predictions[0][0]
    age_probs = predictions[1][0]

    gender_idx = np.argmax(gender_probs)
    age_idx = np.argmax(age_probs)

    gender_result = f"{gender_labels[gender_idx]} – {gender_probs[gender_idx]*100:.1f}%"
    age_result = f"{age_labels[age_idx]} – {age_probs[age_idx]*100:.1f}%"

    return f"Giới tính: {gender_result} | Nhóm tuổi: {age_result}"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    if request.method == "POST":
        image = request.files["image"]
        if image:
            upload_dir = "static/uploaded"
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, image.filename)
            image.save(image_path)
            result = predict(image_path)
    return render_template("index.html", result=result, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
