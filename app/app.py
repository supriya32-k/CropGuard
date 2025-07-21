# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from markupsafe import Markup
import re
import tempfile
import os
from flask import send_from_directory
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "https://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name.lstrip()
    print(city_name)
    print(complete_url)
    # https://api.openweathermap.org/data/2.5/weather?appid=7e565c5c6afa29d0a90a8d814342305d&q=Bhimavaram
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img)).convert('RGB')
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'CropGuard - Home'
    return render_template('index.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'CropGuard - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'CropGuard - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = float(request.form['nitrogen'])
    P = float(request.form['phosphorous'])
    K = float(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', 
        recommendation=response, 
        title=title, 
        crop_name=crop_name, 
        N=N, P=P, K=K)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'CropGuard - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            # Save uploaded image to a temp file for display
            temp_dir = os.path.join(app.root_path, 'static', 'temp_uploads')
            os.makedirs(temp_dir, exist_ok=True)
            temp_filename = next(tempfile._get_candidate_names()) + '.jpg'
            temp_path = os.path.join(temp_dir, temp_filename)
            with open(temp_path, 'wb') as f:
                f.write(img)
            image_url = url_for('static', filename=f'temp_uploads/{temp_filename}')

            prediction_label = predict_image(img)
            disease_info_html = disease_dic.get(prediction_label, '')

            crop_name = 'N/A'
            disease_name = 'N/A'
            cause = 'N/A'

            # Extract crop name
            crop_match = re.search(r'<b>\s*Crop\s*</b>\s*:\s*([^<\n]+)', disease_info_html, re.IGNORECASE)
            if crop_match:
                crop_name = crop_match.group(1).strip()

            # Extract disease name
            disease_match = re.search(r'Disease\s*:\s*([^<\n]+)', disease_info_html, re.IGNORECASE)
            if disease_match:
                disease_name = disease_match.group(1).strip()

            # Extract cause (robust, multiline, up to 'How to prevent' or end)
            cause_match = re.search(r'Cause of disease:?\s*</?b>?\s*:?\s*(.*?)(<br\s*/?>\s*How to prevent|How to prevent|$)', disease_info_html, re.IGNORECASE | re.DOTALL)
            if cause_match:
                cause = cause_match.group(1).strip()
                cause = re.sub(r'^(<br\s*/?>|r/>)\s*', '', cause)
                cause = re.sub(r'<br\s*/?>', ' ', cause)
                cause = re.sub(r'<.*?>', '', cause)
                # Extract all numbered points
                points = re.findall(r'\d+\.\s*[^\d]+(?=\d+\.|$)', cause)
                if points:
                    cause = '\n'.join(point.strip() for point in points)
                else:
                    cause = re.sub(r'\n+', '\n', cause)
                    cause = '\n'.join(line.strip() for line in cause.split('\n'))

            # Extract prevention/cure section
            prevention = 'N/A'
            prevention_match = re.search(r'How to prevent/cure the disease\s*<br\s*/?>\s*(.*)', disease_info_html, re.IGNORECASE | re.DOTALL)
            if prevention_match:
                prevention = prevention_match.group(1).strip()
                prevention = re.sub(r'<.*?>', '', prevention)
                prevention = re.sub(r'<br\s*/?>', ' ', prevention)
                # Extract all numbered points
                points = re.findall(r'\d+\.\s*[^\d]+(?=\d+\.|$)', prevention)
                if points:
                    prevention = '\n'.join(point.strip() for point in points)
                else:
                    prevention = re.sub(r'\n+', '\n', prevention)
                    prevention = '\n'.join(line.strip() for line in prevention.split('\n'))

            prediction_structured = {
                'crop_name': crop_name,
                'disease_name': disease_name,
                'cause': cause,
                'prevention': prevention
            }
            return render_template('disease-result.html', prediction=prediction_structured, title=title, image_url=image_url)
        except Exception as e:
            print("Disease prediction error:", e)
            import traceback; traceback.print_exc()
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
