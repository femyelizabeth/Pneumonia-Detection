from flask import Flask, render_template, request, session, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from werkzeug.utils import secure_filename
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set a secret key for session management

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained models
INCEPTION_MODEL_PATH = 'models/inceptionv3.h5'
VGG16_MODEL_PATH = 'models/pneumonia_detection_model.h5'
RESNET50_MODEL_PATH = 'models/resnet50_model.h5'

inception_model = load_model(INCEPTION_MODEL_PATH)
vgg16_model = load_model(VGG16_MODEL_PATH)
resnet50_model = load_model(RESNET50_MODEL_PATH)

# Function to get X-ray type based on class index
def get_xray_type(argument):
    switcher = {
        1: "NORMAL",
        0: "PNEUMONIA",
    }
    return switcher.get(argument, "Invalid X-ray")

def predict_inception(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = inception_model.predict(img_array)
    confidence = prediction[0][0]
    result = "Pneumonia" if confidence > 0.5 else "Normal"
    confidence_percent = round(confidence * 100, 2) if result == "Pneumonia" else round((1 - confidence) * 100, 2)

    return result, confidence_percent

def predict_vgg16(image_path):
    img = image.load_img(image_path, target_size=(196, 196))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input_vgg16(img_array)

    prediction = vgg16_model.predict(img_array)
    confidence = prediction[0][0]
    result = "Pneumonia" if confidence > 0.5 else "Normal"
    confidence_percent = round(confidence * 100, 2) if result == "Pneumonia" else round((1 - confidence) * 100, 2)

    return result, confidence_percent

def predict_resnet50(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input_resnet50(img_array)

    prediction = resnet50_model.predict(img_array)
    confidence = prediction[0][0]
    result = "Pneumonia" if confidence > 0.5 else "Normal"
    confidence_percent = round(confidence * 100, 2) if result == "Pneumonia" else round((1 - confidence) * 100, 2)

    return result, confidence_percent

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/model_select')
def model_select():
    return render_template('model_select.html')

@app.route('/inception_upload')
def inception_upload():
    session['model'] = 'Inception V3'
    return render_template('upload.html', model='Inception V3')

@app.route('/vgg16_upload')
def vgg16_upload():
    session['model'] = 'VGG-16'
    return render_template('upload.html', model='VGG-16')

@app.route('/resnet50_upload')
def resnet50_upload():
    session['model'] = 'ResNet-50'
    return render_template('upload.html', model='ResNet-50')

@app.route('/upload')
def upload():
    selected_model = session.get('model', 'Inception V3')
    if selected_model == 'Inception V3':
        return redirect(url_for('inception_upload'))
    elif selected_model == 'VGG-16':
        return redirect(url_for('vgg16_upload'))
    elif selected_model == 'ResNet-50':
        return redirect(url_for('resnet50_upload'))
    else:
        return redirect(url_for('model_select'))

@app.route('/all_models_upload')
def all_models_upload():
    return render_template('upload.html', model='All Models')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model_name = request.form['model']
        file = request.files['file']
        filename = secure_filename(file.filename)
        img_path = os.path.join('uploads', filename)
        file.save(img_path)

        if model_name == 'Inception V3':
            pred_result, confidence = predict_inception(img_path)
        elif model_name == 'VGG-16':
            pred_result, confidence = predict_vgg16(img_path)
        elif model_name == 'ResNet-50':
            pred_result, confidence = predict_resnet50(img_path)
        elif model_name == 'All Models':
            inception_result, inception_confidence = predict_inception(img_path)
            vgg16_result, vgg16_confidence = predict_vgg16(img_path)
            resnet50_result, resnet50_confidence = predict_resnet50(img_path)
            results = {
                'Inception V3': {'prediction': inception_result, 'confidence': inception_confidence},
                'VGG-16': {'prediction': vgg16_result, 'confidence': vgg16_confidence},
                'ResNet-50': {'prediction': resnet50_result, 'confidence': resnet50_confidence}
            }
            return render_template('all_models_result.html', filename=filename, results=results)
        else:
            return "Invalid model selection"

        formatted_model = model_name.replace(' ', '_')

        return render_template('result.html', filename=filename, prediction=pred_result, confidence=confidence, model=formatted_model)

if __name__ == '__main__':
    app.run(debug=True)
