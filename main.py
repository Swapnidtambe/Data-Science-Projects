import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.utils import load_img
from io import BytesIO
from PIL import Image
import imageio as iio
from tensorflow.keras.applications.resnet50 import decode_predictions
import json

with open('list.json', 'r') as openfile:
    list = json.load(openfile)


model1 = keras.models.load_model('G:\\projects\\Deep Learning\\BIRDS IMAGE CLASSIFICATION\\results')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    file = request.files["file"]
    filename = file.filename
    file.save(filename)

    image = Image.open(filename)
    image = image.resize((224, 224))
    image = np.array(image)
    size = (224, 224)
    image = tf.image.resize(image, (size)) / 255.0
    image = tf.expand_dims(image, 0)

    prediction = model1.predict(image)

    predict = np.argmax(prediction)
    predict = list[predict]


    return "Bird Name is  "+predict

if __name__ == '__main__':
    app.run(debug=True)