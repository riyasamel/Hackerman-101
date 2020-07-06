

import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import base64
import numpy as np
import io
import tensorflow.keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras_efficientnets import EfficientNetB3
from keras.preprocessing.image import img_to_array
# from flask import request
from flask import jsonify
# from flask import Flask
from PIL import Image
import csv
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
# from keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2 as cv
import pandas as pd
import keras.backend.tensorflow_backend as tb

app = Flask(__name__)
 
 

app.static_folder = 'static'
 
 
 
 
 
print("Loading model")
 
 
global sess
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)
global model
model = load_model('C:/Users/samel/OneDrive/Desktop/testing/Finalmodel.h5')
global graph
graph =tf.compat.v1.get_default_graph()
 
 
 
 
 
@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        file.save('C:/Users/samel/OneDrive/Desktop/testing/uploads/test.jpg')

    return render_template('index.html')


@app.route('/prediction')
def prediction():
    # Step 1
    # my_image = plt.imread(os.path.join('/content/drive/My Drive/flask/uploads', filename))
    print('hi')
    # Step 2
    my_image = plt.imread('C:/Users/samel/OneDrive/Desktop/testing/uploads/test.jpg')
    my_image_re = resize(my_image, (299,299))

    # Step 3
    # with graph.as_default():
    # tf.compat.v1.keras.backend.set_session(sess)
    # set_session(sess)
    x = np.expand_dims(my_image_re, axis=0)

    print(x.shape)
    #bgr_img = cv.imread('C:/Users/owais/Desktop/testing/uploads/test.jpg')
    #rgb_img = cv.resize(cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB) / 255, (299, 299))
    #rgb_img = np.expand_dims(rgb_img, 0)
    tb._SYMBOLIC_SCOPE.value = True
    pro = model.predict(x)
    pro = pro * 100
    print(pro)

    car_annos = pd.read_csv('C:/Users/samel/OneDrive/Desktop/testing/names.csv')
    #annotations=np.array(car_annos)



    ll=[]

    for i, car in enumerate(list(car_annos.iloc[:, 0])):
        element={"car": car, "prob": pro[0][i]}
        ll.append(element)


    save = sorted(ll, key=lambda i: i['prob'], reverse=True)



    pro = {
        "class1": save[0]["car"],
        "class2": save[1]["car"],
        "class3": save[2]["car"],
        "prob1": save[0]["prob"],
        "prob2": save[1]["prob"],
        "prob3": save[2]["prob"],
    }

    os.remove('C:/Users/samel/OneDrive/Desktop/testing/uploads/test.jpg')
    print(pro)
    return render_template('predict.html', pro=pro)
    #return render_template('predict.html')
 
 
 
app.run()





















