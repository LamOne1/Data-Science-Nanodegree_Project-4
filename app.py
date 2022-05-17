import flask
from flask import render_template, redirect
from flask import request, url_for, render_template, redirect
from extract_bottleneck_features import *

from keras.preprocessing import image                  
from keras.applications.resnet50 import preprocess_input

from keras.models import Sequential#, load_model
from keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import os

from keras import backend as K

dogs_names_file = open('dogs_name.txt','r')
dogs_names = dogs_names_file.readlines()


def load_model():
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
    Resnet50_model.add(Dense(133, activation = 'softmax'))
    Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
    return Resnet50_model
    
def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    K.clear_session()

    # obtain predicted vector
    model = load_model()

    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    #return dog_names[np.argmax(predicted_vector)]
    return dogs_names[np.argmax(predicted_vector)]


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

app = flask.Flask(__name__)


bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

#model = load_model('saved_models/weights.best.Resnet50.hdf5', compile=False)



@app.route("/", methods=["POST","GET"])
def predict():

    data = {"success": False}
    path_img = "templates/temp_img.jpg"
    
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            
            image_ = flask.request.files["image"]
            image = image_.save(path_img)
            
            predictions = Resnet50_predict_breed(path_img)
            
            
            data["prediction"] = predictions
            
            return render_template('master.html', data=data, name=path_img)

    return render_template('master.html', data = data, name=path_img)
    
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=3000, debug=True)
