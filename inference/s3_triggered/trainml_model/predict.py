###
# Source: https://learnopencv.com/keras-tutorial-using-pre-trained-imagenet-models/
###

import os
import glob
import json
import re
import numpy as np
import tensorflow as tf

# import the models for further classification experiments
from tensorflow.keras.applications import vgg16

sess = tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(log_device_placement=True)
)

# init the models
vgg_model = vgg16.VGG16(weights="imagenet")


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions


data_dir = os.environ.get("TRAINML_DATA_PATH")
output_dir = os.environ.get("TRAINML_OUTPUT_PATH")


def predict_image(filename):
    # load an image in PIL format
    original = load_img(filename, target_size=(224, 224))

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)

    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = vgg_model.predict(processed_image)
    # print predictions
    # convert the probabilities to class labels
    # we will get top 5 predictions which is the default
    label_vgg = decode_predictions(predictions)

    return label_vgg[0]


for filename in glob.glob(f"{data_dir}/*.JPEG"):
    classes = predict_image(filename)

    output_file_name = re.sub(
        ".JPEG", "_pred.json", os.path.basename(filename)
    )

    print(f"{output_file_name}: {classes}")

    with open(f"{output_dir}/{output_file_name}", "w") as f:
        f.write(json.dumps(str(classes)))