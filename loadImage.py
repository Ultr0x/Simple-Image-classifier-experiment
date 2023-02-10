#import the necessary packages
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import imutils
# Load the model
model = keras.models.load_model("model_selected.h5")
# ui for loading image
image_path="data/bicycles/bicycles.0.jpg"
#image_path = input("Enter the path to the image: ")
# Load the image
img =image.load_img(image_path, target_size=(180, 180))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)
# Classify the input image
print("[INFO] classifying image...")
proba = model.predict(images)[0]
image_copy = cv2.imread(image_path)
output = imutils.resize(image_copy, width=400)
#build label for all the classes
print(proba)
cv2.putText(output, "bicycles: {:.2f}%".format(proba[0]*100), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 255, 0), 2)
cv2.putText(output, "cars: {:.2f}%".format(proba[1]*100), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 255, 0), 2)
cv2.putText(output, "deer: {:.2f}%".format(proba[2]*100), (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 255, 0), 2)
cv2.putText(output, "mountains: {:.2f}%".format(proba[3]*100), (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 255, 0), 2)
idx = np.argmax(proba)
#build label for the class with highest probability
label = "bicycles" if idx == 0 else "cars" if idx == 1 else "deer" if idx == 2 else "mountains"
proba = proba[idx] * 100
cv2.putText(output, label, (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
    1.0, (0, 0, 255), 2)
#print the label
print(label)
# Show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)





