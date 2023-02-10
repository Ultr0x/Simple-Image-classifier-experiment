# Description: This program is used to select a file from the system and classify the image using a pre-trained model and display the result.

# Import the library
import tkinter
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import imutils

# Create an instance of window
win=Tk()
# Set the geometry of the window
win.geometry("600x200")
# Set the title of the window
win.title("Image Classification")
# Load the model
model = keras.models.load_model("model_selected.h5")
# Create a label
Label(win, text="Click the button to classify the image", font='Arial 16 bold').pack(pady=15)

# Function to open a png or jpg in the system
def load_image():
   # global variable to store the image path
   global filepath
   # Open the png or jpg file in the system
   filepath = filedialog.askopenfilename(filetypes=[("Image File", ".png"), ("Image File", ".jpg")])
   classify()

# Load the image 
def classify():
   # Load the image
   img =image.load_img(filepath, target_size=(180, 180))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   images = np.vstack([x])
   classes = model.predict(images, batch_size=10)
   print(classes)
   # Classify the input image
   print("[INFO] classifying image...")
   proba = model.predict(images)[0]
   image_copy = cv2.imread(filepath)
   output = imutils.resize(image_copy, width=400)
   #build label for other classses
   print(proba)
   cv2.putText(output, "bicycles: {:.2f}%".format(proba[0]*100), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
       0.7, (0, 255, 0), 2)
   cv2.putText(output, "cars: {:.2f}%".format(proba[1]*100), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
       0.7, (0, 255, 0), 2)
   cv2.putText(output, "deer: {:.2f}%".format(proba[2]*100), (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
       0.7, (0, 255, 0), 2)
   cv2.putText(output, "mountains: {:.2f}%".format(proba[3]*100), (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
       0.7, (0, 255, 0), 2)
   #print the label
   idx = np.argmax(proba)
   label = "bicycles" if idx == 0 else "cars" if idx == 1 else "deer" if idx == 2 else "mountains"
   proba = proba[idx] * 100
   cv2.putText(output, label, (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
       1.0, (0, 0, 255), 2)
   #print the label
       #print(label)
   # Show the output image
   cv2.imshow("Output", output)
   cv2.waitKey(0)
   #print label
   print(label)

# Create a button to trigger the dialog
button = Button(win, text="Open", command=load_image)
button.pack()
# Start the window
win.mainloop()