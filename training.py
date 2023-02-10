#
# This is the main file for the project. It contains the code for the
# training of the model as well as the code for the testing of the model.
# The model is trained on the dataset that is in the data folder.
#

# This model achieves an accuracy of about 0.7 on the test dataset with given dataset.

# import the necessary packages 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from os import listdir
import cv2

#split the dataset into training, validation and test dataset
training_dataset = keras.preprocessing.image_dataset_from_directory(
  "./data",
  labels='inferred',
  label_mode='categorical',
  color_mode="rgb",
  batch_size=32,
  #image size is 180x180 pixels because it is a small dataset and it is good to have a smaller image size to prevent overfitting of the model as it turned out that 150x150 pixels was too small for the model to learn the features of the images, scoring only 0.6 accuracy on the validation dataset at max
  image_size=(180, 180),
  #validation split is 10% of the dataset as 20% was too much for a small dataset and started to overfit the model relatively quickly
  validation_split=0.1,
  subset="training",
  seed=1
)
validation_dataset = keras.preprocessing.image_dataset_from_directory(
  "./data",
  labels='inferred',
  label_mode='categorical',
  color_mode="rgb",
  batch_size=32,
  image_size=(180, 180),
  validation_split=0.1,
  subset="validation",
  seed=1 
)
num_skipped = 0

# class names of the dataset 
class_names = training_dataset.class_names
print(class_names)

#cache and prefetch the dataset to improve performance 
training_dataset = training_dataset.cache().prefetch(buffer_size=32)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=32)

# Convolutional model was used because it is very efficient for image classification due to the fact that it is able to learn wide range of features of the images
model = keras.Sequential([
    #image augmentation layers to prevent overfitting of the model by randomly flipping the image horizontally, rotating the image and zooming the image
    keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(180, 180, 3)),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
    #preprocessing layers to normalize the data and resize the image to 150x150 pixels 
    keras.layers.experimental.preprocessing.Rescaling(1./255),
    #convolutional layers with 32 and 64 filters and relu activation function to get the output of the neuron and maxpooling layers to reduce the size of the image 
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    #flatten layer to flatten the output of the convolutional layers to 1D array
    keras.layers.Flatten(),
    #dense layer with 128 neurons and relu activation function to get the output of the neuron
    keras.layers.Dense(128,activation="relu"),
    #dropout layer to prevent overfitting of the model 
    keras.layers.Dropout(0.5),
    #output layer with 4 neurons for 4 classes of images (bicycles, cars, deer, mountains) and softmax activation function to get the probabilities of each class for each image in the batch of 32 images
    keras.layers.Dense(4,activation='softmax')
])


#compile the model with rmsprop optimizer, categorical crossentropy loss function and accuracy metric
model.compile(
        optimizer='rmsprop',
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )

#train the model with 20 epochs and validation data as test dataset
history = model.fit(training_dataset,validation_data=validation_dataset,epochs=16)

#summary of the model
model.summary()

#evaluate the model with test dataset
test_loss, test_acc = model.evaluate(validation_dataset)
print(f"Test loss: {test_loss:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

#import matplotlib to plot the training and validation accuracy and loss
import matplotlib.pyplot as plt

#stating the variables for the training and validation accuracy and loss plots
accuracy = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, 17 )

#plot the training and validation accuracy and loss
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()

#show the plots
plt.show()

# Save the model
model.save("model_selected.h5")



