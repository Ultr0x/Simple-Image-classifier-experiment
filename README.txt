Readme

This is a project for the course "Coding Five: Approaches to Machine Intelligence" at the Universit of Arts of London.

The project is a simple image classifier, which classifies images of bicycles,cars,deer and mountains.

There are 2 Models:
model.h5, which was trained on all images given in the data folder, which are not corrupted or invalid
model_selected.h5, which was trained on the hand-picked dataset, ecluding around 100 photos, which were mislabeled or didn't fit the categories.

preprocessing.py is used to preprocess any new data and troubleshot some invalid,corupted data.
training.py is used to train a model a evalue the results
loadImage.py is used to load an Image and see thealgorithms's result for the given image.
classify.py is a script with GUI using loadImage.py,to effortlessly choose an image and show its classification.
classify.py is used to check the information about the enironment, such as the version of tensorflow and keras.
loss.png and accuracy.png show training history results over 30 eochs, form which I decided to choose epoch 17 as the most suitable with highest accuracy and lowest loss.

The accuracy of thez model is around 70% on the test set, which is a good result for a simple model with 4 categories and data set of less than 1500 images.
Main reason for the low accuracy is the small data set, which is not enough to train a model with 4 categories. It would be better to have a data set of at least 5000 images, to get a better accuracy.
The model is also not very deep, which is another reason for the low accuracy but it is also a reason for the low training time.
Biggest factor for the acheived 70% accuracy is the fact that the model was trained on a hand-picked dataset and additional augmentation was used to increase the data set size.

The model is not very good at classifying images of deer, which is probably because the data set is very small and the images are not very different from each other.

Functional API could be used to improve the model, but it would require a lot of time to train the model and more data to get a better accuracy.