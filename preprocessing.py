import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from os import listdir
import cv2

#create a function to rename files and pass if file is already renamed and save to a different directory
def rename_files(directory, category):
    for i, filename in enumerate(os.listdir(directory)):
        try:
            os.rename(directory + "/" + filename, directory + "/" + category + "." + str(i) + ".jpg")
        except:
            continue
    print(f"done renaming {category} files")

rename_files("./data/bicycles", "bicycles")
rename_files("./data/cars", "cars")
rename_files("./data/deer", "deer")
rename_files("./data/mountains", "mountains")
#convert images to RGB and resize to 180x180
def convert_images(directory):
    num_skipped = 0
    for root, dirs, files in os.walk(directory):
      path = root.split(os.sep)

      for file in files:
        _, extension = os.path.splitext(file)

        if extension == ".jpg":
          filepath = root + "/" + file

          try:
            fobj = open(filepath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
          finally:
            fobj.close()

          if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(filepath)
    print(f"Deleted {num_skipped} images")
convert_images("./data/")


#additional troubleshooting to find the invalid files

# if an error "Corrupt JPEG data: xxx extraneous bytes before marker 0xd9" occurs, some files are corrupted and need to be removed
# use this code to search for the corrupted files and then rmemove them manually
#check if the image is valid
def is_valid (directory):
  num_invalid = 0
  for filename in listdir(directory):
      print(directory+"/" +filename)
      try:
          img = cv2.imread(directory +"/"+ filename)
      except:
        continue
      if img is None:
              print("invalid file")
              #remove invalid file
              os.remove(directory+"/" + filename)
              print(filename + " removed")
              num_invalid += 1
  print("number of invalid files: " + str(num_invalid))

is_valid("./data2/bicycles")
is_valid("./data2/cars")
is_valid("./data2/deer")
is_valid("./data2/mountains")

print("done")
#remove invalid files from the directory manually

