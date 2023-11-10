# Import tensorflow
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from sklearn.preprocessing import LabelEncoder


def process_sample(img, process=True):
    # Normalize image pixel values to a float range [0, 1]
    img = (img / 255).astype(np.float32)

    if (process):
      # Convert image from BGR to RGB
      img = img[...,::-1]
      # Make the image dataset squared
      dim = min(img.shape[:-1])
      img = img[(img.shape[0]-dim)//2:(img.shape[0]+dim)//2, (img.shape[1]-dim)//2:(img.shape[1]+dim)//2, :]

      # Resize the image to 224x224 pixels
      #img = tfkl.Resizing(224, 224)(img)
      #img = tfkl.Resizing(96, 96)(img)

    return img

def load_data(folder="public_data.npz", resolution=96, head_only=False, process=True):
    images = []

    loaded = np.load(folder, allow_pickle=True)

    # Iterate through files in the specified folder
    for i, img in enumerate(loaded['data']):
        img = process_sample(img, process=process)

        if img is not None:
            images.append(img)

        if (head_only and i == 9):
           break

    labels = loaded['labels']
    loaded.close()

    if (head_only):
       labels = labels[:10]

    y = LabelEncoder().fit_transform(labels)
    #y = tfk.utils.to_categorical(y, 1)

    return np.array(images), y




def display_random_images(X, y, num_img=10):
  # Create subplots for displaying items
  fig, axes = plt.subplots(2, num_img//2, figsize=(20, 9))
  for i in range(num_img):
      image = randint(0, X.shape[0] - 1)

      ax = axes[i%2, i%num_img//2]
      ax.imshow(np.clip(X[image], 0, 255))  # Display clipped item images
      ax.text(0.5, -0.1, str(image) + ' ' + str(y[image]), size=12, ha="center", transform=ax.transAxes)
      ax.axis('off')
  plt.tight_layout()
  plt.show()




def delete_outliers(X, y):
  shrek = 137
  trololo = 5143

  new_X = []
  new_y = []

  num_outliers = 0

  for i, sample in enumerate(X):
    if (not (np.array_equal(sample, X[shrek]) or np.array_equal(sample, X[trololo]))):
      new_X.append(sample)
      new_y.append(y[i])
    else:
      num_outliers += 1

  return np.array(new_X), np.array(new_y), num_outliers