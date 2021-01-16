# Import packages
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
import platform
import os
import random

# Clear terminal window
os.system("cls")

# Display versions
print("\n[PYTHON AND PACKAGES VERSIONS]")
print("Python", platform.python_version())
print("Numpy", np.__version__)
print("Pandas", pd.__version__)
print("Matplotlib", matplotlib.__version__)
print("Seaborn", sns.__version__)
print("Tensorflow", tf.__version__)
print()

# Read training and test data files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Display first 5 rows of training set
print("\n[FIRST 5 ROWS OF DATASET]")
print(train.head())
print()

# Display shape of training and test set
print("Training set shape:", train.shape)
print("Test set shape:", test.shape)
print()

# Split training set into its x and y variables (independent vs dependent or predicted variables)
train_y = train['label'].astype('float32')
train_x = train.drop(['label'], axis=1).astype('int32')

# Set test data as x variables
test_x = test.astype('float32')

# Display shape of train_y, train_x, test_x
print("Train_x shape:", train_x.shape)
print("Train_y shape:", train_y.shape)
print("Test_x shape:", test_x.shape)

# Reshape Train_x from 784 pixels to 28,28,1 pixels. We will convert images to Grayscale and run CNN on them
train_x = train_x.values.reshape(-1, 28, 28, 1)
# Normalisation of values
train_x = train_x / 255.0

# Repeat for test_x
test_x = test_x.values.reshape(-1, 28, 28, 1)
test_x = test_x / 255.0

# Display shape after reshaping and normalising
print("\nTrain_X shape after reshaping:", train_x.shape)
print("Test_X shape after reshaping:", test_x.shape)

#Using One Hot Encoding to convert vales of train_y to categorical
train_y = tf.keras.utils.to_categorical(train_y, 10)

# Display shape of train_y after converting to categorical
print("Train_Y shape after converting to categorical:", train_y.shape)

# Show that Conversion was done correctly by displaying original training set and train_y after conversion
# For example, if 'label' of original training set shows 1 at index 2,
# this means that row 2 (count goes 0, 1, 2, 3...) should have a value '1' at column 1 (count starts from 0 as well).
# This is how the encoding works
print("\n[ORIGINAL TRAINING SET HEAD]")
print(train['label'].head())
print("\n[FIRST 5 ROWS OF TRAIN_Y AFTER CONVERSION]")
print(train_y[0:5, :])

model = tf.keras.models.load_model('isdp_number_recognition.h5')

predictions = model.predict([test_x])

n = random.randint(-1, 28001)

print("Predicted number:", np.argmax(predictions[n]))

plt.imshow(test_x[n], cmap='Greys')
plt.show()