# 1: Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from sklearn.model_selection import train_test_split

# 2: Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()

# 3: Prepare data sets
dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'])

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Pop off expenses
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# 4: Define batches for training and test model
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

# Here we will call the input_function that was returned to us to get a dataset object we can feed to the model
train_input_fn = make_input_fn(train_dataset, train_labels)
eval_input_fn = make_input_fn(test_dataset, test_labels, num_epochs=1, shuffle=False)

# 5: Define regression model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(train_dataset.shape[1],)),  # Input
    tf.keras.layers.Dense(64, activation='relu'),  # 64 neurons hidden layer
    tf.keras.layers.Dense(32, activation='relu'),  # 32 neurons hidden layer
    tf.keras.layers.Dense(1)  # Output
])

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae', 'mse'])
model.summary()

# 6: Train the model
history = model.fit(train_dataset, train_labels, epochs=200, batch_size=32, validation_split=0.2)

# 7: Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
