
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

"""Example data:
    - An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial
    -The trial had 2100 participants. Half were under 65 zears old, hlaf were 65 or older
    - Around 95% of patients 65 or older experienced side effects
    - Around 95% of patiends under 65 experinced no side effects"""

for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

# for i in train_samples:
#     print(i)
# for i in train_labels:
#     print(i)

## Transform info to numpy array to be able to train our model in keras
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
#############
# Shuffle Data: Eliminate order on data
#############
train_labels, train_samples = shuffle(train_labels, train_samples) # shaffle data so we get rid of the order when we genreated data

#############
# Normalization/Standarization
#############
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

# for i in scaled_train_samples:
#     print(i)  # print elements that have been rescaled

#############
# Create Artoficial NN
#############
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),  # first hidden layer
    Dense(units=32, activation='relu'),  # second hidden layer
    Dense(units=2, activation='softmax')   # output layer
])
# 2 output neurons: beceause either patient experience side effects or it does not
# Softmax returns a probability of having side effects
model.summary()

###################
# Train Artoficial NN
##################
"""We will define the optimizer, which type of loss function we want and how we are going to check our model"""
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # define compiler

model.fit(x=scaled_train_samples,
          y=train_labels, 
          batch_size=10,  # size of the batches in which we divide our data
          epochs=30,  # times we go through all our data
          shuffle=True,  # True by default, shuffles data so it does not learn patterns from when it was generated
          verbose=2  # 
          )  # train model

###################
# Build Validatoin Set
##################
"""
It is useful to use part of our training set for validation to avoid overfitting while training.
This means that each time the loss is computed it computes it also using data that it has not seen before (validation set)

ATTENTION: data here is shaffled after the split between train and validation, this means we will be always getting same 
data for train or same data for validation. It is important to shuffle our data before too to avoid learning patters that might be caused when generating data.
This was done already before at beggining of code.
"""

model.fit(
    x=scaled_train_samples,
    y=train_labels,
    validation_split=0.1,
    batch_size=10,
    epochs=30,
    shuffle=True,
    verbose=2
)
"""If values of loss and accuracy on both train and validation are similar then it means we are not overfitting the model, it is generalizing well then"""


###################
# Neural Networks Predictions
##################








