
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

test_labels =  []
test_samples = []

"""
Same code as before but we now use it to generate the samples that we will predict from the trained model
on file 1_Build_Keras_Model
"""
for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

predictions = model.predict(
     x=scaled_test_samples,
     batch_size=10,
     verbose=0
)  
for i in predictions:
    print(i)  # 2D Output: the probability of experiencing a side effect and probability of not experiencing a side effect

"""Check how if a pecient is predicted to have side effect or not"""
rounded_predictions = np.argmax(predictions, axis=-1)

for i in rounded_predictions:
    print(i)

########################################
# Visualizing Results: Confussion Matrices
########################################

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)


def plot_confusion_matrix(cm, classes,      
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """Function from scikit-learn website"""
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



cm_plot_labels = ['no_side_effects','had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


























