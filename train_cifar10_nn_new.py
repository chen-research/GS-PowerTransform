"""For training a distilled (robust) cnn on the cifar10 images.
Example Usage.

d = CIFAR()
cifar_nn = train_distillation(data=d, 
                              file_name="models/cifar-distilled-100", 
                              params=[64, 64, 128, 128, 256, 256],
                              num_epochs=5, 
                              train_temp=100)
test_loss, test_accuracy = cifar_nn.evaluate(d.test_data, d.test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
"""


from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

cifar10_path = "C:/Users/HP/Desktop/Optimization with Objective Trans and Guassian Smoothing/codes/Draft_Codes/NeuralNet/Data/CIFAR10_Data/"
class CIFAR:
    def __init__(self):

        train_data = np.load(cifar10_path+"x_train.npy")   #60000 train images (32x32x3)
        #train_data = (train_data/255)-0.5
        train_data = (train_data-255/2)/(255/2)
        train_labels = tf.keras.utils.to_categorical(np.load(cifar10_path+"y_train.npy"),10)
        self.test_data = np.load(cifar10_path+"x_test.npy") #10000 test train images (32x32x3)
        #self.test_data = (self.test_data/255)-0.5
        self.test_data = (self.test_data-255/2)/(255/2)
        self.test_labels = tf.keras.utils.to_categorical(np.load(cifar10_path+"y_test.npy"),10)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    print(data.train_data.shape)
    
    model.add(Conv2D(params[0], (3, 3),padding='same',activation="relu",
                            input_shape=data.train_data.shape[1:]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Conv2D(params[1], (3, 3),padding='same',activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(Conv2D(params[2], (3, 3), padding='same',activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Conv2D(params[3], (3, 3), padding='same',activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(Conv2D(params[4], (3, 3), padding='same',activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(params[5],activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(10))

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    #sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    Adamm = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(loss=fn,
                  optimizer=Adamm,#sgd,
                  metrics=['accuracy'])
    
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs, 
              shuffle=True,
              verbose=1
             )
    

    if file_name != None:
        model.save(file_name)

    return model

def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):
    """
    Train a network using defensive distillation.

    Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
    Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
    IEEE S&P, 2016.
    """
    if not os.path.exists(file_name+"_init"):
        # Train for one epoch to get a good starting point.
        train(data, file_name+"_init", params, 1, batch_size)
    
    # now train the teacher at the given temperature
    teacher = train(data, file_name+"_teacher", params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # evaluate the labels at temperature t
    predicted = teacher.predict(data.train_data)
    
    """ #Original uses tensorflow v1
    with tf.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted/train_temp))
        print(y)
        data.train_labels = y
    """
    data.train_labels = tf.nn.softmax(predicted/train_temp) #replace tf v1

    # train the student model at temperature t
    student = train(data, file_name, params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # and finally we predict at temperature 1
    #predicted = student.predict(data.train_data)

    #print(predicted)
    return student #return the distillation neural net