
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from methods import *
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


x_train, y_train = load_data("data/train/")
x_test, y_test = load_data("data/test/")


x_train1 = x_train/255
x_test1 = x_test/255

x_train1 = np.concatenate((x_train1, x_test1))
y_train1 = np.concatenate((y_train, y_test))


x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train1, y_train1, test_size=0.2,
                                                        shuffle=True, random_state=0)


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
          input_shape=(x_train.shape[1], x_train.shape[2], 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.22))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001, beta_1=0.9,
                             beta_2=0.999, epsilon=1e-7),
              metrics=['acc'])

print(model.summary())

tory = model.fit(x_train2, y_train2,
                 batch_size=64,
                 validation_data=(x_test2, y_test2),
                 epochs=40,
                 shuffle=True,
                 callbacks=[callback],
                 verbose=2)
