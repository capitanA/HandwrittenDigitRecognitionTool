from tensorflow import keras
from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import requests

requests.packages.urllib3.disable_warnings()
import ssl
from netHelper import myCallback

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


def trainModel():
    """ I have been tested some other architectures but the one which is uncomment had a better result"""
    # Splitting the MNIST dataset into Train and Test
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocessing the input data
    num_of_trainImgs = x_train.shape[0]  # 60000 here
    num_of_testImgs = x_test.shape[0]  # 10000 here
    img_width = 28
    img_height = 28
    x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
    x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)
    input_shape = (img_height, img_width, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Converting the class vectors to binary class
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Defining the model architecture
    model = Sequential()

    # Defining the hyperparameters of the network
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))

    # Compiling the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])

    callback = myCallback()

    # Fitting the model on training data
    model.fit(x_train, y_train, epochs=12,
                        validation_data=(x_test, y_test),
                        callbacks=[callback])

    # model.fit(x_train, y_train,
    #           batch_size=128,
    #           epochs=20,
    #           verbose=1,
    #           validation_data=(x_test, y_test))
    # Evaluating the model on test data

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('model.h5')


if __name__ == "__main__":
    trainModel()
