# Simple Neural Network 15/01/2018 Ganesh
def basic_nn(x_train, y_train, x_test, y_test):

    from keras.models import Sequential
    from keras.layers import Dense
    import numpy
    from keras.utils import np_utils

###Data Preprocessing ###
    train_test_variables = (x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    pixel_max = numpy.amax(x_train[0])

    # flatten 3 dim array to 2 dim as shape of each data supposed to be 1 dim for simple basic NN
    num_pixel_train = x_train.shape[1] * x_train.shape[2]  # 28 * 28 pixels = 784
    x_train = x_train.reshape(x_train.shape[0], num_pixel_train).astype('float32')  # change type to float
    x_test = x_test.reshape(x_test.shape[0], num_pixel_train).astype('float32')

    # normalise pixel value scale 0-255 to 0-1 for better approximation
    x_train = x_train / pixel_max
    x_test = x_test / pixel_max

    # convert output variable vector of y from 0 to 9 to binary matrix since
    y_train_bin = np_utils.to_categorical(y_train)  # change number to binary
    y_test_bin = np_utils.to_categorical(y_test)
    num_classes = y_test_bin.shape[1]  # size of binary matrix to represent 10 classes
    #print('#classes =', num_classes)



### Basic NN Model build ###
    model = Sequential()
    # 1st input layer (output arrays, input arrays, kernal, activation, units?)
    model.add(Dense(num_pixel_train,
                    input_dim=num_pixel_train,
                    kernel_initializer='normal',
                    activation='relu')) # rectifier activation for neurons in Hidden layer
    # 1st hidden layer (output array, {no need specify input layer}, kernal, activation, units?)
    model.add(Dense(num_classes,
                    kernel_initializer='normal',
                    activation='softmax')) # used on output later to turn
    # outputs into probability values to select 1 class out of the many as prediction

    # compile model
    model.compile(loss='categorical_crossentropy', # logarithmic loss
                  optimizer='adam', # adam gradient descent to learn weights
                  metrics=['accuracy'])



    ### Train & Evaluate MODEL ###

    # Train model to data fit(X data, y class data, validation data to see training improvement, epoch, batch? verbose?)
    model.fit(x_train, y_train_bin,
              validation_data=(x_test, y_test_bin),
              epochs=10,
              batch_size=200, verbose=2)  # batch size for gradient descent update

    # Evaluation of model to determine error in prediction, loss
    scores = model.evaluate(x_test, y_test_bin, verbose=1)  # total percentage accuracy
    print("Prediction Error: %.2f%%" % (100 - scores[1] * 100))

    return model, x_test # have to return the model that is processed. return means that is the output

### Convolution Neural Network Basic ###
def basic_cnn(x_train, y_train, x_test, y_test, num_pixel):

    import numpy
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.utils import np_utils
    from keras import backend as K
    K.set_image_dim_ordering('th')

### Data Preprocessing ###
    train_test_variables = (x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    pixel_max = numpy.amax(x_train[0])

    # flatten 3 dim array to 2 dim as shape of each data supposed to be 1 dim for simple basic NN
    width = x_train.shape[2]
    height = x_train.shape[1]
    x_train = x_train.reshape(x_train.shape[0], num_pixel, height, width).astype('float32')  # change type to float
    x_test = x_test.reshape(x_test.shape[0], num_pixel, height, width).astype('float32')

    # normalise pixel value scale 0-255 to 0-1 for better approximation
    x_train = x_train / pixel_max
    x_test = x_test / pixel_max

    # convert output variable vector of y from 0 to 9 to binary matrix since
    y_train = np_utils.to_categorical(y_train)  # change number to binary
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]  # size of binary matrix to represent 10 classes

### Basic CNN model build ###
    model = Sequential()
    # input layer Conv2D, 32 5x5 feature maps with rectifier activation fn.
    # input structure = [pixels][width][height]
    model.add(Conv2D(32,(5,5), input_shape=(num_pixel, width, height),activation='relu'))

    # pooling layer Maxpool 2x2
    model.add(MaxPooling2D(pool_size=(2,2)))

    # regularization layer Droput
    # randomly exclude 20% of neurons to reduce overfitting
    model.add(Dropout(0.2))

    # convert 2D matrix to vector with Flatten
    # output can now be processed by standard layers
    model.add(Flatten())

    # full layer with 123 neuron and rectifier activation
    model.add(Dense(128, activation='relu'))

    #output layer 10 neuron for 10 classes with softmax activation
    # for 0-1 probability like prediction for each class
    model.add(Dense(num_classes, activation='softmax'))

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ### Train & Evaluate MODEL ###
    model.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=10, batch_size=200, verbose=2)
    scores = model.evaluate(x_test,y_test,verbose=0)
    print(" CNN Error: %.2f%%" % (100-scores[1]*100))

    return model, x_test



### Convolution Neural Network large layers ###
def large_cnn(x_train, y_train, x_test, y_test, num_pixel):

    import numpy
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.utils import np_utils
    from keras import backend as K
    K.set_image_dim_ordering('th')

### Data Preprocessing ###
    train_test_variables = (x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    pixel_max = numpy.amax(x_train[0])

    # flatten 3 dim array to 2 dim as shape of each data supposed to be 1 dim for simple basic NN
    width = x_train.shape[2]
    height = x_train.shape[1]
    x_train = x_train.reshape(x_train.shape[0], num_pixel, height, width).astype('float32')  # change type to float
    x_test = x_test.reshape(x_test.shape[0], num_pixel, height, width).astype('float32')

    # normalise pixel value scale 0-255 to 0-1 for better approximation
    x_train = x_train / pixel_max
    x_test = x_test / pixel_max

    # convert output variable vector of y from 0 to 9 to binary matrix since
    y_train = np_utils.to_categorical(y_train)  # change number to binary
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]  # size of binary matrix to represent 10 classes

### Basic CNN model build ###
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(num_pixel, height, width), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ### Train & Evaluate MODEL ###
    model.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=10, batch_size=200, verbose=2)
    scores = model.evaluate(x_test,y_test,verbose=0)
    print(" Large CNN Error: %.2f%%" % (100-scores[1]*100))

    return model, x_test