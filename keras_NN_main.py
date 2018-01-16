# MINST Training and Testing data Ganesh 14/01/2018
from matplotlib import pyplot as plt
import numpy
from keras.datasets import mnist
from neural_networks_ganesh import basic_nn
from neural_networks_ganesh import basic_cnn
from neural_networks_ganesh import large_cnn

# load mnist data into variables and plot to visualise
(x_train, y_train), (x_test, y_test)= mnist.load_data()
train_test_variables = (x_train.shape, y_train.shape, x_test.shape,y_test.shape)
print(train_test_variables)
plt.imshow(x_train[0])
#plt.show()

### Use built NN model and feed with data
#model, x_test = basic_nn(x_train, y_train, x_test, y_test)
#model, x_test = basic_cnn(x_train, y_train, x_test, y_test,1)
model, x_test = large_cnn(x_train, y_train, x_test, y_test,1)

# predict y with only x data
prediction_binary = model.predict(x_test, verbose=0)
prediction = numpy.argmax(prediction_binary, axis=1)
print(prediction)
print("Manual predict error = ", (1 - sum(prediction == y_test)/len(y_test))*100)
