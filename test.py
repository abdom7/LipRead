import lipreadtrain
import random_data as data
from keras.utils import np_utils, generic_utils


X_train, y_train = data.Train()
X_test,y_test = data.Test()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

net = lipreadtrain.build_network()


lipreadtrain.train(model=net,
                   X_train=X_train, y_train=y_train,
                   X_test=X_test, y_test=y_test)
