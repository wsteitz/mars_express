import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
import keras
from sklearn.pipeline import Pipeline
import pandas as pd

from models.common import FeatureRemover
from models.common import ScaleAndNorm
from models.common import SomeLinearWrapper


class NN:

    def __init__(self, batch_size=64, n_epochs=1):
        self.n_epochs = n_epochs
        self.name = "nn"
        self.batch_size = batch_size

    def fit(self, x, y):
        input_dim = x.shape[1]
        output_dim = y.shape[1]
        model = Sequential()
        model.add(Dense(700, input_dim=input_dim))
        model.add(Activation('tanh'))
        model.add(Dense(700))
        model.add(Activation('tanh'))
        model.add(Dense(300))
        model.add(Activation('tanh'))
        #model.add(Dropout(0.1))
        model.add(Dense(output_dim))
        model.add(Activation('relu'))

        optimizer = keras.optimizers.Adam()
        model.compile(loss='mse', optimizer=optimizer)

        model.fit(x.as_matrix(), y.as_matrix(), batch_size=self.batch_size, verbose=1, nb_epoch=self.n_epochs, shuffle=True)
        self.model = model
        return self

    def predict(self, x):
        pred = self.model.predict(x.as_matrix(), batch_size=self.batch_size, verbose=1)
        pred = np.vstack(pred)
        return pred


nn = Pipeline([
      ("scaleandnorm", ScaleAndNorm()),
      ("nn", NN(batch_size=32, n_epochs=10))
      ])
nn.name = "nn"

nn = SomeLinearWrapper(nn)
