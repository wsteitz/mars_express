import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
import keras
from sklearn.pipeline import Pipeline
import pandas as pd

from models.common import FeatureRemover
from models.common import ScaleAndNorm
from models.common import SomeLinearWrapper

from common import x_all


class MyLSTM:
    
    def __init__(self, batch_size=32, sequence_length=20, n_epochs=1):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.n_epochs = n_epochs
        self.name = "lstm"
        
    def sliding_window(self, df):
        l = self.sequence_length
        res = []
        for i in range(0, len(df) - l, l):
            res.append(list(df.iloc[i:i + l].values))
        return res

    def fit(self, x, y):
        input_dim = x.shape[1]
        output_dim = y.shape[1]
        self.x_train = x

        start = len(x) % (self.batch_size * self.sequence_length)

        x_seq = self.sliding_window(x.iloc[start:])
        y_seq = self.sliding_window(y.iloc[start:])

        model = Sequential()
        model.add(GRU(1024, batch_input_shape=(self.batch_size, self.sequence_length, input_dim), return_sequences=True, stateful=True))
        model.add(Activation("tanh"))
        model.add(GRU(1024, return_sequences=True))
        model.add(Activation("tanh"))
        model.add(GRU(512, return_sequences=True))
        model.add(Activation("tanh"))
        #model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(output_dim)))
        model.add(Activation("linear"))

        optimizer = keras.optimizers.RMSprop(lr=0.002)
        optimizer = keras.optimizers.Nadam(lr=0.002)
        model.compile(loss='mse', optimizer=optimizer)

        model.fit(x_seq, y_seq, batch_size=self.batch_size, verbose=1, nb_epoch=self.n_epochs, shuffle=False)
        self.model = model
        return self


    def predict(self, x):
        # merge the train and the test
        x_merged = pd.concat([self.x_train, x])

        start = len(x_merged.index) % (self.batch_size * self.sequence_length)

        x_seq = self.sliding_window(x_merged.iloc[start:])
        pred = self.model.predict(x_seq, batch_size=self.batch_size, verbose=1)

        pred = np.vstack(pred)
        res = pred[-len(x):, :]
        return res


lstm = Pipeline([
      ("drop", FeatureRemover([c for c in x_all.columns if c[-2] == "-"])),
      ("scaleandnorm", ScaleAndNorm()),
      ("lstm", MyLSTM(batch_size=32, sequence_length=20, n_epochs=10))
      ])
lstm.name = "lstm"

lstm = SomeLinearWrapper(lstm)
