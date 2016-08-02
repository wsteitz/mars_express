My solution for the [Kelvin Mars Express Power Challenge](https://kelvins.esa.int/mars-express-power-challenge/). Ended up in 6th place out of 40 participating teams.


Features
========

Features taken from the provided data:

LTDATA: all features as is
SAAF: all raw variables + lag variables for the solar angles (sa, sx, sy, sz) of the previous 1-4 hours + monthly aggregates
DMOP: hourly counts for each subsystem
EVTF: hourly counts for all events with more than 100 occurances.
FTL: hourly counts for each type of event + one for the flagcomms. Duration (in minutes) of each event per hour.


Additional features:
- days & month in orbit, to capture time related trends
- indicator variable that shows if the spacecraft is ascending or descending
- indicator for new operational mode
- average occultation duration per month


Models
======

* Gradient boosting trees (using xgboost)
  1. dedicated model per powerline (GBT1). This was by far my best performing model.
  2. one huge model, adding the powerlines as a categorical feature (GBT2)

* (Recurent) neural nets (using keras)
  1. 4-layer neural network (NN)
  2. 4-layer LSTM feeding sequences of 20 hours (LSTM)

* Ridge regression (from sklearn) (RIDGE)


For eight of the powerlines with low power consumption, I always used the RIDGE model.


Ensembling
==========

xgboost worked best for me. I tried neural nets and some recurrent nets (LSTMs) with less
satisfying results. Nevertheless, ensembling all these models gave a nice boost.

     0.45 * GBT1 + 0.3 * GTB2 + 0.12 * NN + 0.1 * LSTM + 0.03 * RIDGE


Things that did not help
========================
- running a model on shorter aggregation intervals
- recursive feature removal
- stacking
- learning rate decay & dart booster for xgb


What I've learned
=================
1. xgboost is amazing
2. ensembling. I spend too much time trying to improve my first model. Instead I
   should have started another one, once I hit a wall and my progress stopped.
3. designing a neural network is an art
4. feature engineering is key. Some teams ended up in top positions with a
   rather simple model but good features.


Technical requirements
======================
python3 and some excellent libraries: xgboost, sklearn, keras, theano, pandas
