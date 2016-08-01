from common import *
from models.my import my_model
from models.onegbm import onegbm
from models.common import model_ridge
from models.lstm import lstm
from models.nn import nn
from models.et import et
import argparse
 


models = {"my": my_model,
          "onegbm": onegbm,
          "ridge": model_ridge,
          "lstm": lstm,
          "et": et,
          "nn": nn}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('-l','--powerline', help='powerline', required=False)
    parser.add_argument('-m','--model', help='model', required=True)
    args = parser.parse_args()


    if args.powerline is not None:
        cols = [args.powerline]
    else:
        cols = cols_to_predict
    

    validate_model(models[args.model], cols=cols)
