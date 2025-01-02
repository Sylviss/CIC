import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tools import print_metrics
from argparse import ArgumentParser
import warnings
from tools import *

RANDOM_STATE = 42
warnings.filterwarnings("ignore")

with open("models/vt.pkl","rb") as f:
    vt = pickle.load(f)
with open("models/std.pkl","rb") as f:
    std = pickle.load(f)
with open("models/discretizer.pkl","rb") as f:
    discretizer = pickle.load(f)

def main(path,model_path,name_model):
    model_dict = {"dt":0, "gnb":0, "lr":0, "mnb":1, "mygnb":0, "mymnb":1, "lgbm":0}
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(e)
        return
    if name_model not in model_dict:
        print("Error: model not supported")
        return
    
    try:    
        if model_dict[name_model]==0:
            X_test,y_test = prepare_data_test(path,vt,std)
            
        else:
            X_test,y_test = prepare_data_test(path,vt,discretizer)
            
        y_score = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)
    except Exception as e:        
        print("Data error")
        print(e)
        return 
    print_metrics(model_path,y_test,y_pred,y_score)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path",type=str,required=True,help="Path to the dataset")
    parser.add_argument("--model_path",type=str,required=True,help="Path to the model")
    parser.add_argument("--model_name",type=str,help="Name of the model",required=True)
    args = parser.parse_args()
    main(args.path,args.model_path,args.model_name)