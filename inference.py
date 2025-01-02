import pandas as pd
import numpy as np
import pickle
from tools import *
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore")

def main(dir,model_name,n,name_model):
    with open("models/vt.pkl","rb") as f:
        vt = pickle.load(f)
    with open("models/std.pkl","rb") as f:
        std = pickle.load(f)
    with open("models/discretizer.pkl","rb") as f:
        discretizer = pickle.load(f)
        
    model_dict = {"dt":0, "gnb":0, "lr":0, "mnb":1, "mygnb":0, "mymnb":1, "lgbm":0}
    try:
        with open(model_name, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(e)
    if name_model not in model_dict:
        print("Error: model not supported")
        return
    try:
        if model_dict[name_model]==0:
            X_test,idxs = prepare_data_inference(dir,vt,std)
            
        else:
            X_test,idxs = prepare_data_inference(dir,vt,discretizer)
        y_score = model.predict_proba(X_test)
    except Exception as e:        
        print("Data error")
        print(e)
        return 
    for score,name in zip(y_score[:n],idxs[:n]):
        print(f"{name}: {score[0]} Benign - {score[1]} Malicious")
        if score[0]>0.6:
            print("Prediction: Benign")
        elif score[0]<0.3:
            print("Prediction: Malicious")
        else:
            print("Prediction: Likely to be Malicious")
        print("\n")
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir",type=str,help="Directory of the dataset",default = "./datasets/csv/Visual.csv")
    parser.add_argument("--n",type=int,help="Number of samples to show",default=10)
    parser.add_argument("--model_path",type=str,help="Path to the model",default="models/lgbm.pkl")
    parser.add_argument("--model_name",type=str,help="Name of the model",default="lgbm")
    args = parser.parse_args()
    main(args.dir,args.model_path,args.n,args.model_name)
    
    
    
    