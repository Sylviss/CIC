import pandas as pd
import numpy as np
import pickle
from tools import *
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore")

def main(dir,model_name,n):
    with open("models/vt.pkl","rb") as f:
        vt = pickle.load(f)
    with open("models/std.pkl","rb") as f:
        std = pickle.load(f)
    with open("models/discretizer.pkl","rb") as f:
        discretizer = pickle.load(f)
        
    model_dict = {"dt":0, "gnb":0, "lr":0, "mnb":1, "mygnb":0, "mymnb":1, "lgbm":0}
    
    with open(f"models/{model_name}.pkl", "rb") as f:
        model = pickle.load(f)
        
    if model_dict[model_name]==0:
        X_test,idxs = prepare_data_inference(dir,vt,std)
        
    else:
        X_test,idxs = prepare_data_inference(dir,vt,discretizer)
        
    y_score = model.predict_proba(X_test)
    
    for score,name in zip(y_score[:n],idxs[:n]):
        print(f"{name}: {score[0]} Benign - {score[1]} Malicious")
        if score[0]>0.6:
            print("Prediction: Benign")
        else:
            print("Prediction: Malicious")
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir",type=str,help="Directory of the dataset",default = "./datasets/Visual.csv")
    parser.add_argument("--model_name",type=str,help="Name of the model",default="lgbm")
    parser.add_argument("--n",type=int,help="Number of samples to show",default=10)
    args = parser.parse_args()
    main(args.dir,args.model_name,args.n)
    
    
    
    