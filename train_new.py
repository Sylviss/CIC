import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from NaiveBayes import MultinomialNB as MyMNB
from NaiveBayes import GaussianNB as MyGNB
import pickle
from tools import print_metrics
from argparse import ArgumentParser
import warnings

RANDOM_STATE = 42
warnings.filterwarnings("ignore")

with open("models/vt.pkl","rb") as f:
    vt = pickle.load(f)
with open("models/std.pkl","rb") as f:
    std = pickle.load(f)
with open("models/discretizer.pkl","rb") as f:
    discretizer = pickle.load(f)

def main(path,model,file_name,valid_file):
    name = model
    print("Reading file")
    try:
        a = pd.read_csv(path)
    except Exception as e:
        print(e)
        return
    print("Complete reading file")
    a = a.drop(columns = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', "Timestamp", "Flow ID"], errors='ignore')
    a = a.drop(columns = ['PSH Flag Count', 'ECE Flag Count', 'RST Flag Count', 'ACK Flag Count', 'Fwd Packet Length Min', 'Bwd Packet Length Min', 'Packet Length Min', 'Protocol', 'Down/Up Ratio','Bwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd PSH Flags', 'Bwd URG Flags', 'CWE Flag Count', 'FIN Flag Count', 'Fwd Bulk Rate Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd URG Flags'], errors='ignore')
    a = a.replace([np.inf, -np.inf], 0).fillna(0)
    X,y = a.drop(columns = ["Label"]).to_numpy(),a.Label.to_numpy().astype("int32")
    X = vt.transform(X)
    if model == "dt":
        X = std.transform(X)
        cls = DecisionTreeClassifier()
    elif model == "gnb":
        X = std.transform(X)
        cls = GaussianNB()
    elif model == "lr":
        X = std.transform(X)
        cls = LogisticRegression()
    elif model == "lgbm":
        X = std.transform(X)
        cls = LGBMClassifier(verbosity=-1)
    elif model == "mnb":    
        X = discretizer.transform(X)
        cls = MultinomialNB()
    elif model == "mymnb":
        X = discretizer.transform(X)
        cls = MyMNB()
    elif model == "mygnb":
        X = std.transform(X)
        cls = MyGNB()
    else:
        print("Please choose a correct model")
        return
    if valid_file is None:
        X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE)
    else:
        X_train = X
        y_train = y
        print("Reading validation file")
        try:
            b = pd.read_csv(valid_file)
        except Exception as e:
            print(e)
            return
        print("Complete reading validation file")
        b = b.drop(columns = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', "Timestamp", "Flow ID"], errors='ignore')
        b = b.drop(columns = ['PSH Flag Count', 'ECE Flag Count', 'RST Flag Count', 'ACK Flag Count', 'Fwd Packet Length Min', 'Bwd Packet Length Min', 'Packet Length Min', 'Protocol', 'Down/Up Ratio','Bwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd PSH Flags', 'Bwd URG Flags', 'CWE Flag Count', 'FIN Flag Count', 'Fwd Bulk Rate Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd URG Flags'], errors='ignore')
        b = b.replace([np.inf, -np.inf], 0).fillna(0)
        X_valid,y_valid = b.drop(columns = ["Label"]).to_numpy(),b.Label.to_numpy().astype("int32")
        X_valid = vt.transform(X_valid)
        if model == "dt":
            X_valid = std.transform(X_valid)
        elif model == "gnb":
            X_valid = std.transform(X_valid)
        elif model == "lr":
            X_valid = std.transform(X_valid)
        elif model == "lgbm":
            X_valid = std.transform(X_valid)
        elif model == "mnb":
            X_valid = discretizer.transform(X_valid)
        elif model == "mymnb":
            X_valid = discretizer.transform(X_valid)
        elif model == "mygnb":
            X_valid = std.transform(X_valid)
    try:
        cls.fit(X_train,y_train)
        y_pred = cls.predict(X_valid)
        y_score = cls.predict_proba(X_valid)[:,1]
    except Exception as e:        
        print("Data error")
        print(e)
        return 
    print_metrics(name,y_valid,y_pred,y_score)
    
    with open(f"trained/{file_name}.pkl","wb") as f:
        pickle.dump(cls,f)
    
    print(f"Model saved at trained/{file_name}.pkl")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path",type=str,required=True,help="Path to the dataset")
    parser.add_argument("--valid_path",type=str,help="Path to the validation dataset",default=None)
    parser.add_argument("--model",type=str,required=True,choices=["dt","gnb","lr","lgbm","mnb","mymnb","mygnb"],help="Model to use")
    parser.add_argument("--file_name",type=str,required=True,help="Name of the file to save the model")
    args = parser.parse_args()
    main(args.path,args.model,args.file_name,args.valid_path)