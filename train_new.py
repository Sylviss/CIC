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

def main(path,model,file_name):
    name = model
    a = pd.read_csv(path)
    try:
        a = a.drop(columns = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', "Timestamp"])
        a = a.drop(columns = ['PSH Flag Count', 'ECE Flag Count', 'RST Flag Count', 'ACK Flag Count', 'Fwd Packet Length Min', 'Bwd Packet Length Min', 'Packet Length Min', 'Protocol', 'Down/Up Ratio','Bwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd PSH Flags', 'Bwd URG Flags', 'CWE Flag Count', 'FIN Flag Count', 'Fwd Bulk Rate Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd URG Flags'])
    except KeyError:
        pass
    a = a.replace([np.inf, -np.inf], 0).fillna(0)
    X,y = a.drop(columns = ["Label"]).to_numpy(),a.Label.to_numpy().astype("int32")
    X = vt.transform(X)
    if model == "dt":
        X = std.transform(X)
        model = DecisionTreeClassifier()
    elif model == "gnb":
        X = std.transform(X)
        model = GaussianNB()
    elif model == "lr":
        X = std.transform(X)
        model = LogisticRegression()
    elif model == "lgbm":
        X = std.transform(X)
        model = LGBMClassifier(verbosity=-1)
    elif model == "mnb":
        X = discretizer.transform(X)
        model = MultinomialNB()
    elif model == "mymnb":
        X = discretizer.transform(X)
        model = MyMNB()
    elif model == "mygnb":
        X = std.transform(X)
        model = MyGNB()

    X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE)
        
    model.fit(X_train,y_train)
    y_pred = model.predict(X_valid)
    y_score = model.predict_proba(X_valid)[:,1]
    print_metrics(name,y_valid,y_pred,y_score)
    
    with open(f"models/{file_name}.pkl","wb") as f:
        pickle.dump(model,f)
    
    print(f"Model saved at models/{file_name}.pkl")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path",type=str,required=True,help="Path to the dataset")
    parser.add_argument("--model",type=str,required=True,choices=["dt","gnb","lr","lgbm","mnb","mymnb","mygnb"],help="Model to use")
    parser.add_argument("--file_name",type=str,required=True,help="Name of the file to save the model")
    args = parser.parse_args()
    main(args.path,args.model,args.file_name)