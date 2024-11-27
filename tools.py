from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def print_metrics(model_name,y_true,y_pred,y_score):
    print(model_name)
    accuracy = round(accuracy_score(y_true,y_pred),5)
    f1 = round(f1_score(y_true,y_pred),5)
    roc_auc = round(roc_auc_score(y_true,y_score),5)
    print("Accuracy: ",accuracy)
    print("F1 Score: ",f1)
    print("ROC AUC: ",roc_auc)
    return model_name, accuracy, f1, roc_auc

def draw_curve(y_true_list,y_score_list,model_name_list):
    plt.figure(figsize=(10,10))
    for y_true,y_score,model_name in zip(y_true_list,y_score_list,model_name_list):
        fpr,tpr,_ = roc_curve(y_true,y_score)
        plt.plot(fpr,tpr,label=model_name)
    plt.plot([0,1],[0,1],'--',label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def prepare_data(df_name,vt,hehe):
    df = pd.read_parquet(df_name)
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)].dropna()
    X,y = df.drop(columns = ["Label"]).to_numpy(),df.Label.to_numpy().astype("int32")
    X = vt.transform(X)
    X = hehe.transform(X)
    return X,y

def prepare_data_balanced(df_name,vt,hehe):
    df = pd.read_parquet(df_name)
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)].dropna()
    min_size = df["Label"].value_counts().min()
    df = df.groupby("Label").apply(lambda x: x.sample(min_size)).reset_index(drop=True)
    X,y = df.drop(columns = ["Label"]).to_numpy(),df.Label.to_numpy().astype("int32")
    X = vt.transform(X)
    X = hehe.transform(X)
    return X,y    

def prepare_data_inference(df_name,vt,hehe):
    df = pd.read_csv(df_name)
    idx = df.apply(
    lambda row: f"{min(row['Source IP'], row['Destination IP'])}-{min(row['Source Port'], row['Destination Port'])}-"
                f"{max(row['Source IP'], row['Destination IP'])}-{max(row['Source Port'], row['Destination Port'])}-"
                f"{row['Protocol']}",
    axis=1).to_list()
    df = df.drop(columns = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', "Timestamp"])
    df = df.drop(columns = ['PSH Flag Count', 'ECE Flag Count', 'RST Flag Count', 'ACK Flag Count', 'Fwd Packet Length Min', 'Bwd Packet Length Min', 'Packet Length Min', 'Protocol', 'Down/Up Ratio','Bwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd PSH Flags', 'Bwd URG Flags', 'CWE Flag Count', 'FIN Flag Count', 'Fwd Bulk Rate Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd URG Flags'])
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    X = df.to_numpy()
    X = vt.transform(X)
    X = hehe.transform(X)
    return X,idx