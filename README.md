# Malware-Detection
This project aims to correctly detect network intrusion attack from network packets, collected from CICFlowMeter
# Installation & running guide:
## Prerequisites:
This project assumes Python 3 was used. Used package:
- sklearn
- pandas
- numpy
- lightgbm
## Installation guide:
- Install the source code and extract to a folder
- Install the dataset from the source [here](https://www.kaggle.com/datasets/nguyenhoangsontung/cic-ids-self-collections)
- Open CMD in the main folder, which should be the malware-detection folder. Your current location should look like this:
```bash
C:\abcxyz\...\CIC>
```
## Folder structure:
- Datasets: contains the dataset:
- models: contains the pre-trained models and also the tools for data preprocessing
- trained: contains your model after training
- src:
    + inference_test.ipynb: notebook related to experiment
    + preprocess.ipynb: notebook related to data preprocessing
    + training_cont.ipynb & training_discrete.ipynb: notebooks related to training phase
    + NaiveBayes.py: python module contains the implementation of Naive Bayes
    + train_new.py: cls python files to train new models
    + test.py: cls python files to test models
    + inference.py: cls python files to inference from the data
## Running guide:
- To train a new model, please run:


```bash
python train_new.py 
```


The parameter for the cls are:
- --path: the file path of the train dataset
- --valid_path: the file path of the valid dataset(optional)
- --file_name: the name of the model to be saved
- --model: the model type to be used


For example:
```bash
python train_new.py --path ./datasets/csv/train.csv --valid_path ./datasets/csv/test.csv --file_name first --model dt
```

- To test a model, please run:
```bash
python test.py 
```


The parameter for the cls are:
- --path: the file path of the test dataset
- --model_path: the file path of the trained model
- --model_name: the name of the model to be used


For example:
```bash
python test.py --path ./datasets/csv/test.csv --model_path ./trained/first.pkl --model_name dt
```


- To test a model, please run:
```bash
python inference.py 
```


The parameter for the cls are:
- --dir: the directory path of the inference dataset
- --n: the number of samples to show
- --model_path: the file path of the trained model
- --model_name: the name of the model to be used


For example:
```bash
python inference.py --dir ./datasets/csv/inference.csv --n 100 --model_path ./trained/first.pkl --model_name dt
```
