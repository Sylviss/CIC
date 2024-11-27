import numpy as np
import pandas as pd
import math
class GaussianNB:
    def __init__(self,priors = None):
        self.priors = priors
    
    def fit(self,X,y):
        if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
            X = X.to_numpy()
            
        self.total,self.num_feature = X.shape
        self.classes = np.unique(y)
        self.num_class = len(self.classes)
        
        self.P_c = [0 for _ in self.classes]
        
        self.count_c = np.zeros(self.num_class, dtype=int)
        self.sum_x_c = [np.zeros((self.num_class, self.num_feature)) for _ in range(self.num_class)]
                
        for i,label in enumerate(self.classes):
            class_indices = np.where(y==label)[0]
            self.count_c[i] = class_indices.size
            self.sum_x_c[i] = np.sum(X[class_indices], axis=0)
            
        
        self.dist = []
        
        for i in range(self.num_class):
            mean = self.sum_x_c[i]/self.count_c[i]
            var = ((X[y == self.classes[i]] - mean) ** 2).sum(axis=0) / (self.count_c[i] - 1)
            self.dist.append((mean, var**0.5))
            
        if self.priors !=None:
            for label,value in zip(range(self.num_class),self.priors):
                self.P_c[label]=value
        else:
            for label in range(self.num_class):
                self.P_c[label]=self.count_c[label]/self.total
            
    def predict(self,X):
        res = self.predict_proba(X)
        res = np.argmax(res,axis=1)
        return res
    
    def predict_proba(self,X):
        if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
            X = X.to_numpy()
        res = np.zeros((X.shape[0],self.num_class))
        for i in range(self.num_class):
            mean,std = self.dist[i]
            std = np.where(std == 0, 1e-10, std)
            llh = np.log((1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mean) / std) ** 2))
            llh = np.where(np.isfinite(llh), llh, -1e10)
            class_log_sum = np.sum(llh,axis=1)
            res[:,i] = np.log(self.P_c[i])+class_log_sum
        res = res-np.max(res,axis=1)[:,None]
        res = np.exp(res)
        res = res/np.sum(res,axis=1)[:,None]
        return res
        
class MultinomialNB:
    def __init__(self,alpha=1,priors=None):
        self.alpha = alpha
        self.priors = priors

    def fit(self,X ,y):
        if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
            X = X.to_numpy()
        self.total,self.num_feature = X.shape
        self.classes = sorted(list(set(y)))
        self.class_dict = {label:i for i,label in enumerate(self.classes)}
        self.num_class = len(self.classes)
        self.count_c = {label:0 for label in self.classes}
        self.P_c = {label:0 for label in self.classes}
        self.count_x_c = {label:[{} for _ in range(self.num_feature)] for label in self.classes}
        
        for x,label in zip(X,y):
            for i,value in enumerate(x):
                self.count_x_c[label][i][int(value)]=self.count_x_c[label][i].get(int(value),0)+1
            self.count_c[label]+=1
            
        if self.priors !=None:
            for label,value in zip(self.classes,self.priors):
                self.P_c[label]=value
        else:
            for x in self.classes:
                self.P_c[x]=self.count_c[x]/self.total
                
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        res = []
        for x in X:
            log_predict = {label: np.log(self.P_c[label]) for label in self.classes}

            for label in self.classes:
                for m, value in zip(self.count_x_c[label], x):
                    smoothed_prob = (m.get(int(value), 0) + self.alpha) / (
                        self.count_c[label] + self.alpha * len(m)
                    )
                    log_predict[label] += np.log(smoothed_prob + 1e-9)
            res.append(max(log_predict, key=lambda a: log_predict[a]))

        return res

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        res = np.zeros((X.shape[0], self.num_class))

        for i, x in enumerate(X):
            log_predict = {label: np.log(self.P_c[label]) for label in self.classes}

            for label in self.classes:
                for m, value in zip(self.count_x_c[label], x):
                    smoothed_prob = (m.get(int(value), 0) + self.alpha) / (
                        self.count_c[label] + self.alpha * len(m)
                    )
                    log_predict[label] += np.log(smoothed_prob + 1e-9)
            max_log_prob = max(log_predict.values())
            predict = {label: math.exp(log_predict[label] - max_log_prob)for label in self.classes}
            total_sum = sum(predict.values())
            for label in self.classes:
                res[i][self.class_dict[label]] = predict[label] / total_sum
        return res
