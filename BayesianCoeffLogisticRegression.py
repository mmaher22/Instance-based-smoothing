import random
import numpy as np
import pymc3 as pm
import theano.tensor as Tht
from netcal.metrics import ECE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('once')
def sigmoid(m, b, X):
    #Z = np.dot(row, self.Ws)
    return 1 / (1 + np.exp(-np.dot(X, m)-b))

class BayesianCoeffLogisticRegression():
    def __init__(self, tolerance = 1e-5):
        random.seed(0)
        
    def transform_data(self, X):
        return self.scaler.transform(X)

    def fit(self, X_train, y_train):
		#compute the mean, and standard deviation for feature preprocessing (Gelman, 2008)
        #print(X_train.shape)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
		#to transform data to have standard deviation of 0.5 (Gelman, 2008)
        self.scaler.var_ *= 4
        X_train = self.transform_data(X_train)

        samples = 1000
        with pm.Model() as _:
		    # betas
            alpha = pm.Cauchy('alpha', 0., 10)
            betas = []
            for i in range(X_train.shape[1]):
                betas.append(pm.Cauchy('beta' + str(i), 0., 2.5))
            #beta = pm.Cauchy('beta', 0., 2.5)
		    # logit
            logit_p =  alpha# + beta * X_train)
            for i in range(X_train.shape[1]):
                logit_p += betas[i] * X_train[:, i]#.append(pm.Cauchy('beta' + str(i), 0., 2.5))
            p = Tht.exp(logit_p) / (1 + Tht.exp(logit_p))
		    # likelihood
            _ = pm.Binomial('likelihood', n = 1, p = p, observed = y_train)
		    # inference
            start = pm.find_MAP()
            #step  = pm.NUTS(scaling = start)
            #trace = pm.sample(samples, step, progressbar=False, chains=2, cores=1)
            #summary = pm.summary(trace)['mean']
        self.b_map  = start['alpha']; self.a_map  = []
        for i in range(X_train.shape[1]):
            self.a_map.append(start['beta'+str(i)])
        self.a_map = np.array(self.a_map)
        #self.a_mmse = summary[0]; 		 self.b_mmse = summary[1]
        
    def predict_proba(self, X, mode = 'map'):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = self.transform_data(X)
        if mode == 'map':
            self.a = self.a_map; self.b = self.b_map
        else:
            self.a = self.a_mmse; self.b = self.b_mmse
        preds_probs = sigmoid(self.a, self.b, X)
        return preds_probs#np.squeeze(preds_probs)

    def predict(self, X, threshold = 0.5, mode = 'map'):
        return self.predict_proba(X, mode = mode) >= threshold

    def predict_logloss(self, X, y, mode = 'map'):
        preds_probs = self.predict_proba(X, mode = mode)
        return log_loss(y, preds_probs, labels = [0, 1])
    
    def predict_accuracy(self, X, y, threshold = 0.5, mode = 'map'):
        return accuracy_score(y, self.predict(X, threshold = threshold, mode = mode))
    
    def predict_ece(self, X, y, mode = 'map', bins = 10):
        ece = ECE(bins)
        calibrated_score = ece.measure(self.predict_proba(X, mode = mode), y)
        return calibrated_score
    
    def predict_ece_logloss(self, X, y, bins = 10, mode = 'map'):
        preds_probs = self.predict_proba(X, mode = mode)
        #print(preds_probs, preds_probs.shape)
        ece = ECE(bins)
        calibrated_score = ece.measure(preds_probs, y)
        #print(y, preds_probs)
        return calibrated_score, log_loss(y, preds_probs, labels = [0, 1])