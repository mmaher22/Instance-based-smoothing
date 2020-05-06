import random
import numpy as np
from math import log
from netcal.metrics import ECE
from scipy.optimize import fmin_bfgs
from scipy.special import expit, xlogy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('once')

def sigmoid(m, b, X):
    #Z = np.dot(row, self.Ws)
    return 1 / (1 + np.exp(-np.dot(X, m)-b))

def _sigmoid_calibration(X, y,  T1 = None, tol = 1e-5):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    if T1 is None:
        T = np.zeros(y.shape)
        T[y <= 0] = (prior1 + 1.) / (prior1 + 2.)
        T[y > 0] = 1. / (prior0 + 2.)
        T1 = 1. - T
    else:
        T = 1. - T1
        
    def objective(AB):
        tmp = 0
        for i in range(X.shape[1]):
            tmp += AB[i] * X[:,i]
        tmp += AB[X.shape[1]]
        #P = expit(-(AB[0] * X + AB[1]))
        P = expit(-(tmp))
        loss = -(xlogy(T, P) + xlogy(T1, 1. - P))
        return loss.sum()

    def grad(AB):
        # gradient of the objective function
        tmp = 0
        for i in range(X.shape[1]):
            tmp += AB[i] * X[:,i]
        tmp += AB[X.shape[1]]
        #P = expit(-(AB[0] * X + AB[1]))
        P = expit(-(tmp))
        TEP_minus_T1P = T - P
        dA = np.dot(TEP_minus_T1P, X)
        dB = np.sum(TEP_minus_T1P)
        out_grad = np.append(dA, dB)
        return out_grad
    
    AB0 = np.array([0.] * X.shape[1] + [log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False, gtol = tol, maxiter=10000)
    return AB_[0:-1], AB_[-1]

class CESmoothedLogisticRegression():
    def __init__(self, tolerance = 1e-5):
        self.tolerance = tolerance
        random.seed(0)

    def fun(self, x):
        return 2*x - 1

    def smooth_labels(self, X, y):
        clf2 = LogisticRegression(random_state=0, solver='lbfgs', penalty = 'none', tol=self.tolerance)
        clf2.fit(X, y)
        y_pred = clf2.predict_proba(X)
        y_smoothed = np.zeros(len(y))
        N0 = np.sum(y == 0); N1 = np.sum(y == 1)
        platt_neg = 1 / (N0 + 2)
        platt_pos = 1 / (N1 + 2)
        for i in range(len(y)):
            if y[i] > 0:
                #print(y_pred[i][1])
                y_smoothed[i] = 1 - platt_pos * self.fun(y_pred[i][1])
            else:
                y_smoothed[i] = platt_neg * self.fun(1 - y_pred[i][1])
        return y_smoothed

    def fit(self, X_train, y_train):
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        y_train_smoothed = self.smooth_labels(X_train, y_train)
        self.a, self.b = _sigmoid_calibration(np.squeeze(X_train), y_train, y_train_smoothed, tol = self.tolerance)
	
    def predict_proba(self, X):
        #print(self.b, self.b.shape, self.a, self.a.shape)
        preds_probs = sigmoid(self.a, self.b, X)
        return preds_probs

    def predict(self, X, threshold = 0.5):
        return self.predict_proba(X) >= threshold

    def predict_logloss(self, X, y):
        preds_probs = sigmoid(self.a, self.b, X)
        return log_loss(y, preds_probs, labels = [0, 1])

    def predict_accuracy(self, X, y, threshold = 0.5):
        return accuracy_score(y, self.predict(X, threshold = threshold))

    def predict_ece(self, X, y, bins = 10):
        ece = ECE(bins)
        calibrated_score = ece.measure(self.predict_proba(X), y)
        return calibrated_score
    
    def predict_ece_logloss(self, X, y, bins = 10):
        preds_probs = self.predict_proba(X)
        ece = ECE(bins)
        calibrated_score = ece.measure(preds_probs, y)
        return calibrated_score, log_loss(y, preds_probs, labels = [0, 1])