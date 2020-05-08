import random
import numpy as np
from math import log
from netcal.metrics import ECE
from scipy.optimize import fmin_bfgs
from scipy.special import expit, xlogy
from sklearn.neighbors import KernelDensity
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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

class KDELogisticRegression():
    def __init__(self, kernel = 'gaussian', pos_kernel_bw = 'scott', neg_kernel_bw = 'scott', tolerance = 1e-3):
        self.kernel = kernel
        self.pos_kernel_bw = pos_kernel_bw
        self.neg_kernel_bw = neg_kernel_bw
        self.tolerance = tolerance
        random.seed(0)

    def create_kernel_density_estimators(self, S0, S1):
        #print(S0.shape, S1.shape)
        d = S0.shape[1] # number of features
        #print('d:', d, S1.shape[1], S0.shape[1])
        if self.neg_kernel_bw == 'scott':
            self.neg_kernel_bw = len(S0)**(-1./(d+4))
        elif self.neg_kernel_bw == 'silverman':
            self.neg_kernel_bw = (len(S0) * (d + 2) / 4.)**(-1. / (d + 4))
        self.neg_kde = KernelDensity(kernel = self.kernel, bandwidth=self.neg_kernel_bw).fit(S0)
		
        if self.pos_kernel_bw == 'scott':
            self.pos_kernel_bw = len(S1)**(-1./(d+4))
        elif self.pos_kernel_bw == 'silverman':
            self.pos_kernel_bw = (len(S1) * (d + 2) / 4.)**(-1. / (d + 4))
        self.pos_kde = KernelDensity(kernel = self.kernel, bandwidth=self.pos_kernel_bw).fit(S1)
        print('BW:', self.pos_kernel_bw, self.neg_kernel_bw)

    def smooth_labels(self, X, y):
        y_smoothed = np.zeros(len(y))
        N0 = np.sum(y == 0); N1 = np.sum(y == 1)
        d = X.shape[1]
        platt_neg = 1 / (N0 + 2)
        platt_pos = 1 / (N1 + 2)
        for i in range(len(y)):
            sample = X[i].reshape(1, -1)
            score_pos = self.pos_kde.score_samples(sample)
            score_neg = self.neg_kde.score_samples(sample)
            if y[i] > 0:
                y_smoothed[i] = 1 - platt_pos * score_neg / (score_pos + score_neg)
            else:
                y_smoothed[i] = platt_neg * score_pos / (score_pos + score_neg)
        return y_smoothed

    def fit(self, X_train, y_train):
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        ros = RandomOverSampler(random_state=0)
        X_train2, y_train2 = ros.fit_resample(X_train, y_train)
        if X_train2.ndim == 1:
            X_train2 = X_train2.reshape(-1, 1)
        self.create_kernel_density_estimators(S0 = X_train2[np.where(y_train2 == 0)], S1 = X_train2[np.where(y_train2 == 1)])

        y_train_smoothed = self.smooth_labels(X_train, y_train)
        self.a, self.b = _sigmoid_calibration(np.squeeze(X_train), y_train, y_train_smoothed, tol = self.tolerance)
	
    def predict_proba(self, X):
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