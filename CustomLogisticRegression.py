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

def smooth_labels(y_train, f_pos, f_neg):
    y_train_smoothed = np.zeros(len(y_train))
    for i in range(len(y_train)):
        if y_train[i] > 0:
            y_train_smoothed[i] = 1 - f_pos
        else:
            y_train_smoothed[i] = f_neg
    return y_train_smoothed

def _sigmoid_calibration(X, y,  T1 = None, tol = 1e-3):
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
        return out_grad#np.array([dA, dB])
    
    AB0 = np.array([0.] * X.shape[1] + [log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False, gtol = tol)
    return AB_[0:-1], AB_[-1]

class CustomLogisticRegression():
	def __init__(self, smoothing_factor_pos = 0, smoothing_factor_neg = 0, tolerance = 1e-3, regularization = 'none', regularization_strength = 0, platt_scaling = False):
		self.smoothing_factor_pos = smoothing_factor_pos
		self.smoothing_factor_neg = smoothing_factor_neg
		self.platt = platt_scaling
		self.regularization = regularization
		self.reg_strength = regularization_strength #Inverse of Regularization Strength (Must be positive)
		self.tolerance = tolerance
		random.seed(0)

	def fit(self, X_train, y_train):
		if self.platt == True:
			y_train_smoothed = None
			self.a, self.b = _sigmoid_calibration(X_train, y_train, y_train_smoothed, tol = self.tolerance)
		elif self.smoothing_factor_pos > 0 or self.smoothing_factor_neg > 0:
			y_train_smoothed = smooth_labels(y_train, self.smoothing_factor_pos, self.smoothing_factor_neg)
			self.a, self.b = _sigmoid_calibration(X_train, y_train, y_train_smoothed, tol = self.tolerance)
		else:
			if len(X_train.shape) < 2:
				X_train = X_train.reshape(-1, 1)
                
			if self.regularization == 'l1':
				clf = LogisticRegression(random_state=0, solver='saga', penalty = self.regularization, C = self.reg_strength, tol=self.tolerance)
			else:
				clf = LogisticRegression(random_state=0, solver='lbfgs', penalty = self.regularization, C = self.reg_strength, tol=self.tolerance)
			clf.fit(X_train, y_train)
			self.a = clf.coef_[0]; self.b = clf.intercept_[0]
		#print('COEFFS:', self.a, self.b)
	
	def predict_proba(self, X):
		preds_probs = sigmoid(self.a, self.b, X)
		return preds_probs

	def predict(self, X, threshold = 0.5):
		return self.predict_proba(X) >= threshold

	def predict_logloss(self, X, y):
		preds_probs = self.predict_proba(X)
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
		#print(calibrated_score, y, preds_probs)
		return calibrated_score, log_loss(y, preds_probs, labels = [0, 1])