import random
import numpy as np
from netcal.metrics import ECE
from scipy.stats import beta, uniform, norm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from DatasetGenerator import SyntheticGaussianGenerator, SyntheticCauchyGenerator, SyntheticBetaGenerator
import warnings
warnings.filterwarnings('once')
random.seed(0)

def log_pdf_norm(mu, sigma, sample):
    return norm.logpdf(sample, mu, sigma)
def sigmoid(m, b, X):
    return 1 / (1 + np.exp(-m*X-b))
def optimal_sigmoid_preds(mu0, mu1, sig, X, base_generator_type = 'gaussian'):
    if base_generator_type == 'gaussian':
        W = (mu1 - mu0) / (sig * sig)
        b = (mu0 * mu0 - mu1 * mu1) / (2 * sig * sig)
    else:
        print('A7A')
        if base_generator_type == 'cauchy':
            generator = SyntheticCauchyGenerator(p = 0.5)
            generator.set_cauchy_parameters(mu0, mu1, sig, sig)
        elif base_generator_type == 'beta':
            generator = SyntheticBetaGenerator(p = 0.5)
            generator.set_beta_parameters(mu0, mu1, sig, sig)
        XX, y = generator.generate(int(1e2))
        XX = XX.reshape(-1, 1)
        tmp_clf = LogisticRegression(random_state=0, solver='lbfgs', penalty = 'none', C = 0)
        tmp_clf.fit(XX, y)
        W = tmp_clf.coef_[0]; b = tmp_clf.intercept_[0]
    return sigmoid(W, b, X)


class uniform_prior():
    def __init__(self, l0min, l0max, l1min, l1max, lsmin=1, lsmax=1):
        self.l0min = l0min; self.l0max = l0max #negative class
        self.l1min = l1min; self.l1max = l1max #positive class
        self.lsmin = lsmin; self.lsmax = lsmax #standard deviation

    def sample(self, num_samples):
        np.random.seed(42)
        mu0s   = uniform.rvs(self.l0min, self.l0max - self.l0min, size = num_samples)
        mu1s   = uniform.rvs(self.l1min, self.l1max - self.l1min, size = num_samples)
        sigmas = uniform.rvs(self.lsmin, self.lsmax - self.lsmin, size = num_samples)
        return mu0s, mu1s, sigmas

    def draw_priors(self, priorN = 1e5):
	    #prior of mean of negative distribution
        mu0P = uniform.rvs(self.l0min, self.l0max - self.l0min, size = int(priorN))
	    #prior of mean of positive distribution
        mu1P = uniform.rvs(self.l1min, self.l1max - self.l1min, size = int(priorN))
	    #prior of stdev distribution
        sigmaP = uniform.rvs(self.lsmin, self.lsmax - self.lsmin, size = int(priorN))
        return mu0P, mu1P, sigmaP

    def log_pdf(self, a, b, samples):
        return uniform.logpdf(samples, a, b - a)
    
    def log_pdf_prior(self, mu0s, mu1s):
        return np.sum(self.log_pdf(self.l0min, self.l0max, mu0s)) + np.sum(self.log_pdf(self.l1min, self.l1max, mu1s))

    def log_likelihood(self, mu0P, mu1P, sigP):
        log_pdf0 = np.sum(self.log_pdf(self.l0min, self.l0max - self.l0min, self.mu0P))
        log_pdf1 = np.sum(self.log_pdf(self.l1min, self.l1max - self.l1min, self.mu1P))
        if self.lsmax > self.lsmin:
            log_pdfs = np.sum(self.log_pdf(self.lsmin, self.lsmax - self.lsmin, self.sigmaP))
        else:
            log_pdfs = 0
        return log_pdf0 + log_pdf1 + log_pdfs

class beta_prior():
    def __init__(self, a0, b0, a1, b1, shift=1, lsmin=1, lsmax=1):
        self.a0 = a0; self.b0 = b0
        self.a1 = a1; self.b1 = b1
        self.shift = shift
        self.lsmin = lsmin; self.lsmax = lsmax

    def sample(self, num_samples):
        np.random.seed(42)
        mu0s   = beta.rvs(self.a0, self.b0, size = num_samples)
        mu1s   = beta.rvs(self.a1, self.b1, size = num_samples) + self.shift
        sigmas = uniform.rvs(self.lsmin, self.lsmax - self.lsmin, size = num_samples)
        return mu0s, mu1s, sigmas

    def draw_priors(self, priorN = 1e5):
	    #prior of mean of negative distribution
        mu0P = beta.rvs(self.a0, self.b0, size = int(priorN))
	    #prior of mean of positive distribution
        mu1P = beta.rvs(self.a1, self.b1, size = int(priorN)) + self.shift
	    #prior of distribution stddev is always 1
        sigmaP = np.ones(int(priorN))
        return mu0P, mu1P, sigmaP
    
    def log_pdf(self, a, b, samples):
        return beta.logpdf(samples, a, b)
    
    def log_pdf_prior(self, mu0s, mu1s):
        return np.sum(self.log_pdf(self.a0, self.b0, mu0s)) + np.sum(self.log_pdf(self.a1, self.b1, (mu1s - self.shift)))

    def log_likelihood(self):
        log_pdf0 = np.sum(self.log_pdf(self.a0, self.b0, self.mu0P))
        log_pdf1 = np.sum(self.log_pdf(self.a1, self.b1, (self.mu1P-self.shift)))
        if self.lsmax > self.lsmin:
            log_pdfs = np.sum(self.log_pdf(self.lsmin, self.lsmax - self.lsmin, self.sigmaP))
        else:
            log_pdfs = 0
        return log_pdf0 + log_pdf1 + log_pdfs

class BayesianDataLogisticRegression(object):
    def __init__(self, prior_obj, prior_type='uniform', prior_samples=int(1e5), base_generator_type = 'gaussian'):
        self.prior_type = prior_type
        self.prior = prior_obj
        self.prior_samples = prior_samples
        self.base_generator_type = base_generator_type

    def calc_posts(self, S0, S1):
        log_likelihood  = np.sum(log_pdf_norm([[m] for m in self.mu0P], [[s] for s in self.sigmaP], S0), axis = 1)
        log_likelihood += np.sum(log_pdf_norm([[m] for m in self.mu1P], [[s] for s in self.sigmaP], S1), axis = 1)
        self.posts = log_likelihood + self.prior.log_pdf_prior(self.mu0P, self.mu1P)
        self.posts = np.exp(self.posts - max(self.posts))
        self.posts = np.divide(self.posts, self.posts.sum())

    def fit(self, X_train, y_train):
        self.mu0P, self.mu1P, self.sigmaP = self.prior.draw_priors()
        self.calc_posts(S0 = X_train[np.where(y_train == 0)], S1 = X_train[np.where(y_train == 1)])
        self.mean_mu0_coeff = np.sum(np.multiply(self.posts, self.mu0P)); 
        self.mean_mu1_coeff = np.sum(np.multiply(self.posts, self.mu1P)); 
        self.mean_sigma_coeff = np.sum(np.multiply(self.posts, self.sigmaP))
        
        _, self.mu0_top, self.mu1_top, self.sigma_top = zip(*sorted(zip(self.posts, self.mu0P, self.mu1P, self.sigmaP)))
        self.mu0_top = self.mu0_top[-1]; self.mu1_top = self.mu1_top[-1]; self.sigma_top = self.sigma_top[-1]

    def predict(self, X, threshold = 0.5):
        return self.predict_proba(X) >= threshold

    def predict_logloss(self, X, y):
        preds_probs = self.predict_proba(X)
        #print(y[:10], preds_probs[:10], log_loss(y[:10], preds_probs[:10], labels = [0, 1]))
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

class BayesianDataLogisticRegressionMeanPreds(BayesianDataLogisticRegression):
    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        preds_probs = np.zeros((len(X), 1), dtype=np.float64)
        for mu0, mu1, sig, post in zip(self.mu0P, self.mu1P, self.sigmaP, self.posts):
            prior_output = optimal_sigmoid_preds(mu0, mu1, sig, X, base_generator_type = self.base_generator_type)
            preds_probs = np.add(preds_probs, np.multiply(post, prior_output))
        preds_probs = np.squeeze(preds_probs)
        return preds_probs

class BayesianDataLogisticRegressionMeanCoeffs(BayesianDataLogisticRegression):
    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        preds_probs = optimal_sigmoid_preds(self.mean_mu0_coeff, self.mean_mu1_coeff, self.mean_sigma_coeff, X, base_generator_type = self.base_generator_type)
        return np.squeeze(preds_probs)

class BayesianDataLogisticRegressionMAP(BayesianDataLogisticRegression):
    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        preds_probs = optimal_sigmoid_preds(self.mu0_top, self.mu1_top, self.sigma_top, X, base_generator_type = self.base_generator_type)
        return np.squeeze(preds_probs)