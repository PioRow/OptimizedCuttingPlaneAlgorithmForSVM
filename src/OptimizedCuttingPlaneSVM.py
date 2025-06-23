import numpy as np
from .utils import reduced_problem_solver
import torch
class OCPSVM:
    def __init__(self,tol=1e-7, max_iter=1000, C=1.0,lamb=0.1,verbose=False):
        self.tol = tol
        self.max_iter = max_iter
        self.C = C
        self.lamb=lamb
        self.alphas = None
        self.betas = None
        self.N= None
        self.M=None
        self.W=None
        self.ws_best = None
        self.verbose = verbose

    def _init_loop(self,X,y):
        self.N,self.M= X.shape
        w_c=np.zeros(self.M).reshape(-1,1)
        alpha,beta=self._create_hyperplane(X,y,w_c)
        self.alphas = alpha.reshape(-1,1)
        self.betas=np.array([beta])
        self.W=w_c
        self.ws_best=w_c

    def _create_hyperplane(self, X, y, w_c):
        dot_prods = np.dot(X, w_c).reshape(-1,)
        cond=np.where(dot_prods*y<=1,1,0)
        S=(X.T*y*cond).T
        alpha=-np.mean(S, axis=0)

        beta=self._risk(dot_prods, y) -np.dot(alpha,w_c)
        return alpha, beta
    def _risk(self,dot_prods,y):
        comp=np.where(dot_prods*y<1,1-dot_prods*y,0)
        return np.mean(comp)

    def _line_search(self,w):
        w_prev=self.ws_best[:,-1]
        print(w_prev.shape)
        A_oh=np.linalg.norm(w-w_prev,2)**2
        B_oh=np.dot(w_prev,w-w_prev)
        C_oh=0.5*np.linalg.norm(w_prev,2)**2

        return np.zeros(self.M)

    def fit(self, X, y):
        self.N, self.M = X.shape
        self._init_loop(X, y)
        for i in range(1,self.max_iter):
            w,xi=reduced_problem_solver(self.alphas, self.betas, self.C, self.M)
            w_b=self._line_search()
            w_c=self.lamb*w_b+(1-self.lamb)*w
            break;

    def predict(self,X):
        return np.sign(np.dot(X, self.W))


