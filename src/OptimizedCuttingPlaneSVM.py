import numpy as np
from debugpy.common.log import warning
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from .utils import reduced_problem_solver
# Optimized Cutting Plane SVM
class OCPSVM(BaseEstimator,ClassifierMixin):
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
        #TODO: lucky guess on result

    def _create_hyperplane(self, X, y, w_c,dot_prods=None):
        if dot_prods is None:
            dot_prods = np.dot(X, w_c).reshape(-1,)
        cond=np.where(dot_prods*y<=1,1,0)
        S=(X.T*y*cond).T
        alpha=-np.mean(S, axis=0)

        beta=self._risk(dot_prods, y) -np.dot(alpha,w_c)
        return alpha, beta

    def _risk(self,dot_prods,y):
        comp=np.where(dot_prods*y<1,1-dot_prods*y,0)
        return np.mean(comp)

    def _line_search(self,w,X,y):
        # TODO need revision
        w_prev=self.ws_best[:,-1]


        A_oh=np.linalg.norm(w-w_prev,2)**2
        B_oh=np.dot(w_prev,w-w_prev)

        prev_dots= np.dot(X, w_prev)
        dots=np.dot(X, w)

        Bs=y*self.C*(prev_dots-dots)
        Cs=self.C*(1-prev_dots)
        Ks=-Cs/(Bs+1e-10)
        cond1=(Bs> 0) & (Ks <= 0)
        cond2=(Bs<0)& (Ks > 0)
        cond= (cond1 | cond2)
        max=B_oh+np.sum(Bs*cond)
        if max>0:
            k=0
            return w_prev,prev_dots


        idx=np.argsort(Ks)
        Ks=Ks[idx]
        Bs= Bs[idx]
        Cs= Cs[idx]
        consts=Ks*A_oh+B_oh
        tmp1=Bs.reshape(-1,1)
        tmp2=Ks.reshape(-1,1)
        Mat=(((tmp2@tmp1.T+Cs)>0)*Bs).T
        subdiff=np.sum(Mat,axis=1)
        grads=consts+subdiff
        k = Ks[-1]

        def _eval_for_k(k_0):
            c=(k_0*Bs+Cs>0)*Bs
            constr=np.sum(c)
            const=A_oh*k_0+B_oh
            val= const + constr
            return val
        for i,j in zip(range(len(Ks)-1), range(1, len(Ks))):
            if Ks[j]<0:
                continue
            if np.abs(grads[i])<1e-9:
                k=Ks[i]
                break
            if grads[i]*grads[j]<0:
                tmp=-(subdiff[i]+B_oh)/A_oh
                if Ks[i] < tmp < Ks[j] and tmp>0:
                    k=tmp
                    break
        w_b= w_prev + k * (w - w_prev)
        dots_b=prev_dots+k*(dots-prev_dots)
        w_c=self.lamb*w_b+ (1-self.lamb)*w_prev
        dots_c=self.lamb*dots_b + (1-self.lamb)*dots
        self.ws_best = np.hstack((self.ws_best, w_b.reshape(-1, 1)))
        return w_c,dots_c


    def _objective_function(self, w, dots, y):

        risk = self._risk(dots, y)
        return 0.5 * np.linalg.norm(w) ** 2 + self.C * risk
    def _sub_objective(self,w):
        dot_prods=np.dot(self.alphas.T,w).reshape(-1,)+ self.betas
        r_t=np.max(dot_prods)
        r= max(r_t,0)
        return 0.5 * np.linalg.norm(w) ** 2+self.C*r
    def fit(self, X, y):
        self.N, self.M = X.shape
        self._init_loop(X, y)
        for i in range(1,self.max_iter):
            w,xi=reduced_problem_solver(self.alphas, self.betas, self.C, self.M)
            w_c,dots=self._line_search(w,X,y)
            self.W=w_c
            alpha, beta = self._create_hyperplane(X, y, w_c)
            self.alphas = np.hstack((self.alphas, alpha.reshape(-1, 1)))
            self.betas=np.append(self.betas, beta)
            obj=self._objective_function(w_c,dots,y)
            sub_obj=self._sub_objective(w_c)
            if self.verbose:
                print(f"Iteration {i}: Objective = {obj}, Sub-Objective = {sub_obj}")

            if obj-sub_obj<self.tol:
                if self.verbose:
                    print(f"Convergence reached at iteration {i}.")
                self.is_fitted_=True
                return self
        warning(f"Maximum iterations {self.max_iter} reached without convergence.")
        self.is_fitted_=False
        return self

    def predict(self,X):
        return np.sign(np.dot(X, self.W))
    def score(self, X, y,score_fun=accuracy_score):
        predictions = self.predict(X)
        return score_fun(y, predictions)


