import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import Bounds, LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import seaborn as sns
import sys


import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def weighted_quantile(v,prob,w):
    if(len(w)==0):
        w = np.ones(len(v))
    o = np.argsort(v)
    v = v[o]
    w = w[o]
    i = np.where(np.cumsum(w/np.sum(w)) > prob)[0]
    if(len(i)==0):
        return float('inf') # Can happen with infinite weights
    else:
        return v[np.min(i)]

# conformal prediction with least squares
def CP_LS(X, Y, x, alpha, weights = [], weights_new = 1):
    
    n = len(Y)
    
    if(len(weights)==0):
        weights = np.ones(n+1)
    if(len(weights)==n):
        weights = np.r_[weights,weights_new]
    weights = weights / np.sum(weights)

    XtX = (X.T).dot(X) + np.outer(x,x)
    a = Y - X.dot(np.linalg.solve(XtX,(X.T).dot(Y)))
    b = -X.dot(np.linalg.solve(XtX,x))
    a1 = -x.T.dot(np.linalg.solve(XtX,(X.T).dot(Y)))
    b1 = 1 - x.T.dot(np.linalg.solve(XtX,x))
    
    y_knots = np.sort(np.unique(np.r_[((a-a1)/(b1-b))[b1-b!=0],((-a-a1)/(b1+b))[b1+b!=0]]))
    y_inds_keep = np.where( ((np.abs(np.outer(a1+b1*y_knots,np.ones(n))) > \
       np.abs(np.outer(np.ones(len(y_knots)),a)+np.outer(y_knots,b))) *\
                             weights[:-1] ).sum(1) <= 1-alpha )[0] 
    y_PI = np.array([y_knots[y_inds_keep.min()],y_knots[y_inds_keep.max()]])
    if(weights[:-1].sum() <= 1-alpha):
        y_PI = np.array([-np.inf,np.inf])
    return y_PI, np.mean(y_PI[1] - y_PI[0])


def split_CP_LS(X_tr, Y_tr, X_cal, Y_cal, X_new, alpha, weights = [], weights_new = []):
    
    n = len(Y_cal)
    n_new = X_new.shape[0]
    y_PI = np.zeros(2*n_new).reshape((n_new,2))
    q = np.zeros(n_new)

    XtX = (X_tr.T).dot(X_tr)
    beta_tr = np.linalg.solve(XtX,(X_tr.T).dot(Y_tr))
    a = Y_cal - X_cal.dot(beta_tr)
    for i in range(n_new):
        if(len(weights)==0):
            wts = np.ones(n+1)
        if(len(weights)==n):
            wts = np.r_[weights,weights_new[i]]
        wts = wts / np.sum(wts)
        score = np.append(np.abs(a),float("inf"))
        qt = weighted_quantile(score,1 - alpha,wts)
        q[i] = min(qt, np.max(np.abs(a)))
        # q[i] = qt
    
    mu_new = X_new.dot(beta_tr)
     
    y_PI[:,0] = mu_new - q
    y_PI[:,1] = mu_new + q
    
    return y_PI, 2*np.mean(q)

def est_rho(d, mu_shift, t_idx, mis, clf_pre, marginal_rt):
    
    M = 20
    r_seq = []
    for m in range(M):
        X01, y01, X02, y02 = gen_dataset(10000, 10000, d, mu_shift, t_idx, mis)

        mu_vec = mu_shift*np.ones(d)/np.sqrt(d)
        cov2 = np.eye(d) + t_idx*np.outer(mu_vec, mu_vec)
        
        p01_ = multivariate_normal.pdf(X01, mean=mu_vec, cov=cov2)
        p00_ = multivariate_normal.pdf(X01, mean=np.zeros(d), cov=np.eye(d))
        p11_ = multivariate_normal.pdf(X02, mean=mu_vec, cov=cov2)
        p10_ = multivariate_normal.pdf(X02, mean=np.zeros(d), cov=np.eye(d))

        q0 = (p01_/p00_)
        q = (p11_/p10_)
        q = q/np.mean(q0)
        q0 = q0/np.mean(q0)

        log_prob_fit0 = clf_pre.predict_log_proba(X01)[:,0]
        prob_fit0 = clf_pre.predict_proba(X01)[:,0]
        w0_out0 = log_prob_fit0 - np.log(1 - prob_fit0) + np.log(marginal_rt)
        w0_out_p0 = np.exp(w0_out0)

        log_prob_fit = clf_pre.predict_log_proba(X02)[:,0]
        prob_fit = clf_pre.predict_proba(X02)[:,0]
        w0_out = log_prob_fit - np.log(1 - prob_fit) + np.log(marginal_rt)
        w0_out_p = np.exp(w0_out)
        
        w0_out_p = w0_out_p/np.mean(w0_out_p0)
        w0_out_p0 = w0_out_p0/np.mean(w0_out_p0)
        
        r_hat = np.mean(q0 * np.log(q0/w0_out_p0))
        
        r_seq.append(r_hat)
    
    rho_hat = np.mean(r_seq)
    
    return rho_hat



def entr(w):
    res = np.array([])
    for wi in w:
        if np.abs(wi)<=1e-4:
            res = np.append(res,0)
        elif wi > 0:
            res = np.append(res,wi*np.log(wi))
        else: res = np.append(res,np.inf)
    return res

def log_sum_exp(a):
    return np.log(np.sum(np.exp(a)))

def iso_dro(K,p,r,rho,iso,f_div):
    
    p_mean = np.mean(p)
    p_ = p/p_mean
    alpha = np.sum(r*p_)
    var_r = np.sum(p_*r**2) - alpha**2

    # monotonicity constraint
    A = np.eye(K)
    for k in range(1,K):
        A[k,k-1] = -1

    fun = lambda x: -np.sum(x*r*p_) + alpha
    cons1 = lambda x: np.sum(p_*x)
    
    if f_div == 'bounds':
        u_bnd = 1+rho
        l_bnd = 1/u_bnd
        t_star = (u_bnd - 1)/(u_bnd - l_bnd)
        w_star = l_bnd * np.ones(len(p))
        r_u = np.unique(r)
        if len(r_u) > 1:
            p_u = np.zeros(len(r_u))
            for i in range(len(p_u)):
                p_u[i] = np.mean(r == r_u[i])
            p_sum = 0
            k = 0
            while p_sum < t_star:
                p_sum += p_u[k]
                k += 1
            r_star = r_u[k-1]
            w_star[r > r_star] = u_bnd
            if p_sum > t_star:
                dens = np.sum(p[r == r_star])
                eta = l_bnd + ((u_bnd - l_bnd) * p_sum - (u_bnd - 1))/dens
                w_star[r == r_star] = eta
        else:
            w_star = np.ones(len(p))

        delta = (np.sum(p_*r*w_star) - alpha)*p_mean
    else:
        if f_div == 'chi-sq':
            cons2 = lambda x: np.sum(p_*(x-1)**2)
        elif f_div == 'kl':
            cons2 = lambda x: np.sum(entr(x)*p_)
        C1 = NonlinearConstraint(cons1, 1/p_mean, 1/p_mean)
        C2 = NonlinearConstraint(cons2, -np.inf, rho/p_mean)
        C3 = LinearConstraint(A, np.zeros(K), np.inf*np.ones(K))
        B1 = Bounds(np.zeros(K), np.inf*np.ones(K))
        trial_id = 0
        res_status = 1
        while res_status and (trial_id<=10):
            trial_id = trial_id + 1
            x0 = np.random.uniform(0.2,2,K)
            if iso:
                res = minimize(fun, x0, constraints=(C1,C2,C3), bounds=B1,
                               method='SLSQP',
                               tol = 1e-4) #, options={'disp': True, 'maxiter': 200})
            else:
                res = minimize(fun, x0, constraints=(C1,C2), bounds=B1,
                           method='SLSQP',
                           tol = 1e-4) #, options={'disp': True, 'maxiter': 200})
            w_star = res.x
            res_status = res.status

        if res.status==0:
            delta = (np.sum(p_*r*w_star) - alpha)*p_mean
        else:
            # print('################# infeasible #################')
            delta = np.inf
    
    return w_star, delta




def iso_dual_kl(K,p,r,rho,iso=False):
    p_mean = np.mean(p)
    p = p/p_mean
    alpha = np.sum(r*p)
    var_r = np.sum(p*r**2) - alpha**2
    
    B1 = Bounds(0, np.inf)
    fun = lambda x: x*(rho+np.log(p_mean)+log_sum_exp(np.log(p)+r/x))
    
    trial_id = 0
    res_status=1
    while res_status and (trial_id<=10):
        trial_id = trial_id + 1
        x0 = np.random.uniform(0,1,1)
        res = minimize(fun, x0, bounds=B1, 
                       method='SLSQP',
                       tol = 1e-4, options={'maxiter': 200})
        res_status = res.status
        if res.status==0:
            lam_star = res.x
            delta_star = -alpha*p_mean+lam_star*(rho+np.log(p_mean)+log_sum_exp(np.log(p)+r/lam_star))
        else:
            # print('################# infeasible #################')
            lam_star = np.inf*np.ones(len(x0))
            delta_star = np.inf

    
    return lam_star, delta_star



def iso_dro_recalib(K,p,r,w0,rho,iso,f_div):
    
    p_mean = np.mean(p)
    p_ = p/p_mean
    alpha = np.sum(r*p_)
    var_r = np.sum(p_*r**2) - alpha**2
    w0 = w0 - np.min(w0) + 0.1

    # monotonicity constraint
    A = np.diag(w0)
    for k in range(1,K):
        A[k,k-1] = -w0[k-1]
    # A = np.eye(len(w0))
    # for k in range(1,K):
    #     A[k,k-1] = -1

    fun = lambda x: -np.sum(x*r*p_) + alpha
    cons1 = lambda x: np.sum(p_*x)
    # fun = lambda x: -np.sum(x*r) + alpha
    # cons1 = lambda x: np.sum(x)
    
    cons2 = lambda x: np.sum(entr(x)*p_)
    # cons2 = lambda x: np.sum(entr(x/p_))
    C1 = NonlinearConstraint(cons1, 1/p_mean, 1/p_mean)
    C2 = NonlinearConstraint(cons2, -np.inf, rho/p_mean)
    C3 = LinearConstraint(A, np.zeros(K), np.inf*np.ones(K))
    B1 = Bounds(np.zeros(K), np.inf*np.ones(K))
    trial_id = 0
    res_status = 1
    while res_status and (trial_id<=10):
        trial_id = trial_id + 1
        x0 = np.random.uniform(0.2,2,K)
        if iso:
            res = minimize(fun, x0, constraints=(C1,C2,C3), bounds=B1,
                           method='SLSQP',
                           tol = 1e-4, options={'disp': True, 'maxiter': 200})
        else:
            res = minimize(fun, x0, constraints=(C1,C2), bounds=B1,
                       method='SLSQP',
                       tol = 1e-4, options={'disp': True, 'maxiter': 200})
        w_star = res.x
        res_status = res.status

    if res.status==0:
        delta = (np.sum(p_*r*w_star) - alpha)*p_mean
        # delta = (np.sum(r*w_star) - alpha)*p_mean
    else:
        # print('################# infeasible #################')
        delta = np.inf
    
    return w_star, delta
