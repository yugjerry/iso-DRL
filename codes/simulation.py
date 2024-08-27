import os
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.optimize import Bounds, LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import seaborn as sns
from numpy import random
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.isotonic import IsotonicRegression
from scipy.stats import multivariate_normal

import multiprocessing as mp
from joblib import Parallel, delayed
import seaborn as sns
from tqdm import tqdm
num_cores = mp.cpu_count()

import argparse
from utils import *
from simu_recalib import *


def gen_dataset(n1, n2, d, mu_shift, t_idx, mis):
    
    # source distribution: isotropic gaussian
    mu_vec = mu_shift*np.ones(d)/np.sqrt(d)
    cov2 = np.eye(d) + (t_idx/d)*np.ones((d,d))
    
    X1 = np.random.multivariate_normal(np.zeros(d), np.eye(d), n1)
    X2 = np.random.multivariate_normal(mu_vec, cov2, n2)
    
    beta = np.random.normal(0,1,d).reshape((d,))
    
    if mis:
        mean_y1 = (X1@beta).reshape((n1,)) + np.sin(X1[:,1]) + 0.2*(X1[:,4])**2 + 0.4*(X1[:,3])**3
        mean_y2 = (X2@beta).reshape((n2,)) + np.sin(X2[:,1]) + 0.2*(X2[:,4])**2 + 0.4*(X2[:,3])**3
    else:
        mean_y1 = (X1@beta).reshape((n1,))
        mean_y2 = (X2@beta).reshape((n2,))
 
    y1 = 0.2 * (mean_y1 + np.random.normal(0,1,n1)).reshape((n1,))
    y2 = 0.2 * (mean_y2 + np.random.normal(0,1,n2)).reshape((n2,))
    
    return X1, y1, X2, y2


def simu_synthetic(seed, setting, n1, n2, d, mu_shift, t_idx, mis, alpha, f_div, n_pre, rho_seq, verbose=False):
    
    X1_pre, _, X2_pre, _ = gen_dataset(n_pre, n_pre, d, mu_shift, t_idx, mis)
    X1, y1, X2, y2 = gen_dataset(n1, n2, d, mu_shift, t_idx, mis)

    marginal_rt = X1_pre.shape[0]/X2_pre.shape[0]
    
    X_total_pre = np.concatenate((X2_pre, X1_pre), axis=0)
    label_pre = np.zeros(X2_pre.shape[0] + X1_pre.shape[0])
    label_pre[X2_pre.shape[0]:] = 1


    X1_, y1_, X2_, y2_ = X1, y1, X2, y2
    
    # classification
    clf_pre = LogisticRegression(random_state=0).fit(X_total_pre, label_pre)
    
    if len(rho_seq)==0:
        rho_hat = est_rho(d, mu_shift, t_idx, mis, clf_pre, marginal_rt)
        
        rho_seq = [rho_hat/2, rho_hat]

    if setting == "rho":
        res = compar_generic(seed, X1_, y1_, X2_, y2_, alpha, clf_pre, f_div, mu_shift, t_idx, rho_seq, marginal_rt, oracle=True, verbose=verbose)
        # res = compar(seed, X1_, y1_, X2_, y2_, alpha, clf_pre, f_div, mu_shift, t_idx, rho_seq, marginal_rt, oracle=True, verbose=verbose)
    elif setting == "pre_rt":
        res = compar(seed, X1_, y1_, X2_, y2_, alpha, clf_pre, f_div, mu_shift, t_idx, rho_seq, marginal_rt, oracle=True, verbose=verbose)
    
    return res



print("\nParsing input parameters...")
parser = argparse.ArgumentParser()
parser.add_argument('--n1', type=int, default=1000, help='sample size from source distribution')
parser.add_argument('--n2', type=int, default=500, help='sample size from target distribution')
parser.add_argument('--d', type=int, default=20, help='dimension of covariates')
parser.add_argument('--mu_shift', type=float, default=2, help='shift in Gaussian mean')
parser.add_argument('--t_idx', type=float, default=0.0, help='shift in Gaussian covariance, controlling for misspecification of logistic')
parser.add_argument('--alpha', type=float, default=0.1, help='target coverage rate')
parser.add_argument('--n_pre', type=int, default=50, help='sample size for density ratio estimation')
parser.add_argument('--f_div', type=str, default='kl', help='f-divergence for DRO')
parser.add_argument('--mis', type=bool, default=True, help='controlling for misspecification of OLS')
parser.add_argument('--setting', type=str, default='reweighted', help='rho | pre_rt')
parser.add_argument('--repN', type=int, default=1000, help='number of repetitions')
args = parser.parse_args()

# python3 -m simulation --mu_shift 2 --t_idx 0.0 --setting "pre_rt"
# python3 -m simulation --mu_shift 2 --t_idx 0.0 --setting "rho"


n1, n2, d = args.n1, args.n2, args.d
mu_shift, t_idx = args.mu_shift, args.t_idx
alpha = args.alpha
f_div = args.f_div
mis = args.mis
setting = args.setting
repN = args.repN
n_pre = args.n_pre

# ground-truth \rho
rho_star = 0.5*((mu_shift)**2 + t_idx - np.log(1+t_idx))
print('true KL-div = %s'%(rho_star,))


if setting == "rho":
    print("Setting: varying rho in DRO...\n")
    rho_seq = np.linspace(0.002, 6, 200)
    n_seq = [n_pre]
elif setting == "pre_rt":
    print("Setting: varying data splitting ratio...\n")
    rho_seq = [rho_star]
    n_seq = [20, 40, 60, 80, 100]

if len(rho_seq)>0 and setting == "rho":
    l_ = 3 + 4*len(rho_seq)
    # l_ = 3 + 3*len(rho_seq)
elif len(rho_seq)>0 and setting == "pre_rt":
    l_ = 3 + 3*len(rho_seq)
else:
    l_ = 11


rho_hat_df = pd.DataFrame()

for n_pre in n_seq:

    if __name__ == "__main__":
        with tqdm_joblib(tqdm(desc="Running...", total=repN)) as progress_bar:
            results = Parallel(n_jobs=num_cores)(delayed(simu_synthetic)(i, setting, n1, n2, d, mu_shift, t_idx, mis, alpha, f_div, n_pre, rho_seq) for i in range(repN))
        
        results = np.array(results)
        file_dir1 = './result/result_1/cov_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.csv'% (n1,n2,d,repN,f_div,n_pre,mu_shift,t_idx,mis,setting)
        file_dir2 = './result/result_1/len_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.csv'% (n1,n2,d,repN,f_div,n_pre,mu_shift,t_idx,mis,setting)
        os.makedirs(os.path.dirname(file_dir1), exist_ok=True)
        os.makedirs(os.path.dirname(file_dir2), exist_ok=True)

        df_cov = pd.DataFrame(results[:,:l_])
        df_len = pd.DataFrame(results[:,l_:(2*l_)])

        rho_hat = results[:,(2*l_)]
        rho_hat_df[str(n_pre)] = rho_hat

        df_cov.to_csv(file_dir1, index=False)
        df_len.to_csv(file_dir2, index=False)

