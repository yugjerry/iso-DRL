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


def est_rho_(X01, X02, clf_pre, marginal_rt):
    
    n1 = X01.shape[0]
    n2 = X02.shape[0]
    
    s10 = np.random.uniform(0, 1, n1)
    subsample_idx10 = (s10 < 0.5)
    s20 = np.random.uniform(0, 1, n2)
    subsample_idx20 = (s20 < 0.5)
    X1 = X01[subsample_idx10,:]
    X01 = X01[~subsample_idx10,:]
    X2 = X02[subsample_idx20,:]
    X02 = X02[~subsample_idx20,:]

    # KDE mimic oracle
    bw2 = 0.125
    bw1 = 0.125

    kde2 = KernelDensity(bandwidth=bw2, kernel='gaussian')
    kde2.fit(X02)
    kde1 = KernelDensity(bandwidth=bw1, kernel='gaussian')
    kde1.fit(X01)
    log_density_ratio = kde2.score_samples(X2) - kde1.score_samples(X2)
    log_density_ratio0 = kde2.score_samples(X1) - kde1.score_samples(X1)
    q = np.exp(log_density_ratio)
    q0 = np.exp(log_density_ratio0)
    q = q/np.mean(q0)
    q0 = q0/np.mean(q0)

    w0_out_p0 = np.ones(len(q0))

    r_hat = np.mean(q0 * np.log(q0/w0_out_p0))
    
    return r_hat

def simu_(seed, X_white, y_white, X_red, y_red, alpha, pre_rt, f_div, rho_seq):
    
    n_red = X_red.shape[0]
    n_white = X_white.shape[0]
    
    s_red0 = np.random.uniform(0, 1, n_red)
    subsample_idx_red0 = (s_red0 < 0.3)
    s_white0 = np.random.uniform(0, 1, n_white)
    subsample_idx_white0 = (s_white0 < 0.3)

    X01 = X_white[subsample_idx_white0,:]
    X02 = X_red[subsample_idx_red0,:]
    
    X_white = X_white[~subsample_idx_white0,:]
    y_white = y_white[~subsample_idx_white0]
    X_red = X_red[~subsample_idx_red0,:]
    y_red = y_red[~subsample_idx_red0]
    
    n_red = X_red.shape[0]
    n_white = X_white.shape[0]
    
    rt_white = pre_rt
    s_red = np.random.uniform(0, 1, n_red)
    subsample_idx_red = (s_red < pre_rt)
    s_white = np.random.uniform(0, 1, n_white)
    subsample_idx_white = (s_white < rt_white)
    
    marginal_rt = np.sum(subsample_idx_white) / np.sum(subsample_idx_red)

    X_white_pre = X_white[subsample_idx_white,:]
    X_red_pre = X_red[subsample_idx_red,:]
    X_total_pre = np.concatenate((X_red_pre, X_white_pre), axis=0)
    label_pre = np.zeros(X_white_pre.shape[0] + X_red_pre.shape[0])
    label_pre[X_red_pre.shape[0]:] = 1
    clf_pre = LogisticRegression(random_state=0).fit(X_total_pre, label_pre)

    X_white_ = X_white[~subsample_idx_white,:]
    X_red_ = X_red[~subsample_idx_red,:]
    y_white_ = y_white[~subsample_idx_white]
    y_red_ = y_red[~subsample_idx_red]
    
    rho_hat = est_rho_(X01, X02, clf_pre, marginal_rt)
    
    if len(rho_seq)==0:
        rho_seq = [rho_hat]
    
    df_res = compar(seed, X_white_, y_white_, X_red_, y_red_, alpha, clf_pre, f_div, rho_seq=rho_seq, marginal_rt=marginal_rt)

    return df_res



print("\nParsing input parameters...")
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.1, help='target coverage rate')
parser.add_argument('--pre_rt', type=float, default=0.02, help='ratio of sample for density ratio estimation')
parser.add_argument('--f_div', type=str, default='kl', help='f-divergence for DRO')
parser.add_argument('--setting', type=str, default='reweighted', help='rho | estimated')
parser.add_argument('--repN', type=int, default=1000, help='number of repetitions')
args = parser.parse_args()

# python3 -m simulation_wine --setting "estimated"

alpha = args.alpha
f_div = args.f_div
setting = args.setting
repN = args.repN
pre_rt = args.pre_rt


if setting == "rho":
    print("Setting: varying rho in DRO...\n")
    rho_seq = np.linspace(0.02, 2, 100)
elif setting == "estimated":
    print("Setting: estimated rho...\n")
    rho_seq = []

if len(rho_seq)>0:
    l_ = 2 + 3*len(rho_seq)
else:
    l_ = 5


## Load wine dataset

df_red = pd.read_csv('./wine_quality/winequality-red.csv', delimiter = ';')
df_white = pd.read_csv('./wine_quality/winequality-white.csv', delimiter = ';')
print(f"red wine dataset shape: {df_red.shape}")
print(f"white wine dataset shape: {df_white.shape}")


X_red = np.array(df_red)[:,:11]
X_white = np.array(df_white)[:,:11]
y_red = np.array(df_red.quality)
y_white = np.array(df_white.quality)
n_red = X_red.shape[0]
n_white = X_white.shape[0]

X_total = np.concatenate((X_red, X_white), axis=0)

X_rg = np.max(X_total, axis=0)
X_red = X_red / X_rg
X_white = X_white / X_rg


rho_hat_df = pd.DataFrame()

if __name__ == "__main__":
    with tqdm_joblib(tqdm(desc="Running...", total=repN)) as progress_bar:
        results = Parallel(n_jobs=num_cores)(delayed(simu_)(i, X_white, y_white, X_red, y_red, alpha, pre_rt, f_div, rho_seq) for i in range(repN))
    results = np.array(results)
    file_dir1 = './result/result_wine/cov_wine_%s_%s_%s_%s.csv'% (repN,f_div,pre_rt,setting)
    file_dir2 = './result/result_wine/len_wine_%s_%s_%s_%s.csv'% (repN,f_div,pre_rt,setting)
    os.makedirs(os.path.dirname(file_dir1), exist_ok=True)

    df_cov = pd.DataFrame(results[:,:l_])
    df_len = pd.DataFrame(results[:,l_:(2*l_)])
    df_cov.to_csv(file_dir1, index=False)
    df_len.to_csv(file_dir2, index=False)





