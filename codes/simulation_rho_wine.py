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


def est_rho_(X01, X02):
    
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

def simu_rhohat(seed, X_white, y_white, X_red, y_red):
    
    n_red = X_red.shape[0]
    n_white = X_white.shape[0]
    
    s_red0 = np.random.uniform(0, 1, n_red)
    subsample_idx_red0 = (s_red0 < 0.3)
    s_white0 = np.random.uniform(0, 1, n_white)
    subsample_idx_white0 = (s_white0 < 0.3)

    X01 = X_white[subsample_idx_white0,:]
    X02 = X_red[subsample_idx_red0,:]
    
    rho_hat = est_rho_(X01, X02)
    
    return rho_hat



print("\nParsing input parameters...")
parser = argparse.ArgumentParser()
parser.add_argument('--repN', type=int, default=1000, help='number of repetitions')
args = parser.parse_args()

repN = args.repN

# python3 -m simulation_rho_wine --repN 1000

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


if __name__ == "__main__":
    with tqdm_joblib(tqdm(desc="Running...", total=repN)) as progress_bar:
        results = Parallel(n_jobs=num_cores)(delayed(simu_rhohat)(i, X_white, y_white, X_red, y_red) for i in range(repN))
    
    rho_hat = np.array(results)
    file_dir = './result/result_wine/rhohat_%s.csv'% (repN)
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)

pd.DataFrame(rho_hat).to_csv(file_dir, index=False)





