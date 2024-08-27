import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import Bounds, LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import multivariate_normal

import tqdm

from utils import *
from iso_proj import *

def f(x):
    return x*np.log(x)

def invg(r,rho):
    """
        Caculate g_{f,rho}^{-1}(r)
    """
    eps=1e-10
    left=r
    right=1
    mid=(left+right)/2
    while (right-left>eps):
        ans=mid*f(r/mid)+(1-mid)*f((1-r)/(1-mid))
        if ans<=rho:
            left=mid
        else:
            right=mid
        mid=(left+right)/2
    return mid


def compar(seed, X1_, y1_, X2_, y2_, alpha, clf_pre, f_div, mu_shift=0, t_idx=10, rho_seq=[], marginal_rt=1, verbose = False, oracle = False):
    
    # X1_, y1_: observations from the source distribution
    # X2_, y2_: observations from the target distribution
    # alpha: target level of coverage
    # clf_pre: pre-fitted density ratio between dP^*/dP
    # f_div: f-divergence for uncertainty quantification in DRO
    # mu_shift: mean-shift for Gaussian
    # t_idx: strength of spike effect in covariance
    # rho_seq: sequence of candidate rho's for f-divergence
    # marginal_rt: sample size ratio to correct the density ratio estimate
    # oracle: indicator for synthetic or real data example


    n1_ = X1_.shape[0]
    n2_ = X2_.shape[0]

    # run CP-LS
    a = np.random.rand(n1_)
    mask = (a<=0.5)
    n11 = np.sum(mask)
    X11 = X1_[mask,:]
    y11 = y1_[mask]
    X12 = X1_[~mask,:]
    y12 = y1_[~mask] 

    # run split CP_LS
    a_tr = np.random.rand(X11.shape[0])
    mask_tr = (a_tr<=0.5)
    X_tr = X11[mask_tr,:]
    X_cal = X11[~mask_tr,:]
    y_tr = y11[mask_tr]
    y_cal = y11[~mask_tr]

    y2_PI, len2_ = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, alpha)
    cov_y2_ = (y2_PI[:,0] <= y2_) * (y2_PI[:,1] >= y2_)
    y12_PI, len1_ = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X12, alpha)
    cov_y1_ = (y12_PI[:,0] <= y12) * (y12_PI[:,1] >= y12)

    # wtd-CP
    wts = clf_pre.predict_proba(X_cal)[:,0]/(1-clf_pre.predict_proba(X_cal)[:,0])
    wts2 = clf_pre.predict_proba(X2_)[:,0]/(1-clf_pre.predict_proba(X2_)[:,0])
    y2_PI_wtd, len2_wtd = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, alpha, wts, wts2)
    cov_y2_wtd = (y2_PI_wtd[:,0] <= y2_) * (y2_PI_wtd[:,1] >= y2_)
    
    wts12 = clf_pre.predict_proba(X12)[:,0]/(1-clf_pre.predict_proba(X12)[:,0])
    y12_PI_wtd, len12_wtd = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X12, alpha, wts, wts12)
    cov_y1_wtd = (y12_PI_wtd[:,0] <= y12) * (y12_PI_wtd[:,1] >= y12)

    # oracle wtd-CP
    if oracle:
        d = X2_.shape[1]
        mu_vec = mu_shift*np.ones(d)/np.sqrt(d)
        # cov2 = np.eye(d) + t_idx*np.outer(mu_vec, mu_vec)
        # mu_vec = np.zeros(d)
        # mu_vec[0] = mu_shift
        # cov2 = np.eye(d) + t_idx*np.outer(mu_vec, mu_vec)
        cov2 = np.eye(d) + (t_idx/d)*np.ones((d,d))

        p1 = multivariate_normal.pdf(X_cal, mean=mu_vec, cov=cov2)
        p0 = multivariate_normal.pdf(X_cal, mean=np.zeros(d), cov=np.eye(d))
        p12 = multivariate_normal.pdf(X2_, mean=mu_vec, cov=cov2)
        p02 = multivariate_normal.pdf(X2_, mean=np.zeros(d), cov=np.eye(d))
        wts_o = p1/p0
        wts2_o = p12/p02
        y2_PI_wtd_o, len2_wtd_o = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, alpha, wts_o, wts2_o)
        cov_y2_wtd_o = (y2_PI_wtd_o[:,0] <= y2_) * (y2_PI_wtd_o[:,1] >= y2_)


    # excessive risk
        
    log_prob_fit = clf_pre.predict_log_proba(X12)[:,0]
    prob_fit = clf_pre.predict_proba(X12)[:,0]
    w0 = log_prob_fit - np.log(1 - prob_fit) + np.log(marginal_rt)
    
    rank_x_w0 = w0.argsort()
    w0 = w0[rank_x_w0]
    w0_p = np.exp(w0)
  
    p = w0_p/np.sum(w0_p)
    p0 = np.ones(n1_-n11)/(n1_-n11)

    r = cov_y1_[rank_x_w0]

    iso_reg = IsotonicRegression().fit(w0_p, r)
    r_iso = iso_reg.predict(w0_p)
    
    # r_wtd = cov_y1_wtd[rank_x_w0]
    # iso_wtd_reg = IsotonicRegression().fit(w0_p, r_wtd)
    # r_wtd_iso = iso_wtd_reg.predict(w0_p)

    
    delta = np.zeros(len(rho_seq))
    delta_iso = np.zeros(len(rho_seq))
    len2 = np.zeros(len(rho_seq))
    len2_rob = np.zeros(len(rho_seq))
    len2_iso = np.zeros(len(rho_seq))
    cov_y2 = np.zeros(len(rho_seq)*n2_).reshape((len(rho_seq), n2_))
    cov_y2_rob = np.zeros(len(rho_seq)*n2_).reshape((len(rho_seq), n2_))
    cov_y2_iso = np.zeros(len(rho_seq)*n2_).reshape((len(rho_seq), n2_))

    idx_rho = 0
    for rho in tqdm(rho_seq):
        if f_div == 'kl':
            lam, delta[idx_rho] = iso_dual_kl(n1_-n11,p0,r,rho,False)
            lam_iso, delta_iso[idx_rho] = iso_dual_kl(n1_-n11,p0,r_iso,rho,False)
        else:
            lam, delta[idx_rho] = iso_dro(n1_-n11,p0,r,rho,False,f_div)
            lam_iso, delta_iso[idx_rho] = iso_dro(n1_-n11,p0,r_iso,rho,False,f_div)

   
        y2_PI, len2[idx_rho] = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, max(alpha-delta[idx_rho],0))
        y2_PI_rob, len2_rob[idx_rho] = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, max(1-invg(0.5,rho),0))
        y2_PI_iso, len2_iso[idx_rho] = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, max(alpha-delta_iso[idx_rho],0))
        
        
        cov_y2[idx_rho,:] = (y2_PI[:,0] <= y2_) * (y2_PI[:,1] >= y2_)
        cov_y2_rob[idx_rho,:] = (y2_PI_rob[:,0] <= y2_) * (y2_PI_rob[:,1] >= y2_)
        cov_y2_iso[idx_rho,:] = (y2_PI_iso[:,0] <= y2_) * (y2_PI_iso[:,1] >= y2_)

        cov_rt_y2 = np.mean(cov_y2, axis = 1)
        cov_rt_y2_rob = np.mean(cov_y2_rob, axis = 1)
        cov_rt_y2_iso = np.mean(cov_y2_iso, axis = 1)
        cov_rt_dro = np.concatenate((cov_rt_y2, cov_rt_y2_iso))
        cov_rt_dro = np.concatenate((cov_rt_dro, cov_rt_y2_rob))
        len_dro = np.concatenate((len2, len2_iso))
        len_dro = np.concatenate((len_dro, len2_rob))
        idx_rho += 1

    if verbose:
        print('\nmiscoverage rate on source domain = {} ({})'.format(np.mean(cov_y1_), len1_))
        print('miscoverage rate on target domain = {} ({})'.format(np.mean(cov_y2_), len2_))
        print('miscoverage rate (wtd) on target domain = {} ({})'.format(np.mean(cov_y2_wtd), len2_wtd))
        print('miscoverage rate (oracle-wtd) on target domain = {} ({})'.format(np.mean(cov_y2_wtd_o), len2_wtd_o))
        idx_rho = 0
        for rho in rho_seq:
            print('after adjustion: miscoverage rate on target domain (rho = {}) = {} ({})'.format(rho,cov_rt_y2[idx_rho], len2[idx_rho]))
            print('after adjustion: miscoverage rate on target domain (iso, rho = {}) = {} ({})'.format(rho, cov_rt_y2_iso[idx_rho], len2_iso[idx_rho]))
            idx_rho += 1

    if oracle:
        cov_rt = np.concatenate((np.array([np.mean(cov_y2_),np.mean(cov_y2_wtd),np.mean(cov_y2_wtd_o)]),cov_rt_dro))
        len_ = np.concatenate((np.array([len2_,len2_wtd,len2_wtd_o]), len_dro))
    else:
        cov_rt = np.concatenate((np.array([np.mean(cov_y2_),np.mean(cov_y2_wtd)]),cov_rt_dro))
        len_ = np.concatenate((np.array([len2_,len2_wtd]), len_dro))
        
    res = np.concatenate((cov_rt, len_))
    
    return np.append(res, rho_seq[0])


def compar_generic(seed, X1_, y1_, X2_, y2_, alpha, clf_pre, f_div, mu_shift=0, t_idx=10, rho_seq=[], marginal_rt=1, verbose = False, oracle = False):
    
    # X1_, y1_: observations from the source distribution
    # X2_, y2_: observations from the target distribution
    # alpha: target level of coverage
    # clf_pre: pre-fitted density ratio between dP^*/dP
    # f_div: f-divergence for uncertainty quantification in DRO
    # mu_shift: mean-shift for Gaussian
    # t_idx: strength of spike effect in covariance
    # rho_seq: sequence of candidate rho's for f-divergence
    # marginal_rt: sample size ratio to correct the density ratio estimate
    # oracle: indicator for synthetic or real data example

    n1_ = X1_.shape[0]
    n2_ = X2_.shape[0]

    # run CP-LS
    a = np.random.rand(n1_)
    mask = (a<=0.5)
    n11 = np.sum(mask)
    X11 = X1_[mask,:]
    y11 = y1_[mask]
    X12 = X1_[~mask,:]
    y12 = y1_[~mask] 

    # run split CP_LS
    a_tr = np.random.rand(X11.shape[0])
    mask_tr = (a_tr<=0.5)
    X_tr = X11[mask_tr,:]
    X_cal = X11[~mask_tr,:]
    y_tr = y11[mask_tr]
    y_cal = y11[~mask_tr]

    y2_PI, len2_ = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, alpha)
    cov_y2_ = (y2_PI[:,0] <= y2_) * (y2_PI[:,1] >= y2_)
    y12_PI, len1_ = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X12, alpha)
    cov_y1_ = (y12_PI[:,0] <= y12) * (y12_PI[:,1] >= y12)

    # wtd-CP
    wts = clf_pre.predict_proba(X_cal)[:,0]/(1-clf_pre.predict_proba(X_cal)[:,0])
    wts2 = clf_pre.predict_proba(X2_)[:,0]/(1-clf_pre.predict_proba(X2_)[:,0])
    y2_PI_wtd, len2_wtd = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, alpha, wts, wts2)
    cov_y2_wtd = (y2_PI_wtd[:,0] <= y2_) * (y2_PI_wtd[:,1] >= y2_)
    
    wts12 = clf_pre.predict_proba(X12)[:,0]/(1-clf_pre.predict_proba(X12)[:,0])
    y12_PI_wtd, len12_wtd = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X12, alpha, wts, wts12)
    cov_y1_wtd = (y12_PI_wtd[:,0] <= y12) * (y12_PI_wtd[:,1] >= y12)

    # oracle wtd-CP
    if oracle:
        d = X2_.shape[1]

        mu_vec = mu_shift*np.ones(d)/np.sqrt(d)
        cov2 = np.eye(d) + t_idx*np.outer(mu_vec, mu_vec)
        p1 = multivariate_normal.pdf(X_cal, mean=mu_vec, cov=cov2)
        p0 = multivariate_normal.pdf(X_cal, mean=np.zeros(d), cov=np.eye(d))
        p12 = multivariate_normal.pdf(X2_, mean=mu_vec, cov=cov2)
        p02 = multivariate_normal.pdf(X2_, mean=np.zeros(d), cov=np.eye(d))
        wts_o = p1/p0
        wts2_o = p12/p02
        y2_PI_wtd_o, len2_wtd_o = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, alpha, wts_o, wts2_o)
        cov_y2_wtd_o = (y2_PI_wtd_o[:,0] <= y2_) * (y2_PI_wtd_o[:,1] >= y2_)


    # excessive risk
    p0 = np.ones(n1_-n11)/(n1_-n11)

    r1 = cov_y1_
    r_iso1 = iso_proj(r1, X12, p0*(n1_-n11))

    log_prob_fit = clf_pre.predict_log_proba(X12)[:,0]
    prob_fit = clf_pre.predict_proba(X12)[:,0]
    h0 = log_prob_fit - np.log(1 - prob_fit) + np.log(marginal_rt)
    rank_x_h0 = h0.argsort()
    r2 = cov_y1_#[rank_x_h0]
    iso_reg = IsotonicRegression().fit(h0, r2)
    r_iso2 = iso_reg.predict(h0)
    
    
    delta = np.zeros(len(rho_seq))
    delta_iso1 = np.zeros(len(rho_seq))
    delta_iso2 = np.zeros(len(rho_seq))

    len2 = np.zeros(len(rho_seq))
    len2_rob = np.zeros(len(rho_seq))
    len2_iso1 = np.zeros(len(rho_seq))
    len2_iso2 = np.zeros(len(rho_seq))
    cov_y2 = np.zeros(len(rho_seq)*n2_).reshape((len(rho_seq), n2_))
    cov_y2_rob = np.zeros(len(rho_seq)*n2_).reshape((len(rho_seq), n2_))
    cov_y2_iso1 = np.zeros(len(rho_seq)*n2_).reshape((len(rho_seq), n2_))
    cov_y2_iso2 = np.zeros(len(rho_seq)*n2_).reshape((len(rho_seq), n2_))

    idx_rho = 0
    for rho in rho_seq:
        if f_div == 'kl':
            lam, delta[idx_rho] = iso_dual_kl(n1_-n11,p0,r1,rho,False)
            lam_iso1, delta_iso1[idx_rho] = iso_dual_kl(n1_-n11,p0,r_iso1,rho,False)
            lam_iso2, delta_iso2[idx_rho] = iso_dual_kl(n1_-n11,p0,r_iso2,rho,False)
        else:
            lam, delta[idx_rho] = iso_dro(n1_-n11,p0,r1,rho,False,f_div)
            lam_iso1, delta_iso1[idx_rho] = iso_dro(n1_-n11,p0,r_iso1,rho,False,f_div)
            lam_iso2, delta_iso2[idx_rho] = iso_dro(n1_-n11,p0,r_iso2,rho,False,f_div)

        
        y2_PI, len2[idx_rho] = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, max(alpha-delta[idx_rho],0))
        y2_PI_rob, len2_rob[idx_rho] = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, max(1-invg(1-alpha,rho),0))
        y2_PI_iso1, len2_iso1[idx_rho] = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, max(alpha-delta_iso1[idx_rho],0))
        y2_PI_iso2, len2_iso2[idx_rho] = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, max(alpha-delta_iso2[idx_rho],0))
        
        
        cov_y2[idx_rho,:] = (y2_PI[:,0] <= y2_) * (y2_PI[:,1] >= y2_)
        cov_y2_rob[idx_rho,:] = (y2_PI_rob[:,0] <= y2_) * (y2_PI_rob[:,1] >= y2_)
        cov_y2_iso1[idx_rho,:] = (y2_PI_iso1[:,0] <= y2_) * (y2_PI_iso1[:,1] >= y2_)
        cov_y2_iso2[idx_rho,:] = (y2_PI_iso2[:,0] <= y2_) * (y2_PI_iso2[:,1] >= y2_)

        cov_rt_y2 = np.mean(cov_y2, axis = 1)
        cov_rt_y2_rob = np.mean(cov_y2_rob, axis = 1)
        cov_rt_y2_iso1 = np.mean(cov_y2_iso1, axis = 1)
        cov_rt_y2_iso2 = np.mean(cov_y2_iso2, axis = 1)
        cov_rt_dro = np.concatenate((cov_rt_y2, cov_rt_y2_iso1))
        cov_rt_dro = np.concatenate((cov_rt_dro, cov_rt_y2_iso2))
        cov_rt_dro = np.concatenate((cov_rt_dro, cov_rt_y2_rob))
        len_dro = np.concatenate((len2, len2_iso1))
        len_dro = np.concatenate((len_dro, len2_iso2))
        len_dro = np.concatenate((len_dro, len2_rob))
        idx_rho += 1

    if verbose:
        print('\nmiscoverage rate on source domain = {} ({})'.format(np.mean(cov_y1_), len1_))
        print('miscoverage rate on target domain = {} ({})'.format(np.mean(cov_y2_), len2_))
        print('miscoverage rate (wtd) on target domain = {} ({})'.format(np.mean(cov_y2_wtd), len2_wtd))
        print('miscoverage rate (oracle-wtd) on target domain = {} ({})'.format(np.mean(cov_y2_wtd_o), len2_wtd_o))
        idx_rho = 0
        for rho in rho_seq:
            print('after adjustion: miscoverage rate on target domain (rho = {}) = {} ({})'.format(rho,cov_rt_y2[idx_rho], len2[idx_rho]))
            print('after adjustion: miscoverage rate on target domain (iso-R^d, rho = {}) = {} ({})'.format(rho, cov_rt_y2_iso1[idx_rho], len2_iso1[idx_rho]))
            print('after adjustion: miscoverage rate on target domain (iso-2, rho = {}) = {} ({})'.format(rho, cov_rt_y2_iso2[idx_rho], len2_iso2[idx_rho]))
            idx_rho += 1

    if oracle:
        cov_rt = np.concatenate((np.array([np.mean(cov_y2_),np.mean(cov_y2_wtd),np.mean(cov_y2_wtd_o)]),cov_rt_dro))
        len_ = np.concatenate((np.array([len2_,len2_wtd,len2_wtd_o]), len_dro))
    else:
        cov_rt = np.concatenate((np.array([np.mean(cov_y2_),np.mean(cov_y2_wtd)]),cov_rt_dro))
        len_ = np.concatenate((np.array([len2_,len2_wtd]), len_dro))
        
    res = np.concatenate((cov_rt, len_))
    
    return np.append(res, rho_seq[0])



def compar_robust(seed, X1_, y1_, X2_, y2_, alpha, f_div, mu_shift=0, t_idx=10, rho_seq=[], marginal_rt=1, verbose = False, oracle = False):
    
    # X1_, y1_: observations from the source distribution
    # X2_, y2_: observations from the target distribution
    # alpha: target level of coverage
    # f_div: f-divergence for uncertainty quantification in DRO
    # mu_shift: mean-shift for Gaussian
    # t_idx: strength of spike effect in covariance
    # rho_seq: sequence of candidate rho's for f-divergence
    # marginal_rt: sample size ratio to correct the density ratio estimate
    # oracle: indicator for synthetic or real data example


    n1_ = X1_.shape[0]
    n2_ = X2_.shape[0]

    # run CP-LS
    a = np.random.rand(n1_)
    mask = (a<=0.5)
    n11 = np.sum(mask)
    X11 = X1_[mask,:]
    y11 = y1_[mask]
    X12 = X1_[~mask,:]
    y12 = y1_[~mask] 

    # run split CP_LS
    a_tr = np.random.rand(X11.shape[0])
    mask_tr = (a_tr<=0.5)
    X_tr = X11[mask_tr,:]
    X_cal = X11[~mask_tr,:]
    y_tr = y11[mask_tr]
    y_cal = y11[~mask_tr]

    y2_PI, len2_ = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, alpha)
    cov_y2_ = (y2_PI[:,0] <= y2_) * (y2_PI[:,1] >= y2_)
    y12_PI, len1_ = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X12, alpha)
    cov_y1_ = (y12_PI[:,0] <= y12) * (y12_PI[:,1] >= y12)


    # excessive risk

    p0 = np.ones(n1_-n11)/(n1_-n11)

    r = cov_y1_#[rank_x_w0]


    
    delta = np.zeros(len(rho_seq))
    delta_rob = np.zeros(len(rho_seq))
    len2 = np.zeros(len(rho_seq))
    len2_rob = np.zeros(len(rho_seq))
    cov_y2 = np.zeros(len(rho_seq)*n2_).reshape((len(rho_seq), n2_))
    cov_y2_rob = np.zeros(len(rho_seq)*n2_).reshape((len(rho_seq), n2_))

    idx_rho = 0
    for rho in rho_seq:
        if f_div == 'kl':
            lam, delta[idx_rho] = iso_dual_kl(n1_-n11,p0,r,rho,False)
        else:
            lam, delta[idx_rho] = iso_dro(n1_-n11,p0,r,rho,False,f_div)

   
        y2_PI, len2[idx_rho] = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, max(alpha-delta[idx_rho],0))
        y2_PI_rob, len2_rob[idx_rho] = split_CP_LS(X_tr, y_tr, X_cal, y_cal, X2_, max(1-invg(1-alpha,rho),0))
        
        
        cov_y2[idx_rho,:] = (y2_PI[:,0] <= y2_) * (y2_PI[:,1] >= y2_)
        cov_y2_rob[idx_rho,:] = (y2_PI_rob[:,0] <= y2_) * (y2_PI_rob[:,1] >= y2_)

        cov_rt_y2 = np.mean(cov_y2, axis = 1)
        cov_rt_y2_rob = np.mean(cov_y2_rob, axis = 1)
        cov_rt_dro = np.concatenate((cov_rt_y2, cov_rt_y2_rob))
        len_dro = np.concatenate((len2, len2_rob))
        idx_rho += 1

    cov_rt = cov_rt_dro
    len_ = len_dro
        
    res = np.concatenate((cov_rt, len_))
    
    return np.append(res, rho_seq[0])