import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import Bounds, LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import sys


def part_ord(a,b):
        # 1: a<=b
        # -1: b<=a
        # 0: cannot be compared

        if (a<=b).all():
            return 1
        elif (a>b).all():
            return -1
        else:
            return 0


def iso_proj(r, x, p):
    # r: response vector with len n
    # x: covariate matrix with shape n*d
    # partial order <=: a<=b iff (a<=b).all()

    n = len(r)

    fun = lambda y: np.sum(p * (y - r)**2)

    C = np.eye(n)
    if len(x.shape) > 1:
        A = []
        for i in range(n):
            for j in range(i+1,n):
                C[i,j] = part_ord(x[i,:], x[j,:])
                C[j,i] = -C[i,j]

                if C[i,j]!=0:
                    a = np.zeros(n)
                    a[i] = -1
                    a[j] = 1
                    A.append(a)
        A = np.array(A)
        K = A.shape[0]
    else:
        rank_x = x.argsort()
        A = np.eye(len(x))
        for k in range(1,len(x)):
            A[rank_x[k],rank_x[k-1]] = -1
            K = A.shape[0]


    # print(f"Number of linear constraints: {K}")

    C1 = LinearConstraint(A, np.zeros(K), np.inf*np.ones(K))
    trial_id = 0
    res_status = 1
    while res_status and (trial_id<=10):
        trial_id = trial_id + 1
        y0 = r + np.random.uniform(-1,1,n)
        res = minimize(fun, y0, constraints=(C1,),
                       method='SLSQP', options={'gtol': 1e-6})
        y_star = res.x
        res_status = res.status

    return y_star





