# iso-DRL

Statistical learning under distribution shift is challenging when neither prior knowledge nor fully accessible data from the target distribution is available. Distributionally robust learning (DRL) aims to control the worst-case statistical performance within an uncertainty set of candidate distributions, but how to properly specify the set remains challenging. To enable distributional robustness without being overly conservative, in this paper we propose a shape-constrained approach to DRL, which incorporates prior information about the way in which the unknown target distribution differs from its estimate. 
More specifically, we assume the unknown density ratio between the target distribution and its estimate is isotonic with respect to some partial order. In the population level, we provide a solution to the shape-constrained optimization problem that does not involve the isotonic constraint. In the sample level, we provide consistency results for an empirical estimator of the target in a range of different settings. Empirical studies on both synthetic and real data examples demonstrate the improved accuracy of the proposed shape-constrained approach.


## About

This repo reproduces results in the paper <https://arxiv.org/abs/2407.06867>. The proposed iso-DRL approach is implemented in ```codes/simu_calib.py```.

### Synthetic dataset

The script ```codes/simulation.py``` reproduces results for synthetic datasets. Run the following command:


```bash
python3 -m simulation --mu_shift 2 --t_idx 6 --setting "pre_rt"
```

The argument ```mu_shift``` controls the strength of covariate shift in terms of the Gaussian means and ```t_idx``` controls the strength of the the rank-one perturbation in the covariance matrix of target distribution, which controls the misspecification of the logistic regression in estimating density ratios. The script supports two settings: (1) varying splitting ratio in estimating the density ratio, and (2) varying $\rho$, which is the radius of $f$-divergence ball.


### Real dataset

This paper focus on the wine quality dataset <https://archive.ics.uci.edu/dataset/186/wine+quality> consisting of two groups: white and red wine. To reproduce results, run the following command:

```bash
python3 -m simulation_wine --setting "estimated"
```

The script ```codes/simulation_wine.py``` supports two settings: (1) varying $\rho$, and (2) iso-DRL with estimated $\rho$. To see the distribution of the estimated $\rho$, run the script ```codes/simulation_rho_wine.py```.


### Figures

Figures in the paper are reproduced in the notebook ```plot.ipynb```.






