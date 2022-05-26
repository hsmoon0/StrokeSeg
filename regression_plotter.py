#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:04:03 2021

@author: hsm
"""
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy import linalg

from polyfit import load_example, PolynomRegressor, Constraints
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset



#from statsmodels.nonparametric.kernel_regression.KernelReg import KernelReg as kr
#import pyqt_fit.nonparam_regression as smooth
#from pyqt_fit import npr_methods


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#cfe7b9")

    return ax


# Computations ----------------------------------------------------------------
# Raw Data


mrs = np.load('mRS_020422.npy')
lesion_vol_true = np.load('lesion_vol_true.npy',encoding='ASCII')
lesion_load_true = np.load('ll_true.npy')

ll_methods = np.zeros([79, 4])
ll_methods[:, 0] = np.load('/Users/hsm/.spyder-py3/ll_2d_multi_full_thresh.npy')

wll_methods = np.zeros([79,4])
wll_methods[:, 0] = np.load('/Users/hsm/.spyder-py3/wll_2d_multi_full_thresh.npy')

lv_methods = np.zeros([79, 4])
lv_methods[:, 0] = np.load('/Users/hsm/.spyder-py3/lv_2d_multi_full_thresh.npy')

dice_methods = np.zeros([79, 4])
dice_methods[:, 0] = np.load('/Users/hsm/.spyder-py3/dice_2d_multi_full_thresh.npy')


NIHSS_base = np.load('NIHSS_base.npy')
NIHSS_post = np.load('NIHSS_post.npy')

FMUE_base = np.load('FMUE_base.npy')
FMUE_post = np.load('FMUE_post.npy')

method_str = ['2D Multi','2D Single','3D Multi','3D Single'];


poly_deg = 5


# Outlier Removal ----------------------------------------------------------------
# Clean up data

for ii in range(0,1):
    x_pre = np.copy(wll_methods[:,ii])
    y_pre = np.copy(FMUE_post)

    index = []
    for i in range(0, len(y_pre)):
        if y_pre[i] == 'N/A':
            index.append(i)
        if y_pre[i] == 0:
            index.append(i)
        
    x_pre = np.delete(x_pre, index)
    x_refined = x_pre.astype(float)
    y_pre = np.delete(y_pre, index)
    y_refined = y_pre.astype(float)

    
    y_sort = np.sort(y_refined)
    Q2 = np.median(y_sort)
    Q3 = np.median(y_sort[y_sort>=Q2])
    Q1 = np.median(y_sort[y_sort<=Q2])
    IQR = Q3-Q1
    
    x_sort = np.sort(x_refined)
    Q2_x = np.median(x_sort)
    Q3_x = np.median(x_sort[x_sort>=Q2_x])
    Q1_x = np.median(x_sort[x_sort<=Q2_x])
    IQR_x = Q3_x-Q1_x
    
    
    index_outlier_x = []

    for jj in range(0, len(x_refined)):
        if x_refined[jj]<Q2_x+1.5*IQR_x and y_refined[jj]<Q2-0.3*IQR:
            index_outlier_x.append(jj)
        if x_refined[jj]<Q2_x+1.5*IQR_x and y_refined[jj]>Q2+0.3*IQR:
            index_outlier_x.append(jj)
            
    for jj in range(0, len(x_refined)):
        if x_refined[jj]>Q2_x+1.5*IQR_x and y_refined[jj]<Q1-2*IQR:
            index_outlier_x.append(jj)
        if x_refined[jj]>Q2_x+1.5*IQR_x and y_refined[jj]>Q3+2*IQR:
            index_outlier_x.append(jj)        
    x_refined = np.delete(x_refined, index_outlier_x)
    y_refined = np.delete(y_refined, index_outlier_x)  
    x_refined_poly = x_refined.reshape((-1,1))



# Correlation Calculation ----------------------------------------------------------------
    
    # Modeling with Numpy
    def equation(a, b):
        """Return a 1D polynomial."""
        return np.polyval(a, b) 
    
    #p, cov = np.polyfit(x_refined, y_refined, poly_deg, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
    

    # Monotonically increasing or decreasing polynomial regression
    polyestimator = PolynomRegressor(deg=poly_deg)

    #Set monotonicity as 'dec' for decreasing and 'inc' for increasing
    monotone_constraint = Constraints(monotonicity='dec')
    polyestimator.fit(x_refined_poly, y_refined, loss = 'l2', constraints={0: monotone_constraint})
    p = polyestimator.coeffs_
    p = np.flip(p)
    
    
    y_model = equation(p, x_refined)                                   # model using the fit parameters; NOTE: parameters here are coefficients
    
    # Statistics
    n = y_refined.size                                           # number of observations
    m = p.size                                                 # number of parameters
    dof = n - m                                                # degrees of freedom
    t = stats.t.ppf(0.975, n - m) 
   # print(t)                             # used for CI and PI bands
    
    # Estimates of Error in Data/Model
    resid = y_refined - y_model                           
    chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
    chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2) / dof)    

                # standard deviation of the error
    
    sst = np.sum((y_refined-np.mean(y_refined))**2)
    ssr = np.sum(resid**2)
    r_sq = 1-ssr/sst    
    k = poly_deg + 1

    
    AIC_c = -2*np.log10(ssr/len(y_model))+2*k+(2*k*(k+1))/(len(y_model)-k-1)
    print("AICc of plynomial: " + str(AIC_c))

                
    # corr_matrix = np.corrcoef(y,x)
    # np.sum((y-mean(y))**2)
    # corr = corr_matrix[0,1]
    # r_sq = corr**2
    #print("R: " + str(corr))
    #print()
    
    
    X2 = sm.add_constant(x_refined)
    est = sm.OLS(y_refined, X2)
    est2 = est.fit()
    p_val = est2.pvalues[1]
    print("R-squared: " + str(r_sq) + ",  P-val: " + str(p_val))
    



    # Plotting --------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data
    ax.plot(
        x_refined, y_refined, "o", color="#b9cfe7", markersize=9, 
        markeredgewidth=2, markeredgecolor="darkgreen", markerfacecolor="None"
    )
    
    # Fit
    list1, list2 = zip(*sorted(zip(x_refined, y_model)))
    
    ax.plot(list1, list2, "-", color="0.1", linewidth=2, alpha=0.5, label="Fit")  
    
    x2 = np.linspace(np.min(x_refined), np.max(x_refined), 100)
    y2 = equation(p, x2)
    
    # Confidence Interval (select one)
    plot_ci_manual(t, s_err, n, x_refined, x2, y2, ax=ax)
    #plot_ci_bootstrap(x, y, resid, ax=ax)
    
    # Prediction Interval
    pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x_refined))**2 / np.sum((x_refined - np.mean(x_refined))**2))   
    ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
    ax.plot(x2, y2 + pi, "--", color="0.5")
    

    
    # Figure Modifications --------------------------------------------------------
    # Borders
    ax.spines["top"].set_color("0.5")
    ax.spines["bottom"].set_color("0.5")
    ax.spines["left"].set_color("0.5")
    ax.spines["right"].set_color("0.5")
    ax.get_xaxis().set_tick_params(direction="out")
    ax.get_yaxis().set_tick_params(direction="out")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left() 
    ax.set_facecolor('w')
    
    
    # Labels
    #plt.title("Baseline NIHSS vs w-Lesion Load", fontsize="26", fontweight="bold")
    #plt.ylabel("Baseline NIHSS", fontsize="22")  
    
    #plt.title("90 Days NIHSS vs w-Lesion Size", fontsize="26", fontweight="bold")
    #plt.ylabel("Post NIHSS", fontsize="22") 

    plt.title("90 Days FM-UE vs w-Lesion Load", fontsize="26", fontweight="bold")
    plt.ylabel("Post FM-UE", fontsize="22")  
    
    plt.title(" ", fontsize="26", fontweight="bold")

    #plt.xlabel("Weighted Lesion Load (cc)", fontsize="22")
    plt.xlabel("Weighted Lesion Load (cc)", fontsize="22")
    
    plt.xticks(fontsize= 18)
    plt.yticks(fontsize= 18)
    
    
    #plt.xlim([0,295])
    #plt.xlim([0,23.2])
    #plt.xlim([0,41.25])
    
    #plt.xlim([0,286])
    #plt.xlim([0,22.9])
    #plt.xlim([0,41.25])
    
    #plt.xlim([0,286])
    #plt.xlim([0,22.9])
    plt.xlim([0,41.1])
    
    
    #plt.xlim([0,253])
    #plt.xlim([0,25.45])
    #plt.xlim([0,42.2])
    
    
    #plt.xlim([0,243.5])
    #plt.xlim([0,22.7])
    #plt.xlim([0,37.1])
    
    
    #plt.ylim([0, 28])
    plt.ylim([0, 85])
    #plt.ylim([0, 15])
    

    # plt.ylim([-0.05, 1.3])
    # plt.xlim([0,292])
    
    
    
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    display = (0, 1)
    anyArtist = plt.Line2D((0, 1), (0, 0), color="#65bd5f")    # create custom artists
    legend = plt.legend(
        [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
        [label for i, label in enumerate(labels) if i in display] + ["95% Confidence Limits"],
        loc=8, bbox_to_anchor=(0, -0.31, 1., 0.102), ncol=3, mode="None", fontsize='large', markerscale=200.
    )  
    frame = legend.get_frame().set_edgecolor("0.5")
    
    # Save Figure
    plt.tight_layout()
    plt.grid(b=None, color='None')
   
    #plt.savefig('/Users/hsm/Desktop/Seg_results/CI_limit_legend.png', dpi=700)
