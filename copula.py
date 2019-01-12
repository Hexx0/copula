#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
# @Time : 2019-01-10 16:21 
# @Author : hexiaoxiong
# @Site :  
# @File : paper2.py 

import numpy as np
from scipy import stats
from scipy import optimize


def solve_norm_para(x):
	x = np.array(x)
	n = len(x)
	mu = x.sum() / n
	sigma = ((np.power(x - mu, 2) / (n - 1)).sum()) ** 0.5
	return mu, sigma


class copula(object):
	def __init__(self, x, y, m_dist_type, copula_type):
		self.x = x
		self.y = y
		self.mdt = m_dist_type
		self.ct = copula_type
	
	def solve_copula_para(self, para):
		self.para = para
		xtol = 1e-9
		ftol = 1e-9
		
		def loglikelihood_gumbel(alpha):
			if 0 < alpha <= 1:
				loglikelihood1 = (-((-np.log(u)) ** (1 / alpha) + (-np.log(v)) ** (1 / alpha)) ** alpha).sum()
				loglikelihood2 = ((1 / alpha - 1) * np.log(np.log(u) * np.log(v))).sum()
				loglikelihood3 = (
					np.log(((-np.log(u)) ** (1 / alpha) + (-np.log(v)) ** (
							1 / alpha)) ** alpha + 1 / alpha - 1)).sum()
				loglikelihood4 = (np.log(u * v)).sum()
				loglikelihood5 = ((2 - alpha) * np.log(
					(-np.log(u)) ** (1 / alpha) + (-np.log(v)) ** (1 / alpha))).sum()
			loglikelihood = loglikelihood1 + loglikelihood2 + loglikelihood3 - loglikelihood4 - loglikelihood5
			return -loglikelihood
		
		def loglikelihood_clayton(theta):
			n = len(u)
			if theta > 0:
				loglikelihood1 = n * np.log(1 + theta)
				loglikelihood2 = (1 + theta) * (np.log(u * v)).sum()
				loglikelihood3 = (2 + 1 / theta) * (np.log((1 / u) ** theta + (1 / v) ** theta)).sum()
			loglikelihood = loglikelihood1 - loglikelihood2 - loglikelihood3
			return -loglikelihood
		
		def loglikelihood_frank(lamda):
			n = len(u)
			if lamda != 0:
				loglikelihood1 = n * np.log(lamda * (1 - np.e ** (-lamda)))
				loglikelihood2 = lamda * (u + v).sum()
				loglikelihood3 = 2 * (
					np.log((np.e ** (-lamda) - 1) + (np.e ** (-lamda * u) - 1) * (np.e ** (-lamda * v) - 1))).sum()
			loglikelihood = loglikelihood1 - loglikelihood2 - loglikelihood3
			return -loglikelihood
		
		if self.mdt == 'norm':
			mu1, sigma1 = solve_norm_para(self.x)
			mu2, sigma2 = solve_norm_para(self.y)
			u = stats.norm.cdf(self.x, mu1, sigma1)
			v = stats.norm.cdf(self.y, mu2, sigma2)
			n = len(self.x)
			if self.ct == 'gumbel':
				opt = optimize.fmin(loglikelihood_gumbel, x0 = self.para, xtol = xtol, ftol = ftol)
			elif self.ct == 'clayton':
				opt = optimize.fmin(loglikelihood_clayton, x0 = self.para, xtol = xtol, ftol = ftol)
			elif self.ct == 'frank':
				opt = optimize.fmin(loglikelihood_frank, x0 = self.para, xtol = xtol, ftol = ftol)
				
		
		
		return opt
		
