import pandas as pd
import numpy as np
from dfply import * 
import pytest
import scipy as sp
import pytz
from datetime import timedelta, datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF
from copulas.bivariate.gumbel import Gumbel as gc
from scipy.stats import norm as normal, t, multivariate_normal as mv_normal
from scipy.special import erf, erfinv

from psrmodels.time_collapsed import BivariateHindcastMargin, ConvGenDistribution

def winter_period(dt):
  if dt.month >= 10:
    return dt.year
  else:
    return dt.year - 1

def get_objects(country='uk'):
  gen_file = '../../../data/energy/{c}/generator_data.txt'
  data_path = '../../../data/energy/uk_ireland/InterconnectionData_Rescaled.txt'
  df = pd.read_csv(data_path,sep=' ') 
  df.columns = [x.lower() for x in df.columns]
  #
  df >>= mutate(time = X.date + ' ' +  X.time) >>\
  mutate(time = X.time.apply(lambda x: dt.strptime(x,'%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc))) >> drop(['date']) >> mutate(period = X.time.apply(lambda x: winter_period(x)), uk_net = (X.gbdem_r - X.gbwind_r).apply(lambda x: int(np.floor(x))), ireland_net = (X.idem_r - X.iwind_r).apply(lambda x: int(np.floor(x))))
  #
  gens = [ConvGenDistribution(gen_file.format(c=ctr)) for ctr in ['uk','ireland']]
  #
  #
  return {'gens':gens,'data':df}

##### functions to fit model

def logit(x):
  return 1.0/(1+np.exp(-x))

def qq_plot(mq,xq):
  #
  sns.scatterplot(x=mq,y=xq)
  plt.plot([min(mq), max(mq)], [min(mq), max(mq)], '--', color = '#FF8000')
  plt.show()


def trunc_norm_ll(pars,X):
  mu, presigma = pars
  sigma = np.exp(presigma)
  #threshold = np.min(X)-np.abs(prethreshold)
  ll = normal.logpdf(X,loc=mu,scale=sigma) - normal.logsf(x=0,loc=mu,scale=sigma)# - 0.5*np.log(2*math.pi)
  return -np.mean(ll)

def trunc_norm_q(pars,p):
  mu, sigma = pars
  alpha = (0 - mu)/sigma
  def phi(x):
    #return 0.5*(1 + erf((x-mu)/(sigma*np.sqrt(2)))) 
    return normal.cdf(x,scale=sigma,loc=mu)
  #
  qtl = []
  for p_ in p:
    qtl.append(normal.ppf(phi(alpha) + (1 - phi(alpha))*p_,loc=mu,scale=sigma))
  return np.array(qtl)

def trunc_bivar_norm_ll(pars,X):
  mu1, mu2, presigma1, presigma2, prerho = pars
  sigma1 = np.exp(presigma1)
  sigma2 = np.exp(presigma2)
  rho = np.tanh(prerho)
  cov = rho*sigma1*sigma2
  #
  mu_v = np.array([mu1,mu2])
  cov_m = np.array([[sigma1**2,cov],[cov,sigma2**2]])
  #
  ll = mv_normal.logpdf(X,mean=mu_v,cov=cov_m) - np.log(1 - mv_normal.cdf(np.array([0,0]),mean=mu_v,cov=cov_m)) #normal.logsf(x=threshold,loc=mu,scale=sigma)# - 0.5*np.log(2*math.pi)
  return -np.mean(ll)
  
def trunc_bivar_norm_margin_q(pars,p,axis=0):
  # unidimensional quantiles
  mu1, mu2, sigma1, sigma2, rho = pars
  cov = rho*sigma1*sigma2
  mu_v = np.array([mu1,mu2])
  cov_m = np.array([[sigma1**2,cov],[cov,sigma2**2]])
  zeros = np.zeros((2,))
  def phi(x,y):
    return mv_normal.cdf((x,y),mean=mu_v,cov=cov_m)
  def trunc_bivar_cdf(x,y):
    b = np.minimum((x,y),zeros)
    raw = phi(x,y) - phi(b[0],b[1])
    return raw/(1 - phi(0,0))
  def trunc_bivar_margin_cdf(z):
    if axis == 0:
      x = z
      y = np.inf
    else:
      x = np.inf
      y = z
    return trunc_bivar_cdf(x,y)
  def trunc_bivar_margin_q(p):
    find = lambda z: trunc_bivar_margin_cdf(z) - p
    res = sp.optimize.root(find,x0 = 0)
    return res.x
  #
  qtl = []
  for p_ in p:
    qtl.append(trunc_bivar_margin_q(p_))
  return np.array(qtl).reshape(-1)








# def trunc_bivar_t_ll(pars,X):
#   mu1, mu2, presigma1, presigma2, prerho = pars
#   sigma1 = np.exp(presigma1)
#   sigma2 = np.exp(presigma2)
#   rho = np.tanh(prerho)
#   cov = rho*sigma1*sigma2
#   #
#   mu_v = np.array([mu1,mu2])
#   cov_m = np.array([[sigma1**2,cov],[cov,sigma2**2]])
#   #
#   ll = mv_normal.logpdf(X,mean=mu_v,cov=cov_m) - np.log(1 - mv_normal.logcdf(np.array([0,0]),mean=mu_v,cov=cov_m)) #normal.logsf(x=threshold,loc=mu,scale=sigma)# - 0.5*np.log(2*math.pi)
#   return -np.mean(ll)
  
# def trunc_bivar_t_margin_q(pars,p,axis=0):
#   # unidimensional quantiles
#   mu1, mu2, sigma1, sigma2, rho = pars
#   cov = rho*sigma1*sigma2
#   mu_v = np.array([mu1,mu2])
#   cov_m = np.array([[sigma1**2,cov],[cov,sigma2**2]])
#   zeros = np.zeros((2,))
#   def phi(x,y):
#     return mv_normal.cdf((x,y),mean=mu_v,cov=cov_m)
#   def trunc_bivar_cdf(x,y):
#     b = np.minimum((x,y),zeros)
#     raw = phi(x,y) - phi(b[0],b[1])
#     return raw/(1 - phi(0,0))
#   def trunc_bivar_margin_cdf(z):
#     if axis == 0:
#       x = z
#       y = np.inf
#     else:
#       x = np.inf
#       y = z
#     return trunc_bivar_cdf(x,y)
#   def trunc_bivar_margin_q(p):
#     find = lambda z: trunc_bivar_margin_cdf(z) - p
#     res = sp.optimize.root(find,x0 = 0)
#     return res.x
#   #
#   qtl = []
#   for p_ in p:
#     qtl.append(trunc_bivar_margin_q(p_))
#   return np.array(qtl).reshape(-1)

#### setup
obj = get_objects()
#offset = np.array((margins.margin_quantile(QTH,i=0),margins.margin_quantile(QTH,i=1)))

def get_samples(offset = (0,0), n=20000,year='all',plot=False,**kwargs):
  #
  df_ = obj['data']
  if isinstance(year,int):
    df_ = df_.query('period == {y}'.format(y=year))
  #
  gb_dem = np.array(df_[['gbdem_r']]).astype(np.int64)
  irl_dem = np.array(df_[['idem_r']]).astype(np.int64)
  #
  gb_wind = np.array(df_[['gbwind_r']]).astype(np.int64)
  irl_wind = np.array(df_[['iwind_r']]).astype(np.int64)
  #
  margins = BivariateHindcastMargin(\
    np.concatenate((gb_dem,irl_dem),axis=1),\
    np.concatenate((gb_wind,irl_wind),axis=1),\
    obj['gens'])
  #
  sim = margins.simulate_region(n=n,m=offset,c=0,policy='veto',**kwargs)
  X = -sim + offset + 1
  if plot:
    x,y = X[:,0], X[:,1]
    sns.scatterplot(x=x,y=y)
    plt.show()
  return X


X = get_samples(n=20000,intersection=False,plot=True)
x, y = X[:,0], X[:,1]

pars = np.array([0,0,np.log(np.std(x)),np.log(np.std(y)),np.arctanh(0.2)])
res = sp.optimize.minimize(fun=trunc_bivar_norm_ll,x0=pars,args = (X,))


fitted_pars = np.array([res.x[0],res.x[1],np.exp(res.x[2]),np.exp(res.x[3]),np.tanh(res.x[4])])

covmat = np.array()
q_grid = np.linspace(0.01,0.99,99)
xq = np.quantile(x,q_grid)
mq = trunc_bivar_norm_margin_q(fitted_pars,q_grid)
qq_plot(mq,xq)


xq = np.quantile(y,q_grid)
mq = trunc_bivar_norm_margin_q(fitted_pars,q_grid,axis=1)
qq_plot(mq,xq)



############### 
# test that simulator is working properly

obj_ = get_objects()
obj_["gens"][0].cdf_vals 