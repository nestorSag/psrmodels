import pandas as pd
import numpy as np
from dfply import * 
import pytest
import scipy as sp
import pytz
from datetime import timedelta, datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm as normal, t, multivariate_normal as mv_normal
from scipy.special import erf, erfinv

from psrmodels.time_collapsed import BivariateHindcastMargin, ConvGenDistribution

# this script is called by R to enable fitting a truncated normal tail model 
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

def truncated_model_cdf(m,cdf_func):
  norm_constant = cdf_func((0,np.Inf)) + cdf_func((np.Inf,0)) - cdf_func((0,0))
  if np.any(np.array(m) <= 0):
    return cdf_func(m)/norm_constant
  else:
    #raw = self.h_cdf(m=(0,m[1])) + self.h_cdf(m=(m[0],0)) - self.h_cdf(m=(0,0))
    raw = cdf_func((0,m[1])) + cdf_func((m[0],0)) - cdf_func((0,0))
    return raw/norm_constant

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
  survival_origin = 1 + mv_normal.cdf(np.array([0,0]),mean=mu_v,cov=cov_m) - normal.cdf(0,loc=mu1,scale=sigma1) - normal.cdf(0,loc=mu2,scale=sigma2)
  #ll = mv_normal.logpdf(X,mean=mu_v,cov=cov_m) - np.log(1 - mv_normal.cdf(np.array([0,0]),mean=mu_v,cov=cov_m)) #normal.logsf(x=threshold,loc=mu,scale=sigma)# - 0.5*np.log(2*math.pi)
  ll = mv_normal.logpdf(X,mean=mu_v,cov=cov_m) - np.log(1 - survival_origin) 
  return -np.mean(ll)

def rectangular_bivar_norm_ll(pars,X):
  mu1, mu2, presigma1, presigma2, prerho = pars
  sigma1 = np.exp(presigma1)
  sigma2 = np.exp(presigma2)
  rho = np.tanh(prerho)
  cov = rho*sigma1*sigma2
  #
  mu_v = np.array([mu1,mu2])
  cov_m = np.array([[sigma1**2,cov],[cov,sigma2**2]])
  #
  ll = mv_normal.logpdf(X,mean=mu_v,cov=cov_m) - mv_normal.cdf(np.array([0,0]),mean=mu_v,cov=cov_m)
  return -np.mean(ll)


def trunc_bivar_norm_margin_q(pars,p,axis=0):
  # unidimensional quantiles
  mu1, mu2, sigma1, sigma2, rho = pars
  cov = rho*sigma1*sigma2
  mu_v = np.array([mu1,mu2])
  cov_m = np.array([[sigma1**2,cov],[cov,sigma2**2]])
  # def phi(x,y):
  #   return mv_normal.cdf((x,y),mean=mu_v,cov=cov_m)
  def trunc_bivarh_cdf(x,y):
    return truncated_model_cdf((x,y),lambda m: mv_normal.cdf(m,mean=mu_v,cov=cov_m))
    # zeros = np.zeros((2,))
    # b = np.minimum((x,y),zeros)
    # raw = phi(x,y) - phi(b[0],b[1])
    # return raw/(1 - phi(0,0))
  def trunc_bivar_marginh_cdf(z):
    if axis == 0:
      x = z
      y = np.inf
    else:
      x = np.inf
      y = z
    return trunc_bivarh_cdf(x,y)
  def trunc_bivar_margin_q(p):
    find = lambda z: trunc_bivar_marginh_cdf(z) - p
    res = sp.optimize.root(find,x0 = 0)
    return res.x
  #
  qtl = []
  for p_ in p:
    qtl.append(trunc_bivar_margin_q(p_))
  return np.array(qtl).reshape(-1)

def get_hindcast_model(year):
  data_obj = get_objects()
  df_ = data_obj['data']
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
    data_obj['gens'])

  return margins

def get_samples(offset = (0,0), n=20000,year='all',plot=False,**kwargs):
  #
  margins = get_hindcast_model(year)

  X = margins.simulate_region(n=n,m=offset,c=0,policy='veto',**kwargs)
  #X = -sim + offset + 1
  #X = - sim
  #X = sim
  if plot:
    x,y = X[:,0], X[:,1]
    sns.scatterplot(x=x,y=y)
    plt.show()

  return X

def fit_truncated_normal(year):

  X = get_samples(n=25000,intersection=False,plot=False,year=year)
  x, y = X[:,0], X[:,1]

  pars = np.array([0,0,np.log(np.std(x)),np.log(np.std(y)),np.arctanh(0.5)])
  res = sp.optimize.minimize(fun=trunc_bivar_norm_ll,x0=pars,args = (X,))

  #ecdf = BivariateHindcastMargin.bivar_ecdf(X)
  #res = sp.optimize.minimize(fun=trunc_bivar_norm_cdfl2 ,x0=pars,args = (X,ecdf))

  fitted_pars = np.array([res.x[0],res.x[1],np.exp(res.x[2]),np.exp(res.x[3]),np.tanh(res.x[4])])

  return {"fitted_pars":fitted_pars,"data":X}

def fit_rectangular_normal(year,offset):

  X = get_samples(offset=offset, n=25000,intersection=True,plot=False,year=year)
  x, y = X[:,0], X[:,1]

  pars = np.array([0,0,np.log(np.std(x)),np.log(np.std(y)),np.arctanh(0.5)])

  # operate on X with offset substracted
  res = sp.optimize.minimize(fun=trunc_bivar_norm_ll,x0=pars,args = (X - np.array(offset),))

  #ecdf = BivariateHindcastMargin.bivar_ecdf(X)
  #res = sp.optimize.minimize(fun=trunc_bivar_norm_cdfl2 ,x0=pars,args = (X,ecdf))

  fitted_pars = np.array([res.x[0],res.x[1],np.exp(res.x[2]),np.exp(res.x[3]),np.tanh(res.x[4])])

  ## add offset 
  fitted_pars[0] += offset[0]
  fitted_pars[1] += offset[1]

  return {"fitted_pars":fitted_pars,"data":X}


def get_logcdf_df(lbs,ubs,n_bins,distclass,years):
  obj = get_objects()

  qtls = [np.linspace(lbs[i],ubs[i],n_bins[i]) for i in range(2)]

  row_year = []
  row_q = []
  row_area = []
  row_logp = []

  for year in years:
    df = obj['data']
    if year != 'all':
      df = df.query('period == {y}'.format(y=year))

    demand = df[["gbdem_r","idem_r"]]
    wind = df[["gbwind_r","iwind_r"]]
    if demand.shape[1] != 2 or wind.shape[1] != 2:
      raise Exception("wrong demand or wind shapes")
    dist = distclass(
      np.array(demand),
      np.array(wind),
      obj['gens'])

    for area in ['GB','IRL']:
      if area == 'GB':
        idx = 0
      else:
        idx = 1
      m = [np.Inf,np.Inf]
      for q in qtls[idx]:
        m[idx] = q
        row_year.append(str(year))
        row_area.append(area)
        row_q.append(q)
        row_logp.append(np.log(dist.cdf(m))/np.log(10))

  return pd.DataFrame({'year':row_year,'area':row_area,'q':row_q,'logp':row_logp})

def get_hindcast_logcdf_df(lbs,ubs,n_strides,years):

  return get_logcdf_df(lbs,ubs,n_strides,BivariateHindcastMargin,years)

class ShortfallRegionHindcast(object):

  def __init__(self,hindcast_model):
    self.model = hindcast_model
    self.shortfall_region_prob = self.h_cdf(m=(0,np.Inf)) + self.h_cdf(m=(np.Inf,0)) - self.h_cdf(m=(0,0))

  def h_cdf(self,m):
    return self.model.cdf(m)

  # def cdf(self,m):
  #   if np.any(np.array(m) <= 0):
  #     return self.h_cdf(m)/self.shortfall_region_prob
  #   else:
  #     raw = self.h_cdf(m=(0,m[1])) + self.h_cdf(m=(m[0],0)) - self.h_cdf(m=(0,0))
  #     return raw/self.shortfall_region_prob

  def cdf(self,m):
    return truncated_model_cdf(m,self.h_cdf)#,self.shortfall_region_prob)

  def _create_marginal_m(self,m,i):
    if i == 0:
      m_ = (m,np.Inf)
    else:
      m_ = (np.Inf,m)
    return m_

  def marginal_cdf(self,m,i=0):
    m = self._create_marginal_m(m,i)
    return self.cdf(m)

  def marginal_quantile(self,p,i=0,x0=None):
    find = lambda m: self.cdf(self._create_marginal_m(m,i)) - p
    if x0 is None:
      x0 = 0
    res =  sp.optimize.root(find,x0=x0)
    return res.x

def get_sr_logcdf_df(lbs,ubs,n_strides,years):

  class WrapperClass(object):

    def __init__(self,demand,wind,gendists):
      hindcast_model = BivariateHindcastMargin(demand,wind,gendists)
      self.model = ShortfallRegionHindcast(hindcast_model)

    def cdf(self,m):
      return self.model.cdf(m)

  return get_logcdf_df(lbs,ubs,n_strides,WrapperClass,years)



