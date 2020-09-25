import numpy as np
from scipy.stats import genpareto as gp
from scipy.stats import expon as expdist

import pymc3 as pm
import matplotlib.pyplot as plt

import theano
import theano.tensor as tt

from .UnivariateEVMargin import * 

from _c_ext_univarmargins import ffi, lib as C_CALL

class UnivariateBayesianEVMargin(UnivariateEVMargin):
  """Univariate time-collapsed margin model where net demand tails are extrapolated using a Generalized Pareto and it is fitted through Bayesian estimation

    **Parameters**:
    
    `gen` (`ConvGenDistribution`): available conventional generation object

    `nd_data` (`numpy.npdarray`): vector of net demand values

    `u` (`float`): thresold for GP approximation 

    `n_posterior_samples` (`int`): number of samples from posterior

    `plot_trace` (`boolean`): should the posterior plots be shown? 

    """
  def __init__(self,gen,nd_data,u,n_posterior_samples=2500,plot_trace=False,seed=1):

    super().__init__(gen,nd_data,u,None,None)

    nd_data = np.array(nd_data)
    exceedances = nd_data[nd_data > u] - u
    print("getting posterior samples...")
    self._get_posterior_samples(exceedances,n_posterior_samples,plot_trace,seed)
    

  def cdf(self,m):
    """calculate margin CDF values

    **Parameters**:
    
    `m` (`float`): point to evaluate on

    """
    return C_CALL.bev_margin_cdf(
      np.int64(m),
      np.float64(self.u),
      np.float64(self.p),
      np.int64(self.n_posterior),
      ffi.cast("double *",self.posterior_sigma.ctypes.data),
      ffi.cast("double *",self.posterior_xi.ctypes.data),
      np.int64(self.n),
      np.int64(self.gen.max),
      ffi.cast("long *",self.nd_vals.ctypes.data),
      ffi.cast("double *",self.gen.cdf_vals.ctypes.data)
      )

  # def _nd_tail_cdf(self,x):
  #   return np.mean(gp.cdf(x,c=self.posterior_xi,loc=self.u,scale=self.posterior_sigma))

  def epu(self):

    epu = C_CALL.bev_epu(
                        np.float64(self.u),
                        np.float64(self.p),
                        np.int64(self.n_posterior),
                        ffi.cast("double *",self.posterior_sigma.ctypes.data),
                        ffi.cast("double *",self.posterior_xi.ctypes.data),
                        np.int64(self.n),
                        np.int64(self.gen.max),
                        ffi.cast("long *",self.nd_vals.ctypes.data),
                        ffi.cast("double *",self.gen.cdf_vals.ctypes.data),
                        ffi.cast("double *",self.gen.expectation_vals.ctypes.data))

    if epu == -1:
      epu = np.Inf
      
    return epu

  def _get_posterior_samples(self,obs,n_samples,seed):

    def tegpd_logp(x,xi,sigma):
      #returns the sum of log-liklihoods
      if xi != 0:
        return tt.sum(-tt.log(sigma) - (1.0/xi+1)*tt.log(1+xi/sigma*x))
      else:
        return tt.sum(-tt.log(sigma) - x/sigma)

  
    obs = np.array(obs)
    
    x_max = np.max(obs)
    #Bayesian specification and inference
    gp_model = pm.Model()

    with gp_model:
      #xi, sigma = get_priors(prior_name)
      xi = pm.Normal('xi',mu=0,sigma=1)
      sigma = pm.HalfFlat('sigma')

      X = pm.DensityDist("tegpd",tegpd_logp,observed={"xi":xi,"sigma":(-xi*x_max).clip(0,np.Inf) + sigma,"x":obs})
      trace = pm.sample(n_samples,random_seed = seed)
      
    #get sampled posteriors
    xi_samples = trace.get_values("xi")
    sigma_samples = trace.get_values("sigma") + (-xi_samples*x_max).clip(0,np.Inf)
    mat = np.concatenate([
      np.array(sigma_samples).reshape(-1,1),
      np.array(xi_samples).reshape(-1,1)],axis=1)

    print("Done")
    self.posterior_sigma = np.ascontiguousarray(mat[:,0],dtype=np.float64)
    self.posterior_xi = np.ascontiguousarray(mat[:,1],dtype=np.float64)

    self.n_posterior = len(self.posterior_sigma)

  def _simulate_nd(self,n):

    samples = np.empty((n,))

    u = np.random.binomial(1,self.p,n)
    n_samples_below = np.sum(u)

    below_obs = self.nd_vals[self.nd_vals <= u]
    n_obs_below = range(len(below_obs))

    row_idx = np.random.choice(n_obs_below,size=n_samples_below)

    samples[u==1] = below_obs[row_idx]

    n_samples_tail = n - n_samples_below

    posterior_idx = np.random.choice(range(self.n_posterior),n_samples_tail)
    posterior_xi_sample = self.posterior_xi[posterior_idx]
    posterior_sigma_sample = self.posterior_sigma[posterior_idx]
    
    samples[u==0] = gp.rvs(c=posterior_xi_sample,loc=self.u,scale=posterior_sigma_sample,size=n_samples_tail)

    return samples