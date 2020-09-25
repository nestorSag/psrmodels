import numpy as np
from scipy.stats import genpareto as gp
from scipy.stats import expon as expdist

from .UnivariateHindcastMargin import *

from _c_ext_univarmargins import ffi, lib as C_CALL

class UnivariateEVMargin(UnivariateHindcastMargin):
  """Univariate time-collapsed margin model where net demand tails are extrapolated using a Generalized Pareto with point parameter estimates

    **Parameters**:
    
    `gen` (`ConvGenDistribution`): available conventional generation object

    `nd_data` (`numpy.npdarray`): vector of net demand values

    `precompute` (`boolean`): should the margin CDF values be precomputed? 

    `u` (`float`): thresold for GP approximation 

    `sigma` (`float`): scale parameter for GP approximation

    `xi` (`float`): shape parameter for GP approximation

    """
  def __init__(self,gen,nd_data,u,sigma,xi):

    super().__init__(gen,nd_data)
    self.gen = gen
    self.n = len(self.nd_vals)

    self.u = u

    self.sigma = sigma

    #if np.abs(xi) < 1e-8: #to avoid numerical instability
    #  xi = 0

    self.xi = xi

    self.p = C_CALL.h_nd_cdf(
      np.float64(self.u),
      np.int64(self.n),
      ffi.cast("long *",self.nd_vals.ctypes.data)
      )

  # def _fit(self):

  #   """Fit EV model using method from: Park, M.H.; Kim, J.H.T. Estimating extreme tail risk measures with generalized Pareto distribution.
  #      Comput. Stat. Data Anal. 2016, 98, 91–104
    
  #   """

  #   exceedances = self.nd_vals[self.nd_vals > u] - u

  #   pars = np.ascontiguousarray([np.log(np.sd(exceedances)),0],dtype=np.float64) #array of initial parameters

  #   return C_CALL.fit_ev_model(
  #     np.int64(self.n),
  #     ffi.cast("double *",pars.ctypes.data),
  #     ffi.cast("double *",exceedances.ctypes.data)
  #     )

  #   return list(pars)
        
  def cdf(self,m):
    """calculate margin CDF values

    **Parameters**:
    
    `m` (`float`): point to evaluate on

    """
    return C_CALL.ev_margin_cdf(
      np.int64(m),
      np.float64(self.u),
      np.float64(self.p),
      np.float64(self.sigma),
      np.float64(self.xi),
      np.int64(self.n),
      np.int64(self.gen.max),
      ffi.cast("long *",self.nd_vals.ctypes.data),
      ffi.cast("double *",self.gen.cdf_vals.ctypes.data)
      )

  def epu(self):

    epu = C_CALL.ev_epu(
                        np.float64(self.u),
                        np.float64(self.p),
                        np.float64(self.sigma),
                        np.float64(self.xi),
                        np.int64(self.n),
                        np.int64(self.gen.max),
                        ffi.cast("long *",self.nd_vals.ctypes.data),
                        ffi.cast("double *",self.gen.cdf_vals.ctypes.data),
                        ffi.cast("double *",self.gen.expectation_vals.ctypes.data))

    if epu == -1:
      epu = np.Inf
      
    return epu

  def _simulate_nd(self,n):

    samples = np.empty((n,))

    u = np.random.binomial(1,self.p,n)
    n_samples_below = np.sum(u)

    below_obs = self.nd_vals[self.nd_vals <= u]
    n_obs_below = range(len(below_obs))

    row_idx = np.random.choice(n_obs_below,size=n_samples_below)

    samples[u==1] = below_obs[row_idx]

    n_samples_tail = n - n_samples_below

    samples[u==0] = gp.rvs(c=self.xi,loc=self.u,scale=self.sigma,size=n_samples_tail)

    return samples
