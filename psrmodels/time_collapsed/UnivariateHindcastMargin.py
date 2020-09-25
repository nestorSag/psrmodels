import numpy as np

from _c_ext_univarmargins import ffi, lib as C_CALL

from .ConvGenDistribution import *

class UnivariateHindcastMargin(object):
  """Univariate time-collapsed hindcast model

    **Parameters**:
    
    `gen` (`ConvGenDistribution`): available conventional generation object

    `nd_data` (`numpy.npdarray`): vector of net demand values

    """
  def __init__(self,gen,nd_data):

    if not isinstance(gen,ConvGenDistribution):
      raise Exception("gen is not an instance of ConvGenDistribution")
      
    self.gen = gen
    self.nd_vals = np.ascontiguousarray(nd_data).clip(min=0).astype(np.int64)
    self.n = len(self.nd_vals)

    self.min = -np.max(self.nd_vals)
    self.max = self.gen.max - np.min(self.nd_vals)
        
  def cdf(self,x):
    """calculate margin CDF values

    **Parameters**:
    
    `x` (`numpy.ndarray`): point to evaluate on

    """
    if x >= self.max:
      return 1.0
    elif x < self.min:
      return 0.0
    else:
      return C_CALL.h_margin_cdf(
        np.int64(x),
        np.int64(self.n),
        np.int64(self.gen.min),
        np.int64(self.gen.max),
        ffi.cast("long *",self.nd_vals.ctypes.data),
        ffi.cast("double *",self.gen.cdf_vals.ctypes.data)
        )

  def pdf(self,x):
    """calculate margin PDF values

    **Parameters**:
    
    `x` (`numpy.ndarray`): point to evaluate on

    """
    return self.cdf(x) - self.cdf(x-1)

  def lolp(self):
    """calculate loss of load probability
    
    """

    return self.cdf(-1)

  def lole(self):
    """calculate loss of load expectation

    """

    return self.n * self.lolp()

  def epu(self):
    """calculate expected power unserved

    **Parameters**:
    
    `x` (`numpy.ndarray`): point to evaluate on

    """
    return  C_CALL.h_epu(
              np.int64(self.n),
              np.int64(self.gen.min),
              np.int64(self.gen.max),
              ffi.cast("long *",self.nd_vals.ctypes.data),
              ffi.cast("double *",self.gen.cdf_vals.ctypes.data),
              ffi.cast("double *",self.gen.expectation_vals.ctypes.data))

  def eeu(self):
    """calculate expected energy unserved

    """

    return self.n * self.epu()

  def _simulate_nd(self,n):

    row_range = range(len(self.nd_vals))
    row_idx = np.random.choice(row_range,size=n)

    return self.nd_vals[row_idx]

  def simulate(self,n,seed=1):
    """Simulate from hindcast distribution

    **Parameters**:
    
    `n` (`n`): number of simulations

    `seed` (`int`): random seed

    """

    np.seed(seed)
    gen_simulation = self.gen.simulate(n).reshape((n,1))
    nd_simulation = self._simulate_nd(n).reshape((n,1))

    margin_simulation = gen_simulation - nd_simulation

    return {"margin":margin_simulation,"generation":gen_simulation, "net_demand":nd_simulation}
                        

