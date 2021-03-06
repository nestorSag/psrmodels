import numpy as np

from _c_ext_univarmargins import ffi, lib as C_CALL

from .ConvGenDistribution import *

import warnings

from scipy.optimize import bisect

class UnivariateHindcastMargin(object):
  """Univariate time-collapsed hindcast model

    **Parameters**:
    
    `gen` (`ConvGenDistribution`): available conventional generation object

    `nd_data` (`numpy.npdarray`): vector of net demand values

    `season_length` (`int`): Peak season length. if `None`, defaults to data length

  """
  def __init__(self,gen,nd_data, season_length=None):

    if not isinstance(gen,ConvGenDistribution):
      raise Exception("gen is not an instance of ConvGenDistribution")
      
    self.gen = gen
    self.nd_vals = np.ascontiguousarray(nd_data).astype(np.int32)#.clip(min=0)
    self.n = len(self.nd_vals)
    self.season_length = season_length if season_length is not None else self.n

    self.min = -np.max(self.nd_vals)
    self.max = self.gen.max - np.min(self.nd_vals)

  def renewables_efc(self,demand,renewables,metric="lole",tol=0.01):
    """calculate efc of wind fleet

    **Parameters**:
    
    `demand` (`numpy.ndarray`): array of demand observations

    `renewables` (`numpy.ndarray`): vector of renewable generation observations

    `metric` (`str` or function): baseline risk metric to perform the calculations; if `str`, the instance's method with matching name will be used; of a function, it needs to take a `UnivariateHindcastMargin` object as a parameter

    `tol` (`float`): absolute error tolerance with respect to true baseline risk metric for bisection function
    """

    if np.any(renewables < 0):
      raise Exception("renewable generation observations contain negative values.")

    if self.gen.fc != 0:
      warnings.warn("available generation's firm capacity is nonzero.")

    def get_risk_function(metric):

      if isinstance(metric,str):
        return lambda x: getattr(x,metric)()
      elif callable(metric):
        return metric

    with_wind_obj = UnivariateHindcastMargin(self.gen,demand - renewables)
    with_wind_risk = get_risk_function(metric)(with_wind_obj)

    def bisection_target(x):
      self.gen += x
      with_fc_obj = UnivariateHindcastMargin(self.gen,demand)
      with_fc_risk = get_risk_function(metric)(with_fc_obj)
      self.gen += (-x)
      #print("fc: {x}, with_fc_risk:{wfcr}, with_wind_risk: {wwr}".format(x=x,wfcr=with_fc_risk,wwr=with_wind_risk))
      return (with_fc_risk - with_wind_risk)/with_wind_risk

    diff_to_null = bisection_target(0)
    delta = 500

    #print("diff to null: {x}".format(x=diff_to_null))

    if diff_to_null == 0: #itc is equivalent to null interconnection riskwise
      return 0.0
    else:      
      # find suitalbe search intervals that are reasonably small
      if diff_to_null > 0: #interconnector adds risk => negative firm capacity
        rightmost = delta
        leftmost = 0
        while bisection_target(rightmost) > 0 :
          rightmost += delta
      else:
        leftmost = -delta
        rightmost = 0
        while bisection_target(leftmost) < 0:
          leftmost -= delta
      
      #print("finding efc in [{a},{b}]".format(a=leftmost,b=rightmost))
    efc, res = bisect(f=bisection_target,a=leftmost,b=rightmost,full_output=True,xtol=tol/2,rtol=tol/(2*with_wind_risk))
    #print("EFC: {efc}".format(efc=efc))
    if not res.converged:
      print("Warning: EFC estimator did not converge.")
    #print("efc:{efc}".format(efc=efc))
    return int(efc)
 
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
      return C_CALL.empirical_power_margin_cdf_py_interface(
        np.int32(x),
        np.int32(self.n),
        np.int32(self.gen.min),
        np.int32(self.gen.max),
        ffi.cast("int *",self.nd_vals.ctypes.data),
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

    return self.n * self.lolp() /(self.n/self.season_length)

  def _rescaled_cvar(self,x):

    return  C_CALL.empirical_cvar_py_interface(
              np.int32(x),
              np.int32(self.n),
              np.int32(self.gen.min),
              np.int32(self.gen.max),
              ffi.cast("int *",self.nd_vals.ctypes.data),
              ffi.cast("double *",self.gen.cdf_vals.ctypes.data),
              ffi.cast("double *",self.gen.expectation_vals.ctypes.data))

  def cvar(self, x):
    """calculate conditional value at risk for the energy unserved distribution conditioned to being non-zero

    **Parameters**:
    
    `x` (`int`): absolute value of power margin shortfall's lower bound

    """
    if x < 0:
      raise Exception("x must be a non-negative number.")

    # raw = self._rescaled_cvar(x)
    # if conditional:
    #   return raw/self.cdf(-x-1)
    # else:
    #   return raw
    raw = self._rescaled_cvar(x)
    return raw/self.cdf(-x-1)

  def eeu(self):
    """calculate expected energy unserved

    """

    return self.n * self.cvar(0) * self.cdf(-1) /(self.n/self.season_length)

  def _simulate_nd(self,n):

    row_range = range(len(self.nd_vals))
    row_idx = np.random.choice(row_range,size=n)

    return self.nd_vals[row_idx]

  def quantile(self,q):

    """Returns quantiles of the power margin distribution

    **Parameters**:
    
    `q` (`float`): quantile


    """
    def bisection(x):
      return (self.cdf(x) - q)/q
    
    delta = 1000

    lower = 0
    upper = 0

    while bisection(lower) >= 0:
      lower -= delta

    while bisection(upper) <= 0:
      upper += delta

    return int(bisect(f=bisection,a=lower,b=upper))

  def simulate(self,n,seed=1):
    """Simulate from hindcast distribution

    **Parameters**:
    
    `n` (`n`): number of simulations

    `seed` (`int`): random seed

    """

    np.random.seed(seed)
    gen_simulation = self.gen.simulate(n).reshape((n,1))
    nd_simulation = self._simulate_nd(n).reshape((n,1))

    margin_simulation = gen_simulation - nd_simulation

    return pd.DataFrame({"margin":margin_simulation.reshape(-1),"generation":gen_simulation.reshape(-1), "net_demand":nd_simulation.reshape(-1)})
                        

