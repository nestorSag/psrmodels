import numpy as np
from scipy.optimize import bisect

import pandas as pd
import math

from .BivariateConvGenDist import * 
from .UnivariateHindcastMargin import *

from _c_ext_bivarmargins import ffi, lib as C_CALL

from deprecated import deprecated
from warnings import warn

class BivariateHindcastMargin(object):
  """Main class for risk calculations in a time-collapsed 2-area hindcast model.
  It receives demand and wind generation data as opposed to just net demand data, to make it posible to calculate
  risk metrics under a share policy, in which available power flow depends on demand as well as net demand.

  **Parameters**:
    
  `demand` (`numpy.ndarray`): matrix of demand values where columns are areas and rows are observations.

  `renewables` (`numpy.ndarray`): matrix of renewable generation values where columns are areas and rows are observations

  `gen_dists` (`ConvGenDistribution`, `pd.DataFrame` or `str`): list with information to build conventional generation objects: either the objects themselves, or input to the ConvGenDistribution class constructors
    
  `kwargs` Additional parameters to be passed to ConvGenDistribution constructor, such as column separators.
  """

  def __init__(self,demand,renewables,gen_dists,**kwargs):

    self.net_demand = np.ascontiguousarray((demand - renewables),dtype=np.int64).clip(min=0) #no negative net demand
    self.renewables = renewables
    self.demand = np.ascontiguousarray(demand,dtype=np.int64)
    self.gen_dists = self._parse_gendists(gen_dists,**kwargs)
    self.n = self.net_demand.shape[0]

    self.MARGIN_BOUND = int(np.iinfo(np.int64).max / 2)

  def _parse_gendists(self,data,**kwargs):
    gen_dists = []
    for entry in data:
      # if isinstance(entry, str) or isinstance(entry,pd.DataFrame):
      #   print("Passing to ConvGenDistribution constructor..")
      #   output = ConvGenDistribution(entry,**kwargs)
      # else:
      #   output = entry
      if isinstance(entry,ConvGenDistribution):
        output = entry
      else:
        print("Passing to ConvGenDistribution constructor..")
        output = ConvGenDistribution(entry,**kwargs)
      gen_dists.append(output)
    return gen_dists

  @staticmethod
  def bivar_ecdf(X):
    n = X.shape[0]
    ecdf = np.ascontiguousarray(np.empty((n,)),dtype=np.float64)
    C_CALL.bivar_ecdf(
      ffi.cast("double *",ecdf.ctypes.data),
      ffi.cast("double *",np.ascontiguousarray(X,dtype=np.float64).ctypes.data),
      np.int64(n))

    return ecdf

  @staticmethod
  def _triangle_prob(bigen, origin,length):
    """Recursive calculation of probability mass for the interior of a right, symmetric triangular lattice.
    This function is vestigial from previous package versions and is here only to test it; no method in this class
    uses this function anymore and instead call _trapezoid_prob

    **Parameters**:

    `bigen` (`BivariateConvGenDist`) bivariate available conventional generation object 
    
    `origin` (`list`): right angle coordinate in the plane

    `length` (`int`): length of triangle legs

    """
    origin = np.ascontiguousarray(origin,dtype=np.int64)

    return C_CALL.triangle_prob(
                      np.int64(origin[0]),
                      np.int64(origin[1]),
                      np.int64(length),
                      np.int64(bigen.X1.min),
                      np.int64(bigen.X2.min),
                      np.int64(bigen.X1.max),
                      np.int64(bigen.X2.max),
                      ffi.cast("double *",bigen.X1.cdf_vals.ctypes.data),
                      ffi.cast("double *",bigen.X2.cdf_vals.ctypes.data))


  @staticmethod
  def _is_shortfall_region(m1,m2,c,policy):
    """returns true if coordinate is in the shortfall region of area 1 for the given policy

    **Parameters**:
    
    `m1` (`int`): Area 1 margin

    `m2` (`int`): Area 2 margin

    `c` (`int`): Interconnector capacity

    `policy` (`string`): either 'share' or 'veto'

    """
    if policy == "share":
      return m1 < (c if m2 <= -c else -m2 if abs(m2) <= c else -c)
    elif policy == "veto":
      return m1 < (0 if m2 <= 0 else -m2 if m2 <= c else -c)
    else:
      raise Exception("policy not recognised")

  @staticmethod
  def _get_share_flow(m1,m2,d1,d2,c):
    """returns available flow to area 1 under a share policy and given demand and margin values

    **Parameters**:
    
    `m1` (`int`): Area 1 margin

    `m2` (`int`): Area 2 margin

    `d1` (`int`): Demand in area 1

    `d2` (`int`): Demand in area 2

    `c` (`int`): Interconnector capacity

    """
    if m1 + m2 < 0 and m1 < c and m2 < c:
      return min(max(float(d1)/(d1+d2)*m2 - float(d2)/(d1+d2)*m1,-c),c)
    else:
      return BivariateHindcastMargin._get_veto_flow(m1,m2,c)

  @staticmethod
  def _get_veto_flow(m1,m2,c):
    """returns available flow to area 1 under a veto policy and margin values

    **Parameters**:
    
    `m1` (`int`): Area 1 margin

    `m2` (`int`): Area 2 margin

    `c` (`int`): Interconnector capacity

    """
    if m1 > 0 and m2 < 0:
      return -min(c,m1,-m2)
    elif m1 < 0 and m2 > 0:
      return min(c,m2,-m1)
    else:
      return 0

  @staticmethod
  def _get_share_pu(m1,m2,d1,d2,c):
    """returns power unserved under a share policy for area 1

    **Parameters**:
    
    `m1` (`int`): Area 1 margin

    `m2` (`int`): Area 2 margin

    `d1` (`int`): Demand in area 1

    `d2` (`int`): Demand in area 2

    `c` (`int`): Interconnector capacity

    """

    if not BivariateHindcastMargin._is_shortfall_region(m1,m2,c,"share"):
      return 0
    else:
      return - (m1 + BivariateHindcastMargin._get_share_flow(m1,m2,d1,d2,c))

  @staticmethod
  def _get_veto_pu(m1,m2,c):
    """returns power unserved for area 1 under a veto policy

    **Parameters**:
    
    `m1` (`int`): Area 1 margin

    `m2` (`int`): Area 2 margin

    `c` (`int`): Interconnector capacity

    """

    if not BivariateHindcastMargin._is_shortfall_region(m1,m2,c,"veto"):
      return 0
    else:
      return - (m1 + BivariateHindcastMargin._get_veto_flow(m1,m2,c))

  def pdf(self,x,**kwargs):
    """calculate bivariate PDF

    **Parameters**:
    
    `x` (`numpy.ndarray`): margin values

    `kwargs` : additional arguments to be passed to the CDF function (system parameters like policy)


    """

    x = np.int64(x)
    return self.cdf(x,**kwargs) - (self.cdf(x - (1,0),**kwargs) - self.cdf(x - (0,1),**kwargs) + self.cdf(x - (1,1),**kwargs))


  def margin_cdf(self,x,i=0,**kwargs):
    """calculate bivariate margin CDF values based on a component index

    **Parameters**:
    
    `x` (`numpy.ndarray`): margin values

    `x` (`numpy.ndarray`): component index

    `kwargs` : additional arguments to be passed to the CDF function (system parameters like policy)


    """

    m = (x,self.MARGIN_BOUND) if i == 0 else (self.MARGIN_BOUND,x)
    return self.cdf(m=m,**kwargs)

  # def marginal_cdf(self,x,i=0):

  #   """calculate marginal power margin CDF for one of the areas

  #   **Parameters**:
    
  #   *x* (*np.ndarray*): margin values

  #   *i* (*int*): area index

  #   """
  #   margin = UnivariateHindcastMargin(self.gen_dists[i],self.net_demand[:,i])
  #   return margin.cdf(x)
    
  # def marginal_pdf(self,x,i=0):

  #   """calculate marginal power margin PDF for one of the areas

  #   **Parameters**:
    
  #   *x* (*np.ndarray*): margin values

  #   *i* (*int*): area index

  #   """

  #  return self.marginal_cdf(x,i) - self.marginal_cdf(x-1,i)

  def _swap_axes(self):
    """swap area indices

    """

    self.demand = np.flip(self.demand,axis=1)
    self.net_demand = np.flip(self.net_demand,axis=1)
    self.gen_dists = [self.gen_dists[1],self.gen_dists[0]]
    #self.gen_dists = self.gen_dists[::-1]

  def lole(self,c,policy,axis=0,get_pointwise_risk=False):
    """calculate LOLE for one of the areas

    **Parameters**:
    
    `c` (`numpy.ndarray`): interconnector capacity

    `policy` (`str`): either 'veto' or 'share'

    `axis (`int*): area index for which this will be calculated

    `get_pointwise_risk (`bool`): If `True`, returns pandas dataframe with LOLE contributions historic observation

    """

    # if axis == 1:
    #   self._swap_axes()

    m = [np.Inf,np.Inf]
    m[axis] = -1
    #m = (-1,np.Inf)
    lolp_vals = self.cdf(m=m,c=c,policy=policy,get_pointwise_risk=get_pointwise_risk) 
    if get_pointwise_risk:
      lole = lolp_vals
    else:
      lole = self.n * lolp_vals

    return lole

    #lole = self._lole(c=c,policy=policy,get_pointwise_risk=get_pointwise_risk)

    # if axis == 1:
    #   self._swap_axes()


  # @deprecated(version="1.0.0",reason="use lole() instead")
  # def lole(self,c,policy,axis=0,get_pointwise_risk=False):

  #   return self.lole(c,policy,axis,get_pointwise_risk)

  # def _lole(self,c,policy="share",get_pointwise_risk=False):
  #   """Returns LOLE for area 1

  #   **Parameters**:
    
  #   `dist` (`BivariateHindcastMargin`): bivariate hindcast object

  #   `c` (`int`): Interconnector capacity

  #   `policy` (`str`): Either 'share' or 'veto'

  #   `get_pointwise_risk` (`str`): return pandas DataFrame with net demand values and corresponding pointwise risk measurements
  #   """
  #   r = self.lolp(c,policy,get_pointwise_risk)
  #   if isinstance(r,pd.DataFrame):
  #     return r
  #   else:
  #     return self.n * r

  # def lolp(self,c,policy="share",get_pointwise_risk=False):
  #   """Returns LOLP for area 1

  #   **Parameters**:
    
  #   `dist` (`BivariateHindcastMargin`): bivariate hindcast object

  #   `c` (`int`): Interconnector capacity

  #   `policy` (`str`): Either 'share' or 'veto'

  #   `get_pointwise_risk` (`str`): return pandas DataFrame with net demand values and corresponding point-wise risk measurements. Does not work for a null interconnection capacity

  #   """
  #   m = (-1,np.Inf)
  #   return self.cdf(m=m,c=c,policy=policy,get_pointwise_risk=get_pointwise_risk)  


  def eeu(self,c,policy,axis=0,get_pointwise_risk=False):
    """calculate EEU for one of the areas

    **Parameters**:
    
    `c` (`int`): Interconnector capacity

    `policy` (`str`): either 'veto' or 'share'

    `axis` (`int`): area index for which this will be calculated 

    `get_pointwise_risk (`bool`): If `True`, returns pandas dataframe with EEU contributions for each historic observation

    """

    if axis == 1:
      self._swap_axes()
    
    epu_vals = self.epu(c=c,policy=policy,get_pointwise_risk=get_pointwise_risk)

    if axis == 1:
      self._swap_axes()

    if get_pointwise_risk:
      return epu_vals
    else:
      return self.n * epu_vals

    #return epu

  # def _eeu(self,c,policy="share",get_pointwise_risk=False):
  #   """Returns EEU for area 1

  #   **Parameters**:
    
  #   `dist` (`BivariateHindcastMargin`): bivariate hindcast object

  #   `c` (`int`): Interconnector capacity

  #   `policy` (`str`): Either 'share' or 'veto'

  #   `get_pointwise_risk` (`str`): return pandas DataFrame with net demand values and corresponding pointwise risk measurements
  #   """
  #   r = self.epu(c,policy,get_pointwise_risk)
  #   if isinstance(r,pd.DataFrame):
  #     return r
  #   else:
  #     return self.n * r

  def epu(self,c,policy="share",get_pointwise_risk=False):
    """Returns EPU for area 1

    **Parameters**:
    
    `c` (`int`): Interconnector capacity

    `policy` (`str`): Either 'share' or 'veto'

    `get_pointwise_risk` (`str`): return pandas DataFrame with EPU for each historic observation

    """
    if c > 0 or get_pointwise_risk:
      #self._check_null_fc()
      X1 = self.gen_dists[0]
      X2 = self.gen_dists[1]
      n = self.n

      EPU = 0

      if get_pointwise_risk:
        nd0 = []
        nd1 = []
        pu = []
        
      for i in range(n):
        #print(i)
        v1, v2 = self.net_demand[i,:]
        d1, d2 = self.demand[i,:]
        if policy == "share":
          point_EPU = self._cond_EPU_share(X1,X2,d1,d2,v1,v2,c)
        else:
          point_EPU = self._cond_EPU_veto(X1,X2,v1,v2,c)

        EPU += point_EPU
        
        if get_pointwise_risk:
          nd0.append(v1)
          nd1.append(v2)
          pu.append(point_EPU)

      if get_pointwise_risk:
        pw_df = pd.DataFrame({"nd0":nd0,"nd1":nd1,"value":pu})
        return pw_df 
      else:
        return EPU/n
        #return self.season_hours * EPU/n


    else:
      # if interconnector capacity is zero, use UnivariateHindcastMargin to compute risks
      # as it does it more efficiently
      margin = UnivariateHindcastMargin(self.gen_dists[0],self.net_demand[:,0])
      return margin.epu()
  


  def system_eeu(self,c,**kwargs):

    """calculate system-wide EEU

    **Parameters**:
    
    `c` (`int`): Interconnector capacity

    """
    return self.eeu(c,axis=0,**kwargs) + self.eeu(c,axis=1,**kwags)

  def _simulate_net_demand(self,n,seed=None):
    """simulate net demand

    **Parameters**:
    
    `n` (`int`): number of samples

    """
    if seed is not None:
      np.random.seed(seed)

    return self.net_demand[np.random.choice(range(self.n),size=n),:]

  def efc(self,**kwargs):
    """This method calculates the equivalent firm capacity of interconnection for a given power system. It has been deprecated; use the `itc_efc` method instead. When called, this function calls `itc_efc`, passing all arguments along.
    """
    warn("This method is deprecated; use methods itc_efc or convgen_ifc instead.")
    return self.itc_efc(**kwargs)

  def convgen_efc(self, cap, prob, gen_axis, fc_axis, c,policy,metric="lole",axis=0,tol=0.1,**kwargs):
    """Returns the amount of firm capacity that needs to be added to area `fc_axis` such that area `axis` has the same risk (as defined by `metric`) than if new conventional generation was installed in `gen_axis`.

    **Parameters**:
    
    `cap` (`int`): maximum available capacity of new generator in MW

    `prob` (`float`): availability probability for the new generator

    `gen_axis` (`int`): Axis to which the new generator will be added

    `fc_axis` (`int`): Area to which firm capacity will be added

    `c` (`int`): interconnection capacity

    `policy` (`str`): Either 'share' or 'veto'

    `axis` (`int`): area for which risk will be calculated

    `metric` (`string`): name of the instance's method that will be used to measure risk. Only 'LOLE' and 'EEU' are supported

    `tol` (`float`): absolute error tolerance from true target risk value

    """
    if not (axis in [0,1]):
      raise Exception("axis value must be 0 or 1")

    if not (gen_axis in [0,1]):
      raise Exception("gen_axis value must be 0 or 1")

    if not (fc_axis in [0,1]):
      raise Exception("fc_axis value must be 0 or 1")

    if prob > 1 or prob < 0:
      raise Exception("Availability value must be between 0 and 1")

    other = 1 - gen_axis
    gen_data = self.gen_dists[gen_axis].original_data

    # create augmented conv gen distribution
    new_row = pd.Series([cap,prob],index=["Capacity","Availability"])
    augmented_data = gen_data.append(new_row,ignore_index=True)
    augmented_gen_dist = ConvGenDistribution(augmented_data)

    # create augmented conv gen bivariate distribution and get risk metric in agumented system
    new_bivariate_dist = [0,1] #place holder values
    new_bivariate_dist[gen_axis] = augmented_gen_dist
    new_bivariate_dist[other] = self.gen_dists[other]
    new_margin_dist = BivariateHindcastMargin(self.demand,self.renewables,new_bivariate_dist)
    new_metric_func = getattr(new_margin_dist,metric)
    new_metric_val = new_metric_func(c=c,policy=policy,axis=axis)

    print("new metric val: {x}".format(x=new_metric_val))
    ### take original system and add firm capacity until we get new_metric_val
    # define bisection algorithm's bounds
    if cap >= 0:
      leftmost = 0
      rightmost = cap
    else:
      rightmost = 0
      leftmost = cap

    # clone object to prevent any side effects
    original_bivariate_dist = [ConvGenDistribution(self.gen_dists[area].original_data) for area in range(2)]
    
    # define objective for bisection algorithm
    def find_efc(x):
      # add firm capacity
      original_bivariate_dist[fc_axis] += x
      # create bivariate margin distribution object
      dist = BivariateHindcastMargin(self.demand,self.renewables,original_bivariate_dist)
      metric_func = getattr(dist,metric)
      metric_val = metric_func(c=c,policy=policy,axis=axis)

      # reset firm capacity to 0
      original_bivariate_dist[fc_axis] += (-original_bivariate_dist[fc_axis].fc)

      print("Adding {x}, getting val {v}".format(x=x,v=metric_val))
      return metric_val - new_metric_val

    efc, res = bisect(f=find_efc,a=leftmost,b=rightmost,full_output=True,xtol=tol/2,rtol=tol/(2*new_metric_val))
    if not res.converged:
      print("Warning: EFC estimator did not converge.")
    #print("efc:{efc}".format(efc=efc))
    return int(efc)


  def itc_efc(self,c,policy,metric="lole",axis=0,tol=0.1,**kwargs):
    """Returns equivalent firm capacity of interconnector in one area

    **Parameters**:
    
    `c` (`int`): interconnection capacity

    `policy` (`str`): Either 'share' or 'veto'

    `axis` (`int`): area for which risk will be calculated

    `metric` (`string`): name of the instance's method that will be used to measure risk

    `tol` (`float`): absolute error tolerance from true target risk value

    """
    if not metric in ["lole","eeu"]:
      raise Exception("Only 'lole' or 'eeu' supported as risk metrics")
    
    #target value
    with_itc = getattr(self,metric)(c=c,axis=axis,policy=policy,**kwargs)

    def compare_itc_to_fc(k):
      self.gen_dists[axis] += k ## add firm capacity
      univar = UnivariateHindcastMargin(self.gen_dists[axis],self.net_demand[:,axis])
      #risk_metric = getattr(univar,metric)
      without_itc = getattr(univar,metric)()
      #k_fc_risk =  with_itc - without_itc
      self.gen_dists[axis] += (-k) #reset firm capacity to 0
      return with_itc - without_itc

    diff_to_null = compare_itc_to_fc(0)

    # now find the root of compare_itc_to_fc by bisection

    # is the interconnector adding risk?
    if diff_to_null == 0: #itc is equivalent to null interconnection riskwise
      return 0.0
    else:
      # find suitalbe search intervals that are reasonably small
      if diff_to_null > 0: #interconnector adds risk => negative firm capacity
        rightmost = 0
        leftmost = -c
        while compare_itc_to_fc(leftmost) > 0 :
          leftmost -= c
      else:
        leftmost = 0
        rightmost = c
        while compare_itc_to_fc(rightmost) < 0:
          rightmost += c
      
      #print("finding efc in [{a},{b}]".format(a=leftmost,b=rightmost))
      efc, res = bisect(f=compare_itc_to_fc,a=leftmost,b=rightmost,full_output=True,xtol=tol/2,rtol=tol/(2*with_itc))
      if not res.converged:
        print("Warning: EFC estimator did not converge.")
      #print("efc:{efc}".format(efc=efc))
      return int(efc)

  def _trapezoid_prob(self,X,ulc,c):

    """Compute the probability mass of a trapezoidal segment of the plane
    # The trapezoid os formed by stacking a right triangle on top of a rectangle
    # where the hypotenuse is facing to the right

    **Parameters**:

    `X` (`BivariateConvGenDist`) bivariate available conventional generation object 
    
    `ulc` (`list`): upper left corner

    `c` (`int`): width of trapezoid

    """
    return C_CALL.trapezoid_prob(
                        np.int64(ulc1),
                        np.int64(ulc2),
                        np.int64(c),
                        np.int64(X.X1.min),
                        np.int64(X.X2.min),
                        np.int64(X.X1.max),
                        np.int64(X.X2.max),
                        ffi.cast("double *",X.X1.cdf_vals.ctypes.data),
                        ffi.cast("double *",X.X2.cdf_vals.ctypes.data))

  def _cond_EPU_share(self,FX1,FX2,d1,d2,v1,v2,c):
    """Returns EPU conditional on given demand and wind generations under a share policy

    **Parameters**:
    
    `FX1` (`ConvGenDistribution`): available conventional generation distribution object for area 1

    `FX2` (`ConvGenDistribution`): available conventional generation distribution object for area 2

    `d1` (`int`): demand in area 1

    `d2` (`int`): demand in area 2

    `v1` (`int`): net demand in area 1 (demand - renewable generation)

    `v2` (`int`): net demand in area 2

    `c` (`int`): Interconnector capacity

    """

    return C_CALL.cond_epu_share(
                    np.int64(d1),
                    np.int64(d2),
                    np.int64(v1),
                    np.int64(v2),
                    np.int64(c),
                    np.int64(FX1.min),
                    np.int64(FX2.min),
                    np.int64(FX1.max),
                    np.int64(FX2.max),
                    ffi.cast("double *",FX1.cdf_vals.ctypes.data),
                    ffi.cast("double *",FX2.cdf_vals.ctypes.data),
                    ffi.cast("double *",FX1.expectation_vals.ctypes.data))
  
  def _cond_EPU_veto(self,FX1,FX2,v1,v2,c):
    """Returns EPU conditional on given demand and wind generations under a veto policy

    **Parameters**:
    
    `FX1` (`ConvGenDistribution`): available conventional generation distribution object for area 1

    `FX2` (`ConvGenDistribution`): available conventional generation distribution object for area 2

    `v1` (`int`): net demand in area 1 (demand - renewable generation)

    `v2` (`int`): net demand in area 2

    `c` (`int`): Interconnector capacity

    """

    return C_CALL.cond_epu_veto(
                    np.int64(v1),
                    np.int64(v2),
                    np.int64(c),
                    np.int64(FX1.min),
                    np.int64(FX2.min),
                    np.int64(FX1.max),
                    np.int64(FX2.max),
                    ffi.cast("double *",FX1.cdf_vals.ctypes.data),
                    ffi.cast("double *",FX2.cdf_vals.ctypes.data),
                    ffi.cast("double *",FX1.expectation_vals.ctypes.data))
  
  def _check_null_fc(self):
    if np.any([d.fc != 0 for d in self.gen_dists] + [d.min != 0 for d in self.gen_dists]):
        raise Exception("Bivariate calculations do not support non-zero firm capacity")

  def simulate_region(self,n,m,c,policy,intersection=True,seed=1):
    """ Simulate region of post interconnector power margins

    **Parameters**:
    
    `n` (`int`): number of simulations

    `m` (`tuple`): Upper bound that delimits the region for each component

    `c` (`tuple`): Interconnection capacity

    `policy` (`str`): Either 'share' or 'veto'

    `intersection` (`bool`): if `True`, simulate from region given by `m[0] <= m_0 AND m[1] <= m_1` inequality; otherwise from region `m[0] <= m_0 OR m[1] <= m_1`

    `seed` (`int`): random seed
    """

    def get_prob_df(m,c,policy,intersection):
      if intersection:
        df = self.cdf(m=m,c=c,policy=policy,get_pointwise_risk=True)
      else:
        if m[0] >= self.MARGIN_BOUND or m[1] >= self.MARGIN_BOUND:
          # the union of anything with constraint <= infinity is the whole plane
          df = self.cdf(m=(self.MARGIN_BOUND,self.MARGIN_BOUND),c=c,policy=policy,get_pointwise_risk=True)
        else:
          df1 = self.cdf(m=(self.MARGIN_BOUND,m[1]),c=c,policy=policy,get_pointwise_risk=True)
          df2 = self.cdf(m=(m[0],self.MARGIN_BOUND),c=c,policy=policy,get_pointwise_risk=True)
          df3 = self.cdf(m=m,c=c,policy=policy,get_pointwise_risk=True)
          union_prob = df1["value"] + df2["value"] - df3["value"]
          df = pd.DataFrame({"value": union_prob,"nd0":df3["nd0"],"nd1":df3["nd1"]})

      df["d0"] = self.demand[:,0]
      df["d1"] = self.demand[:,1]
      df = df.sort_values(by="value",ascending=True)#.query("value >= 0") #sometimes rounding errors may 
      # produce negative probabilities in the order of -1e-60
      #print(df)
      return df

    np.random.seed(seed)

    m = np.clip(m,a_min=-self.MARGIN_BOUND,a_max=self.MARGIN_BOUND)
    m1, m2 = m

    X1 = self.gen_dists[0]
    X2 = self.gen_dists[1]

    X = BivariateConvGenDist(X1,X2) #system-wide conv. gen. distribution

    simulated = np.ascontiguousarray(np.zeros((n,2)),dtype=np.int64)

    ### calculate conditional probability of each historical observation given
    ### margin value tuple m
    df = get_prob_df(m=m,c=c,policy=policy,intersection=intersection)
    probs = df["value"]
    total_prob = np.sum(probs)
    if total_prob <= 1e-8:
      raise Exception("Region has probability lower than 1e-8; too small to simulate accurately")
    else:
      probs = np.array(probs)/total_prob
      
      df["row_weights"] = np.random.multinomial(n=n,pvals=probs,size=1).reshape((df.shape[0],))
      ## only pass rows which induce at least one simulated value
      df = df.query("row_weights > 0")

      row_weights = np.ascontiguousarray(df["row_weights"],dtype=np.int64)

      net_demand = np.ascontiguousarray(df[["nd0","nd1"]],dtype=np.int64)

      demand = np.ascontiguousarray(df[["d0","d1"]],dtype=np.int64)

      C_CALL.region_simulation(
        np.int64(n),
        ffi.cast("long *",simulated.ctypes.data),
        np.int64(X.X1.min),
        np.int64(X.X2.min),
        np.int64(X.X1.max),
        np.int64(X.X2.max),
        ffi.cast("double *",X.X1.cdf_vals.ctypes.data),
        ffi.cast("double *",X.X2.cdf_vals.ctypes.data),
        ffi.cast("long *",net_demand.ctypes.data),
        ffi.cast("long *",demand.ctypes.data),
        ffi.cast("long *",row_weights.ctypes.data),
        np.int64(net_demand.shape[0]),
        np.int64(m1),
        np.int64(m2),
        np.int64(c),
        int(seed),
        int(intersection),
        int(policy == "share"))

      return simulated

  def simulate_conditional(self,n,cond_value,cond_axis,c,policy,seed=1):
    """ Simulate power margins in one area conditioned to a particular value in the other area

    **Parameters**:
    
    `n` (`int`): number of simulated values

    `cond_value` (`int`): conditioning power margin value

    `cond_axis` (`int`): Conditioning component

    `c` (`tuple`): Interconnection capacity

    `policy` (`str`): Either 'share' or 'veto'

    `seed` (`int`): random seed
    """

    np.random.seed(seed)
    m1 = np.clip(cond_value,a_min=-self.MARGIN_BOUND,a_max=self.MARGIN_BOUND)
    m2 = self.MARGIN_BOUND

    if cond_axis == 1:
      self._swap_axes()

    X1 = self.gen_dists[0]
    X2 = self.gen_dists[1]

    X = BivariateConvGenDist(X1,X2) #system-wide conv. gen. distribution

    simulated = np.ascontiguousarray(np.zeros((n,2)),dtype=np.int64)

    ### calculate conditional probability of each historical observation given
    ### margin value tuple m
    df = self.cdf(m=(m1,m2),c=c,policy=policy,get_pointwise_risk=True)    
    df["value"] = df["value"] - self.cdf(m=(m1-1,m2),c=c,policy=policy,get_pointwise_risk=True)["value"]

    df["d0"] = self.demand[:,0]
    df["d1"] = self.demand[:,1]

    ## rounding errors can make probabilities negative of the order of 1e-60
    df = df.query("value > 0")
    df = df.sort_values(by="value",ascending=True)
    probs = df["value"]
    total_prob = np.sum(probs)
    
    if total_prob <= 1e-8:
      raise Exception("Region has probability lower than 1e-8; too small to simulate accurately")
    else:
      probs = np.array(probs)/total_prob
      
      df["row_weights"] = np.random.multinomial(n=n,pvals=probs,size=1).reshape((df.shape[0],))
      ## only pass rows which induce at least one simulated value
      df = df.query("row_weights > 0")

      row_weights = np.ascontiguousarray(df["row_weights"],dtype=np.int64)

      net_demand = np.ascontiguousarray(df[["nd0","nd1"]],dtype=np.int64)

      demand = np.ascontiguousarray(df[["d0","d1"]],dtype=np.int64)

      C_CALL.conditioned_simulation(
          np.int64(n),
          ffi.cast("long *",simulated.ctypes.data),
          np.int64(X.X1.min),
          np.int64(X.X2.min),
          np.int64(X.X1.max),
          np.int64(X.X2.max),
          ffi.cast("double *",X.X1.cdf_vals.ctypes.data),
          ffi.cast("double *",X.X2.cdf_vals.ctypes.data),
          ffi.cast("long *",net_demand.ctypes.data),
          ffi.cast("long *",demand.ctypes.data),
          ffi.cast("long *",row_weights.ctypes.data),
          np.int64(net_demand.shape[0]),
          np.int64(m1),
          np.int64(c),
          int(seed),
          int(policy == "share"))

    if cond_axis == 1:
      self._swap_axes()

    return simulated[:,1] #first column has variable conditioned on (constant value)

  def cdf(self,m,c=0,policy="share",get_pointwise_risk=False):

    """Evaluate the CDF of bivariate power margins for a given system configuration under hindcast.

    **Parameters**:
    
    `m` (`tuple`, `list`, or `numpy.ndarray`) point to evaluate in power margin space

    `c` (`int`): Interconnector capacity

    `policy` (`str`): Either 'share' or 'veto'

    `get_pointwise_risk` (`str`): return pandas DataFrame with shortfall probabilities induced by each historic observation

    """

    m = np.clip(m,a_min=-self.MARGIN_BOUND,a_max=self.MARGIN_BOUND)
    m1, m2 = m
    #self._check_null_fc()

    gendist1, gendist2 = self.gen_dists
    #X2 = self.gen_dists[1]

    #X = BivariateConvGenDist(X1,X2) #system-wide conv. gen. distribution
   
    n = self.n

    cdf = 0

    if get_pointwise_risk:
      nd0 = []
      nd1 = []
      cdf_list = []
      
    for i in range(n):
      #print(i)
      v1, v2 = self.net_demand[i,:]

      d1, d2 = self.demand[i,:]

      point_cdf = C_CALL.get_cond_cdf(
                      np.int64(gendist1.min),
                      np.int64(gendist2.min),
                      np.int64(gendist1.max),
                      np.int64(gendist2.max),
                      ffi.cast("double *",gendist1.cdf_vals.ctypes.data),
                      ffi.cast("double *",gendist2.cdf_vals.ctypes.data),
                      np.int64(m1),
                      np.int64(m2),
                      np.int64(v1),
                      np.int64(v2),
                      np.int64(d1),
                      np.int64(d2),
                      np.int64(c),
                      np.int64(policy == "share"))

      #print("point cdf: {x}, index: {i}".format(x=point_cdf, i=i))
      cdf += point_cdf

      #print(v1)
      if get_pointwise_risk:
          nd0.append(v1)
          nd1.append(v2)
          cdf_list.append(point_cdf)

    if get_pointwise_risk:
      pw_df = pd.DataFrame({"nd0":nd0,"nd1":nd1,"value":cdf_list})
      #print(pw_df)
      return pw_df
    else:   
      return cdf/n

    # else:
    #   # if interconnector capacity is zero, use UnivariateHindcastMargin to compute risks
    #   # as it does it more efficiently
    #   margin = UnivariateHindcastMargin(self.gen_dists[0],self.net_demand[:,0])
    #   return margin.lolp()

  def system_lolp(self,c,get_pointwise_risk=False):
    """Returns system-wise LOLP

    **Parameters**:
    
    `c` (`int`): Interconnector capacity

    `policy` (`str`): Either 'share' or 'veto'

    `get_pointwise_risk` (`str`): return pandas DataFrame with LOLPs for each historic observation

    """

    # calculate LOLP
    X1 = self.gen_dists[0]
    X2 = self.gen_dists[1]

    X = BivariateConvGenDist(X1,X2) #system-wide conv. gen. distribution
    
    n = self.n

    LOLP = 0

    if get_pointwise_risk:
      nd0 = []
      nd1 = []
      lolp = []
      
    for i in range(n):
      #print(i)
      v1, v2 = self.net_demand[i,:]
      # system-wide LOLP does not depend on the policy
      #point_LOLP = X.cdf((v1-c-1,math.inf)) + X.cdf((math.inf,v2-c-1)) - X.cdf((v1-c-1,v2-c-1)) + BivariateHindcastMargin._triangle_prob(X,(v1-c-1,v2-c-1),2*c+1)
      point_LOLP = X.cdf((v1-c-1,math.inf)) + X.cdf((math.inf,v2-c-1)) - X.cdf((v1+c,v2-c-1)) + self._trapezoid_prob(X,(v1-c-1,v2+c),2*c)
      LOLP += point_LOLP

      if get_pointwise_risk:
        nd0.append(v1)
        nd1.append(v2)
        lolp.append(point_LOLP)

    if get_pointwise_risk:
      pw_df = pd.DataFrame({"nd0":nd0,"nd1":nd1,"value":lolp})
      return pw_df
    else:    
      return LOLP/n

  def _simulate_conditional_demand_ratio(self,n,cond_value,cond_axis,c,policy,seed=1):

    np.random.seed(seed)
    m1 = np.clip(cond_value,a_min=-self.MARGIN_BOUND,a_max=self.MARGIN_BOUND)
    m2 = self.MARGIN_BOUND

    if cond_axis == 1:
      self._swap_axes()

    ### calculate conditional probability of each historical observation given
    ### margin value tuple m
    df = self.cdf(m=(m1,m2),c=c,policy=policy,get_pointwise_risk=True)    
    #df["value"] = df["value"] - self.cdf(m=(m1-1,m2),c=c,policy=policy,get_pointwise_risk=True)["value"]

    df["d0"] = self.demand[:,0]
    df["d1"] = self.demand[:,1]

    ## rounding errors can make probabilities negative of the order of 1e-60
    df = df.query("value > 0")
    df = df.sort_values(by="value",ascending=True)
    probs = df["value"]
    total_prob = np.sum(probs)
    
    if total_prob <= 1e-8:
      raise Exception("Region has probability lower than 1e-8; too small to simulate accurately")
    else:
      probs = np.array(probs)/total_prob
      
      sample_indices = np.random.choice(range(df.shape[0]),size=n,replace=True,p=probs)
      d0 = np.array(df["d0"])[sample_indices]
      d1 = np.array(df["d1"])[sample_indices]

      return d0/(d0 + d1)


  def system_lole(self,c,get_pointwise_risk=False):
    """Returns system-wise LOLE

    **Parameters**:
    
    `c` (`int`): Interconnector capacity

    `policy` (`str`): Either 'share' or 'veto'

    `get_pointwise_risk` (`str`): return pandas DataFrame with system LOLE contributions for each historic observation
    """
    return self.n * self.system_lolp(c,get_pointwise_risk)


  def margin_quantile(self,q,i=0,c=0,policy="veto"):

    """Returns quantile of any of the margin distributions

    **Parameters**:
    
    `q` (`float`): quantile

    `i (`int*): area index for which this will be calculated

    `c` (`int`): Interconnector capacity

    `policy` (`str`): Either 'share' or 'veto'

    """
    def bisection(x):
      y = np.empty((2,))
      y[1-i] = np.Inf
      y[i] = x
      return self.cdf(y,c=c,policy=policy) - q
    
    step = 1000

    lower = 0
    upper = 0

    while bisection(lower) >= 0:
      lower -= step

    while bisection(upper) <= 0:
      upper += step

    return bisect(f=bisection,a=lower,b=upper)

  # def gen_efc(self,c,policy,metric="LOLE",axis=0,**kwargs):
  #   """Returns equivalent firm capacity of interconnector in one area

  #   **Parameters**:
    
  #   *c* (*int*): interconnection capacity

  #   *policy* (*str*): Either 'share' or 'veto'

  #   *axis* (*int*): area for which this will be calculated

  #   *metric* (*string*): name of the instance's method that will be used to measure risk

  #   """
  #   if not metric in ["LOLE","EEU"]:
  #     raise Exception("Only LOLE or EPU supported as risk metrics")
    
  #   with_itc = getattr(self,metric)(c=c,axis=axis,policy=policy,**kwargs)
    


