import numpy as np
from scipy.optimize import bisect

class MonteCarloEstimator(object):
  """ Class that contains method to calculate Monte Carlo estimates for reliability analysis from a sample of observed pre-interconnection power margin values
  """
  @staticmethod
  def get_efc_metric_function(metric,obj,**kwargs):
      if isinstance(metric,str):
        return getattr(obj,metric)
      elif callable(metric):
        return lambda obs,c,policy,axis: metric(obj=obj,obs=obs,c=c,policy=policy,axis=axis,**kwargs) #return curried metric set_metric_function
      else:
        raise Exception("'metric' has to be either a function or a string")

  # @staticmethod
  def find_shortfalls(self,obs,axis,c,policy):
    if axis not in [0,1]:
      raise Exception("axis value must be 0 or 1")
    if policy not in ["veto","share"]:
      raise Exception("Policy not recognised.")
    other = 1 - axis
    shortfall_region_limit = 0 if policy == "veto" else c
    shortfalls = np.logical_or(obs[:,axis] < -c,np.logical_and(obs[:,axis] < shortfall_region_limit,obs[:,other] < -obs[:,axis]))

    return shortfalls

  # @staticmethod
  # def _lolp(self,shortfalls):
  #   return np.mean(shortfalls)

  #@staticmethod
  def lole(self,obs,season_length=3360,c=1000,policy="veto",axis=0,demand=None):
    """returns LOLE estimate

    **Parameters**:
    
    `obs` (`numpy.ndarray`): Matrix with observations of power margin values, with a column per area

    `season_length` (`int`): Length of season under consideration. Defaults to length of single UK peak season in hours.

    `c` (`int`): Interconnector capacity

    `policy` (`string`): either 'share' or 'veto'

    `axis` (`int`): area for which this will be calculated

    `demand` (`numpy.ndarray`): Matrix with observations of demand values, with a column per area. Can be `None` if policy equals `veto`.

    """
    #shortfalls = self.find_shortfalls(obs,axis,c,policy)

    #return np.mean(shortfalls) * season_length
    m = (-1,np.Inf) if axis == 0 else (np.Inf,-1)
    #print("from MC m = {m}, axis ={a}".format(m=m,a=axis))
    return self.cdf(x=m,obs=obs,c=c,policy=policy,demand=demand) * season_length

  #@staticmethod
  def _power_flow(self,obs,c,policy,to_axis,demand=None):
    m, n = obs.shape
    other = 1 - to_axis

    def veto_flow(row):
      if np.all(row <= 0) or np.all(row >= 0):
        return 0.0
      elif row[other] > 0 and row[to_axis] < 0:
        return np.min([row[other],-row[to_axis],c])
      else:
        return -np.min([c,-row[other],row[to_axis]])

    def share_flow(extended_row):
      demand = extended_row[2:4]
      row = extended_row[0:2]
      if np.sum(row) < 0 and np.all(row < c):
        r = demand[to_axis]/np.sum(demand,axis=1)
        flow = min(max(r*obs[other] - (1-r)*obs[to_axis],-c),c)
      else:
        return veto_flow(row)

    shortfall_idx = np.logical_or(obs[:,0] < 0, obs[:,1] < 0)
    shortfalls = obs[shortfall_idx,:]

    if policy == "veto":
      flow = np.apply_along_axis(lambda row: float(veto_flow(row)),arr=shortfalls,axis=1)
    else:
      if demand is not None:
        r = demand[:,to_axis]/np.sum(demand,axis=1)
        flow = np.apply_along_axis(lambda row: float(share_flow(row)),arr=np.concatenate([shortfalls,demand[shortfall_idx,:]],axis=1),axis=1)
        #flow = np.minimum(np.maximum(r*obs[:,other] - (1-r)*obs[:,to_axis],-c),c)
      else:
        raise Exception("demand not supplied")

    total_flow = np.zeros((m,))
    total_flow[shortfall_idx] = flow
    return total_flow

  #@staticmethod
  def _epu_vector(self,obs,c=1000,policy="veto",axis=0,demand=None):
    """returns vector of EPU values for each observed power margin value

    **Parameters**:
    
    `obs` (`numpy.ndarray`): Matrix with observations of power margin values, with a column per area

    `c` (`int`): Interconnector capacity

    `policy` (`string`): either 'share' or 'veto'

    `axis` (`int`): area for which this will be calculated

    `demand` (`numpy.ndarray`): Matrix with observations of demand values, with a column per area

    """
    if policy == "share" and demand is None:
      raise Exception("share policy requires demand observations to be passed.")

    shortfalls = self.find_shortfalls(obs,axis,c,policy)

    obs_shortfall = obs[shortfalls,:]

    pu = obs_shortfall[:,axis] + self._power_flow(obs=obs_shortfall,c=c,policy=policy,demand=demand[shortfalls,:] if demand is not None else None,to_axis=axis)
    return pu

  #@staticmethod 
  def eeu(self,obs,season_length=3360,c=1000,policy="veto",axis=0,demand=None):
    """returns EEU estimate

    **Parameters**:
    
    `obs` (`numpy.ndarray`): Matrix with observations of power margin values, with a column per area

    `season_length` (`int`): Length of season under consideration. Defaults to length of single UK peak season in hours.

    `c` (`int`): Interconnector capacity

    `policy` (`string`): either 'share' or 'veto'

    `axis` (`int`): area for which this will be calculated

    `demand` (`numpy.ndarray`): Matrix with observations of demand values, with a column per area. Can be `None` if policy equals `veto`.

    """
    return - np.sum(self._epu_vector(obs=obs,c=c,policy=policy,axis=axis,demand=demand))/obs.shape[0] * season_length

  #@staticmethod
  def system_lole(self,obs,season_length=3360,c=1000,policy="veto"):
    """returns system-wide LOLE estimate, treating the system as a single area

    **Parameters**:
    
    `obs` (`numpy.ndarray`): Observation matrix with a column per area

    `season_length` (`int`): Length of season under consideration. Defaults to length of single UK peak season in hours.

    `c` (`int`): Interconnector capacity

    `policy` (`string`): either 'share' or 'veto'

    `axis` (`int`): area for which this will be calculated

    """

    shortfalls_a0 = self.find_shortfalls(obs,0,c,policy)
    shortfalls_a1 = self.find_shortfalls(obs,1,c,policy)

    return np.mean(np.logical_or(shortfalls_a0,shortfalls_a1)) * season_length

  #@staticmethod
  def system_eeu(self,obs,season_length=3360,c=1000,policy="veto",demand=None):
    """returns system-wide EEU estimate, treating the system as a single area

    **Parameters**:
    
    `obs` (`numpy.ndarray`): Observation matrix with a column per area

    `season_length` (`int`): Length of season under consideration. Defaults to length of single UK peak season in hours.

    `c` (`int`): Interconnector capacity

    `policy` (`string`): either 'share' or 'veto'

    `axis` (`int`): area for which this will be calculated

    """
    pu_a0 = - self._epu_vector(obs,c=c,policy=policy,axis=0,demand=demand)
    pu_a1 = - self._epu_vector(obs,c=c,policy=policy,axis=1,demand=demand)

    return (np.sum(pu_a0) + np.sum(pu_a1))/obs.shape[0] * season_length

  #@staticmethod

  def _get_post_itc(self,obs,c,policy,**kwargs):
    m,n = obs.shape
    if c > 0:
      flow_to_0 = self._power_flow(obs=obs,c=c,policy=policy,to_axis=0,**kwargs)
      obs = np.concatenate([(obs[:,0] + flow_to_0).reshape(m,1),(obs[:,1] - flow_to_0).reshape(m,1)],axis=1)

    return obs

  def cdf(self,x,obs,c=1000,policy="veto",**kwargs):
    """returns empirical CDF

    **Parameters**:
    
    `obs` (`numpy.ndarray`): Observation matrix with a column per area

    `c` (`int`): Interconnector capacity

    `policy` (`string`): either 'share' or 'veto'

    """
    obs = self._get_post_itc(obs=obs,c=c,policy=policy,**kwargs)
    
    return np.mean(np.logical_and(obs[:,0] <= x[0],obs[:,1] <= x[1]))

  def cvar(self,obs,bound,c=1000,policy="veto",metric="lole",axis=0):
    """Calculates conditional value at risk given that `obs < bound` for a vector of observations

    **Parameters**:

    `obs` (`numpy.ndarray`): Observation matrix with a column per area
    
    `bound` (`float`): Upper bound used for conditioning

    `c` (`int`): interconnection capacity

    `policy` (`str`): Either 'share' or 'veto'

    `metric` (`string` or function): Baseline risk metric that will be used to calculate EFC. If a `string`, use matching method from the appropriate `BivariateHindcastMargin` instance (for example, "`1lole`" or "`eeu`"); if a function, it needs to take as parameters a `BivariateHindcastMargin` instance `obj`, interconnection capacity `c`, axis `axis` and  policy `policy`, and optionally additional arguments. This is useful for using more complex metrics such as quantiles.

    `axis` (`int`): area for which risk will be calculated

    """

    obs = self._get_post_itc(obs,c,policy)
    return np.mean(obs[obs[:,axis]<bound,axis])

  #@staticmethod
  def itc_efc(self,obs,c=1000,policy="veto",metric="lole",axis=0,tol=0.1,**kwargs):
    """Returns equivalent firm capacity of interconnector in one area

    **Parameters**:

    `obs` (`numpy.ndarray`): Observation matrix with a column per area
    
    `c` (`int`): interconnection capacity

    `policy` (`str`): Either 'share' or 'veto'

    `metric` (`string` or function): Baseline risk metric that will be used to calculate EFC. If a `string`, use matching method from the appropriate `BivariateHindcastMargin` instance (for example, "`1lole`" or "`eeu`"); if a function, it needs to take as parameters a `BivariateHindcastMargin` instance `obj`, interconnection capacity `c`, axis `axis` and  policy `policy`, and optionally additional arguments. This is useful for using more complex metrics such as quantiles.

    `axis` (`int`): area for which risk will be calculated

    `tol` (`float`): absolute error tolerance from true EFC value

    `kwargs` : additional parameters to be passed to the risk metric function

    """
    obs = obs.astype(dtype=np.float64)
    with_itc = MonteCarloEstimator.get_efc_metric_function(metric,self,**kwargs)(obs=obs,c=c,policy=policy,axis=axis)
    #print("with_itc: {x}".format(x=with_itc))

    def find_efc(x):
      #print("x: {x}".format(x=x))
      obs[:,axis] += x
      without_itc = MonteCarloEstimator.get_efc_metric_function(metric,self,**kwargs)(obs=obs,c=0,policy=policy,axis=axis)
      obs[:,axis] -= x
      return with_itc - without_itc

    diff_to_null = find_efc(0)
    delta = max(500,c)
    if diff_to_null == 0: #itc is equivalent to null interconnection riskwise
      return 0
    else:
      # find suitalbe search intervals that are reasonably small
      if diff_to_null > 0: #interconnector adds risk => negative firm capacity
        rightmost = 0
        leftmost = delta
        while find_efc(leftmost) > 0 :
          leftmost -= delta
      else:
        leftmost = 0
        rightmost = delta
        while find_efc(rightmost) < 0:
          rightmost += delta
      #print("bisect  algorithm...")
      efc, res = bisect(f=find_efc,a=leftmost,b=rightmost,full_output=True,xtol=tol/2,rtol=tol/(2*with_itc))
      if not res.converged:
        print("Warning: EFC estimator did not converge.")
      #print("efc:{efc}".format(efc=efc))
      return int(efc)

  #@staticmethod
  def convgen_efc(self, obs, cap, prob, gen_axis, fc_axis, c=1000,policy="veto",metric="lole",axis=0,tol=0.1,**kwargs):
    """Returns the amount of firm capacity that needs to be added to area `fc_axis` such that area `axis` has the same risk (as defined by `metric`) than if new conventional generation was installed in `gen_axis`.

    **Parameters**:
    
    `obs` (`numpy.ndarray`): Observation matrix with a column per area

    `cap` (`int`): maximum available capacity of new generator in MW

    `prob` (`float`): availability probability for the new generator

    `gen_axis` (`int`): Axis to which the new generator will be added

    `fc_axis` (`int`): Area to which firm capacity will be added

    `c` (`int`): interconnection capacity

    `policy` (`str`): Either 'share' or 'veto'

    `axis` (`int`): area for which risk will be calculated

    `metric` (`string` or function): Baseline risk metric that will be used to calculate EFC. If a `string`, use matching method from the appropriate `BivariateHindcastMargin` instance (for example, "`1lole`" or "`eeu`"); if a function, it needs to take as parameters a `BivariateHindcastMargin` instance `obj`, interconnection capacity `c`, axis `axis` and  policy `policy`, and optionally additional arguments. This is useful for using more complex metrics such as quantiles.

    `tol` (`float`): absolute error tolerance from target risk value

    `kwargs` : additional parameters to be passed to the risk metric function

    """
    if not (axis in [0,1]):
      raise Exception("axis value must be 0 or 1")

    if not (gen_axis in [0,1]):
      raise Exception("gen_axis value must be 0 or 1")

    if not (fc_axis in [0,1]):
      raise Exception("fc_axis value must be 0 or 1")

    if prob > 1 or prob < 0:
      raise Exception("Availability value must be between 0 and 1")

    if cap < 0:
      raise Exception("Generation must be positive")

    obs = obs.astype(dtype=np.float64)
    # get risk when adding new generator
    new_gen_simulations = cap * np.random.binomial(n=1,p=prob,size=obs.shape[0])
    #pint(new_gen_simulations)
    obs[:,gen_axis] += new_gen_simulations
    with_gen = MonteCarloEstimator.get_efc_metric_function(metric,self,**kwargs)(obs = obs,c=c,policy=policy,axis=axis)
    #print("with_gen: {x}".format(x=with_gen))

    # get efc of new generator
    obs[:,gen_axis] -= new_gen_simulations 
    #print("without_gen: {x}".format(x=getattr(self,metric)(obs = obs,c=c,policy=policy,axis=axis,**kwargs)))

    def find_efc(x):
      obs[:,fc_axis] += x
      with_fc = MonteCarloEstimator.get_efc_metric_function(metric,self,**kwargs)(obs = obs,c=c,policy=policy,axis=axis)
      #with_fc = getattr(self,metric)(obs=obs,c=c,policy=policy,axis=axis,**kwargs)
      obs[:,fc_axis] -= x
      #print("x: {x}, with_fc: {with_fc}, with_gen: {with_gen}".format(x=x,with_fc=with_fc,with_gen=with_gen))
      return with_gen - with_fc

    efc, res = bisect(f=find_efc,a=0,b=cap,full_output=True,xtol=tol/2,rtol=tol/(2*with_gen))
    if not res.converged:
      print("Warning: EFC estimator did not converge.")
    #print("efc:{efc}".format(efc=efc))
    return int(efc)

  def renewables_efc(self,renewables, without_renewables,c=1000,policy="veto",metric="lole",axis=0,tol=0.1,**kwargs):
    """Returns the amount of firm capacity that needs to be added to replace installed renewable generation in terms of a risk metric

    **Parameters**:

    `renewables` (`numpy.ndarray`): Observation matrix for renewable generation with a column per area

    `without_renewables` (`numpy.ndarray`): Observation matrix for supply and demand other than renewables generation, with a column per area
    
    `c` (`int`): interconnection capacity

    `policy` (`str`): Either 'share' or 'veto'

    `metric` (`string` or function): Baseline risk metric that will be used to calculate EFC. If a `string`, use matching method from the appropriate `BivariateHindcastMargin` instance (for example, "`1lole`" or "`eeu`"); if a function, it needs to take as parameters a `BivariateHindcastMargin` instance `obj`, interconnection capacity `c`, axis `axis` and  policy `policy`, and optionally additional arguments. This is useful for using more complex metrics such as quantiles.

    `axis` (`int`): area for which risk will be calculated

    `tol` (`float`): absolute error tolerance from true EFC value

    `kwargs` : additional parameters to be passed to the risk metric function

    """

    with_rnw = MonteCarloEstimator.get_efc_metric_function(metric,self,**kwargs)(obs=without_renewables+renewables,c=c,policy=policy,axis=axis)
    #print("with_itc: {x}".format(x=with_itc))

    def find_efc(x):
      #print("x: {x}".format(x=x))
      without_renewables[:,axis] += x
      without_rnw = MonteCarloEstimator.get_efc_metric_function(metric,self,**kwargs)(obs=without_renewables,c=c,policy=policy,axis=axis)
      without_renewables[:,axis] -= x
      return with_rnw - without_rnw

    diff_to_null = find_efc(0)
    delta = max(c,500)
    if diff_to_null == 0: #renewables is equivalent to null interconnection riskwise
      return 0
    else:
      # find suitalbe search intervals that are reasonably small
      if diff_to_null > 0: #interconnector adds risk => negative firm capacity
        rightmost = 0
        leftmost = -delta
        while find_efc(leftmost) > 0 :
          leftmost -= delta
      else:
        leftmost = 0
        rightmost = delta
        while find_efc(rightmost) < 0:
          rightmost += delta
      #print("bisect  algorithm...")
      efc, res = bisect(f=find_efc,a=leftmost,b=rightmost,full_output=True,xtol=tol/2,rtol=tol/(2*with_itc))
      if not res.converged:
        print("Warning: EFC estimator did not converge.")
      #print("efc:{efc}".format(efc=efc))
      return int(efc)






