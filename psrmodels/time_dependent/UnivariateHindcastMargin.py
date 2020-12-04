import numpy as np
import pandas as pd
from _c_ext_timedependence import ffi, lib as C_CALL

from scipy.optimize import bisect

class UnivariateHindcastMargin(object):

  """Univariate hindcast time-dependent margin simulator
  
    **Parameters**:

    `demand` (`numpy.array`): Matrix of demands with one column per area 

    `renewables` (`numpy.array`): Matrix of renewable generation with one column per area 

    `gen_dists` (`list`): List of `time_dependent.ConvGenDistribution` objects corresponding to the areas

    `n_simulations` (`int`): number of peak seasons to simulate

  """
  def __init__(self,demand,renewables,gen_dist,n_simulations=1000):

    self.set_w_d(demand,renewables)

    self.gen = gen_dist

    self.n_sim = n_simulations

  def set_w_d(self,demand,renewables):

    """Set new demand and renewable generation matrices
  
      **Parameters**:

      `demand` (`numpy.array`): Matrix of demands with one column per area 

      `renewables` (`numpy.array`): Matrix of renewable generation with one column per area 

    """
    self.net_demand = np.ascontiguousarray((demand - renewables).clip(min=0),dtype=np.float64) #no negative net demand
    self.renewables = renewables
    self.demand = np.ascontiguousarray(demand).clip(min=0).astype(np.float64)
    self.n = self.net_demand.shape[0]

  def _get_gen_simulation(self,seed,use_saved,**kwargs):

    output = np.ascontiguousarray(self.gen.simulate(n_sim=self.n_sim,n_timesteps=self.n-1,seed=seed,use_buffer=use_saved))

    return output

  def simulate(self,seed=1,use_saved=True,**kwargs):
    """Simulate pre-interconnection power margins
  
      **Parameters**:

      `seed` (`int`): random seed

      `use_saved` (`bool`): use saved state of simulated generation values. Save them if they don't exist yet.
        
      **Returns**:

      numpy.array of simulated values
    """
    generation = self._get_gen_simulation(seed,use_saved)

    # overwrite gensim array with margin values

    C_CALL.calculate_pre_itc_margins_py_interface(
        ffi.cast("double *", generation.ctypes.data),
        ffi.cast("double *",self.net_demand.ctypes.data),
        np.int32(self.n),
        np.int32(generation.shape[0]),
        np.int32(1))

    return generation
    #### get simulated outages from n_sim*self.n hours

  def simulate_shortfalls(
    self,
    seed=1,
    use_saved=True,
    raw=False,
    stf_bound=0):

    """Run simulation and return only shortfall events
  
      **Parameters**:

      `seed` (`int`): random seed

      `use_saved` (`boolean`): if `True`, use saved simulation samples from previous runs

      `raw` (`boolean`): whether to get raw or processed data. See below

      `stf_bound` (`float`): bound below which is considered a shortfall; defaults to 0
      
      **Returns**:

      if raw is False, pandas DataFrame object with columns:

        `margin`: shortfall size

        `shortfall_event_id`: identifier that groups consecutive hourly shortfalls. This allows to measure shortfall durations

        `period_time_id`: time id with respect to simulated time length. This allows to find common shortfalls across areas.

      If raw is True pandas DataFrame object with columns:

        `m<i>`: margin value of area i

        `time_cyclical`: time of ocurrence in peak season

        `time_id`: time of occurrence in simulation

    """
    sampled = self.simulate(seed,use_saved)
    df = self._process_shortfall_data(sampled,raw,stf_bound)
    return df

  def _process_shortfall_data(self,sampled,raw=False,stf_bound=0):
    
    m,n = sampled.shape
    #add global time index with respect to simulations
    sampled = np.concatenate((sampled,np.arange(m).reshape(m,1)),axis=1)
    #filter non-shortfalls
    sampled = sampled[np.any(sampled<stf_bound,axis=1),:]
    sampled = pd.DataFrame(sampled)
    # name columns 
    sampled.columns = ["m"] + ["time_id"]
    
    # also need time index with respect to simulated periods
    sampled["time_cyclical"] = sampled["time_id"]%self.n

    #reformat simulations in tidy column format
    if raw:
      return sampled
    else:
      formated_df = self._get_shortfall_clusters(sampled)

      return formated_df

  def _get_shortfall_clusters(self,df):
    # I call a shortfall cluster a shortfall of potentially multiple consecutive hours
    current_margin = "m"

    # the first shortfall in a cluster must have at least 1 timestep between it and the previous shortfall
    cond1 = np.array((df["time_cyclical"] - df["time_cyclical"].shift()).fillna(value=2) != 1)

    # shortfalls that are not the first of their clusters must be in the same peak season simulation
    # than the previous shortfall in the same cluster

    #first check if previous shortfall is in same peak season simulation
    peak_season_idx = (np.array(df["time_id"])/self.n).astype(np.int64)
    lagged_peak_season_idx =  (np.array(df["time_id"].shift().fillna(value=2))/self.n).astype(np.int64)
    same_peak_season = peak_season_idx != lagged_peak_season_idx
    # check that peak season is the same but only for shortfalls that are not first in their cluster
    cond2 = np.logical_not(cond1)*same_peak_season

    df["shortfall_event_id"] = np.cumsum(cond1 + cond2)

    formatted_df = df[[current_margin,"shortfall_event_id","time_id"]].rename(columns={current_margin:"margin"})
    return formatted_df[["margin","shortfall_event_id","time_id"]] 

  def simulate_eu(
    self,
    seed=1,
    use_saved=True):

    """Simulate season-agreggated energy unserved. Seasons without shortfall events below said level are ignored
  
      **Parameters**:

      `seed` (`int`): random seed

      `use_saved` (`boolean`): if `True`, use saved simulation samples from previous runs

    """

    df = self.simulate_shortfalls(seed,use_saved,True,0)
    df["season"] = (df["time_id"]/self.n).astype(np.int32)
    df.groupby(by="season").agg({"m":"sum"})
    return - np.array(df["m"])

  def lole(self, **kwargs):
    """calculates Monte Carlo estimate of LOLE

    **Parameters**:
    
    `**kwargs` : Additional parameters to be passed to `simulate_shortfalls`

    """
    return self.simulate_shortfalls(**kwargs).shape[0]/self.n_sim

  def eeu(self,**kwargs):

    """calculates Monte Carlo estimate of EEU

    **Parameters**:
    
    `**kwargs` : Additional parameters to be passed to `simulate_shortfalls`

    """
    return np.sum(self.simulate_eu(**kwargs))/self.n_sim

  def cvar(self,bound,**kwargs):

    """calculate conditional value at risk for the energy unserved distribution conditioned to being non-zero

    **Parameters**:
    
    `bound` (`float`): absolute value of power margin shortfall's lower bound

    """
    if bound < 0:
      raise Error("bound has to be a non-negative number")
    sample = self.simulate_eu(**kwargs)
    sample = sample[sample > bound]
    return np.mean(sample)

  def renewables_efc(self,demand,renewables,metric="lole",tol=0.01):
      """calculate efc of wind fleer

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

      original_demand = self.demand
      original_renewables = self.renewables
      self.set_w_d(demand,renewables)
      #with_wind_obj = UnivariateHindcastMargin(self.gen,demand,renewables)
      with_wind_risk = get_risk_function(metric)(self)

      def bisection_target(x):
        self.gen += x
        #with_fc_obj = UnivariateHindcastMargin(self.gen,demand,0*renewables)
        self.set_w_d(demand,np.zeros(renewables.shape))
        with_fc_risk = get_risk_function(metric)(self)
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

      ## set original demand and wind before returinin
      self.set_w_d(original_demand,original_renewables)
      return efc






        
