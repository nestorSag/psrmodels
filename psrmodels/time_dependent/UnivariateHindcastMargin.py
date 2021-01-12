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
  def __init__(self,demand,renewables,gen_dist,n_simulations=1000,seed=1):

    self.shortfall_history = None

    self.set_w_d(demand,renewables)

    self.gen = gen_dist

    self.n_sim = n_simulations

    self.seed = seed


  def set_w_d(self,demand,renewables):

    """Set new demand and renewable generation matrices
  
      **Parameters**:

      `demand` (`numpy.array`): Matrix of demands with one column per area 

      `renewables` (`numpy.array`): Matrix of renewable generation with one column per area 

    """

    # to avoid throwing away samples, merge new net demand data in old samples to update them
    if self.shortfall_history is not None and demand.reshape((-1,1)).shape[0] == self.demand.reshape((-1,1)).shape[0]:
      colnames = self.shortfall_history.columns
      new_data_df = pd.DataFrame({"new_net_demand":(demand-renewables).reshape(-1),"season_time":np.arange(len(demand))})
      old_data_df = pd.DataFrame({"old_net_demand":(self.demand-self.renewables).reshape(-1),"season_time":np.arange(len(self.demand))})
      joint_shortfalls = self.shortfall_history.merge(new_data_df,on="season_time").merge(old_data_df,on="season_time")
      delta = joint_shortfalls["old_net_demand"] - joint_shortfalls["new_net_demand"]
      if np.all(delta >= 0):
        # 
        joint_shortfalls["m"] = joint_shortfalls["m"] + delta
        joint_shortfalls.dropna(axis=0,inplace=False)
        self.shortfall_history = joint_shortfalls[colnames]
      else:
        # if we don't drop historic samples now, there will be a gap in the EU distribution that will be underrepresented
        # so results will be wrong
        print("starting new shortfall history dataset")
        self.shortfall_history = None

    self.net_demand = np.ascontiguousarray((demand - renewables),dtype=np.float32) #no negative net demand
    self.renewables = renewables
    self.demand = np.ascontiguousarray(demand).astype(np.float32)#.clip(min=0)
    self.season_length = self.net_demand.shape[0]
    # mark buffered data as stale

  def _get_gen_simulation(self,**kwargs):

    output = np.ascontiguousarray(self.gen.simulate(n_sim=self.n_sim,n_timesteps=self.season_length-1,seed=self.seed,use_buffer=True))

    return output

  def simulate(self):
    """Simulate pre-interconnection power margins
  
      **Parameters**:
        
      **Returns**:

      numpy.array of simulated values
    """
    generation = self._get_gen_simulation()

    # overwrite gensim array with margin values

    C_CALL.calculate_pre_itc_margins_py_interface(
        ffi.cast("float *", generation.ctypes.data),
        ffi.cast("float *",self.net_demand.ctypes.data),
        np.int32(self.season_length),
        np.int32(generation.shape[0]),
        np.int32(1))

    return generation
    #### get simulated outages from n_sim*self.n hours

  def get_shortfalls(
    self,
    raw=True,
    cold_start=False):

    """Run simulation and return only shortfall events
  
      **Parameters**:

      `raw` (`boolean`): whether to get raw or processed data. See below

      `cold_start` (`boolean`): Bypass previously computed samples and simulate new ones
      
      **Returns**:

      if raw is False, pandas DataFrame object including columns:

        `margin`: shortfall size

        `shortfall_event_id`: identifier that groups consecutive hourly shortfalls. This allows to measure shortfall durations

        `period_simulation_time`: time id with respect to simulated time length. This allows to find common shortfalls across areas.

      If raw is True pandas DataFrame object including columns:

        `m`: margin value of area i

        `season_time`: time of ocurrence in peak season

        `simulation_time`: time of occurrence in simulation

    """

    # buffer simulated shortfall events for performance
    if self.shortfall_history is not None and not cold_start: 
      df = self.shortfall_history.copy()
      # only take those samples that were taken with a larger FC than current FC
      # otherwise we risk over representing extreme EU observations under current FC
      df = df[df["fc"]<=self.gen.fc]

      #update shortfall samples to current FC value
      df["m"] += (-df["fc"] + self.gen.fc)
      df = df[df["m"] < 0] #return only shortfalls under the new firm capacity value
      if df.shape[0] == 0:
        print("no suitable shortfall data found in history; computing new batch")
        df = self.get_shortfalls(raw=raw,cold_start=True)
    else:
      #print("simulating shortfalls...")
      sampled = self.simulate_shortfalls()
      df = self._process_shortfall_batch(sampled)

    if not raw:
      df = self._group_shortfalls(df)

    return df

  def simulate_shortfalls(self,max_tries = 5):
    """Returns data frame of simualted shortfall events
  
      **Parameters**:

      `max_tries` (`int`): number of times the model should simulate additional samples if not a single shortfall event is found (in case of extremely low shortfall probabilities and/or low n_simulations parameter)
        
      **Returns**:

      numpy.array of simulated values
    """
    original_seed = self.seed
    tries = 0
    n_shortfalls = 0
    while n_shortfalls == 0 and tries < max_tries:
      if tries > 0:
        print("No shortfalls found; retrying.")
        self.seed += 1
      df = self.simulate()
      #print(df)
      m,n = df.shape
      #add global time index with respect to simulations
      df = np.concatenate((df,np.arange(m).reshape(m,1)),axis=1)
      df = df[np.any(df<0,axis=1),:]
      #print(df)
      n_shortfalls = df.shape[0]
      tries += 1
    if n_shortfalls == 0:
      raise Exception("No shortfalls were found in {k} runs of {n} simulated seasons with the current data; shortfall probability of n_simulations parameter might be too low".format(k=max_tries,n=self.n_sim))

    df = pd.DataFrame(df)
    # name columns 
    df.columns = ["m"] + ["simulation_time"]

    return df

  def _process_shortfall_batch(self,sampled):
    
    # add metadata
    sampled["fc"] = self.gen.fc
    if self.shortfall_history is not None:
      sampled["batch"] = len(np.unique(self.shortfall_history["batch"])) + 1
    else:
      sampled["batch"] = 1
    # also need time index with respect to simulated periods
    sampled["season_time"] = (sampled["simulation_time"]%self.season_length).astype(np.int32)

    previously_computed_seasons = 0 if self.shortfall_history is None else self.n_sim*len(np.unique(self.shortfall_history["batch"]))
    sampled["season"] = (sampled["simulation_time"]/self.season_length).astype(np.int32) + previously_computed_seasons

    # store new data in buffer or create it if it doesn't exist
    if self.shortfall_history is not None:
      self.shortfall_history = pd.concat([self.shortfall_history,sampled])
    else:
      self.shortfall_history = sampled

    return sampled

    # if raw:
    #   #self.shortfall_historys = sampled
    #   #self.shortfall_historys_params = {"stf_bound":stf_bound,"fc":self.gen.fc}
    #   return sampled
    # else:
    #   formated_df = self._group_shortfalls(sampled)

    #   return formated_df

  def _group_shortfalls(self,df):
    # I call a shortfall cluster a shortfall of potentially multiple consecutive hours
    current_margin = "m"

    # the first shortfall in a cluster must have at least 1 timestep between it and the previous shortfall
    cond1 = np.array((df["season_time"] - df["season_time"].shift()).fillna(value=2) != 1)

    # shortfalls that are not the first of their clusters must be in the same peak season simulation
    # than the previous shortfall in the same cluster

    #first check if previous shortfall is in same peak season simulation
    peak_season_idx = (np.array(df["simulation_time"])/self.season_length).astype(np.int64)
    lagged_peak_season_idx =  (np.array(df["simulation_time"].shift().fillna(value=2))/self.season_length).astype(np.int64)
    same_peak_season = peak_season_idx != lagged_peak_season_idx
    # check that peak season is the same but only for shortfalls that are not first in their cluster
    cond2 = np.logical_not(cond1)*same_peak_season

    df["shortfall_event_id"] = np.cumsum(cond1 + cond2)

    formatted_df = df[[current_margin,"shortfall_event_id","simulation_time"]].rename(columns={current_margin:"margin"})
    return formatted_df

  def simulate_eu(
    self,
    min_samples = 100):

    """Simulate season-agreggated energy unserved. Seasons without shortfall events below said level are ignored
  
      **Parameters**:

      `min_samples` (`int`): minimum acceptable number of season-aggregated energy unserved events to compute the estimate; if needed, the model samples more data.

    """

    df = self.get_shortfalls(raw=True)
    n_distinct_seasons = len(np.unique(df["season"]))

    if min_samples > self.n_sim:
      print("min_samples larger than number of simulated seasons; ignoring.")
    elif n_distinct_seasons < min_samples:
      original_seed = self.seed
      while n_distinct_seasons < min_samples:
        print("Insuficient samples ({n}); simulating {k} additional seasons to get at least {ms} points".format(k=self.n_sim,ms=min_samples,n=n_distinct_seasons))
        self.seed += 1
        new_df = self.get_shortfalls(raw=True,cold_start=True)
        #new_df["season"] = (new_df["simulation_time"]/self.season_length).astype(np.int32) + self.season_length*counter
        df = pd.concat([df,new_df])

        n_distinct_seasons = len(np.unique(df["season"]))

      #self.shortfall_history = df
      self.seed = original_seed

    df = df.groupby(by="season").agg({"m":"sum"})
    return - np.array(df["m"])

  def lole(self):
    """calculates Monte Carlo estimate of LOLE

    **Parameters**:
    
    `**kwargs` : Additional parameters to be passed to `get_shortfalls`

    `min_event_samples` : resample until getting at least this much valid shortfalls before computing estimate

    """
    df = self.get_shortfalls(raw=True)
    total_effective_seasons = self.n_sim*len(np.unique(df["batch"]))
    return df.shape[0]/total_effective_seasons

  def eeu(self,min_samples=100):

    """calculates Monte Carlo estimate of EEU

    **Parameters**:
    
    `min_samples` (`int`): minimum acceptable number of season-aggregated energy unserved events to compute the estimate; if needed, the model samples more data.

    """
    eu_samples = self.simulate_eu(min_samples)
    df = self.get_shortfalls(raw=True)
    total_effective_seasons = self.n_sim*len(np.unique(df["batch"]))
    return np.sum(eu_samples)/total_effective_seasons

  def cvar(self,bound,min_samples=100):

    """calculate conditional value at risk for the energy unserved distribution conditioned to being non-zero

    **Parameters**:
    
    `bound` (`float`): absolute value of power margin shortfall's lower bound

    """
    if bound < 0:
      raise Error("bound has to be a non-negative number")
    sample = self.simulate_eu(min_samples)
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

      self.set_w_d(demand,np.zeros(renewables.shape))
      def bisection_target(x):
        self.gen += x
        #with_fc_obj = UnivariateHindcastMargin(self.gen,demand,0*renewables)
        with_fc_risk = get_risk_function(metric)(self)
        self.gen += (-x)
        #print("fc: {x}, with_fc_risk:{wfcr}, with_wind_risk: {wwr}".format(x=x,wfcr=with_fc_risk,wwr=with_wind_risk))
        y = (with_fc_risk - with_wind_risk)/with_wind_risk
        #print("fc: {x}, risk: {y}".format(x=x,y=y))
        return y

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






        
