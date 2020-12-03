import numpy as np
import pandas as pd
from _c_ext_timedependence import ffi, lib as C_CALL

class UnivariateHindcastMargin(object):

  """Univariate hindcast time-dependent margin simulator
  
    **Parameters**:

    `demand` (`numpy.array`): Matrix of demands with one column per area 

    `renewables` (`numpy.array`): Matrix of renewable generation with one column per area 

    `gen_dists` (`list`): List of `time_dependent.ConvGenDistribution` objects corresponding to the areas

  """
  def __init__(self,demand,renewables,gen_dist):

    self.set_w_d(demand,renewables)

    self.gen_dist = gen_dist

  def set_w_d(self,demand,renewables):

    """Set new demand and renewable generation matrices
  
      **Parameters**:

      `demand` (`numpy.array`): Matrix of demands with one column per area 

      `renewables` (`numpy.array`): Matrix of renewable generation with one column per area 

    """
    self.net_demand = np.ascontiguousarray((demand - renewables).clip(min=0),dtype=np.float64) #no negative net demand
    self.wind = renewables
    self.demand = np.ascontiguousarray(demand).clip(min=0).astype(np.float64)
    self.n = self.net_demand.shape[0]

  def _get_gen_simulation(self,n_sim,seed,use_saved,**kwargs):

    output = np.ascontiguousarray(self.gen_dist.simulate(n_sim=n_sim,n_timesteps=self.n-1,seed=seed,use_buffer=use_saved))

    return output

  def simulate(self,n_sim,seed=1,use_saved=True,**kwargs):
    """Simulate pre-interconnection power margins
  
      **Parameters**:

      `n_sim` (`int`): number of peak seasons to simulate

      `seed` (`int`): random seed

      `use_saved` (`bool`): use saved state of simulated generation values. Save them if they don't exist yet.
        
      **Returns**:

      numpy.array of simulated values
    """
    generation = self._get_gen_simulation(n_sim,seed,use_saved)

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
    n_sim,
    seed=1,
    use_saved=True,
    raw=False):

    """Run simulation and return only shortfall events
  
      **Parameters**:

      `n_sim` (`int`): number of peak seasons to simulate

      `seed` (`int`): random seed

      `use_saved` (`boolean`): if `True`, use saved simulation samples from previous runs

      `raw` (`boolean`): whether to get raw or processed data. See below
      
      **Returns**:

      if raw is False, pandas DataFrame object with columns:

        `margin`: shortfall size

        `area`: area in which the shortfall occurred

        `shortfall_event_id`: identifier that groups consecutive hourly shortfalls. This allows to measure shortfall durations

        `period_time_id`: time id with respect to simulated time length. This allows to find common shortfalls across areas.

      If raw is True pandas DataFrame object with columns:

        `m<i>`: margin value of area i

        `time_cyclical`: time of ocurrence in peak season

        `time_id`: time of occurrence in simulation

    """
    sampled = self.simulate(n_sim,seed)
    df = self._process_shortfall_data(sampled,raw)
    return df

  def _process_shortfall_data(self,sampled,raw=False):
    
    m,n = sampled.shape
    #add global time index with respect to simulations
    sampled = np.concatenate((sampled,np.arange(m).reshape(m,1)),axis=1)
    #filter non-shortfalls
    sampled = sampled[np.any(sampled<0,axis=1),:]
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
    n_sim,
    seed=1,
    use_saved=True):

    """Simulate aggregated energy unserved events per peak season. Peak season without shortfall events are ignored
  
      **Parameters**:

      `n_sim` (`int`): number of peak seasons to simulate

      `seed` (`int`): random seed

      `use_saved` (`boolean`): if `True`, use saved simulation samples from previous runs

    """

    df = self.simulate_shortfalls(n_sim,seed,use_saved,True)
    df["season"] = (df["time_id"]/self.n).astype(np.int32)
    df.groupby(by="season").agg({"m":"sum"})
    return - np.array(df.groupby(by="season").agg({"m":"sum"})["m"])
      
