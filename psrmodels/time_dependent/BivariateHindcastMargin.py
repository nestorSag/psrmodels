import numpy as np
import pandas as pd
from _c_ext_timedependence import ffi, lib as C_CALL

class BivariateHindcastMargin(object):

  """Bivariate hindcast time-dependent margin simulator
  
    **Parameters**:

    `demand` (`numpy.array`): Matrix of demands with one column per area 

    `renewables` (`numpy.array`): Matrix of renewable generation with one column per area 

    `gen_dists` (`list`): List of `time_dependent.ConvGenDistribution` objects corresponding to the areas

    `n_sim` (`int`): number of peak seasons to simulate

  """
  def __init__(self,demand,renewables,gen_dists,n_simulations=1000):

    self.set_w_d(demand,renewables)

    self.gen = gen_dists

    self.n_sim = n_simulations

    #self.gensim = None
    #self.gensim_nsim = None

  def set_w_d(self,demand,renewables):

    """Set new demand and renewable generation matrices
  
      **Parameters**:

      `demand` (`numpy.array`): Matrix of demands with one column per area 

      `renewables` (`numpy.array`): Matrix of renewable generation with one column per area 

    """
    if demand.shape[1] != 2 or renewables.shape[1] != 2:
      raise Exception("data matrices must have exactly 2 columns")
    self.net_demand = np.ascontiguousarray((demand - renewables),dtype=np.float64) #no negative net demand
    self.renewables = renewables
    self.demand = np.ascontiguousarray(demand).clip(min=0).astype(np.float64)
    self.n = self.net_demand.shape[0]

  def _get_gen_simulation(self,seed,save,**kwargs):

    gensim = np.ascontiguousarray(
      np.concatenate(
        [self.gen[i].simulate(n_sim=self.n_sim,n_timesteps=self.n-1,seed=seed+i,use_buffer=save) for i in range(len(self.gen))],
        axis=1
        )
      )

    return gensim

  def simulate_pre_itc(self,seed=1,use_saved=True,**kwargs):
    """Simulate pre-interconnection power margins
  
      **Parameters**:

      `seed` (`int`): random seed

      `use_saved` (`bool`): use saved state of simulated generation values. Save them if they don't exist yet.
        
      **Returns**:

      numpy.array of simulated values
    """
    # if use_saved:
    #   if self.gensim is None:
    #     # create state and make a copy
    #     self.gensim = self._get_gen_simulation(n_sim,seed,True,**kwargs)
    #     gensim = self.gensim.copy()
    #   else:
    #     if self.gensim_nsim < n_sim:
    #       # run only the necessary simulations and append, then create a copy
    #       new_sim = self._get_gen_simulation(n_sim-self.gensim_nsim,seed,True,**kwargs)
    #       self.gensim = np.ascontiguousarray(np.concatenate((self.gensim,new_sim),axis=0)).astype(np.float64)
    #       self.gensim_nsim = n_sim
    #       gensim = self.gensim.copy()
    #     else:
    #       #create a copy of a slice, since there are more simulations than necessary
    #       gensim = self.gensim[0:(self.n*n_sim),:].copy()  
    # else:
    #   gensim = self._get_gen_simulation(n_sim,seed,False,**kwargs)

    gensim = self._get_gen_simulation(seed,use_saved,**kwargs)

    # overwrite gensim array with margin values

    C_CALL.calculate_pre_itc_margins_py_interface(
        ffi.cast("double *", gensim.ctypes.data),
        ffi.cast("double *",self.net_demand.ctypes.data),
        np.int32(self.n),
        np.int32(gensim.shape[0]),
        np.int32(self.net_demand.shape[1]))

    return gensim
    #### get simulated outages from n_sim*self.n hours

  def simulate_post_itc(self,c,policy,seed=1,use_saved=True,**kwargs):
    """Simulate post-interconnection power margins
  
      **Parameters**:

      `c` (`int`): interconnection capacity

      `policy` (`string`): shortfall sharing policy. Only 'veto' (no shortfall sharing) and 'share' (demand-proportional shortfall sharing) are supported

      `seed` (`int`): random seed

      `use_saved` (`bool`): use saved state of simulated generation values. Save them if they don't exist yet.
      
      **Returns**:

      numpy.array of simulated values
    """
    if(self.net_demand.shape[1] != 2):
      raise Exception("Data needs to have exactly 2 columns")

    pre_itc = self.simulate_pre_itc(seed,use_saved,**kwargs)

    #override pre_itc array with margin values
    if policy == "veto":
      C_CALL.calculate_post_itc_veto_margins_py_interface(
          ffi.cast("double *", pre_itc.ctypes.data),
          np.int64(pre_itc.shape[0]),
          np.int32(self.net_demand.shape[1]),
          np.float64(c))

    elif policy == "share":

      C_CALL.calculate_post_itc_share_margins_py_interface(
          ffi.cast("double *", pre_itc.ctypes.data),
          ffi.cast("double *", self.demand.ctypes.data),
          np.int64(self.n),
          np.int64(pre_itc.shape[0]),
          np.int32(self.net_demand.shape[1]),
          np.float64(c))

    else:

      raise Exception("Policy not recognised")

    return pre_itc

  def simulate_shortfalls(
    self,
    c=1000,
    policy="veto",
    seed=1,
    raw=False,
    stf_bound=0,
    **kwargs):

    """Run simulation and return only shortfall events
  
      **Parameters**:

      `c` (`int`): interconnection capacity

      `policy` (`string`): shortfall sharing policy. Only 'veto' (no shortfall sharing) and 'share' (demand-proportional shortfall sharing) are supported

      `seed` (`int`): random seed

      `use_saved` (`bool`): use saved state of simulated generation values. Save them if they don't exist yet.
      
      `raw` (`bool`): return dataframe with margin values in all areas, when at least one of them have a shortfall.

      `stf_bound` (`float`): bound below which is considered a shortfall; defaults to 0

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
    post_itc = self.simulate_post_itc(c=c,policy=policy,seed=seed,use_saved=True,**kwargs)
    df = self._process_shortfall_data(post_itc,raw=raw,stf_bound=stf_bound)
    return df

  def _process_shortfall_data(self,post_itc,raw=False,stf_bound=0):
    
    m,n = post_itc.shape
    #add time index with respect to simulations
    post_itc = np.concatenate((post_itc,np.arange(m).reshape(m,1)),axis=1)
    #filter non-shortfalls
    post_itc = post_itc[np.any(post_itc<stf_bound,axis=1),:]
    post_itc = pd.DataFrame(post_itc)
    # name columns 
    post_itc.columns = ["m" + str(i) for i in range(n)] + ["time_id"]
    
    # also need time index with respect to simulated periods
    post_itc["time_cyclical"] = post_itc["time_id"]%self.n

    #reformat simulations in tidy column format
    if raw:
      return post_itc
    else:
      formated_dfs = []
      for area_id in range(n):
        formated_dfs.append(self._get_shortfall_clusters(post_itc,area_id))

      return pd.concat(formated_dfs,axis=0)

  def _get_shortfall_clusters(self,df,area_id):
    # I call a shortfall cluster a shortfall of potentially multiple consecutive hours
    current_margin = "m" + str(area_id)
    #find first shortfall time for each shortfall event
    area_df = df[df[current_margin] < 0]
    # characterize consecutive shortfall clusters events 

    # the first shortfall in a cluster must have at least 1 timestep between it and the previous shortfall
    cond1 = np.array((area_df["time_cyclical"] - area_df["time_cyclical"].shift()).fillna(value=2) != 1)

    # shortfalls that are not the first of their clusters must be in the same peak season simulation
    # than the previous shortfall in the same cluster

    #first check if previous shortfall is in same peak season simulation
    peak_season_idx = (np.array(area_df["time_id"])/self.n).astype(np.int64)
    lagged_peak_season_idx =  (np.array(area_df["time_id"].shift().fillna(value=2))/self.n).astype(np.int64)
    same_peak_season = peak_season_idx != lagged_peak_season_idx
    # check that peak season is the same but only for shortfalls that are not first in their cluster
    cond2 = np.logical_not(cond1)*same_peak_season

    area_df["shortfall_event_id"] = np.cumsum(cond1 + cond2)

    formatted_df = area_df[[current_margin,"shortfall_event_id","time_id"]].rename(columns={current_margin:"margin"})
    formatted_df["area"] = area_id
    return formatted_df[["margin","area","shortfall_event_id","time_id"]] 

  def simulate_eu(
    self,
    c=1000,
    policy="veto",
    axis=0,
    seed=1):

    """Simulate season-agreggated energy unserved. Seasons without shortfall events below said level are ignored
  
      **Parameters**:

      `c` (`int`): interconnection capacity

      `policy` (`string`): shortfall sharing policy. Only 'veto' (no shortfall sharing) and 'share' (demand-proportional shortfall sharing) are supported

      `axis` (`int`): area for which this will be simulated

      `seed` (`int`): random seed

      `use_saved` (`boolean`): if `True`, use saved simulation samples from previous runs

    """

    df = self.simulate_shortfalls(c=c,policy=policy,seed=seed,raw=True,stf_bound=0)
    df["season"] = (df["time_id"]/self.n).astype(np.int32)
    df = df.groupby(by="season").agg({("m" + str(axis)):"sum"})
    return - np.array(df[("m" + str(axis))])

  def lole(self, c=1000, policy="veto", axis=0, **kwargs):
    """calculates Monte Carlo estimate of LOLE

    **Parameters**:
    
    `c` (`int`): interconnection capacity

    `policy` (`string`): shortfall sharing policy. Only 'veto' (no shortfall sharing) and 'share' (demand-proportional shortfall sharing) are supported

    `axis` (`int`): area for which this will be simulated

    `**kwargs` : Additional parameters to be passed to `simulate_shortfalls`

    """
    samples = self.simulate_shortfalls(c=c,policy=policy,raw=True,**kwargs)[("m" + str(axis))]
    samples = samples[samples < 0]

    return samples.shape[0]/self.n_sim

  def eeu(self, c=1000, policy="veto", axis=0, **kwargs):

    """calculates Monte Carlo estimate of EEU

    **Parameters**:
    
    `c` (`int`): interconnection capacity

    `policy` (`string`): shortfall sharing policy. Only 'veto' (no shortfall sharing) and 'share' (demand-proportional shortfall sharing) are supported

    `axis` (`int`): area for which this will be simulated

    `**kwargs` : Additional parameters to be passed to `simulate_shortfalls`

    """
    return np.sum(self.simulate_eu(c=c,policy=policy,axis=axis,**kwargs))/self.n_sim

  def cvar(self,bound, c=1000, policy="veto", axis=0,**kwargs):

    """calculate conditional value at risk for the energy unserved distribution conditioned to being non-zero

    **Parameters**:
    
    `bound` (`float`): absolute value of power margin shortfall's lower bound

    `c` (`int`): interconnection capacity

    `policy` (`string`): shortfall sharing policy. Only 'veto' (no shortfall sharing) and 'share' (demand-proportional shortfall sharing) are supported

    `axis` (`int`): area for which this will be simulated

    """
    if bound < 0:
      raise Error("bound has to be a non-negative number")
    sample = self.simulate_eu(c=c,policy=policy,axis=axis,**kwargs)
    sample = sample[sample > bound]
    return np.mean(sample)


      
