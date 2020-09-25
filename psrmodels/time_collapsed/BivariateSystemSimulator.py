import numpy as np
import pandas as pd
import functools

from .BivariateConvGenDist import * 

class BivariateSystemSimulator(object):
  """This class outputs risk metrics MC estimates from a general net demand simulation model, for a 2-area system unver a veto policy
  
  **Parameters**:

  `gen_dists` (`list`): list of available conventional generation distributions

  `net_demand_dist` (`object`): object from which  the system's net demand is simualated, e.g. HindcastModel, BivariateLogisticNetDemand. It has to have a simulate() method

  """

  def __init__(self,gen_dists,net_demand_dist):

    self.gens = gen_dists

    self.net_demand = net_demand_dist

    self.d = len(self.gens)

  def _pu(self,m):
    m[m>=0] = 0
    return -m
  
  def _veto_flow(self,m,c):
    """calculate available flow in a veto policy for given margins and interconnector capacity

    **Parameters**:

    `m` (`numpy.ndarray`): margin values

    `c` (`int`): interconnector capacity

    """
    i = 0
    j = 1

    delta = np.zeros((self.d,))
    
    # calculate flow to i
    if m[i] < 0 and m[j] > 0:
      flow = np.min(np.array([-m[i],m[j],c]))
    elif m[i] > 0 and m[j] < 0:
      flow = -np.min(np.array([m[i],-m[j],c]))
    else:
      flow = 0

    delta[i] = flow
    delta[j] = -flow
    
    return m + delta

  def _simulate(self,n,seed=1,**kwargs):

    nd = self.net_demand.simulate(n=n,seed=seed,**kwargs)

    bigen = BivariateConvGenDist(self.gens[0],self.gens[1])
    gen = bigen.simulate(n=n,seed=seed)

    return gen, nd

  def simulate(self,n,seed=1,**kwargs):
    """Simulate pre-interconnector margins 
  
    **Parameters**:

    `n` (`int`): number of samples

    `seed` (`int`): random seed

    Additional arguments are passed directly to the simulate() method from the net demand model class, and the arguments are also stringified and added as an string column in the results data frame

    """

    gen, nd = self._simulate(n,seed,**kwargs)

    return gen - nd

  def simulate_veto(self,n,c,seed=1,**kwargs):

    return np.apply_along_axis(lambda m: self._veto_flow(m,c),1,self.simulate(n,seed,**kwargs))


  def veto_risk(self,n,c,seed,period_length=3360,**kwargs):

    """Simulate veto policy and returns MC estimates of LOLE and EPU for different system areas wrapped in a SimulationResults object
  
    **Parameters**:

    `n` (`int`): number of samples

    `c` (`int`): interconnector capacity

    `seed` (`int`): random seed

    `period_length` (`int`): number of time steps in period of interests. Default to 3,360: the number of hours in a peak season in GB

    Additional arguments are passed directly to the simulate() method from the net demand model class, and the arguments are also stringified and added as an string column in the results data frame

    """
    
    gen, nd = self._simulate(n,seed,**kwargs)

    margins = gen - nd

    post_itc_margins = np.apply_along_axis(lambda m: self._veto_flow(m,c),1,margins)

    lole_vector = np.apply_along_axis(lambda m: (m < 0).astype(int),1,post_itc_margins)

    eeu_vector = np.apply_along_axis(lambda m: self._pu(m),1,post_itc_margins)
  
    system_lole_vector = np.sum(lole_vector,axis=1)
    system_eeu_vector = np.sum(eeu_vector,axis=1)

    mean_vals = period_length * np.concatenate([
      np.mean(lole_vector,axis=0).reshape((2,)),
      np.mean(eeu_vector,axis=0).reshape((2,)),
      np.mean(system_lole_vector).reshape((1,)),
      np.mean(system_eeu_vector).reshape((1,))])

    sd_vals = period_length * np.concatenate([
      (np.std(lole_vector,axis=0)/np.sqrt(n)).reshape((2,)),
      (np.std(eeu_vector,axis=0)/np.sqrt(n)).reshape((2,)),
      (np.std(system_lole_vector)/np.sqrt(n)).reshape((1,)),
      (np.std(system_eeu_vector)/np.sqrt(n)).reshape((1,))])

    area_vector = ["A1","A2","A1","A2","SYS","SYS"]
    metric_vector = ["LOLE","LOLE","EEU","EEU","LOLE","EEU"]

    results_df = pd.DataFrame({
      "mean_val":mean_vals,
      "mean_sd":sd_vals,
      "area":area_vector,
      "metric":metric_vector})

    results_df["c"] = c
    results_df["n"] = n
    results_df["params"] = str(kwargs)
    results_df["net_demand_info"] = self.net_demand.__class__.__name__

    # results["lolp"] = {"mean":np.mean(lolp_vector,axis=0),"sd":np.std(lolp_vector,axis=0)/np.sqrt(n)}
    # results["epu"] = {"mean":np.mean(epu_vector,axis=0),"sd":np.std(epu_vector,axis=0)/np.sqrt(n)}

    # results["system_lolp"] = {"mean":np.mean(system_lolp_vector),"sd":np.std(system_lolp_vector)/np.sqrt(n)}
    # results["system_epu"] = {"mean":np.mean(system_epu_vector),"sd":np.std(system_epu_vector)/np.sqrt(n)}

    # res = SimulationResults(
    #   results=results,
    #   n=k+bc,
    #   sim_class=self.net_demand.__class__.__name__,
    #   d=self.d,
    #   c=c,
    #   **kwargs)
  
    # results_list.append(res)

    # loss_events_idx = system_lole_vector > 0

    # loss_events_df = pd.DataFrame(
    #   np.concatenate((
    #     nd[loss_events_idx,:],
    #     margins[loss_events_idx,:],
    #     eeu_vector[loss_events_idx,:]),
    #     axis=1),
    #   columns = ["nd0","nd1","gen0","gen1","pu0","pu1"])
    
    return results_df#, loss_events_df
