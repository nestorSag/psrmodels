import numpy as np
import pandas as pd

from _c_ext_timedependence import ffi, lib as C_CALL

class ConvGenDistribution(object):

  """Available conventional generation distribution object, taken as an aggregation of Markov Chains representing generators
  
  **Parameters**:

  `gens_info` (`dict` or `pandas.DataFrame`): Either a data frame with columns 'Capacity', 'Availability' and 'TTR' (time to repair), or a dictionary with keys 'transition_probs' (a list of transition matrices) and 'states_list' (a list of state sets corresponding to transition matrices)

  """

  #def __init__(self,states_list, transition_probs):
  def __init__(self,gens_info):

    if isinstance(gens_info,pd.DataFrame):
      
      gens_info = self._map_df_to_dict(gens_info)

    if isinstance(gens_info,dict) and "transition_probs" in gens_info.keys() and "states_list" in gens_info.keys():
      self._build_from_dict(gens_info)
    else:
      raise Exception("gens_info have to be either a data frame with columns 'Capacity', 'Availability' and 'TTR' (time to repair), or a dictionary with keys 'transition_probs' (a list of transition matrices) and 'states_list' (a list of state sets corresponding to transition matrices)")

  def _build_from_dict(self,gens_info):

    self.transition_prob_array = np.ascontiguousarray(gens_info["transition_probs"],dtype=np.float64)

    self.states_array = np.ascontiguousarray(gens_info["states_list"],dtype=np.float64)
    
    self.n_gen = len(self.states_array)
    self.n_states = len(self.states_array[0]) #take first element as baseline

    if len(self.states_array) != len(self.transition_prob_array):
      raise Exception("Length of states list do not match length of transition matrices list")

    for i in range(self.n_gen):
      if self.transition_prob_array[i].shape != (self.n_states,self.n_states):
        raise Exception("Transition matrices have varying shapes. They should have the same shape.")


  def _map_df_to_dict(self,gens_df):

    # mat must be a df with columns: Capacity, Availability, TTR, casted to matrix
    def row_to_mc_matrix(row):
      pi, ttr = row
      alpha = 1 - 1/ttr
      a11 = 1 - (1-pi)*(1-alpha)/pi
      mat = np.array([[a11,1-a11],[1-alpha,alpha]])
      return mat

    mat = np.array(gens_df[["Capacity","Availability","TTR"]])
    states_list = [[x,0] for x in mat[:,0]]
    transition_prob_list = np.apply_along_axis(row_to_mc_matrix,1,mat[:,1:3])

    return {"states_list":states_list,"transition_probs":transition_prob_list}

  def simulate(self,n_sim,n_timesteps,x0_list=None,seed=1,simulate_streaks=True):


    """Simulate traces of available conventional generation
    
      **Parameters**:

      `n_sim` (`int`): number of traces to simulate

      `n_timesteps` (`int`): number of transitions to simulate in each trace

      `x0_list`: `list` of initial state values. If `None`, they are sampled from the statinary distributions

      `seed` (`int`): random seed

      `simulate_streaks` (`bool`): simulate transition time lengths only. Probably faster if any of the states have a stationary probability larger than 0.5

    """

    # sanitise inputs
    if n_sim <= 0 or not isinstance(n_sim,int):
      raise Exception("Invalid 'n_sim' value or type")

    if n_timesteps <= 0 or not isinstance(n_timesteps,int):
      raise Exception("Invalid 'n_timesteps' value or type")

    if seed <= 0 or not isinstance(seed,int):
      raise Exception("Invalid 'seed' value or type")

    # validate list of initial values
    if x0_list is None:
      # if initial values are None, generate from stationary distributions
      np.random.seed(seed)
      x0_list = np.ascontiguousarray(self._get_stationary_samples()).astype(np.float64)
    else:
      if len(x0_list) != self.n_gen:
        raise Exception("Number of initial values do not match number of generators")

      for i in range(self.n_gen):
        if x0_list[i] not in self.states_array[i]:
          raise Exception("Some initial values are not valid for the corresponding generator")
        if self.transition_prob_array[i].shape[0] != len(self.states_array[i]) or self.transition_prob_array[i].shape[1] != len(self.states_array[i]):
          raise Exception("Some state sets do not match the shape of corresponding transition matrix")

    # set output array
    output_length = n_timesteps+1 #initial state + n_timesteps
    output = np.ascontiguousarray(np.empty((output_length,n_sim)),dtype=np.float64)

    # set initial values array
    initial_values = np.ascontiguousarray(x0_list,dtype=np.float64)

    # call C program


    C_CALL.simulate_mc_power_grid(
      ffi.cast("double *",output.ctypes.data),
      ffi.cast("double *",self.transition_prob_array.ctypes.data),
      ffi.cast("double *",self.states_array.ctypes.data),
      ffi.cast("double *",initial_values.ctypes.data),
      np.int64(self.n_gen),
      np.int64(n_sim),
      np.int64(n_timesteps),
      np.int64(self.n_states),
      np.int32(seed),
      np.int32(simulate_streaks))

    return output.reshape((-1,1))

  def _get_stationary_samples(self):

    sample = []
    for Q,states in zip(self.transition_prob_array,self.states_array):
      pi = self._find_stationary_dist(Q)
      s = np.random.choice(states,size=1,p=pi)
      sample.append(s)

    return sample

  def _find_stationary_dist(self,Q):
    # from somewhere in stackoverflow
    
    evals, evecs = np.linalg.eig(Q.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    if evec1.shape[1] == 0:
      raise Exception("Some generators might not have a stationary distribution")
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real

    return stationary
