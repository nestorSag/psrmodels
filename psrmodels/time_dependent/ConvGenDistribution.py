import warnings

import numpy as np
import pandas as pd

from _c_ext_timedependence import ffi, lib as C_CALL

class ConvGenDistribution(object):

  """Available conventional generation distribution object, taken as an aggregation of Markov Chains representing generators
  
  **Parameters**:

  `gens_info` (`str`, `dict` or `pandas.DataFrame`): Either a csv file path, a data frame with columns 'Capacity', 'Availability' and 'TTR' (time to repair), or a dictionary with keys 'transition_probs' (a list of transition matrices) and 'states_list' (a list of state 'np.ndarray's corresponding to transition matrices)


  `seed` (`int`): If not none, this will be the random seed used by the object throughout its lifetime, overriding other random seeds that are passed as arguments. 


  """

  #def __init__(self,states_list, transition_probs):
  def __init__(self,gens_info,seed=None):

    self.seed = seed
    if self.seed is not None:
      print(f"A random seed of {seed} was provided to ConvGenDistribution at instantiation time, and will remain fixed throughout the instance's lifetime, ignoring seeds passed as arguments.")

    if isinstance(gens_info,str):
      data = pd.read_csv(gens_info,sep=",")
      gens_info = self._map_df_to_dict(data)

    if isinstance(gens_info,pd.DataFrame):
      
      gens_info = self._map_df_to_dict(gens_info)

    if isinstance(gens_info,dict) and "transition_probs" in gens_info.keys() and "states_list" in gens_info.keys():
      self._build_from_dict(gens_info)
    else:
      raise Exception("gens_info have to be either a data frame with columns 'Capacity', 'Availability' and 'TTR' (time to repair), or a dictionary with keys 'transition_probs' (a list of transition matrices) and 'states_list' (a list of state sets corresponding to transition matrices)")

  def _build_from_dict(self,gens_info):

    self.fc = 0 #initialise firm capacity as zero
    self.min = 0 #min possible generation
    self.max = sum([x[0] for x in gens_info["states_list"]]) #max possible generation

    self.transition_prob_array = np.ascontiguousarray(gens_info["transition_probs"],dtype=np.float32)

    self.states_array = np.ascontiguousarray(gens_info["states_list"],dtype=np.float32)
    
    self.n_gen = len(self.states_array)
    self.n_states = len(self.states_array[0]) #take first element as baseline

    # buffers for saved samples
    self.saved_sample = None
    self.saved_sample_params = None

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
      mat = np.array([[a11,1-a11],[1-alpha,alpha]],dtype=np.float32)
      mat = mat / np.sum(mat,axis=1)
      # this is to ensure that all chains are probability distributions
      #remainder = np.clip(1.0 - np.sum(mat,axis=1),a_min=0,a_max=np.Inf)
      #mat[:,0] += remainder
      return mat

    mat = np.array(gens_df[["Capacity","Availability","MTTR"]],dtype=np.float32)
    states_list = [[x,0] for x in mat[:,0]]
    transition_prob_list = np.apply_along_axis(row_to_mc_matrix,1,mat[:,1:3])

    return {"states_list":states_list,"transition_probs":transition_prob_list}

  def add_fc(self,fc):

    """ Add firm capacity to the system

      `fc` (`int`): capacity to be added

    """
    fc_ = np.int32(fc)
    self.max += fc_
    self.min += fc_
    self.fc += fc_

  def __add__(self, k):

    """ adds a constant (100% available) integer capacity to the generation system; this is useful to get equivalent firm capacity values 
  
    **Parameters**:

    `k` (`int`): capacity to be added

    """
    # if there is no generator with 100% availability, add one
    self.add_fc(k)
    return self

  def _same_params_as_buffer(self,n_transitions,x0_list,seed):
    return self.saved_sample_params["n_transitions"] == n_transitions and \
      self.saved_sample_params["x0_list"] == x0_list and \
      self.saved_sample_params["seed"] == seed

  def simulate(self,n_sim,n_transitions,x0_list=None,seed=1,simulate_streaks=True,use_buffer=True):

    if self.seed is not None:
      #.warn("Using random seed set at instantiation time; ignoring seed from argument list.")
      seed = self.seed
    
    """Simulate traces of available conventional generation
    
      **Parameters**:

      `n_sim` (`int`): number of traces to simulate

      `n_transitions` (`int`): number of transitions to simulate in each trace

      `x0_list`: `list` of initial state values. If `None`, they are sampled from the statinary distributions

      `seed` (`int`): random seed

      `simulate_streaks` (`bool`): simulate transition time lengths only. Probably faster if any of the states have a stationary probability larger than 0.5

      `use_buffer` (`bool`): save results in an internal buffer for further use without having to sample again; if buffer exists already, reuse results

    """

    max_array_size = int(2**31-1) #2**31-1 -> C integer range
    if n_sim*(n_transitions+1) >= max_array_size:
      raise Exception("Resulting arrays are too large; for the provided data, n_sim cannot be larger than {x} at each individual run".format(x=int(max_array_size/(n_transitions+1))))
    timesteps_in_season = n_transitions + 1 #t0 + n_transitions timesteps = [t0,...,tn]
    if use_buffer:
      if self.saved_sample is None or not self._same_params_as_buffer(n_transitions,x0_list,seed):
        # if there is no buffered data or if its stale, create new one and return a copy
        self.saved_sample = self.simulate(n_sim,n_transitions,x0_list,seed,simulate_streaks,False)
        self.saved_sample_params = {"n_sim":n_sim,"n_transitions":n_transitions,"x0_list":x0_list,"seed":seed,"fc":self.fc}
        #print("ConvGenDistribution: returing fresh samples")
        return self.saved_sample.copy()
      else:
        # if there is valid buffered data
        if self.saved_sample_params["n_sim"] < n_sim:
          # run only the necessary simulations and append, then create a copy
          new_seed = int(self.saved_sample_params["seed"] + np.random.randint(low = 1, high = 1000,size=1)[0])
          print("Buffer is not large enough for required number of samples; generating additional ones with new seed to avoid duplication. Be aware that this affects reproducibility.")
          new_sim = self.simulate(n_sim-self.saved_sample_params["n_sim"],n_transitions,x0_list,new_seed,simulate_streaks,False) + self.saved_sample_params["fc"]
          self.saved_sample = np.ascontiguousarray(np.concatenate((self.saved_sample,new_sim),axis=0)).astype(np.float32)
          self.saved_sample_params["n_sim"] = n_sim
          #print("ConvGenDistribution: returing precomputed samples plus fresh samples")
          return self.saved_sample.copy() + (self.fc - self.saved_sample_params["fc"])
        else:
          #create a copy of a slice, since there are more simulations than necessary
          #print("ConvGenDistribution: returing precomputed samples")
          return self.saved_sample[0:(timesteps_in_season*n_sim),:].copy() + (self.fc - self.saved_sample_params["fc"])
    else:

      # sanitise inputs
      if n_sim <= 0 or not isinstance(n_sim,int):
        raise Exception("Invalid 'n_sim' value or type")

      if n_transitions <= 0 or not isinstance(n_transitions,int):
        raise Exception("Invalid 'n_transitions' value or type")

      if seed <= 0 or not isinstance(seed,int):
        raise Exception("Invalid 'seed' value or type")

      # validate list of initial values
      if x0_list is None:
        # if initial values are None, generate from stationary distributions
        np.random.seed(seed)
        x0_list = np.ascontiguousarray(self._get_stationary_samples()).astype(np.float32)
      else:
        if len(x0_list) != self.n_gen:
          raise Exception("Number of initial values do not match number of generators")

        for i in range(self.n_gen):
          if x0_list[i] not in self.states_array[i]:
            raise Exception("Some initial values are not valid for the corresponding generator")
          if self.transition_prob_array[i].shape[0] != len(self.states_array[i]) or self.transition_prob_array[i].shape[1] != len(self.states_array[i]):
            raise Exception("Some state sets do not match the shape of corresponding transition matrix")

      # set output array
      output = np.ascontiguousarray(np.zeros((n_sim,timesteps_in_season),dtype=np.float32),dtype=np.float32)
      #print(f"ConvGenDist: output array of size {output.shape}")
      #print("output shape: {s}".format(s=output.shape))
      #print("output before: {o}".format(o=output))

      # set initial values array
      initial_values = np.ascontiguousarray(x0_list,dtype=np.float32)

      # call C program

      C_CALL.simulate_mc_power_grid_py_interface(
        ffi.cast("float *",output.ctypes.data),
        ffi.cast("float *",self.transition_prob_array.ctypes.data),
        ffi.cast("float *",self.states_array.ctypes.data),
        ffi.cast("float *",initial_values.ctypes.data),
        np.int32(self.n_gen),
        np.int32(n_sim),
        np.int32(n_transitions),
        np.int32(self.n_states),
        np.int32(seed),
        np.int32(simulate_streaks))

      return output.reshape((-1,1)) + self.fc

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
