import numpy as np
import pandas as pd
import time

class ConvGenDistribution(object):

  """Available conventional generation distribution class, binned to 1MW. 
  
  **Parameters**:

  `gen_data` (`pandas.DataFrame` or `str`): Data Frame (or filename to load it) with 'Capacity' (in MW) and 'Availability' (a probability value) columns for each generator

  `sep` (`str`): column separator to use if data has to be loaded

  `bin_size` (`int`): optional MW bin size for the distribution. Defaults to 1 (untested feature)

  """

  def __init__(self,gen_data,sep=" ",bin_size = 1):

    if isinstance(gen_data,str):
      data = pd.read_csv(gen_data,sep=sep)
    elif isinstance(gen_data,pd.DataFrame):
      data = gen_data # assuming gen_data is a pandas DataFrame
    else:
      raise Exception("gen_data must be either a file path or a Pandas DataFrame; passed object has type {t}".format(t=str(type(gen_data))))
    #self.max = np.sum(data["Capacity"])
    capacities = np.array(data["Capacity"])
    availabilities = np.array(data["Availability"])

    if np.any(availabilities < 0) or np.any(availabilities > 1):
      raise Exception("Availabilities bust between 0 and 1")
      
    self.original_data = data
    self.original_capacities = capacities #this is stored to avoid rounding error mounting up after rescaling
    self.rescaling_factor = 1
    self._setup(capacities,availabilities,bin_size)

    # do internal setup given the data
  def _setup(self,capacities,availabilities,bin_size=1):
    """Sets up the relevant arrays: CDF values and partial sums of E[X]
  
    **Parameters**:

    `capacities` (`numpy.array`): capacity values

    `availabilities` (*np.array`): availability values

    """
    if np.any(availabilities > 1) or np.any(availabilities <= 0):
      raise Exception("Some generators have negative or 0% availability")

    capacities = np.array(capacities).astype(np.int32)
    availabilities = np.array(availabilities)

    is_fc = (availabilities==1) #firm capacity 
    self.fc = np.sum(capacities[is_fc])

    if self.fc > 0:
      print("Firm capacity (100% available generators) will be trated as a shift value and not included in capacity and availability arrays")
    
    #treat capacities as integers (uint16 <= 65,532)
    not_fc = np.logical_not(is_fc)
    self.capacity_vals = capacities[not_fc]
    self.availability_vals = availabilities[not_fc]

    # influde shift induced by firm capacity in min and max attributes
    self.max = int(np.sum(self.capacity_vals[self.capacity_vals>=0])) + self.fc
    self.min = int(np.sum(self.capacity_vals[self.capacity_vals<0])) + self.fc

    # firm capacity only shifts the pribability mass function, so exclude them from the convolution
    self.cdf_vals = np.ascontiguousarray(self._convolve_generators(),dtype=np.float64)
    self.pdf_vals = np.array([self.pdf(x) for x in range(self.min,self.max+1)])
    # compute expectation values array
    self.expectation_vals = np.ascontiguousarray(self._compute_expectations(),dtype=np.float64)
    #self.expectation_vals = np.ascontiguousarray(np.cumsum(np.array([x * self.pdf(x) for x in range(self.min,self.max+1)])),dtype=np.float64)
    self.n_gen = len(self.availability_vals)
    
    if bin_size > 1:
      self._scale(bin_size)

    self.bin_size = bin_size
    #print("Gen. convolution took {x} seconds".format(x = time.time() - t0))

  def _compute_expectations(self,test=False):

    return np.cumsum(np.array([x * self.pdf(x) for x in range(self.min,self.max+1)]))

  def add_fc(self,fc):

    """ Add firm capacity to the system

      `fc` (`int`): capacity to be added

    """
    fc_ = np.int32(fc)
    self.max += fc_
    self.min += fc_
    self.fc += fc_

    self.expectation_vals = np.ascontiguousarray(self._compute_expectations(),dtype=np.float64)

  def __add__(self, k):

    """ adds a constant (100% available) integer capacity to the generation system; this is useful to get equivalent firm capacity values 
  
    **Parameters**:

    `k` (`int`): capacity to be added

    """
    # if there is no generator with 100% availability, add one
    self.add_fc(k)
    return self

  def _rescale(self,x):

    """ rescales generator capacities by a constant factor
  
    **Parameters**:

    `k` (`int`): capacity to be added

    """
    # if there is no generator with 100% availability, add one

    new_caps = (self.rescaling_factor*self.original_capacities).astype(np.int32)
    self._setup(new_caps,self.availability_vals,self.bin_size)

  def __mul__(self,x):
    """ multiplication method to allow rescaling of generator capacities
  
    **Parameters**:

    `x` (`int`): rescaling factor

    """

    self.rescaling_factor *= x
    self._rescale(x)
    return self

  def _E0(self,x):
    """Return partial expectation: 0*p_0 + ... + x*p_x
  
    **Parameters**:

    `x` (`int`): upper bound

    """

    if x > self.max:
      return self.expectation_vals[-1]
    elif x < self.min:
      return 0
    else:
      return self.expectation_vals[int(x) - self.min]
  
  def expectation(self,fro=0,to=None):
    """Return partial expectation: a*p_a + ... + b*p_b
  
    **Parameters**:

    `fro` (`int`): lower bound

    `to` (`int`): upper bound

    """
    if to is None:
      to = self.max
    return self._E0(to) - self._E0(fro-1)
      
  def _scale(self,bin_size):
    """Bins distribution in coarser grid
  
    **Parameters**:

    `bin_size` (`int`): bin size

    """
    k = int(np.ceil(self.max/bin_size))
    new_cdf = np.empty(k)
    for i in range(k):
      new_cdf[i] = self.cdf(bin_size*i)
      
    self.cdf_vals = np.array(new_cdf)
    self.max = k
    
  def _convolve_generators(self):
    """compute CDF values and save them in a list

    **Parameters**:

    `capacities` (`numpy.array`): capacity values

    `availabilities` (`numpy.array`): availability values

    """
    zero_idx = np.abs(self.min) #this is in case there are generators with negative generation
    f_length = self.max+1 - self.min
    f = np.zeros((f_length,),dtype=np.float64)
    f[zero_idx] = 1.0
    for c,p in zip(self.capacity_vals,self.availability_vals):
      if c >= 0:
        suffix = f[0:f_length-c]
        preffix = np.zeros((c,))
      else:
        preffix = f[np.abs(c):f_length]
        suffix = np.zeros((np.abs(c),))
      f = (1-p) * f + p * np.concatenate([preffix,suffix])
    F = np.cumsum(f/np.sum(f))
    return F

  def cdf(self,x):
    """returns CDF value 
  
    **Parameters**:

    `x` (`int`): value to evaluate CDF on

    """

    if x >= self.max:
      return 1.0
    elif x < self.min:
      return 0.0
    else:
      return self.cdf_vals[int(x) - self.min]

  def pdf(self,x):
    """returns PDF value 
  
    **Parameters**:

    `x` (`int`): value to evaluate PDF on

    """

    return self.cdf(x) - self.cdf(x-1)

  def simulate(self,n,lb=-np.Inf,ub=np.Inf,seed=None):
    """Simulate from this distribution 
  
    **Parameters**:

    `n` (`int`): number of samples

    `lb` (`int`): simulated samples are above this lower bound

    `ub` (`int`): simulated samples are below this upper bound

    `seed` (`int`): random seed

    """
    if seed is not None:
      np.random.seed(seed)
    
    n = int(n)

    lb = int(max(self.min,lb))
    ub = int(min(self.max,ub))

    lb_idx = lb - self.min
    ub_idx = lb_idx + ub - lb

    p = self.pdf_vals[lb_idx:ub_idx+1]
    p = p/np.sum(p)

    return np.random.choice(range(lb,ub+1),size=n,p=p).reshape(n,1)

    #return np.random.choice(range(self.min,self.max+1),size=n,p=self.pdf_vals).reshape(n,1)

  # def equals(self,other):
  #   print("checking for equality")
  #   for internal in dir(self):
  #     attr = getattr(self,internal)
  #     if isinstance(attr,list) or isinstance(attr,np.ndarray):
  #       eql = list(attr) == list(getattr(other,internal))
  #       if not eql:
  #         print("not equal: {x}".format(x=internal))
  #     if isinstance(attr,int) or isinstance(attr,float):
  #       eql = attr == getattr(other,internal)
  #       if not eql:
  #         print("not equal: {x}".format(x=internal))

class ClippedGenDist(ConvGenDistribution):

  """This class takes a shifted generation distribution and caps it to have a minimum value of zero. This is so we can shift distribution negatively without negative generation as a side effect

  
  **Parameters**:

  `gendist` (`ConvGenDistribution`): `ConvGenDistribution` instance, possibly with negative firm capacity

  """
  def __init__(self,gendist):

    self.cdf_vals= np.ascontiguousarray(np.copy(gendist.cdf_vals))
    self.expectation_vals = np.ascontiguousarray(np.copy(gendist.expectation_vals))
    self.max = gendist.max
    self.min = gendist.min
    self.fc = gendist.fc
    self._normalize()

  def _normalize(self):

    if self.min < 0:
      clipped_prob = self.cdf_vals[np.abs(self.min)-1]
      self.cdf_vals = self.cdf_vals[np.abs(self.min)::]
      self.min = 0
      self.fc = 0
      # smooth out the would-be-negative probability mass on all non-negative domain points
      self.cdf_vals -= clipped_prob
      self.cdf_vals += np.cumsum(clipped_prob/len(self.cdf_vals)*np.ones(self.cdf_vals.shape))
      self.cdf_vals = np.ascontiguousarray(self.cdf_vals)
      self.expectation_vals = np.ascontiguousarray(self._compute_expectations(),dtype=np.float64)

