
from .ConvGenDistribution import ConvGenDistribution

import numpy as np

class BivariateConvGenDist(object):
    
  """Wrapper for two independent conventional generation distributions; behaves like a bivariate distribution
  **Parameters**:
    
  `X1` (`ConvGenDistribution`): conventional generation distribution from area 1

  `X2` (`ConvGenDistribution`): conventional generation distribution from area 2 
  """

  def __init__(self,X1,X2):

    if not isinstance(X1,ConvGenDistribution) or not isinstance(X2,ConvGenDistribution):
      raise Exception("inputs are not ConvGenDistribution instances: X1: " + str(type(X1)) + ", X2: " + str(type(X2)))
    self.X1 = X1
    self.X2 = X2
    
  def pdf(self,x):

    """calculate PDF 
    **Parameters**:
      
    `x` (`iterable`): length 2 iterable of integers
    """
    return self.X1.pdf(x[0])*self.X2.pdf(x[1])
    
  def cdf(self,x):
    """calculate PDF 
    **Parameters**:
      
    `x` (`iterable`): length 2 iterable of integers
    """
    return self.X1.cdf(x[0])*self.X2.cdf(x[1])

  def simulate(self,n,seed=None):
    """calculate PDF 
    **Parameters**:
      
    `n` (`int`): number of simulated values

    `seed` (`int`): random seed; defaults to 999 if left as `None`
    """
    if seed is None:
      seed = 999
    return np.concatenate([
      self.X1.simulate(n=n,seed=seed+1).reshape(n,1), 
      self.X2.simulate(n=n,seed=seed+2).reshape(n,1)],
      axis=1)