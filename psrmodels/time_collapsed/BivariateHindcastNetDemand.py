import numpy as np

class BivariateHindcastNetDemand(object):
  """Bivariate hindcast net demand distribution for a 2-area system in a time collapsed model

    **Parameters**:
    
    `X` (`numpy.ndarray`): net demand matrix, where each column is an area and each observation is a row

    """
  def __init__(self,X):
    #self.X = np.clip(X,a_min=0,a_max=np.Inf).astype(np.int32)
    self.X = X.clip(min=0)
    self.n = X.shape[0]
    
  def simulate(self,n,seed=1):
    np.random.seed(seed)
    return self.X[np.random.choice(range(self.n),size=n),:]

  def pdf(self,x):
    """calculate bivariate margin PDF

    **Parameters**:
    
    `x` (`numpy.ndarray`): margin values

    """

    return self.cdf(x) - (self.cdf(x - (1,0)) + self.cdf(x - (0,1)) - self.cdf(x - (1,1)))
  	

  def cdf(self,x):
  	"""calculate bivariate margin CDF

    **Parameters**:
    
    `x` (`numpy.ndarray`): margin values

    """
  	
  	return np.sum((self.X[:,0] <= x[0])*(self.X[:,1] <= x[1] ))/self.n
