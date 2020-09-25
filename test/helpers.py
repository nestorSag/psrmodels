import numpy as np
from scipy.stats import poisson

def winter_period(dt):
  if dt.month >= 10:
    return dt.year
  else:
    return dt.year - 1

# class UniformLattice(object):

#   def cdf(self,x):
#     x = np.array(x)
#     if (x>=0).all():
#       return np.prod(x+1)
#     else:
#       return 0

#   def pdf(self,x):
#     x = np.array(x)
#     if (x>=0).all():
#       return 1.0
#     else:
#       return 0

# class MultivariatePoisson(object):

#   def __init__(self):
#     self.rate = 10
    
#   def cdf(self,x):
#     x = np.array(x)
#     if (x>=0).all():
#       return np.prod(np.apply_along_axis(lambda x: poisson.cdf(x,mu=self.rate),0,x))
#     else:
#       return 0

#   def pdf(self,x):
#     x = np.array(x)
#     if (x>=0).all():
#       return np.prod(np.apply_along_axis(lambda y: poisson.pmf(y,mu=self.rate),0,x))
#     else:
#       return 0
    
