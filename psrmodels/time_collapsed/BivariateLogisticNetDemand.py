import numpy as np
from scipy.stats import genpareto as gp
from scipy.stats import expon as expdist
from scipy.stats import gumbel_r as gumbel
from scipy.stats import gaussian_kde

from scipy.special import lambertw

class BivariateLogisticNetDemand(object):
  """Semi-parametric bivariate net demand model with a logistic extreme value model at the tails.
  It does not performs estimation, instead receiving the estimated parameters as input
  
  **Parameters**:

  `X` (`numpy.ndarray`): net demand data matrix with two columns

  `p` (`float`): threshold probability for each of the components

  `alpha` (`float`): estimated dependence parameter for logistic extreme value model

  `shapes` (`numpy.ndarray`): shape estimates for margin GP approximations

  `scales` (`numpy.ndarray`): scale estimates for margin GP approximation

  """
  
  def __init__(self,X,p,alpha,shapes,scales):

    self.n, self.d = X.shape
    self.X = X.clip(min=0)

    #self.u = np.array(u)
    #self.p = np.array([np.sum(self.X[:,i] <= self.u[i])/self.n for i in range(self.d)])

    self.p = float(p)
    self.u = np.quantile(self.X,self.p,axis=0)
    
    self.alpha = alpha
    self.shapes = np.array(shapes)
    self.scales = np.array(scales)
    self.endpoints = np.array([self.u[i] - self.scales[i]/self.shapes[i] if shapes[i] < 0 else np.Inf for i in range(self.d)])
    self.kde = None #kernel density estimates for the marginals (they are used for plotting only)

  def _dy_dx(self,x,i):
    #derivative of y = -log(-log(F(x))) w.r.t. x
    # x is a vector
    # i = component index

    eta = self.shapes[i]
    sigma = self.scales[i]
    mu = self.u[i]
    p = self.p
    
    # copied from wolfram alpha as text
    if x[i] >= mu:
      if eta != 0:
        d = -((1 - p)*((eta*(x[i] - mu))/sigma + 1)**(-1/eta - 1))/(sigma*((1 - p)*(1 - ((eta*(x[i] - mu))/sigma + 1)**(-1/eta)) + p)*np.log((1 - p)*(1 - ((eta*(x[i] - mu))/sigma + 1)**(-1/eta)) + p))
      else:
        d = -np.exp(mu/sigma)/(((p - 1)*np.exp(mu/sigma) + np.exp(x[i]/sigma))*np.log((p - 1)*np.exp((mu - x[i])/sigma) + 1))

    else:
      # the derivatives dont exist below the threshold, so they are imputed by a Gaussian KDE
      if self.kde is None:
        self.kde = []
        self.kde_normval = []
        for j in range(self.d):
          self.kde.append(gaussian_kde(self.X[:,j]))
          # record normalizing constant necessary so that the density is continuous
          kde_cdf = self.kde[j].integrate_box_1d(-np.Inf,self.u[j])
          self.kde_normval.append(self._dy_dx(self.u,j)/(-(1-p)/(kde_cdf*np.log(kde_cdf))*self.kde[j](self.u[j])[0]))
          
      #d = -(1-p)/(self._sp_margin_cdf(x[i],i)*np.log(self._sp_margin_cdf(x[i],i)))*self.kde[i](x[i])[0]*self.kde_normval[i]

      kde_cdf = self.kde[i].integrate_box_1d(-np.Inf,x[i])
      
      d = -(1-p)/(kde_cdf*np.log(kde_cdf))*self.kde[i](x[i])[0]*self.kde_normval[i]
            
    return d
  
  def _margin_tail_cdf(self,x,i):
    # CDF of GP approximation (no need to weight it by p, that's done elsewhere)
    # i = component index
    if self.shapes[i] != 0:
      return gp.cdf(x,c=self.shapes[i],loc=self.u[i],scale=self.scales[i])
    else:
      return expdist.cdf(x,loc=self.u[i],scale=self.scales[i])

  def _margin_tail_pdf(self,x,i):
    # density of GP approximation (no need to weight it by p, that's done elsewhere)
    # i = component index
    if self.shapes[i] != 0:
      return gp.pdf(x,c=self.shapes[i],loc=self.u[i],scale=self.scales[i])
    else:
      return expdist.pdf(x,loc=self.u[i],scale=self.scales[i])
    
  def _sp_margin_cdf(self,x,i=0):
    # x is a scalar
    # semiparametric marginal cdf
    if x <= self.u[i]:
      return self._mecdf(x,i=i)
    else:
      p = self.p
      return p + (1-p)*self._margin_tail_cdf(x,i)

  def _sp_margin_pdf(self,x,i=0):
    # only non-zero above margin threshold
    if x <= self.u[i]:
      return 0
    else:
      return (1-self.p)*self._margin_tail_pdf(x,i)

  def _jecdf(self,x):
    # joint empirical cdf
    return np.sum((self.X[:,0] <= x[0])*(self.X[:,1] <= x[1] ))/self.X.shape[0]

  def _mecdf(self,x,i=0):
    # marginal empirical cdf
    return np.sum(self.X[:,i] <= np.array(x))/self.X.shape[0]

  def _to_gumbel(self,x):
    # transform vector to gumbel marginals componentwise using semiparametric marginal estimates
    return np.array([-np.log(-np.log(self._sp_margin_cdf(x[i],i=i))) for i in range(self.d)])
  
  def _cond_ext_pdf(self,x,i):
    # conditional extreme value cdf
    # x = point to evaluate
    # i = largest component (quantile-wise)

    
    # transform data to Gumbel margins first
    y = self._to_gumbel(x)
    j = 1 if i == 0 else 0 #the not-so-large component

    # integrating the assymptotic density Y_j|Y_i = y_i w.r.t. a std. Gumbel from y[i] to infty
    # and using the fact that Gumbel ~ exp(1)
    
    return gumbel.cdf(y[j],loc=0,scale=1) + np.exp(-y[j]) - np.sum(np.exp(-y/self.alpha))**self.alpha

  def _cond_ext_pdf(self,x,i):
    # conditional extreme value pdf
    # x = point to evaluate
    # i = largest component (quantile-wise)

    # transform data to Gumbel margins first
    y = self._to_gumbel(x)
    j = 1 if i == 0 else 0 #the not-so-large component

    # dF/(dx1dx2) = dF/(dy1dy2)*dy1/dx1*dy2/dx2

    # copied from wolfram alpha as plain text
    dF_dy1dy2 = -((self.alpha - 1)*np.exp(y[i]/self.alpha + y[j]/self.alpha)*(np.exp(-y[i]/self.alpha) + np.exp(-y[j]/self.alpha))**self.alpha)/(self.alpha*(np.exp(y[i]/self.alpha) + np.exp(y[j]/self.alpha))**2)

    dyi_dxi = self._dy_dx(x,i)
    dyj_dxj = self._dy_dx(x,j)

    return dF_dy1dy2 * dyi_dxi * dyj_dxj
  
  def _jtcdf(self,x):
    # joint tail cdf

    # transform data to Gumbel margins first
    y = self._to_gumbel(x)
    return np.exp(-np.sum(np.exp(-1.0/self.alpha*y))**self.alpha)

  def _jtpdf(self,x):
    # joint tail pdf

    # transform data to Gumbel margins first
    y = self._to_gumbel(x)

    # both are exceedances so the order doesnt matter
    i = 0
    j = 1
    # dF/(dx1dx2) = dF/(dy1dy2)*dy1/dx1*dy2/dx2
    dyi_dxi = self._dy_dx(x,i)
    dyj_dxj = self._dy_dx(x,j)
    
    # from wolfram alpha as plain text
    density = (np.exp(y[i]/self.alpha - (np.exp(-y[i]/self.alpha) + np.exp(-y[j]/self.alpha))**self.alpha + y[j]/self.alpha)*(np.exp(-y[i]/self.alpha) + np.exp(-y[j]/self.alpha))**self.alpha*(-self.alpha + self.alpha*(np.exp(-y[i]/self.alpha) + np.exp(-y[j]/self.alpha))**self.alpha + 1))/(self.alpha*(np.exp(y[i]/self.alpha) + np.exp(y[j]/self.alpha))**2)
    
    return density * dyi_dxi * dyj_dxj
    
  def cdf(self,x):
    """ Get model cumulative density function (CDF)
    
    **Parameters**:

    `x` (`numpy.ndarray`): point to evaluate the model's CDF in

    """
    
    if np.all(x > self.u):
      return self._jtcdf(x)
    
    elif np.all(x <= self.u):
      return self._jecdf(x)
    
    else:
      i = int(np.argwhere(x > self.u))
      return self._cond_ext_pdf(x,i)
      #return self._jtcdf(x)

  def pdf(self,x):
    """ Get model probability density function (PDF)
    
    **Parameters**:

    `x` (`numpy.ndarray`): point to evaluate the model's PDF in

    """
    
    if np.all(x <= self.u):
      return 0
    
    elif np.all(x >= self.u):
      return self._jtpdf(x)
    
    else:
      i = int(np.argwhere(x > self.u))
      return self._cond_ext_pdf(x,i)
      #return self._jtpdf(x)
      

  def raster(self,nx=100,ny=100,cdf=True,llim=None,ulim=None,beta=0.995):
    """ Get model PDF or CDF contour lines raster, in the scale of the original data.
    
    **Parameters**:

    `nx` (`int`): number of grid points in the x axis

    `ny` (`int`): number of grid points in the y axis

    `cdf` (`bool`): Plot CDF. If False, plot PDF

    `llim` (`float`): lower plot threshold. If None, taken as minus infinity

    `ulim` (`float`): upper plot threshold. If None, taken as infinity

    `beta` (`float`): If estimated enpoint of original data is infinite, use a quantile of beta as upper plot endpoint

    """

    
    def compute_upper_limits(beta=beta):
      # compute beta quantile as plot limits
      lims = []
      gamma = np.exp(2**(self.alpha)*np.log(beta))
      for i in range(self.d):
        p = self.p
        if self.shapes[i] < 0:
          val = self.endpoints[i]
        elif self.shapes[i] > 0:
          val = self.u[i] + self.scales[i]*(((1-(gamma-p)/(1-p))**(-self.shapes[i])-1)/self.shapes[i])
        else:
          val = 1.5 * (self.u[i] + self.scales[i]*(-np.log(1-(gamma-p)/(1-p))))

        lims.append(val)

      return lims
    
    # nx, ny = number of grid points in each axis
    # llim = lower plot limits
    # ulim upper plot limits
    
    # bound plot
    
    llim = [-np.Inf for i in range(self.d)] if llim is None else llim

    ulim = [np.Inf for i in range(self.d)] if ulim is None else ulim

    llim = [max(llim[i],min(self.X[:,i])) for i in range(self.d)]

    ubounds = compute_upper_limits()

    ulim = [min(ulim[i],ubounds[i]) for i in range(self.d)]

    #print(llim)
    #print(ulim)
    
    x = np.linspace(llim[0],ulim[0],nx)
    y = np.flip(np.linspace(llim[1],ulim[1],ny))

    vals = []
    
    for x_ in x:
      for y_ in y:
        val = self.cdf((x_,y_)) if cdf else self.pdf((x_,y_))
        vals.append(val)

    z = np.array(vals).reshape((int(ny),int(nx)),order="F")
    
    return {"x":x,"y":y,"z":np.nan_to_num(z)}

  def margin_quantiles(self,probs):
    """get quantiles of semiparametric CDFs, componentwise
    
    **Parameters**:

    `probs` (`iterable`): probabilities for which quantiles will be evaluated at each component

    """

    x = []
    for i in range(self.d):
      p = self.p
      if probs[i] <= p:
        val = np.quantile(X[:,i],probs[i])
      elif self.shapes[i] != 0:
        val = self.u[i] + self.scales[i]*(((1-(probs[i]-p)/(1-p))**(-self.shapes[i])-1)/self.shapes[i])
      else:
        val = self.u[i] + self.scales[i]*(-np.log(1-(probs[i]-p)/(1-p)))

      x.append(val)

    return np.array(x)
      
  def _from_gumbel(self,m):
    # transform gumbel simulated values to original data distribution

    def original_scale(v,i):
      # v = vector of Gumbel quantiles
      # i = component index
      p = self.p
      index = v > p
      above = v[index]
      below = v[np.logical_not(index)]

      if self.shapes[i] != 0:
        above = self.u[i] + self.scales[i]*(((1-(above-p)/(1-p))**(-self.shapes[i])-1)/self.shapes[i])
      else:
        above = self.u[i] + self.scales[i]*(-np.log(1-(above-p)/(1-p)))

      
      below = np.quantile(self.X[:,i],below)

      v[index] = above
      v[np.logical_not(index)] = below

      return v

    m_= np.empty(shape=m.shape)
    
    for i in range(self.d):
      gamma = np.exp(-np.exp(-m[:,i]))
      m_[:,i] = original_scale(gamma,i)

    return m_

  def _simulate_exceedances(self,n,threshold=None,exs_prob=None,seed=None,asymptotic=False):
    # simulate from extreme value model at the tails
    if seed is not None:
      np.random.seed(int(seed))

    n = int(n)

    #determine threshold values and probabilities
    threshold, exs_prob = self._format_th_params(threshold,exs_prob)

    gumbel_threshold = self._to_gumbel(threshold)
    
    maximums = -np.log(-np.log(exs_prob*np.random.uniform(size=n)+1-exs_prob)) + self.alpha*np.log(2)
    comp = np.random.binomial(1,0.5,n)

    if asymptotic:
      other = np.array([self._sample_cond_ext_dist(np.Inf,uq=self._joint_cond_ext_pdf(0,np.Inf)) for x in maximums]).reshape(n,1)
    else:
      other = np.array([self._sample_cond_ext_dist(x,uq=self._joint_cond_ext_pdf(0,x)) for x in maximums]).reshape(n,1)

    xs0 = maximums[comp==0].reshape(-1,1)
    xs1 = maximums[comp==1].reshape(-1,1)

    other0 = other[comp==0] + xs0
    other1 = other[comp==1] + xs1

    mat0 = np.concatenate((xs0,other0),axis=1)
    mat1 = np.concatenate((other1,xs1),axis=1)
   
    return self._from_gumbel(np.concatenate((mat0,mat1),axis=0))
    

  def _joint_cond_ext_pdf(self,z,y):
    return (1+np.exp(-z/self.alpha))**(self.alpha-1)*np.exp(np.exp(-y)*(1-(1+np.exp(-z/self.alpha))**self.alpha))
  
  def _sample_cond_ext_dist(self,exs,lq=0,uq=1):
    # sample conditional extremes distribution given an exceedance in the other component
    gamma = np.exp(-exs)
    beta = np.log(np.random.uniform(size=1,low=lq,high=uq)) - gamma
    eta = beta/(self.alpha-1) - 1.0/self.alpha*lambertw(-gamma*self.alpha/(self.alpha-1)*np.exp(self.alpha*beta/(self.alpha-1))).real
    z = -self.alpha*np.log(np.exp(eta)-1)
    return z

  def _simulate_empirical(self,n,exs_prob=None,threshold=None,seed=None):
    # get uniform sample from observations below thresholds
    if seed is not None:
      np.random.seed(int(seed))
      
    n = int(n)

    #determine threshold values and probabilities
    threshold, exs_prob = self._format_th_params(threshold,exs_prob)
      
    #threshold = self.u if threshold is None else threshold
    # matrix with observations below the thresholds
    th = np.array(threshold)
    fX = np.array([row for row in self.X if np.all(row <= th)])
    
    rand_i = np.random.randint(low=0,high=fX.shape[0],size=n)
    return fX[rand_i,:]

    
  def simulate(self,n=1000,exs_prob=None,threshold=None,seed=1):
    """simulate net demand observations
    
    **Parameters**:

    `n` (`int`): number of samples

    `exs_prob` (`float`): Exceedance probability that characterises extreme value model region (assuming region bounds are equal in all components)

    `threshold` (`float`): threshold after which extreme value model region starts (in Gumbel scale, applied symmetrically to components). If exs_prob and threshold are specified, exs_prob takes precedence

    `seed` (`int`): random seed

    """
    np.random.seed(seed)
    n = int(n)

    #determine threshold values and probabilities
    threshold, exs_prob = self._format_th_params(threshold,exs_prob)
      
    # get margin thresholds corresponding to joint excess probability given by exs_prob, assuming both threshold values are equal
    
    
    # b[i] == 1 <==> i-th simulation comes from empirical portion of CDF
    empirical = np.random.binomial(1,1-exs_prob,n)

    n_empirical = np.sum(empirical)

    n_tails = n - n_empirical
    
    #s1 = self._simulate_exceedances(n_cond,g_th,seed)
    s1 = self._simulate_empirical(n_empirical,threshold = threshold)
    s2 = self._simulate_exceedances(n_tails,threshold = threshold)
    
    print("{n} samples from tail".format(n=n_tails))
    sim = np.concatenate((s1,s2),axis=0)
    np.random.shuffle(sim)
    return sim

  def _format_th_params(self,threshold,exs_prob):
    # format input into valid model thresholds and exceedance probabilities
    # returns conssitent threshold and exceedance probabilities
    
    if exs_prob is None and threshold is None:
      exs_prob = 1-self.cdf(self.u)
      threshold = self._from_gumbel((self.alpha*np.log(2) - np.log(-np.log(1-exs_prob))) * np.ones((1,2))).reshape((2,))
      print("Using marginal thresholds as bivariate model threshold")
    elif exs_prob is not None:
      if exs_prob > 1-self.cdf(self.u):
        print("exs_prob too low. Using margin thresholds as joint threshold value")
        exs_prob = 1-self.cdf(self.u)
      threshold = self._from_gumbel((self.alpha*np.log(2) - np.log(-np.log(1-exs_prob))) * np.ones((1,2))).reshape((2,))
    else:
      if np.any(threshold < self.u):
        print("thresholds too low. Using margin thresholds as joint threshold value")
        for i in range(self.d):
          threshold[i] = max(self.u[i],threshold[i])
      exs_prob = 1-self.cdf(threshold)

    return np.array(threshold), exs_prob


