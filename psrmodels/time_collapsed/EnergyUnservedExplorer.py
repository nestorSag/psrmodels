import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import bisect, minimize, NonlinearConstraint

from scipy.stats import genpareto

class EnergyUnservedExplorer(object):

	"""This class outputs some exploratory plots and statistics from extreme value theory, for a bivariate hindcast power margin distribution
  
  **Parameters**:

  `bivariate_hindcast_model` (`BivariateHindcastMargin`): instance of `BivariateHindcastMargin`

  """

	def __init__(self,bivariate_hindcast_model):

		self.quantile_grid_delta = 0.05

		self.set_model(bivariate_hindcast_model)

	def set_model(self,model):
		self.precomputed_quantiles = {}
		self.precomputed_nc = {}
		self.hindcast = model
		self.hindcast_limits = [-np.max(self.hindcast.net_demand[:,i]) for i in range(2)]

	### implictly work with the negative of the margins so that
	### energy unserved can be treated as a right side tail
	def _inverted_hindcast_margin_cdf(self,x):
		x0, x1 = x
		return 1 - self.hindcast.marginal_cdf(-x0-1,i=0) - self.hindcast.marginal_cdf(-x1-1,i=1) + self.hindcast.cdf(-np.array(x)-1)
	
	def system_eu_ecdf(self,x):
		return self._inverted_hindcast_margin_cdf(x) - self._inverted_hindcast_margin_cdf(np.minimum(0,x))
	
	def _inverted_hindcast_marginal_cdf(self,x,i=0):
		return 1 - self.hindcast.marginal_cdf(-x-1,i=i)

	def system_eu_mecdf(self,x,i=0):
		return self._inverted_hindcast_marginal_cdf(x,i) - self._inverted_hindcast_marginal_cdf(min(x,0),i)

	# create quantile grid
	def _create_quantile_grid(self,c,policy):
		# create or get dictionary with quantiles of marginal tail cdfs
		if (c,policy) not in self.precomputed_quantiles.keys():
			Q = np.linspace(0,1,int(1/self.quantile_grid_delta)+1) #5% intervals
			quantiles = np.zeros((len(Q),2))
			#print("precomputing quantiles...")
			for i in range(quantiles.shape[0]):
				for j in range(2):
					quantiles[i,j] = self._jtemq(Q[i],j,c,policy)

			self.precomputed_quantiles[(c,policy)] = quantiles
			#print("finished...")

	# get else create normalising constant (joint tail probability)
	def _get_else_create_nc(self,c,policy):
		# create or get dictionary with quantiles of marginal tail cdfs
		if (c,policy) not in self.precomputed_nc.keys():
			nc = self.hindcast.cdf((-1,-1),c=c,policy=policy)
			self.precomputed_nc[(c,policy)] = nc
		else:
			nc = self.precomputed_nc[(c,policy)]

		return nc  

	# joint tail empirical marginal quantile function
	def _jtemq(self,q,i,c,policy):
		if q == 0:
			return 0
		elif q == 1:
			return max(self.hindcast.net_demand[:,i])
		else:
			if (c,policy) in self.precomputed_quantiles.keys():
				quantiles = self.precomputed_quantiles[(c,policy)][:,i]
				b_idx = int(np.ceil(q/self.quantile_grid_delta))
				b = quantiles[b_idx]
				a = quantiles[b_idx-1] if b_idx > 0 else 0
			else:
				a = 0
				b = max(self.hindcast.net_demand[:,i])

			f = lambda k: -q + self._jtemcdf(k,i=i,c=c,policy=policy)

			#print("q: {q}, a: {a}, b: {b}".format(q=q,a=a,b=b))
			#print("f(a): {a}, f(b): {b}".format(a=f(a),b=f(b)))
			r = bisect(f= f,a=a,b=b,xtol=0.5)
			return int(r)

	# joint tail empirical marginal cdf
	def _jtemcdf(self,m,i,c,policy):
		nc = self._get_else_create_nc(c,policy)
		if i == 0:
			m_ = (-m-1,-1)
		else:
			m_ = (-1,-m-1)
		#
		#print(m_)
		return 1 - self.hindcast.cdf(m=m_,c=c,policy=policy)/nc

	# joint tail empirical cdf
	def _jtecdf(self,x,c,policy):
		nc = self._get_else_create_nc(c,policy)
		x0, x1 = x
		area_prob =\
		  self.hindcast.cdf(m=(-1,-1),c=c,policy=policy) -\
		  self.hindcast.cdf(m=(-x0-1,-1),c=c,policy=policy) -\
		  self.hindcast.cdf(m=(-1,-x1-1),c=c,policy=policy) +\
		  self.hindcast.cdf(m=-np.array(x)-1,c=c,policy=policy)
		return area_prob/nc

	def pickands(self,t,c,policy):
		"""Computes Pickands dependence function for an array of values in [0,1], for the distribution of power margins conditioned to both areas having a shortfall
  
	  **Parameters**:

	  `t` (`numpy.ndarray`): Array of values in [0,1]

	  `c` (`int`): Interconnection capacity

	  `policy` (`string`): Either 'veto' or 'share'

	  """
		q0 = self._jtemq(np.exp(-(1-t)),i=0,c=c,policy=policy)
		q1 = self._jtemq(np.exp(-t),i=1,c=c,policy=policy)
		return -np.log(self._jtecdf(x=(q0,q1),c=c,policy=policy))

	def plot_empirical_pickands(self,c=0,policy="veto",grid_size=30):
		"""Creates a plot of the Pickands dependence function for the post-interconnection power margin distribution conditional distribution `m[0] <= m_0 AND m[1] <= m_1` (both areas having a shortfall)
  
	  **Parameters**:

	  `c` (`int`): Interconnection capacity

	  `policy` (`string`): Either 'veto' or 'share'

	  `grid_size` (`int`): Number of points in the [0,1] grid


	  """
		self._create_quantile_grid(c,policy)

		pickands = []
		T = np.linspace(0,1,grid_size)
		for t in T:
			pickands.append(self.pickands(t,c,policy))

		pickands[0] = 1
		pickands[-1] = 1
		sns.lineplot(T,pickands)
		# add pickand function bounds
		plt.plot([0,0.5], [1,0.5], '--', linewidth=1,color="grey")
		plt.plot([0.5,1], [0.5,1], '--', linewidth=1,color="grey")
		plt.plot([0,1], [1,1], '--', linewidth=1,color="grey")
		plt.show()

	def extremal_coefficient(self,c,policy):
		"""Returns the extremal coefficient for the post-interconnection power margin distribution conditioned to `m[0] <= m_0 AND m[1] <= m_1`
  
	  **Parameters**:

	  `c` (`int`): Interconnection capacity

	  `policy` (`string`): Either 'veto' or 'share'


	  """
		return self.pickands(0.5,c=c,policy=policy)

	def fit_marginal_models(self,n=100,c=0,policy="veto",qq_plots=False,seed=1):

		"""Fit Generalised Pareto models to the negative portion of post-interconnection power margin distributions and returns fitted parameters
  
	  **Parameters**:

	  `n` (`int`): Number of simulated samples with which to fit the models

	  `c` (`int`): Assumed interconnection capacity

	  `policy` (`string`): Either 'veto' or 'share'

	  `qq_plots` (`bool`): if `True`, outputs QQ-plots of fitted models

	  `seed` (`int`): random seed


	  """
		a1_shortfalls = self.hindcast.simulate_region(n=n,m=(-1,np.Inf),c=c,policy=policy)
		a2_shortfalls = self.hindcast.simulate_region(n=n,m=(np.Inf,-1),c=c,policy=policy)

		a1_shortfalls = -a1_shortfalls[:,0]
		a2_shortfalls = -a2_shortfalls[:,1]

		pars0 = self._cdf_univar_gp_fitter(a1_shortfalls,upper_endpoint_lb = -self.hindcast_limits[0])
		pars1 = self._cdf_univar_gp_fitter(a2_shortfalls,upper_endpoint_lb = -self.hindcast_limits[1])

		a1_scale_est = np.exp(pars0.x[0])
		a1_shape_est = pars0.x[1]

		a2_scale_est = np.exp(pars1.x[0])
		a2_shape_est = pars1.x[1]

		#a1_scale_std = np.exp(pars0.hess_inv[0,0]/np.sqrt(n))
		#a1_shape_std = pars0.hess_inv[1,1]/np.sqrt(n)

		#a2_scale_std = np.exp(pars1.hess_inv[0,0]/np.sqrt(n))
		#a2_shape_std = pars1.hess_inv[1,1]/np.sqrt(n)

		if qq_plots:
			q_grid = np.linspace(0.01,0.99,99)
			eq1 = np.quantile(a1_shortfalls,q = q_grid)
			eq2 = np.quantile(a2_shortfalls,q = q_grid)

			fq1 = genpareto(loc=0,scale=a1_scale_est,c=a1_shape_est).ppf(q=q_grid)
			fq2 = genpareto(loc=0,scale=a2_scale_est,c=a2_shape_est).ppf(q=q_grid)

			fig = plt.figure(figsize=(5,5))
			ax = fig.add_subplot(211)
			ax.scatter(eq1, fq1, color = '#004C99')
			ax.plot([0, max(eq1)], [0, max(eq1)], '--', color = '#FF8000')
			#ax.xlim(lineStart, lineEnd)
			#ax.ylim(lineStart, lineEnd)
			ax.set_xlabel('Empirical quantiles')
			ax.set_ylabel('Fitted quantiles')
			ax.set_title('Area 1')
			##plt.axis('scaled')

			ax = fig.add_subplot(212)
			ax.scatter(eq2, fq2, color = '#004C99')
			ax.plot([0, max(eq2)], [0, max(eq2)], '--', color = '#FF8000')
			#ax.xlim(lineStart, lineEnd)
			#ax.ylim(lineStart, lineEnd)
			ax.set_xlabel('Empirical quantiles')
			ax.set_ylabel('Fitted quantiles')
			ax.set_title('Area 2')
			##plt.axis('scaled')

			plt.tight_layout()
			plt.show()


		# res = {
		#   "area1":{
		#     "pars":{
		#       "scale":a1_scale_est,
		#       "c":a1_shape_est
		#     },
		#     "std":{
		#       "scale":a1_scale_std,
		#       "c":a1_shape_std
		#     }
		#   },
		#   "area2":{
		#     "pars":{
		#       "scale":a2_scale_est,
		#       "c":a2_shape_est
		#     },
		#     "std":{
		#       "scale":a2_scale_std,
		#       "c":a2_shape_std
		#     }
		#   }
		# }

		res = {
		  "a1_scale":a1_scale_est,
		  "a1_shape":a1_shape_est,
		  "a2_scale":a2_scale_est,
		  "a2_shape":a2_shape_est}

		return res

	def _cdf_univar_gp_fitter(self,x,upper_endpoint_lb=None):

		def loglikelihood(par):
			logscale = par[0]
			shape = par[1]
			dist = genpareto(loc=0,scale=np.exp(logscale),c=shape)
			return -np.mean(dist.logpdf(x))

		def upper_endpoint_constraint(par):
			logscale = par[0]
			shape = par[1]
			return np.exp(logscale) - max(-shape*upper_endpoint_lb,0)

		logscale0 = np.log(np.std(x))
		shape0 = 0

		par0 = np.array([logscale0,shape0])

		if upper_endpoint_lb is not None:
			const = cons = {'type':'ineq', 'fun': upper_endpoint_constraint, 'lb': 0, 'ub': np.Inf}
		
		res = minimize(
			fun=loglikelihood,
			x0=par0,
			constraints = () if upper_endpoint_lb is None else const)


		return res



