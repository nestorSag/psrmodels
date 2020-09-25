from dfply import * 
from datetime import timedelta, datetime as dt
import pytz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from collections import OrderedDict

from scipy.optimize import bisect

import sys, os
#sys.path.append('..')

from psrmodels.time_collapsed import *

from helper_functions import *

from multiprocessing import Pool

FIGURE_FOLDER = "figures/EU_analysis/"
DATA_FOLDER = "data/EU_analysis/"

# this script creates heatmaps of the IRL-GB interconnector's EFC plotted on a 2D plane where
# the x and y axis are the wind generation capacity in GB and IRL respectively. It shifts generation
# negatively 


# this class takes a shifted generation distribution and caps it to have a minimum value of zero
# so it doesn't allow negative generation values
# this is so we can shift distribution negatively without negative generation as a side effect
class ClippedGenDist(ConvGenDistribution):

  def __init__(self,gendist):

    self.cdf_vals = np.ascontiguousarray(gendist.cdf_vals)
    self.expectation_vals = np.ascontiguousarray(gendist.expectation_vals)
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


def compute(
  caps,
  target_wpf,
  periods,
  grid_size = 20,
  rel_error_tol = 0.01):
  
  ev_results = []
  cond_resuts = []
  policies=["veto","share"]
  metrics=["LOLE"]
  initial_windgen = [15000,3000]
  areas = ["GB","IRL"]

  n_areas = 2
  GB = 0
  IRL = 1
  # set baseline objects
  data_path = "../../../data/energy/uk_ireland/InterconnectionData_Rescaled.txt"

  #ceil demands and floor wind to avoid underestimating risk
  if not os.path.exists(FIGURE_FOLDER):
    os.makedirs(FIGURE_FOLDER)

  if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

  df = read_data(data_path) >> mutate(
    gbdem_r = X.gbdem_r.apply(lambda x: int(np.ceil(x))),
    idem_r = X.idem_r.apply(lambda x: int(np.ceil(x))),
    gbwind_r = X.gbwind_r.apply(lambda x: int(np.floor(x))),
    iwind_r = X.iwind_r.apply(lambda x: int(np.floor(x))))

  gb_gen_file = "../../../data/energy/uk/generator_data.txt"
  irl_gen_file = "../../../data/energy/ireland/generator_data.txt"

  gb_gen_df = pd.read_csv(gb_gen_file,sep=" ")
  irl_gen_df = pd.read_csv(irl_gen_file,sep=" ")

  for year in periods:

    if year != "all":
      baseline_demand = np.array(df.query("period == {p}".format(p=year))[["gbdem_r","idem_r"]])
      baseline_wind = np.array(df.query("period == {p}".format(p=year))[["gbwind_r","iwind_r"]])
    else:
      baseline_demand = np.array(df[["gbdem_r","idem_r"]])
      baseline_wind = np.array(df[["gbwind_r","iwind_r"]])

    baseline_gens = [ConvGenDistribution(file) for file in [gb_gen_file,irl_gen_file]]

    for metric in metrics:

      new_genshift_dict = {}
      axis_vals = [[],[]]
      wpfs = [np.empty(grid_size+1,) for i in range(n_areas)]

      def system_risk(alpha,gendist,demand,wind):
          #print(demand)
          clipped_gen = ClippedGenDist(gendist + alpha)
          obj = UnivariateHindcastMargin(clipped_gen,demand - wind)
          metric_func = getattr(obj,metric)
          mf = metric_func()
          #print("alpha: {a}, rescaling: {rf}, mf: {mf}".format(a=alpha,rf=gendist.rescaling_factor,mf=mf))
          gendist += (-gendist.fc) #reset fc
          #print("new rescaling factor (should be 1): {x}".format(x=gendist.rescaling_factor))
          return mf
      
      system_baseline_risk = [system_risk(0,baseline_gens[i],baseline_demand[:,i],baseline_wind[:,i]) for i in range(n_areas)]
    
      for j in range(n_areas):

        grid_stepsize = initial_windgen[j]*(target_wpf - 1)/grid_size
        grid_points = [initial_windgen[j] + i*grid_stepsize for i in range(grid_size+1)]

        axis_vals[j] = np.array([x/1000 for x in grid_points]) #expressed in GW

        for i in range(grid_size+1):
          #### Setup smaller conventional generation system and replace with wind
          area_gendist = baseline_gens[j]

          # for this smaller conventiona system, find the amount of wind that replaces decomissioned generators
          # in terms of baseline risk
          added_wpf = grid_points[i]/initial_windgen[j]
          wpfs[j][i] = added_wpf
          area_wind = added_wpf*baseline_wind[:,j]

          def system_risk_diff(alpha):
            return system_risk(alpha,area_gendist,baseline_demand[:,j],area_wind) - system_baseline_risk[j]

          system_risk_change = system_risk_diff(0)
          #print(system_risk_change)
          new_risk = system_risk_change + system_baseline_risk[j]
          #print("baseline risk: {x}".format(x=system_baseline_risk[j]))
          #print("new risk: {x}".format(x=new_risk))
          if np.abs(system_risk_change) <= new_risk*rel_error_tol:
            new_genshift_factor = 0
          else:
            delta = 250
            rightmost = 0
            leftmost = - delta
            while system_risk_diff(leftmost) < 0:
              leftmost -= delta

            new_genshift_factor, info = bisect(f=system_risk_diff,a=leftmost,b=rightmost,full_output=True,xtol=1)

          ##print("adjusted risk: {x}".format(x=system_risk_diff(new_genshift_factor) + system_baseline_risk[j]))

          new_genshift_dict[(j,i)] = new_genshift_factor

      # once the hypothetical wind capacities corresponding to each of the smaller systems have been calculated, build 2D grid
      x_axis, y_axis = axis_vals
      itc_efc_vals = np.empty((grid_size+1,grid_size+1))
      
      for i in range(grid_size+1):
        #### Setup smaller conventional generation system and replace with wind
        baseline_gens[GB] += (-baseline_gens[GB].fc)

        area1_gendist = ClippedGenDist(baseline_gens[GB] + (new_genshift_dict[(GB,i)]))

        for j in range(grid_size+1):
          #### Setup smaller conventional generation system and replace with wind
          baseline_gens[IRL] += (-baseline_gens[IRL].fc)
          area2_gendist = ClippedGenDist(baseline_gens[IRL] + new_genshift_dict[(IRL,j)])

          gens = [area1_gendist,area2_gendist]

          demand = baseline_demand

          wind = np.array([wpfs[GB][i]*baseline_wind[:,GB],wpfs[IRL][j]*baseline_wind[:,IRL]]).T

          bivariate_model = BivariateHindcastMargin(demand,wind,gens)
          explorer = EnergyUnservedExplorer(bivariate_model)

          print("i: {i}, j: {j}".format(i=i,j=j))
          for c in caps:
            for policy in policies:
              ev_info = explorer.fit_marginal_models(n=10000,c=c,policy=policy)
              ev_info["extremal_coef"] = explorer.extremal_coefficient(c=c,policy=policy)
              ev_info["c"] = c
              ev_info["policy"] = policy
              ev_info["gb_wpf"] = wpfs[GB][i]
              ev_info["irl_wpf"] = wpfs[IRL][j]
              ev_results.append(ev_info)

          for k in range(2):
            cond_info = {}
            cond_info["shortfall_area"] = k
            for v in np.linspace(0,1000,6):
              cond_info["shortfall_size"] = v
              cond_vals = bivariate_model.simulate_conditional(n=5000,cond_value=-v,cond_axis=k,c=0,policy="veto")
              cond_info["pm_mean"] = np.mean(cond_vals)
              cond_info["pm_q975"] = np.quantile(cond_vals,0.975)
              cond_info["pm_q025"] = np.quantile(cond_vals,0.025)
              cond_results.append(cond_info)
                # except:
                #   print("something went wrong: year: {y}, gb_wpf: {gbw}, irl_wpf: {irlw}, c: {c}, policy: {policy}".format(y=year,gbw=wpfs[GB][i],irlw=wpfs[IRL][j],c=c,policy=policy))
                
              
              
              ##print("appended")

      df = pd.DataFrame(ev_results)
      df.to_csv(DATA_FOLDER + "ev_results_{y}.csv".format(y=str(periods)))

      df = pd.DataFrame(cond_results)
      df.to_csv(DATA_FOLDER + "cond_results_{y}.csv".format(y=str(periods)))

def compute_par(par_list):
  caps, target_wpf, periods, grid_size = par_list
  compute(caps=caps,target_wpf=target_wpf,periods=periods,grid_size=grid_size)

if __name__=="__main__":
  caps = np.linspace(0,2500,6)
  #caps = [0]
  target_wp=2
  years=range(2007,2014)
  # compute(
  #  caps = [500],
  #  target_wpf=2,
  #  periods=[2007],
  #  grid_size=10)

  par_list_list = [[caps,2,[year],10] for year in years]

  par = Pool(4)
  par.map(compute_par,par_list_list)
