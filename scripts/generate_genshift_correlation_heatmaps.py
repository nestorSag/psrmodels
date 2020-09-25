from dfply import * 
from datetime import timedelta, datetime as dt
import pytz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from collections import OrderedDict

from scipy.optimize import bisect
from scipy.interpolate import interp1d

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri


from psrmodels.time_collapsed import BivariateHindcastMargin, ConvGenDistribution, UnivariateHindcastMargin

import sys, os
#sys.path.append('')
from helper_functions import *

FIGURE_FOLDER = "figures/PMAPS_2020/"
DATA_FOLDER = "data/PMAPS_2020/"
N_SAMPLE = 25000

# this script creates heatmaps of the IRL-GB power margins' Gaussian copula correlation parameter plotted on a 2D plane where
# the x and y axis are the wind generation capacity in GB and IRL respectively. It shifts generation
# negatively 


# this class takes a shifted generation distribution and caps it to have a minimum value of zero
# so it doesn't allow negative generation values
# this is so we can shift distribution negatively without negative generation as a side effect
class ClippedGenDist(ConvGenDistribution):

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


def draw_itc_efc(
  c,
  target_wpf,
  periods,
  grid_size = 10,
  policies=["veto","share"],
  metrics=["LOLE","EEU"],
  initial_windgen = [15000,3000],
  areas = ["GB","IRL"],
  rel_error_tol = 0.01):
  
  n_areas = 2
  GB = 0
  IRL = 1
  # set baseline objects
  data_path = "../../../data/energy/uk_ireland/InterconnectionData_Rescaled.txt"

  rpy2.robjects.numpy2ri.activate()

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

      #baseline_demand = [gbdem_r, idem_r]
      #baseline_wind = [gbwind_r, iwind_r]

    baseline_gens = [ConvGenDistribution(file) for file in [gb_gen_file,irl_gen_file]]

    for policy in policies:

      for metric in metrics:

        new_genshift_dict = {}
        axis_vals = [[],[]]
        wpfs = [np.empty(grid_size+1,) for i in range(n_areas)]

        def system_risk(alpha,gendist,demand,wind):
            clipped_gen = ClippedGenDist(gendist + alpha)
            obj = UnivariateHindcastMargin(clipped_gen,demand - wind)
            metric_func = getattr(obj,metric)
            mf = metric_func()
            gendist += (-gendist.fc) #reset fc
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
            print("baseline risk: {x}".format(x=system_baseline_risk[j]))
            if np.abs(system_risk_change) <= new_risk*rel_error_tol:
              new_genshift_factor = 0
            else:
              delta = 250
              rightmost = 0
              leftmost = - delta
              while system_risk_diff(leftmost) < 0:
                leftmost -= delta

              new_genshift_factor, info = bisect(f=system_risk_diff,a=leftmost,b=rightmost,full_output=True,xtol=1)

            new_genshift_dict[(j,i)] = new_genshift_factor
        # once the hypothetical wind capacities corresponding to each of the smaller systems have been calculated, build 2D grid
        
        print("Filling heatmap...")
        x_axis, y_axis = axis_vals
        itc_efc_vals = np.empty((grid_size+1,grid_size+1))

        for i in range(grid_size+1):
          #### Setup smaller conventional generation system and replace with wind
          #print("check 1")
          #baseline_gens[GB].equals(aux_baseline_gens[GB])
          #print("check 2")
          #baseline_gens[GB].equals(aux_baseline_gens[GB])
          area1_gendist = ClippedGenDist(baseline_gens[GB] + (new_genshift_dict[(GB,i)]))
          baseline_gens[GB] += (-baseline_gens[GB].fc)
          #print("check 3")
          #baseline_gens[GB].equals(aux_baseline_gens[GB])

          for j in range(grid_size+1):
            #### Setup smaller conventional generation system and replace with wind
            baseline_gens[IRL] += (-baseline_gens[IRL].fc)
            area2_gendist = ClippedGenDist(baseline_gens[IRL] + new_genshift_dict[(IRL,j)])

            gens = [area1_gendist,area2_gendist]

            demand = baseline_demand

            wind = np.array([wpfs[GB][i]*baseline_wind[:,GB],wpfs[IRL][j]*baseline_wind[:,IRL]]).T

            bivariate_model = BivariateHindcastMargin(demand,wind,gens)

            #get shortfall region sample in power margin scale
            sample = bivariate_model.simulate_region(n=N_SAMPLE,m=(0,0),policy="veto",c=0,intersection=False)

            U = np.empty((sample.shape[0],2))

            for component in (0,1):
              lims = {"min":min(sample[:,component]),"max":max(sample[:,component])}
              grid = np.linspace(start=lims["min"] - 1,stop=lims["max"] + 1,num=int((lims["max"] - lims["min"] + 2)/10) + 1)
              cdf = np.array([bivariate_model.margin_cdf(m,i=component) for m in grid])
              interpolator = interp1d(x=grid,y=cdf)
              U[:,component] = interpolator(sample[:,component])

            # call R copula functions to fit the models

            nr,nc = U.shape
            Ur = ro.r.matrix(U, nrow=nr, ncol=nc)
            ro.r.assign("U", Ur)

            val = ro.r("""
            # if not installed, install copula package

            list.of.packages <- c('copula')
            new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,'Package'])]
            if(length(new.packages)) install.packages(new.packages)

            library(copula)
            gc = normalCopula()
            tcfit = summary(fitCopula(gc,U))
            fitted = tcfit$coefficients[1]
              """)

            print("val: {x}".format(x=val[0]))

            itc_efc_vals[j,i] = val[0]
          
          np.save(DATA_FOLDER + "matrix_" + "itc_genshift2_correlation_heatmap_{y}year_{m}metric".format(m=metric,y=year),itc_efc_vals)

          plt.figure(figsize=(8,8))

          plt.rc('xtick',labelsize=18)
          plt.rc('ytick',labelsize=18)
          contours = plt.contourf(x_axis,y_axis,itc_efc_vals)
          plt.contour(contours)
          plt.xlabel("GB wind capacity (GW)",fontsize=20)
          plt.ylabel("IRL wind capacity (GW)",fontsize=20)
          cbar = plt.colorbar(contours)
          cbar.ax.tick_params(labelsize=14)
          cbar.ax.set_title(label="Rho",fontdict={'fontsize':18})

          plt.savefig(FIGURE_FOLDER + "itc_genshift2_correlation_heatmap_{y}year_{m}metric.png".format(m=metric,y=year),bbox_inches='tight')
          plt.close()

def draw_itc_efc_par(par_list):
  c, target_wpf, periods, policies, metrics = par_list
  draw_itc_efc(c=c,target_wpf=target_wpf,periods=periods,policies=policies,metrics=metrics)

if __name__=="__main__":
  periods = [2010]
  policies = ["veto"]
  metrics = ["LOLE","EEU"]
  
  # par_list_list = [
  #   [0,2,["all"],["veto"],["LOLE"]],
  #   [0,2,["all"],["share"],["LOLE"]],
  #   [0,2,["all"],["veto"],["EEU"]],
  #   [0,2,["all"],["share"],["EEU"]]]

  # par = Pool(4)
  # par.map(draw_itc_efc_par,par_list_list)

  draw_itc_efc(
   c = 0,
   grid_size=1,
   target_wpf=2,
   periods=periods,
   policies=policies,
   metrics=metrics)
