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

from psrmodels.time_collapsed import BivariateHindcastMargin, ConvGenDistribution, UnivariateHindcastMargin

from helper_functions import *

FIGURE_FOLDER = "figures/PMAPS_2020/"
DATA_FOLDER = "data/PMAPS_2020/"

# this script creates heatmaps of the IRL-GB interconnector's EFC plotted on a 2D plane where
# the x and y axis are the wind generation capacity in GB and IRL respectively, assuming such capacity
# replaces conventional generation gradually so as to keep a given risk metric fixed 

## to avoid the lumpiness of removing generators, demand is rescaleda s well as wind

def draw_itc_efc(
  c,
  target_wpf,
  periods,
  grid_size = 10,
  policies=["veto","share"],
  metrics=["LOLE","EEU"],
  remove_larges_first=True,
  initial_windgen = [15000,3000],
  areas = ["GB","IRL"],
  rel_error_tol = 0.01):
  
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
  # ndg = []

  # for i in range(n_areas):
  #   m = gen_dfs[i].query("cumulative_cap > {x}".format(x=initial_windgen[i]*(target_wpf-1))).shape[0]
  #   ndg.append(gen_dfs[i].shape[0] - m)

  # print("Grid size: {m} x {n}".format(m=ndg[0],n=ndg[1]))
  for year in periods:

    if year != "all":
      baseline_demand = np.array(df.query("period == {p}".format(p=year))[["gbdem_r","idem_r"]])
      baseline_wind = np.array(df.query("period == {p}".format(p=year))[["gbwind_r","iwind_r"]])
    else:
      baseline_demand = np.array(df[["gbdem_r","idem_r"]])
      baseline_wind = np.array(df[["gbwind_r","iwind_r"]])

      #baseline_demand = [gbdem_r, idem_r]
      #baseline_wind = [gbwind_r, iwind_r]

    print(baseline_demand.shape)
    baseline_gens = [ConvGenDistribution(file) for file in [gb_gen_file,irl_gen_file]]

    for policy in policies:

      for metric in metrics:

        new_genrescale_dict = {}
        axis_vals = [[],[]]
        wpfs = [np.empty(grid_size+1,) for i in range(n_areas)]

        print("year: {y}, metric: {m}, policy: {p}, ".format(y=year,m=metric,p=policy))
        def system_risk(alpha,gendist,demand,wind):
            #print(demand)
            obj = UnivariateHindcastMargin(gendist * alpha,demand - wind)
            metric_func = getattr(obj,metric)
            mf = metric_func()
            #print("alpha: {a}, rescaling: {rf}, mf: {mf}".format(a=alpha,rf=gendist.rescaling_factor,mf=mf))
            gendist *= (1.0/gendist.rescaling_factor) #reset fc
            #print("new rescaling factor (should be 1): {x}".format(x=gendist.rescaling_factor))
            return mf
        
        system_baseline_risk = [system_risk(1,baseline_gens[i],baseline_demand[:,i],baseline_wind[:,i]) for i in range(n_areas)]
      
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

            system_risk_change = system_risk_diff(1)
            #print(system_risk_change)
            new_risk = system_risk_change + system_baseline_risk[j]
            print("baseline risk: {x}".format(x=system_baseline_risk[j]))
            print("new risk: {x}".format(x=new_risk))
            if np.abs(system_risk_change) <= new_risk*rel_error_tol:
              new_genrescale_factor = 1
            else:
              delta = (grid_stepsize) * 0.25 / area_gendist.max
              rightmost = 1
              leftmost = 1 - delta
              while system_risk_diff(leftmost) < 0:
                leftmost -= delta

              #print("gen rescaling factor: {x}".format(x=area_gendist.rescaling_factor))
              #print("src after = {x}".format(x=system_risk_diff(1)))
              #print("[{a},{b}] => [{x},{y}]".format(a=leftmost,b=rightmost,x=system_risk_diff(leftmost),y=system_risk_diff(rightmost)))
              new_genrescale_factor, info = bisect(f=system_risk_diff,a=leftmost,b=rightmost,full_output=True)

            print("adjusted risk: {x}".format(x=system_risk_diff(new_genrescale_factor) + system_baseline_risk[j]))

            new_genrescale_dict[(j,i)] = new_genrescale_factor

        # once the hypothetical wind capacities corresponding to each of the smaller systems have been calculated, build 2D grid
        
        print("Filling heatmap...")
        x_axis, y_axis = axis_vals
        itc_efc_vals = np.empty((grid_size+1,grid_size+1))

        for axis in [0,1]:

          #print("axis: {a}".format(a=axis_vals))
          #print(new_windcap_dict.keys())
          #print("ndg: {ndg}".format(ndg=ndg))

          area = ("GB" if axis == 0 else "IRL")
          
          for i in range(grid_size+1):
            #### Setup smaller conventional generation system and replace with wind
            baseline_gens[GB] *= (1.0/baseline_gens[GB].rescaling_factor)

            area1_gendist = baseline_gens[GB] * (new_genrescale_dict[(GB,i)])

            for j in range(grid_size+1):
              print("axis: {a}, GB: {xi}, IRL: {xj}".format(xi=i,xj=j,a=axis))
              #### Setup smaller conventional generation system and replace with wind
              baseline_gens[IRL] *= (1.0/baseline_gens[IRL].rescaling_factor)
              area2_gendist = baseline_gens[IRL] * (new_genrescale_dict[(IRL,j)])

              gens = [area1_gendist,area2_gendist]

              demand = baseline_demand

              wind = np.array([wpfs[GB][i]*baseline_wind[:,GB],wpfs[IRL][j]*baseline_wind[:,IRL]]).T

              bivariate_model = BivariateHindcastMargin(demand,wind,gens)

              # GB = x axis, so second coordinate in matrix notation
              val = bivariate_model.efc(c=c,policy=policy,metric=metric,axis=axis)
              print("axis: {a}, val: {v}, GB: {xi}, IRL: {xj}".format(xi=x_axis[i],xj=y_axis[j],v=val,a=axis))
              itc_efc_vals[j,i] = val
          
          np.save(DATA_FOLDER + "matrix_" + "itc_genrescale_efc_vs_wp_heatmap_{p}policy_{y}year_{m}metric_{a}".format(m=metric,p=policy,y=year,a=area),itc_efc_vals)

          plt.figure(figsize=(8,8))

          plt.rc('xtick',labelsize=18)
          plt.rc('ytick',labelsize=18)
          contours = plt.contourf(x_axis,y_axis,itc_efc_vals)
          plt.contour(contours)
          plt.xlabel("GB wind capacity (GW)",fontsize=20)
          plt.ylabel("IRL wind capacity (GW)",fontsize=20)
          cbar = plt.colorbar(contours)
          cbar.ax.tick_params(labelsize=14)
          cbar.ax.set_title(label="EFC",fontdict={'fontsize':18})

          plt.savefig(FIGURE_FOLDER + "itc_genrescale_efc_vs_wp_heatmap_{p}policy_{y}year_{m}metric_{a}.png".format(m=metric,p=policy,y=year,a=area),bbox_inches='tight')
          plt.close()

if __name__=="__main__":
  #periods = list(range(2007,2014))
  periods = ["all"]
  #periods = list(range(2007))
  #periods = [2007]
  #wp_factors = [1 + x/10.0 for x in range(11)]
  policies = ["veto","share"]
  metrics = ["LOLE","EEU"]
  
  draw_itc_efc(
    c = 1000,
    target_wpf=2,
    periods=periods,
    policies=policies,
    metrics=metrics,
    remove_larges_first=True)
