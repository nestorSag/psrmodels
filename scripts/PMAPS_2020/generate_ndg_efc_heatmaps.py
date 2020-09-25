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
def get_gens_dfs(file_list,remove_larges_first):

  df_list = []
  for file in file_list:
    df = pd.read_csv(file,sep=" ").sort_values(ascending = not remove_larges_first, by = "Capacity")
    df["cumulative_cap"] = np.cumsum(df["Capacity"])
    df_list.append(df)

  return df_list

def draw_itc_efc(
  c,
  target_wpf,
  periods,
  policies=["veto","share"],
  metrics=["LOLE","EEU"],
  remove_larges_first=True,
  initial_windgen = [15000,3000],
  areas = ["GB","IRL"],
  rel_error_tol = 0.005):
  
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

  gen_dfs = get_gens_dfs([gb_gen_file,irl_gen_file],remove_larges_first)

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

    baseline_gens = [ConvGenDistribution(file) for file in [gb_gen_file,irl_gen_file]]

    for policy in policies:

      # arrange generator data 

      # get number of generators that need to be decomissioned and replaced by wind so that the wind 
      # capacity grows to a proportion of target_wpf with respect to the initial_windgen values

      # ndg = number of generators that have to be decomissioned to allow total wind capacity to 
      # scale in the desired proportion (say, 200% of its initial overall capacity)

      for metric in metrics:

        new_windcap_dict = {}
        removed_caps = [[],[]]
        axis_vals = [[],[]]

        print("year: {y}, metric: {m}, policy: {p}, ".format(y=year,m=metric,p=policy))
        def system_risk(alpha,gendist,demand,wind):
            #print(demand)
            obj = UnivariateHindcastMargin(gendist,demand - alpha*wind)
            metric_func = getattr(obj,metric)
            return metric_func()
        
        system_baseline_risk = [system_risk(1,baseline_gens[i],baseline_demand[:,i],baseline_wind[:,i]) for i in range(n_areas)]
      
        for j in range(n_areas):

          new_windcap_factor = 1
          i = 0
          n_gen = gen_dfs[j].shape[0]

          while new_windcap_factor < target_wpf and i <  n_gen:
            #### Setup smaller conventional generation system and replace with wind
            removed_capacity = np.sum(gen_dfs[j]["Capacity"][0:i])
            #print("removed capacity: {c}".format(c=removed_capacity))
            area_gens_df = gen_dfs[j].iloc[i::,]
            area_gendist = ConvGenDistribution(area_gens_df)

            # for this smaller conventiona system, find the amount of wind that replaces decomissioned generators
            # in terms of baseline risk
            def system_risk_diff(alpha):
              return system_risk(alpha,area_gendist,baseline_demand[:,j],baseline_wind[:,j]) - system_baseline_risk[j]

            system_risk_change = system_risk_diff(1)
            #print(system_baseline_risk[j])
            #print(system_risk_change)
            new_risk = system_risk_change + system_baseline_risk[j]
            print("baseline risk: {x}".format(x=system_baseline_risk[j]))
            #print("new risk: {x}".format(x=new_risk))

            if system_risk_change <= new_risk*rel_error_tol:
              new_windcap_factor = 1
            else:
              delta = removed_capacity/initial_windgen[j] 
              leftmost = 1
              rightmost = 1 + delta
              while system_risk_diff(rightmost) > 0:
                rightmost += delta

              #print("Finding equivalente wind factor")
              new_windcap_factor, info = bisect(f=system_risk_diff,a=leftmost,b=rightmost,full_output=True)
              #print("decom: {x}, [{a},{b}], est.: {f}".format(x=removed_capacity, a=leftmost,b=rightmost,f=new_windcap_factor))

            print("adjusted risk: {x}".format(x=system_risk_diff(new_windcap_factor) + system_baseline_risk[j]))
            area_wind = new_windcap_factor*baseline_wind[:,j]

            new_windcap_dict[(j,i)] = area_wind

            removed_caps[j].append(removed_capacity)
            axis_vals[j].append(initial_windgen[j]*new_windcap_factor/1000) #expressed in GW

            i += 1

        # once the hypothetical wind capacities corresponding to each of the smaller systems have been calculated, build 2D grid
        
        #but first save all the relvant info to a file
        flat_area = ["GB" for x in axis_vals[GB]] + ["IRL" for x in axis_vals[IRL]]
        flat_axis_vals = [x for x in axis_vals[GB]] + [x for x in axis_vals[IRL]]
        flat_removed_caps = [x for x in removed_caps[GB]] + [x for x in removed_caps[IRL]]

        iter_data = {"metric":metric,"year":year,"policy":policy,"removed_caps":flat_removed_caps,"added_wind_caps":flat_axis_vals,"area":flat_area}
        pd.DataFrame(iter_data).to_csv(DATA_FOLDER + "iterdata" + "itc_efc_vs_wp_heatmap_{p}policy_{y}year_{m}metric".format(m=metric,p=policy,y=year))
        

        print("Filling heatmap...")
        x_axis, y_axis = [np.array(entry) for entry in axis_vals]
        ndg = (len(x_axis),len(y_axis))
        itc_efc_vals = np.empty(list(reversed(ndg)))

        for axis in [0,1]:

          print("axis: {a}".format(a=axis_vals))
          print(new_windcap_dict.keys())
          print("ndg: {ndg}".format(ndg=ndg))

          area = ("GB" if axis == 0 else "IRL")
          
          for i in range(ndg[GB]):
            #### Setup smaller conventional generation system and replace with wind
            area_gens_df = gen_dfs[GB].iloc[i::,]
            area1_gendist = ConvGenDistribution(area_gens_df)

            for j in range(ndg[IRL]):
              print("axis: {a}, GB: {xi}, IRL: {xj}".format(xi=i,xj=j,a=axis))
              #### Setup smaller conventional generation system and replace with wind
              area_gens_df = gen_dfs[IRL].iloc[j::,]
              area2_gendist = ConvGenDistribution(area_gens_df)

              gens = [area1_gendist,area2_gendist]

              wind = np.array([new_windcap_dict[(GB,i)],new_windcap_dict[(IRL,j)]]).T

              bivariate_model = BivariateHindcastMargin(baseline_demand,wind,gens)

              # GB = x axis, so second coordinate in matrix notation
              val = bivariate_model.efc(c=c,policy=policy,metric=metric,axis=axis)
              print("axis: {a}, val: {v}, GB: {xi}, IRL: {xj}".format(xi=x_axis[i],xj=y_axis[j],v=val,a=axis))
              itc_efc_vals[j,i] = val
          
          np.save(DATA_FOLDER + "matrix_" + "itc_ndg_efc_vs_wp_heatmap_{p}policy_{y}year_{m}metric_{a}".format(m=metric,p=policy,y=year,a=area),itc_efc_vals)

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

          plt.savefig(FIGURE_FOLDER + "itc_ndg_efc_vs_wp_heatmap_{p}policy_{y}year_{m}metric_{a}.png".format(m=metric,p=policy,y=year,a=area),bbox_inches='tight')
          plt.close()

if __name__=="__main__":
  #periods = list(range(2007,2014))
  periods = ["all"]
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
