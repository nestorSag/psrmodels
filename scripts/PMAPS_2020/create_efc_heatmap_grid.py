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

sys.path.append('scripts/')
from helper_functions import *

FIGURE_FOLDER = "figures/PMAPS_2020/"
DATA_FOLDER = "data/PMAPS_2020/"

GBWINDCAP = 15000
IRLWINDCAP = 3000

def draw_grid(
  x_axis = None, #have to be dictionary
  y_axis = None, #have to be dictionary
  years = [2010],
  policies=["veto","share"],
  metrics=["LOLE","EEU"],
  areas = ["GB","IRL"],
  prefix="_netdem"):
  
  for year in years:
    for area in areas:
      i = 1
      plt.figure(figsize=(8,8))
      for metric in metrics:
        for policy in policies:
          plot_id = int(str(len(metrics)) + str(len(policies)) + str(i))

          itc_efc_vals = np.load(DATA_FOLDER + "matrix_" + "itc{prefix}_efc_vs_wp_heatmap_{p}policy_{y}year_{m}metric_{a}.npy".format(m=metric,p=policy,y=year,a=area,prefix=prefix))

          plt.subplot(plot_id)
          plt.title("Policy: {p}, metric: {m}".format(p=policy,m=metric, a = area))
          #plt.rc('xtick',labelsize=18)
          #plt.rc('ytick',labelsize=18)
          x_axis_ = np.linspace(1,2,itc_efc_vals.shape[1])*GBWINDCAP if x_axis is None else x_axis
          y_axis_ = np.linspace(1,2,itc_efc_vals.shape[0])*IRLWINDCAP if y_axis is None else y_axis

          #print(x_axis)
          #print(y_axis)
          # if area == "IRL":
          #   vmin = 0
          #   vmax = 1000
          # else:
          #   vmin = 590
          #   vmax = 700
          # #print(itc_efc_vals.shape)
          # contours = plt.contourf(x_axis_,y_axis_,itc_efc_vals,vmin=vmin,vmax=vmax)
          # plt.contour(contours,vmin=vmin,vmax=vmax)

          contours = plt.contourf(x_axis_/1000,y_axis_/1000,itc_efc_vals/1000)
          plt.contour(contours)
          plt.xlabel("GB wind capacity (GW)")
          plt.ylabel("IRL wind capacity (GW)")
          cbar = plt.colorbar(contours)
          #cbar.ax.tick_params(labelsize=14)
          cbar.ax.set_title(label="EFC")
          i += 1
      #plt.show()
      plt.tight_layout()
      plt.savefig(FIGURE_FOLDER + "itc{prefix}_efc_vs_wp_heatmap_grid_{y}year_{a}.png".format(y=year,a=area,prefix=prefix),bbox_inches='tight')
      plt.close()



if __name__=="__main__":
  #periods = list(range(2007,2014))
  periods = ["all"]
  #periods = [2007]
  #wp_factors = [1 + x/10.0 for x in range(11)]
  policies = ["veto","share"]
  metrics = ["LOLE","EEU"]
  areas = ["GB","IRL"]

  #x_axis = np.array([15 + i*15/10 for i in range(11)])
  #y_axis = np.array([3 + i*3/10 for i in range(11)])

  prefix = "_genshift2"
  draw_grid(
    #x_axis,
    #y_axis,
    years=periods,
    policies=policies,
    metrics=metrics,
    areas=areas,
    prefix=prefix)
