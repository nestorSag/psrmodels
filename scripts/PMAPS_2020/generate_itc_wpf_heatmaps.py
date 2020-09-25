from dfply import * 
from datetime import timedelta, datetime as dt
import pytz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from collections import OrderedDict

import sys, os

#generate heatmaps of interconnection's EFC on wind generation capacity space

from psrmodels.time_collapsed import BivariateHindcastMargin, ConvGenDistribution

from helper_functions import *

FOLDER = "figures/PMAPS_2020/"

def main(periods,cap_range,wp_factors=[1],policies=["veto","share"],areas=["GB","IRL"],metrics=["LOLE","EPU"]):
  if not os.path.exists("data/"):
    os.makedirs("data/")
    
      #read and transform data
  uk_gen_file = "../../../data/energy/uk/generator_data.txt"
  ire_gen_file = "../../../data/energy/ireland/generator_data.txt"

  uk_gen = ConvGenDistribution(uk_gen_file)
  ire_gen = ConvGenDistribution(ire_gen_file)

  data_path = "../../../data/energy/uk_ireland/InterconnectionData_Rescaled.txt"

  #ceil demands and floor wind to avoid underestimating risk
  df = read_data(data_path) >> mutate(
    gbdem_r = X.gbdem_r.apply(lambda x: int(np.ceil(x))),
    idem_r = X.idem_r.apply(lambda x: int(np.ceil(x))),
    gbwind_r = X.gbwind_r.apply(lambda x: int(np.floor(x))),
    iwind_r = X.iwind_r.apply(lambda x: int(np.floor(x))))
  
  #periods = df["period"].unique()
  for period in periods:

    for policy in policies:

      for area in areas:
        axis = int(area=="IRL")
        for metric in metrics:
          x = np.sort(wp_factors)
          y = np.sort(cap_range)
          z = np.zeros((len(cap_range),len(wp_factors)))
          i = 0
          for wpf in wp_factors:
            rows = []
            demands = np.array(df.query("period == {p}".format(p=period))[["gbdem_r","idem_r"]])
            winds = wpf*np.array(df.query("period == {p}".format(p=period))[["gbwind_r","iwind_r"]])

            dist = BivariateHindcastMargin(demands,winds,[uk_gen,ire_gen])
            j = 0
            method = getattr(dist,metric)
            for itc_cap in cap_range:
              z[j,i] = method(itc_cap,axis=axis,policy=policy)
              j += 1

            print("row: {i}".format(i=i))
            i += 1

          plt.figure(figsize=(8,8))

          plt.rc('xtick',labelsize=18)
          plt.rc('ytick',labelsize=18)
          contours = plt.contourf(x,y,z)
          plt.contour(contours)
          plt.xlabel("WPF",fontsize=25)
          plt.ylabel("Interconnector capacity (GW)",fontsize=20)
          ticks, _ = plt.yticks()
          ylabs = ["{x_}".format(x_=x/1000.0) for x in ticks]
          plt.yticks(ticks,ylabs)
          cbar = plt.colorbar(contours)
          cbar.ax.tick_params(labelsize=14)
          cbar.ax.set_title(label=metric,fontdict={'fontsize':18})

          plt.savefig("figures/itc_wpf_heatmap_{p}policy_{y}year_{a}area_{m}metric.png".format(m=metric,p=policy,y=period,a=area),bbox_inches='tight')

if __name__=="__main__":
  #periods = list(range(2007,2014))
  periods = list(range(2007,2014))
  itc_caps = [200*x for x in range(11)]
  #wp_factors = [1 + x/10.0 for x in range(11)]
  wp_factors = np.arange(1,2.1,0.1)
  policies = ["share","veto"]
  areas = ["IRL","GB"]
  metrics = ["EEU"]
  main(periods,itc_caps,wp_factors,policies,areas,metrics)
