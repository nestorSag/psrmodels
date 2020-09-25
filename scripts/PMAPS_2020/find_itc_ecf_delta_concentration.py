import pandas as pd
import numpy as np
from dfply import * 
import pytz
from datetime import timedelta, datetime as dt

from psrmodels.time_collapsed import BivariateHindcastMargin, ConvGenDistribution, BivariateHindcastMarginCalculator

from helper_functions import *

import matplotlib.pyplot as plt
import seaborn as sns

#this script finds the k points where the change in risk is concentrated as we go from a low to high interconnection capacity
FIGURES_FOLDER = "figures/PMAPS_2020/"
DATA_FOLDER = "data/PMAPS_2020/"

# area indices 
GB = 0
IRL = 1
def generate_data(c0,c1,years,policies,axes,metrics):
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
  
  for year in years:

    demands = np.array(df.query("period == {y}".format(y=year))[["gbdem_r","idem_r"]])
    winds = np.array(df.query("period == {y}".format(y=year))[["gbwind_r","iwind_r"]])

    model = BivariateHindcastMargin(demands,winds,[uk_gen,ire_gen])

    for metric in metrics:

      data_getter = getattr(model,metric)

      for policy in policies:

        for axis in axes:
          area = "GB" if axis == 0 else "IRL"
          df_c0 = data_getter(c=c0,policy=policy,axis=axis,get_pointwise_risk=True).reset_index()
          print(df_c0)
          
          #df_c0 = df_c0.reset_index()
          df_c1 = data_getter(c=c1,policy=policy,axis=axis,get_pointwise_risk=True).reset_index()

          #print(year)
          #print(sum(df_c0["value"]))
          #print(sum(df_c1["value"]))
          
          raw_data_df = df.query("period == {y}".format(y=year))[["idem_r","gbdem_r","iwind_r","gbwind_r"]].reset_index()
          raw_data_df["c0_risk"] = df_c0["value"]
          raw_data_df["c1_risk"] = df_c1["value"]
          raw_data_df["risk_diff"] = df_c1["value"] - df_c0["value"]

          
          #print(df_c0)
          #print(df_c0["nd0"][0])
          #print(model.gen_dists[axis].cdf(df_c0["nd" + str(1-axis)][0]))

          raw_data_df["gen_quantile_irl"] = [model.gen_dists[IRL].cdf(x) for x in df_c0["nd" + str(1-axis)]]
          raw_data_df["gen_quantile_gb"] = [model.gen_dists[GB].cdf(x) for x in df_c0["nd" + str(axis)]]
          raw_data_df["year"] = year

          #print(df_c0["value"])
          #print(raw_data_df["c0_risk"])

          raw_data_df.to_csv(DATA_FOLDER + "risk_delta_{y}year_{p}policy_{m}metric_{a}area.csv".format(y=year,p=policy,m=metric,a=area))

  # simulate shortfalls in the UK

if __name__=="__main__":
  c0 = 600
  c1 = 2500

  years = list(range(2007,2014))
  axes = [1]
  metrics = ["EEU"]
  policies = ["share"]
  #c = 1000
  generate_data(c0,c1,years,policies,axes,metrics)
  
  #plot_shortfall_diffs(period,c)
