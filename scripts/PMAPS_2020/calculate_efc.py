from dfply import * 
from datetime import timedelta, datetime as dt
import pytz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from collections import OrderedDict

import sys, os
#sys.path.append('..')

from psrmodels.time_collapsed import BivariateHindcastMargin, ConvGenDistribution

from helper_functions import *

DATA_FOLDER = "data/PMAPS_2020/"
FIGURE_FOLDER = "figures/PMAPS_2020/"
#this script calculates bivariate hindcast LOLE and EEU exactly for varying interconnection capacities
def main(periods,cap_range,wp_factors=[1],policies=["veto","share"]):
  
  if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    
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
  for wpf in wp_factors:

    for period in periods:

      if not os.path.isfile("data/bivariate_hindcast_efc_{wpf}wpf_{y}y.csv".format(wpf=str(wpf),y=period)):

        rows = []
        if period != "all":
          demands = np.array(df.query("period == {p}".format(p=period))[["gbdem_r","idem_r"]])
          winds = wpf*np.array(df.query("period == {p}".format(p=period))[["gbwind_r","iwind_r"]])
        else:
          demands = np.array(df[["gbdem_r","idem_r"]])
          winds = wpf*np.array(df[["gbwind_r","iwind_r"]])

        dist = BivariateHindcastMargin(demands,winds,[uk_gen,ire_gen])
        # these inner loops are made to create a dataframe in tidy format
        for policy in policies:

          for itc_cap in cap_range:

            for metric in ["LOLE","EEU"]:
              
              for area in ["GB","IRL"]:

                print("metric: {m}, area: {a}, policy: {p}, c: {c}".format(m=metric,a=area,p=policy,c=itc_cap))

                axis = int(area == "IRL")
                value = dist.efc(c=itc_cap,policy=policy,metric=metric,axis=axis)

                row = (
                  period,
                  policy,
                  itc_cap,
                  value,
                  metric,
                  area,
                  wpf)

                rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df.columns = ["year","policy","capacity","value","metric","area","wpf"]

        results_df.to_csv(DATA_FOLDER + "bivariate_hindcast_efc_{wpf}wpf_{y}y.csv".format(wpf=str(wpf),y=period),index=False)

if __name__ == "__main__":
  #periods = [2007,2008,2009,2010,2011,2012,2013]
  periods = ["all"]
  cap_range = [250*x for x in range(11)]
  policies = ["veto","share"]
  wp_factors = [1,2]
  # main function creates csv file with yearly data and creates some basic plots
  main(periods,cap_range,wp_factors,policies)
