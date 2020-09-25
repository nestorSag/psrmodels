from dfply import * 
from datetime import timedelta, datetime as dt
import pytz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys, os

from psrmodels.time_collapsed import BivariateHindcastMargin, ConvGenDistribution
from helper_functions import *

import matplotlib.pyplot as plt
import seaborn as sns

# this script generates files with pointwise risk metrics for hindcast
def main(policies,cap_range,wp_factors):
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
  periods = np.unique(df["period"])

  for wpf in wp_factors:
    for period in periods:

      demands = np.array(df.query("period == {p}".format(p=period))[["gbdem_r","idem_r"]])
      winds = wpf*np.array(df.query("period == {p}".format(p=period))[["gbwind_r","iwind_r"]])

      dist = BivariateHindcastMargin(demands,winds,[uk_gen,ire_gen])

      for policy in policies:
        for itc_cap in cap_range:
          
          pw_data_file="data/pw_eeu_gb_{y}p_{p}policy_{c}c_{wpf}wpf.csv".format(p=policy,y=period,c=itc_cap,wpf=str(wpf))
          df_ = dist.eeu(c=itc_cap,policy=policy,axis=0,get_pointwise_risk=True)
          #print(df_)
          df_.to_csv(pw_data_file, index=False)

          pw_data_file="data/pw_eeu_irl_{y}p_{p}policy_{c}c_{wpf}wpf.csv".format(p=policy,y=period,c=itc_cap,wpf=str(wpf))
          df_ = dist.eeu(c=itc_cap,policy=policy,axis=1,get_pointwise_risk=True)
          df_.to_csv(pw_data_file, index=False)

          pw_data_file="data/pw_lole_gb_{y}p_{p}policy_{c}c_{wpf}wpf.csv".format(p=policy,y=period,c=itc_cap,wpf=str(wpf))
          df_ = dist.lole(c=itc_cap,policy=policy,axis=0,get_pointwise_risk=True)
          df_.to_csv(pw_data_file, index=False)

          pw_data_file="data/pw_lole_irl_{y}p_{p}policy_{c}c_{wpf}wpf.csv".format(p=policy,y=period,c=itc_cap,wpf=str(wpf))
          df_ = dist.lole(c=itc_cap,policy=policy,axis=1,get_pointwise_risk=True)
          df_.to_csv(pw_data_file, index=False)

          # pw_data_file="data/pw_system_lole_gb_{p}p_{c}c_{wpf}wpf.csv".format(p=period,c=itc_cap,wpf=str(wpf))
          # dist.system_lole(itc_cap,get_pointwise_risk=True)
          # df_.to_csv(pw_data_file, index=False)

          # pw_data_file="data/pw_system_lole_irl_{p}p_{c}c_{wpf}wpf.csv".format(p=period,c=itc_cap,wpf=str(wpf))
          # dist.system_lole(itc_cap,get_pointwise_risk=True)
          # df_.to_csv(pw_data_file, index=False)


if __name__=="__main__":
  cap_range = [0,1000]
  policies = ["veto","share"]
  wp_factors = [1,2] #wind penetration factors (simulate different installed capacities)
  main(policies,cap_range,wp_factors)
