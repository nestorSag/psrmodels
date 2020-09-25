import pandas as pd
import numpy as np
from dfply import * 
import pytz
from datetime import timedelta, datetime as dt

from psrmodels.time_collapsed import BivariateHindcastMargin, ConvGenDistribution

from helper_functions import *

import matplotlib.pyplot as plt
import seaborn as sns

# this script simulates shortfalls under a veto policy and compares what would have happened under a share policy (larger or smaller loss)
FIGURES_FOLDER = "figures/PMAPS_2020/"

def generate_data(period,c):
  if not os.path.exists("data/"):
    os.makedirs("data/")
    
  season_hours = 20*7*24
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
  #period = 2010
  #c = 1000
  policy = "share"
  #eeu_mc_n = 200
  
  # simulate shortfalls in the UK
  demands = np.array(df.query("period == {p}".format(p=period))[["idem_r","gbdem_r"]])
  winds = np.array(df.query("period == {p}".format(p=period))[["iwind_r","gbwind_r"]])

  area_dist_ire = BivariateHindcastMargin(demands,winds,[ire_gen,uk_gen])

  uk_pu = calc.mc_epu(area_dist_ire,c,compare_policies=True)

  return pd.DataFrame(uk_pu)

def plot_shortfall_diffs(df):
  #print(df)

  #plt.rc('xtick',labelsize=18)
  #plt.rc('ytick',labelsize=18)
  
  df >>= mutate(change_ = X.share2 - X.share1) >> mutate(change = X.change_.apply(lambda x: "Worsened" if x >0 else "Improved" if x < 0 else 'None')) >> mutate(size = abs(X.change_))

  print("capacity1 -> capacity2 produces a change in EPU of {x}".format(x = - df.query("change == 'Improved'")["size"].sum() + df.query("change == 'Worsened'")["size"].sum()))
  
  fig = plt.figure(figsize=(8,8))

  #ymin, ymax = df["m2"].min(), df["m2"].max()
  #xmin, xmax = df["m1"].min(), df["m1"].max()

  xmin = -2500
  ymax = 7000
  xmax = 2500
  ymin = -7000

  ax = fig.add_subplot(111)
  ax.axvline(x=0,ymin=ymin,ymax=ymax,color="grey",linestyle="--")
  ax.axhline(y=0,xmin=xmin,xmax=xmax,color="grey",linestyle="--")
  sns_plot = sns.scatterplot(x="m1",y="m2",hue="change",style="change",size="size",data=df,palette=[ "#481567","#FDE725","#238A8D"]) #neutral, worsened, improved
  ax.plot([xmin,ymax],[xmax,ymin],'--',color="black")
  
  sns_plot.set(ylim=(ymin,ymax),xlim=(xmin,xmax))

  plt.legend(title_fontsize=20,fontsize=16,loc='upper right')
  ax.set_xlabel("IRL pre-ITC margin (GW)",fontsize=25)
  ax.set_ylabel("GB pre-ITC margin (GW)",fontsize=22)
  ax.tick_params(labelsize=18)

  ylabs = ["{y_}".format(y_=y/1000.0) for y in ax.get_yticks()]
  ax.set_yticklabels(ylabs)

  xlabs = ["{x_}".format(x_=x/1000.0) for x in ax.get_xticks()]
  ax.set_xticklabels(xlabs)
  
  plt.savefig(FIGURES_FOLDER + "policy_shortfall_comparison.png")
  
if __name__=="__main__":
  period = 2011
  #c = 1000
  df1 = generate_data(period,600)
  df2 = generate_data(period,2500)

  df1.columns = ["m1","m2","share1","veto1"]
  df2.columns = ["m1","m2","share2","veto2"]

  # control of random seeds ensure that data in both dataframes
  # use same demand, net demand and convgen values for each of the simulations ran, so we
  # can join both just on m1 and m2 without problem.
  # otherwise we would need to also join on demands since given margins can arise from
  # a combination of generation, net demands and demands
  df = df1.merge(df2,"outer",on=["m1","m2"]).fillna(0)

  plot_shortfall_diffs(df)
  
  #plot_shortfall_diffs(period,c)
