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

#this script calculates bivariate hindcast LOLE and EEU exactly for varying interconnection capacities
def main(periods,cap_range,wp_factors=[1],policies=["veto","share"]):
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
  for wpf in wp_factors:

    for period in periods:

      if not os.path.isfile(DATA_FOLDER + "bivariate_hindcast_results_{wpf}wpf_{y}y.csv".format(wpf=str(wpf),y=period)):

        rows = []
        demands = np.array(df.query("period == {p}".format(p=period))[["gbdem_r","idem_r"]])
        winds = wpf*np.array(df.query("period == {p}".format(p=period))[["gbwind_r","iwind_r"]])

        dist = BivariateHindcastMargin(demands,winds,[uk_gen,ire_gen])
        # these inner loops are made to create a dataframe in tidy format
        for policy in policies:
          for itc_cap in cap_range:
            print("processing policy: {p}, capacity: {c}".format(p=policy,c=itc_cap))

            for metric in ["LOLE","EEU"]:
              for area in ["GB","IRL","System"]:

                if area in ["GB","IRL"]:
                  axis = int(area == "IRL")
                  method = getattr(dist,metric)
                  value = method(itc_cap,axis=axis,policy=policy)
                else:
                  if metric == "LOLE":
                    method = getattr(dist,"system_LOLE")
                    value = method(itc_cap)
                  else:
                    method = getattr(dist,"EEU")
                    value = method(itc_cap,axis=0,policy=policy) + method(itc_cap,axis=1,policy=policy)
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

        results_df.to_csv(DATA_FOLDER + "bivariate_hindcast_results_{wpf}wpf_{y}y.csv".format(wpf=str(wpf),y=period),index=False)

      #generate_plots(wp_factors,period)

def generate_plots(wp_factors,period):

  if not os.path.exists("figures/"):
    os.makedirs("figures/")

  for wpf in wp_factors:
    df = pd.read_csv(DATA_FOLDER + "bivariate_hindcast_results_{wpf}wpf_{y}y.csv".format(wpf=wpf,y=period))

    #sns.set(rc={"lines.linewidth": 0.7})

    plt.figure(figsize=(8,8))
    
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)

    ax = sns.lineplot(
      data=df.query("metric == 'LOLE'"),
      x="capacity",
      y="value",
      hue="area",
      style="policy")

    sns.scatterplot(
      data=df.query("metric == 'LOLE'"),
      x="capacity",
      y="value",
      hue="area",
      style="area",
      size="area",
      sizes=(80,80))

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(list(by_label.values())[0::], list(by_label.keys())[0::],title_fontsize=40,fontsize=22)

    #plt.show()
    #plt.legend(title_fontsize=40,fontsize=22)
    ax.set_xlabel("Interconnector capacity (GW)",fontsize=25)
    ax.set_ylabel("LOLE (hrs.)",fontsize=20)
    #ax.tick_params(labelsize=15)

    xlabs = ["{x_}".format(x_=x/1000.0) for x in ax.get_xticks()]
    ax.set_xticklabels(xlabs)
    
    plt.savefig("figures/LOLE_{wpf}wpf_{y}y.png".format(wpf=wpf,y=period))
    
    #plt.subplot(1,3,1)
    plt.legend(title_fontsize=40,fontsize=22)
    plt.figure(figsize=(8,8))

    ax = sns.lineplot(
      data=df.query("country == 'GB' and value == 'EEU'"),
      x="capacity",
      y="value",
      hue="policy")
    
    sns.scatterplot(
      data=eeu_mean_df.query("country == 'GB' and value == 'EEU'"),
      x="capacity",
      y="value",
      hue="policy",
      legend=False)#.set_title("EEU vs interconnector capacity (GB)")

    plt.legend(title_fontsize=40,fontsize=25)
    ax.set_xlabel("Interconnector capacity (GW)",fontsize=25)
    ax.set_ylabel("EEU (GW)",fontsize=20)
    #ax.tick_params(labelsize=15)

    ylabs = ["{y_}".format(y_=y/1000.0) for y in ax.get_yticks()]
    ax.set_yticklabels(ylabs)

    xlabs = ["{x_}".format(x_=x/1000.0) for x in ax.get_xticks()]
    ax.set_xticklabels(xlabs)
    

    plt.savefig("figures/EEU_GB_{wpf}wpf_{y}y.png".format(wpf=wpf,y=period))  
    #print(eeu_mean_df)
    #plt.subplot(1,3,2)
    plt.figure(figsize=(8,8))

    ax = sns.lineplot(
      data=eeu_mean_df.query("country == 'Ireland' and value == 'EEU'"),
      x="capacity",
      y="value",
      hue="policy")
    
    sns.scatterplot(
      data=eeu_mean_df.query("country == 'Ireland' and value == 'EEU'"),
      x="capacity",
      y="value",
      hue="policy",
      legend=False)#.set_title("EEU vs interconnector capacity (Ireland)" )

    plt.legend(title_fontsize=40,fontsize=22)
    ax.set_xlabel("Interconnector capacity (GW)",fontsize=25)
    ax.set_ylabel("EEU (GW)",fontsize=20)

    ylabs = ["{y_}".format(y_=y/1000.0) for y in ax.get_yticks()]
    ax.set_yticklabels(ylabs)

    xlabs = ["{x_}".format(x_=x/1000.0) for x in ax.get_xticks()]
    ax.set_xticklabels(xlabs)
    
    #ax.tick_params(labelsize=15)
    plt.savefig("figures/EEU_Ireland_{wpf}wpf_{y}y.png".format(wpf=wpf,y=period))

    #plt.subplot(1,3,3)
    plt.legend(title_fontsize=40,fontsize=22)
    plt.figure(figsize=(8,8))

    ax = sns.lineplot(
      data=eeu_mean_df.query("country == 'system' and value == 'EEU'"),
      x="capacity",
      y="value",
      color = "#18691D")
    
    sns.scatterplot(
      data=eeu_mean_df.query("country == 'system' and value == 'EEU'"),
      x="capacity",
      y="value",
      color = "#18691D")
      #.set_title("EEU vs capacity (System)" ) 

    #.set_title("EEU vs capacity (System)" 
    ax.set_xlabel("Interconnector capacity (GW)",fontsize=25)
    ax.set_ylabel("EEU (GW)",fontsize=20)
    #ax.tick_params(labelsize=15)

    ylabs = ["{y_}".format(y_=y/1000.0) for y in ax.get_yticks()]
    ax.set_yticklabels(ylabs)

    xlabs = ["{x_}".format(x_=x/1000.0) for x in ax.get_xticks()]
    ax.set_xticklabels(xlabs)
    
    plt.savefig("figures/EEU_system_{wpf}wpf_{y}y.png".format(wpf=wpf,y=period))



    
    ####
    ####
    #### EEU plot for all systems
    ax = sns.lineplot(
      data=eeu_mean_df.query("country == 'Ireland' and value == 'EEU'"),
      x="capacity",
      y="value",
      color="#EE854B",
      style="policy",
      legend=False)
    
    sns.scatterplot(
      data=eeu_mean_df.query("country == 'Ireland' and value == 'EEU'"),
      x="capacity",
      y="value",
      color="#EE854B",
      marker="X",
      s=80,
      legend=False)#.set_title("EEU vs interconnector capacity (Ireland)" )

    sns.lineplot(
      data=eeu_mean_df.query("country == 'GB' and value == 'EEU'"),
      x="capacity",
      y="value",
      color="#4779D0",
      style="policy",
      legend=False)
    
    sns.scatterplot(
      data=eeu_mean_df.query("country == 'GB' and value == 'EEU'"),
      x="capacity",
      y="value",
      color="#4779D0",
      marker="o",
      s=80,
      legend=False)#.set_title("EEU vs interconnector capacity (GB)")

    sns.lineplot(
      data=eeu_mean_df.query("country == 'system'  and value == 'EEU'"),
      x="capacity",
      y="value",
      color = "#6BCC65",
      legend=False)
    
    sns.scatterplot(
      data=eeu_mean_df.query("country == 'system'  and value == 'EEU'"),
      x="capacity",
      y="value",
      color = "#6BCC65",
      marker="s",
      s=80,
      legend=False)
      #.set_title("EEU vs capacity (System)" )

    ttl1_lgnd = mlines.Line2D([], [], color='black', label='area',linewidth=0)

    ttl2_lgnd = mlines.Line2D([], [], color='black', label='policy',linewidth=0)
    
    irl_lgnd = mlines.Line2D([], [], color='#EE854B', marker='X',
                             markersize=15, label='Ireland', linewidth=0)

    gb_lgnd = mlines.Line2D([], [], color='#4779D0', marker='o',
                            markersize=15, label='GB', linewidth=0)

    sys_lgnd = mlines.Line2D([], [], color='#6BCC65', marker='s',
                             markersize=15, label='System', linewidth=0)

    veto_lgnd = mlines.Line2D([], [], color='black', label='veto', linestyle ="-")

    share_lgnd = mlines.Line2D([], [], color='black', label='share', linestyle="--")

    plt.legend(bbox_to_anchor=(0.3,1),handles=[ttl1_lgnd,gb_lgnd,irl_lgnd,sys_lgnd,ttl2_lgnd,veto_lgnd,share_lgnd],title_fontsize=35,fontsize=20,ncol=2)
    
    ax.set_xlabel("Interconnector capacity (GW)",fontsize=25)
    ax.set_ylabel("EEU (GW)",fontsize=20)
    #ax.tick_params(labelsize=15)

    ylabs = ["{y_}".format(y_=y/1000.0) for y in ax.get_yticks()]
    ax.set_yticklabels(ylabs)

    xlabs = ["{x_}".format(x_=x/1000.0) for x in ax.get_xticks()]
    ax.set_xticklabels(xlabs)
    

    plt.savefig("figures/EEU_all_{wpf}wpf_{y}y.png".format(wpf=wpf,y=period))

    plt.close()

if __name__ == "__main__":
  periods = [2007,2008,2009,2010,2011,2012,2013]
  cap_range = [250*x for x in range(11)]
  policies = ["veto","share"]
  wp_factors = [1]
  # main function creates csv file with yearly data and creates some basic plots
  main(periods,cap_range,wp_factors,policies)
