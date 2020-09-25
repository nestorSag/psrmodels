import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys, os, functools
#sys.path.append('..')

from psrmodels.time_collapsed import UnivariateHindcastMargin, ConvGenDistribution

from helper_functions import *

import scipy as sp

def main(drop_list,period,bin_size):
  #ceil demands and floor wind to avoid underestimating risk
  ire_gen_file = "../../data/energy/ireland/generator_data.txt"

  # usind 2010 only
  data = read_data() >> mutate(
    gbdem_r = X.gbdem_r.apply(lambda x: int(np.ceil(x))),
    idem_r = X.idem_r.apply(lambda x: int(np.ceil(x))),
    gbwind_r = X.gbwind_r.apply(lambda x: int(np.floor(x))),
    iwind_r = X.iwind_r.apply(lambda x: int(np.floor(x))))

  data = data.query("period == {p}".format(p=period))

  net_demand = np.clip(np.array(data["idem_r"] - data["iwind_r"]),a_min=0,a_max=np.Inf).astype(int)
  
  ### create dataframes for different convgen configurations
  
  gen_df = pd.read_csv(ire_gen_file,sep=" ")
  #ire_df = ire_df.sort_values(by=["Capacity"])

  #drop_list = [[51],[51,48]] #indices of generators to drop in each scenario
  # get capacities and avilabilities of dropped units
  drop_list_caps = [np.array(gen_df.filter(items = dropped,axis=0)["Capacity"]) for dropped in drop_list]
  drop_list_probs = [np.array(gen_df.filter(items = dropped,axis=0)["Availability"]) for dropped in drop_list]

  # generate plot labels
  label_list = [" baseline (-0MW)"] + ["{n} gen(s). (-{x}MW)".format(n=len(drop_list[i]),x=np.sum(drop_list_caps[i])) for i in range(len(drop_list))]

  # [0] accounts for the baseline case
  # calculate lost power expectation
  drop_list_expectation = [0] + [np.sum(x*y) for x,y in zip(drop_list_caps,drop_list_probs)]
  
  # create gen dataframes of generators without dropped units
  gen_df_list = [gen_df] + [gen_df.drop(drop_ids) for drop_ids in drop_list]

  # create list of UnivariateHindcastMargin calculator objects for the different generation scenarios
  hc_list = [UnivariateHindcastMargin(ConvGenDistribution(df),net_demand,precompute=True) for df in gen_df_list]

  # support and margin PDF for each convgen scenario
  m_support_list = [np.array(list(range(hc_list[i].min,hc_list[i].max+1))) for i in range(len(hc_list))]

  m_pdf_list = [[hc_list[i].pdf(int(x)) for x in m_support_list[i]] for i in range(len(m_support_list))]

  m_support_list = [drop_list_expectation[i] + np.array(list(range(hc_list[i].min,hc_list[i].max+1))) for i in range(len(hc_list))]

  label_column = [[label_list[i] for j in range(len(m_support_list[i]))] for i in range(len(m_support_list))]

  dist_df = pd.DataFrame({
    "binned":functools.reduce(lambda a,b: list(a)+list(b),m_support_list),
    "pdf":functools.reduce(lambda a,b: list(a)+list(b),m_pdf_list),
    "loss":functools.reduce(lambda a,b: list(a)+list(b),label_column)})

  #dist_df.to_csv("figures/dist_df1.csv",index=False)
  dist_df["unbinned_cdf"] = dist_df["pdf"].groupby(dist_df["loss"]).transform("cumsum")
  dist_df["unbinned_logcdf"] =  np.log10(dist_df["unbinned_cdf"])
  dist_df["unbinned_margin"] = dist_df["binned"]

  unbinned_df = dist_df.copy(deep=True).query("unbinned_logcdf >= -6")
  unbinned_df["unbinned_logcdf"] = np.log10(3360) + unbinned_df["unbinned_logcdf"]

  ax = sns.lineplot(x="unbinned_margin",y="unbinned_logcdf",data=unbinned_df,hue="loss")
  #ax.axvline(x=0,linestyle="--",color="grey")
  plt.title("shifted logcdf (LOLE scale) when removing units")
  ax.set(xlabel="margin",ylabel="LOLE scale")
  plt.savefig("figures/convgen_shifted_lole_scale_plot")

  plt.clf()

  
  dist_df["binned"] = dist_df["binned"].apply(lambda x: np.floor(x/bin_size))
  dist_df = dist_df.groupby(by=["binned","loss"]).sum().reset_index()
  dist_df["cdf"] = dist_df["pdf"].groupby(dist_df["loss"]).transform("cumsum")
  dist_df["logcdf"] = np.log10(dist_df["cdf"])
  #dist_df.to_csv("figures/dist_df2.csv",index=False)

  lole_df = dist_df.query("binned==0")
  print(lole_df)
  lole_df["LOLE"] = 3360*lole_df["cdf"]
  ax = sns.barplot(x="loss",y="LOLE",data=lole_df)
  plt.title("shifted margin LOLE when removing units")
  ax.set(xlabel="scenario",ylabel="LOLE")
  plt.savefig("figures/convgen_shifted_lole_plot")

  plt.clf()
  
  #dist_df = dist_df.query("logcdf >= -6")
  
  ax = sns.lineplot(x="binned",y="pdf",data=dist_df,hue="loss")
  plt.title("shifted margin pdf when removing units")
  ax.set(xlabel="margin (binned at a scale of {x}MW)".format(x=bin_size),ylabel="pdf")
  plt.savefig("figures/convgen_shifted_pdf_plot")

  plt.clf()

  #compare_shifts(df_list,label_list,k,"ireland")

if __name__=="__main__":
  main(drop_list=[[51],[51,48]],period=2010,bin_size=20)
