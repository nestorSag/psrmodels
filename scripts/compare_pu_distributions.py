import pandas as pd
import numpy as np
from dfply import *

from psrmodels.time_collapsed import BivariateLogisticNetDemand, BivariateHindcastNetDemand, BivariateSystemSimulator, ConvGenDistribution

from helper_functions import * 

#Script designed to be called from an R session where a particular EV model parameters is passed to simulate_pu
def simulate(n,c):
  #
  data = read_data().query("period ==2010") >>\
      mutate(\
        gb = X.gbdem_r - X.gbwind_r, \
        ireland = X.idem_r - X.iwind_r) >> \
      select(["gb","ireland"])
  #
  dt = np.array(data).astype(np.int32)
  #
  gb_gen_file = "../../../data/energy/uk/generator_data.txt"
  ire_gen_file = "../../../data/energy/ireland/generator_data.txt"
  uk_gen = ConvGenDistribution(gb_gen_file)
  ire_gen = ConvGenDistribution(ire_gen_file)
  gens = [uk_gen,ire_gen]
  #
  h = BivariateHindcastNetDemand(dt)
  #
  hindsim = BivariateSystemSimulator(gens,h)
  #
  #hindcast = hindsim.simulate_veto(n,c,1)
  #
  # return only shortfall events
  return hindsim.veto_risk(n,c,1)

if __name__=="__main__":
  df1, df2 = simulate(100000,1000)
  #print(df1)
  #print(df2)