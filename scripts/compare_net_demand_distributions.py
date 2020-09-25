import sys
import pandas as pd
import numpy as np
from dfply import *

from psrmodels.time_collapsed import BivariateLogisticNetDemand, BivariateHindcastNetDemand, BivariateSystemSimulator, ConvGenDistribution

from helper_functions import * 

# this is mean to be load by a python client from R, not run from the command line
# the simulate() function runs simulations of hindcast and EVT models of net demand
def simulate_risk(dt,n,c,year,wpf,p,shapes,scales,alpha):
  #
  #
  dt = np.array(dt).clip(min=0).astype(np.int32)
  gb_gen_file = '../../data/energy/uk/generator_data.txt'
  ire_gen_file = '../../data/energy/ireland/generator_data.txt'
  uk_gen = ConvGenDistribution(gb_gen_file)
  ire_gen = ConvGenDistribution(ire_gen_file)
  gens = [uk_gen,ire_gen]
  #
  h = BivariateHindcastNetDemand(dt)
  l = BivariateLogisticNetDemand(X=dt,p=p,alpha=alpha,shapes=shapes,scales=scales)
  #
  lsim = BivariateSystemSimulator(gens,l)
  hindsim = BivariateSystemSimulator(gens,h)
  df = pd.concat((lsim.veto_risk(n,c,1),hindsim.veto_risk(n,c,1)),axis=0)
  #
  #hindcast = hindsim.simulate_veto(n,c,1)
  #
  # return only shortfall events
  return df
