import pandas as pd
import numpy as np
from dfply import *

import functools
import sys
sys.path.append('.')

from helper_functions import *

from psrmodels.time_collapsed import ConvGenDistribution, BivariateLogisticNetDemand, BivariateHindcastNetDemand, BivariateSystemSimulator

def simulate_veto(X,n,m,cs,thresholds,shapes,scales,alpha,exs_probs):
  #n = batch size
  #m = number of batches to stack
  #cs = list of interconnector capacities to try
  # threshold, shapes, scales, alpha = parameters for logistic model
  # exs_probs = probabilities of extreme events (drawn from EV model)
  
  #data = read_data().query("period ==2010") >> mutate(uk = X.gbdem_r - X.gbwind_r, ireland = X.idem_r - X.iwind_r) >> select(["uk","ireland"])

  #X = np.array(data)

  g = BivariateLogisticNetDemand(X,thresholds,alpha,shapes,scales)

  uk_gen_file = "../../data/energy/uk/generator_data.txt"
  ire_gen_file = "../../data/energy/ireland/generator_data.txt"
  uk_gen = ConvGenDistribution(uk_gen_file)
  ire_gen = ConvGenDistribution(ire_gen_file)
  gens = [uk_gen,ire_gen]

  logsim = BivariateSystemSimulator(gens,g)

  results_list = []
  events_dict = {}

  if isinstance(exs_probs,float):
    exs_probs = [exs_probs]
  if isinstance(cs,float):
    cs = [cs]
    
  for p in exs_probs:
    for c in cs:
      print("Simulating c = {c} and p = {p}".format(c=c,p=p))
      res, events = logsim.veto_risk(n,c,m,seed=1,exs_prob=p)
      results_list.append(res)
      events_dict[(c,p)] = events

  results = functools.reduce(lambda x,y: x+y,results_list)
  
  return {"lolp":results.lolp,
          "epu":results.epu,
          "slolp":results.slolp,
          "sepu":results.sepu,
          "events":events_dict}
