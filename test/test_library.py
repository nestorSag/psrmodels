import pandas as pd
import numpy as np
from dfply import * 
import pytest
import scipy as sp
import pytz
from datetime import timedelta, datetime as dt

from psrmodels.time_collapsed import * 
import psrmodels.time_dependent as td

from scipy.stats import poisson

def winter_period(dt):
  if dt.month >= 10:
    return dt.year
  else:
    return dt.year - 1

def get_objects(country="uk"):
  gen_file = "test/data/energy/{c}/generator_data.txt".format(c=country)
  data_path = "test/data/energy/uk_ireland/InterconnectionData_Rescaled.txt"
  df = pd.read_csv(data_path,sep=" ") 
  df.columns = [x.lower() for x in df.columns]
  #
  df >>= mutate(time = X.date + " " +  X.time) >>\
  mutate(time = X.time.apply(lambda x: dt.strptime(x,"%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc))) >> drop(["date"]) >> mutate(period = X.time.apply(lambda x: winter_period(x)), uk_net = (X.gbdem_r - X.gbwind_r).apply(lambda x: int(np.floor(x))), ireland_net = (X.idem_r - X.iwind_r).apply(lambda x: int(np.floor(x))))
  #
  gen = ConvGenDistribution(gen_file)
  #
  gen_df = pd.read_csv(gen_file, sep=" ")
  gen_df["Capacity"] = -gen_df["Capacity"]
  mirror_gen = ConvGenDistribution(gen_df)
  #
  return {"gen":gen,"data":df,"mirror_gen":mirror_gen}

def get_bivariate_margins_object(y=2010):
  
  obj = get_objects()
  df_ = obj["data"].query("period == {y}".format(y=y))
  gb_gen = obj["gen"]
  irl_gen = get_objects("ireland")["gen"]
  #
  gb_dem = np.array(df_[["gbdem_r"]]).round().astype(np.float64)
  irl_dem = np.array(df_[["idem_r"]]).round().astype(np.float64)
  gb_wind = np.array(df_[["gbwind_r"]]).round().astype(np.float64)
  irl_wind = np.array(df_[["iwind_r"]]).round().astype(np.float64)
  #
  margins = BivariateHindcastMargin(\
  demand=np.concatenate([gb_dem,irl_dem],axis=1),\
  renewables=np.concatenate([gb_wind,irl_wind],axis=1),\
  gen_dists=[gb_gen,irl_gen])
  #
  return margins

def test_UnivariateHindcastMargin():
  
  obj = get_objects()

  true_values_uk = {\
       2007:0.73030655,\
       2008:2.54567463,\
       2009:2.79082304,\
       2010:8.04225239,\
       2011:1.89215985,\
       2012:4.90238648,\
       2013:0.09852472}

  for year in true_values_uk:
    h = UnivariateHindcastMargin(obj["gen"],np.array(obj["data"].query("period == {y}".format(y=year))["uk_net"]))
    assert h.n*h.cdf(0) == pytest.approx(true_values_uk[year],1e-7)

  obj = get_objects("ireland")

  true_values_ireland = {\
       2007:0.78562182,\
       2008:3.95649830,\
       2009:4.22307607,\
       2010:9.36447318,\
       2011:0.56158819,\
       2012:1.46867929,\
       2013:0.62098436}

  for year in true_values_ireland:
    h = UnivariateHindcastMargin(obj["gen"],np.array(obj["data"].query("period == {y}".format(y=year))["ireland_net"]))
    assert h.n*h.cdf(0) == pytest.approx(true_values_ireland[year],1e-7)
    #print("ireland: {y} - {x}".format(y=year,x=x))

  

def test_generation_dist():
  # true values and calcualted values shouldn't differ by more than 1e-10 on average
  obj = get_objects(country="ireland")
  gen_vals_path = "test/data/energy/ireland/pmf.txt"
  pmf_vals = np.array(pd.read_csv(gen_vals_path,sep=" ")["freq"])
  k = min(pmf_vals.size,obj["gen"].cdf_vals.size)
  s = 0
  for i in range(k):
    assert obj["gen"].pdf(i) == pytest.approx(pmf_vals[i],1e-12)
    #s += abs(pmf_vals[i] - obj["gen"].pdf(i))

  ### test adding/removing firm capacity
  expectation_allowed_error = 1e-60
  cdf_allowed_error = 1e-16 #cumulative sum, so can't ensure more than say 1e-16 precission
  fc = 500
  baseline = obj["gen"]
  fc_gen = get_objects(country="ireland")["gen"] + fc# required to avoid selfreferencing

  ### check that PMF has shifted to the right
  k = obj["gen"].max
  for i in range(k):
    assert obj["gen"].pdf(i) == fc_gen.pdf(i+fc)

  ### check that expectation array is well calculated
  for i in range(k):
    assert obj["gen"].pdf(i)*(i+fc) == pytest.approx(fc_gen.expectation(i+fc,i+fc),expectation_allowed_error)

  fc = -500
  fc_gen = get_objects(country="ireland")["gen"] + fc# required to avoid selfreferencing
  ### check that PMF has shifted to the left
  for i in range(k):
    assert obj["gen"].pdf(i) == fc_gen.pdf(i+fc)

  ### check that expectation array is well calculated
  for i in range(k):
    assert obj["gen"].pdf(i)*(i+fc) == pytest.approx(fc_gen.expectation(i+fc,i+fc),expectation_allowed_error)
  
  # check that negative capacity is added correctly 
  #mirror_revcdf = 1 - np.flip(obj["mirror_gen"].cdf_vals)
  for i in range(k):
    assert obj["gen"].cdf(i) == pytest.approx(1 - obj["mirror_gen"].cdf(-i-1),cdf_allowed_error)


def test_convgen_integration():
  
  origin = (0,0)
  d = len(origin)
  min_n = 5
  max_n = 10

  ## mock classes to test triangle_prob method

  class MockConvGenDist(object):
      def __init__(self,cdf_vals):
        self.cdf_vals = cdf_vals
        self.max = len(cdf_vals) - 1
        self.min = 0

  class MockBivariateConvGenDist(object):

    def __init__(self,cdf_vals1,cdf_vals2):
      self.X1 = MockConvGenDist(np.ascontiguousarray(cdf_vals1,dtype=np.float64))
      self.X2 = MockConvGenDist(np.ascontiguousarray(cdf_vals2,dtype=np.float64))


  # test divide and conquer procedure for integrating triangular sections
  uniform_cdf_vals = np.cumsum([1 for x in range(max_n+1)])
  uniform_convgen = MockBivariateConvGenDist(uniform_cdf_vals,uniform_cdf_vals)

  rate = 1
  poisson_cdf_vals = np.cumsum([poisson.pmf(x,rate) for x in range(max_n+1)])

  ps_convgen = MockBivariateConvGenDist(poisson_cdf_vals,poisson_cdf_vals)

  for n in range(min_n,max_n):
    assert BivariateHindcastMargin._triangle_prob(uniform_convgen,origin,n) == (n-1)*n/2

  assert BivariateHindcastMargin._triangle_prob(ps_convgen,origin,2) == poisson.pmf(1,rate)**d
  assert BivariateHindcastMargin._triangle_prob(ps_convgen,origin,3) == (poisson.pmf(1,rate)**d + d*poisson.pmf(1,rate)*poisson.pmf(2,rate))

  #test validity of plane divisions and flow equations for both policies
  c = 2
  plc = "veto"
  #veto policy

  #shortfall region

  #should be true
  points = [(-1,-1),(-3,3)]
  for p in points:
    complete_tuple = p + (c,plc)
    assert BivariateHindcastMargin._is_shortfall_region(*complete_tuple)

  #should be false
  points = [(1,1),(0,0),(-1,1),(-2,2),(-2,3),(0,-1),(1,-1)]
  for p in points:
    complete_tuple = p + (c,plc)
    assert not BivariateHindcastMargin._is_shortfall_region(*complete_tuple)


  #flow

  #should be true
  points = [(-1,-1),(0,0),(1,1),(2,-2),(1,-2),(-2,2),(-3,2),(-3,3)]
  correct_flow = [0,0,0,-2,-1,2,2,2]

  for i in range(len(points)):
    complete_tuple = points[i] + (c,)
    assert BivariateHindcastMargin._get_veto_flow(*complete_tuple) == correct_flow[i]



  #share policy
  plc = "share"
  #shortfall region
  
  #should be true
  points = [(-1,-1),(1,-2),(-3,3)]
  for p in points:
    complete_tuple = p + (c,plc)
    assert BivariateHindcastMargin._is_shortfall_region(*complete_tuple)

  #should be false
  points = [(1,1),(0,0),(-1,1),(-2,2),(-2,3),(1,-1)]
  for p in points:
    complete_tuple = p + (c,plc)
    assert not BivariateHindcastMargin._is_shortfall_region(*complete_tuple)


  #flow

  #should be true
  def itc(m1,m2,d1,d2,c):
    return max(min(d1/(d1+d2)*m2 - d2/(d1+d2)*m1,c),-c)
  
  d_1 = 1.0
  d_2 = d_1/6
  points = [(-1,-1),(0,0),(2,-2),(1,-2),(-2,2),(-3,2)]
  for p in points:
    complete_tuple = p + (d_1,d_2,c,)
    assert BivariateHindcastMargin._get_share_flow(*complete_tuple) == itc(*complete_tuple)

  
def test_time_dependent_margins():

  obj = get_objects()

  td_convgen_file = "test/data/energy/{c}/synthetic_sequential_convgen_data.csv"
  
  gen_df = pd.read_csv(td_convgen_file.format(c="ireland"))
  irl_td_convgen = td.ConvGenDistribution(gen_df)
  gb_td_convgen = td.ConvGenDistribution(pd.read_csv(td_convgen_file.format(c="uk")))
  
  c = 1000
  n_sim = 10
  for y in range(2007,2014):
    df_ = obj["data"].query("period == {y}".format(y=y))
    gb_dem = np.array(df_[["gbdem_r"]]).round().astype(np.float64)
    irl_dem = np.array(df_[["idem_r"]]).round().astype(np.float64)

    gb_wind = np.array(df_[["gbwind_r"]]).round().astype(np.float64)
    irl_wind = np.array(df_[["iwind_r"]]).round().astype(np.float64)

    if y == 2007:
      td_h = td.BivariateHindcastMargin(\
        np.concatenate((gb_dem,irl_dem),axis=1),\
        np.concatenate((gb_wind,irl_wind),axis=1),\
        [gb_td_convgen,irl_td_convgen])
      convgen_sim = td_h._get_gen_simulation(n_sim,1,True)
    else:
      td_h.set_w_d(np.concatenate((gb_dem,irl_dem),axis=1),np.concatenate((gb_wind,irl_wind),axis=1))
      convgen_sim = td_h.gensim

    # model output
    netdem = td_h.net_demand
    margins = td_h.simulate_pre_itc(n_sim,1)
    veto_margins = td_h.simulate_post_itc(n_sim,c,"veto",1)
    share_margins = td_h.simulate_post_itc(n_sim,c,"share",1)

    veto_shortfalls = td_h._process_shortfall_data(veto_margins)
    share_shortfalls = td_h._process_shortfall_data(share_margins)

    # true values (assuming calc object is tested)
    #true_margins = np.array(convgen_sim) - np.array(netdem)
    netdem_sim = np.empty(convgen_sim.shape)
    for i in range(n_sim):
      netdem_sim[(i*td_h.n):((i+1)*td_h.n),:] = np.array(netdem)
    
    true_margins = convgen_sim - netdem_sim
    veto_flow_to_a1 = np.apply_along_axis(func1d = lambda x: float(BivariateHindcastMargin._get_veto_flow(m1=x[0],m2=x[1],c=c)),axis=1,arr=true_margins).reshape(-1,1)
    true_veto_margins = true_margins + np.concatenate(
      (veto_flow_to_a1,
       -veto_flow_to_a1),
      axis=1)

    dem_sim = np.empty(convgen_sim.shape)
    for i in range(n_sim):
      dem_sim[(i*td_h.n):((i+1)*td_h.n),:] = np.concatenate((gb_dem,irl_dem),axis=1)


    margins_demand = np.concatenate((true_margins,dem_sim),axis=1)
    share_flow_to_a1 = np.apply_along_axis(func1d = lambda x: float(BivariateHindcastMargin._get_share_flow(m1=x[0],m2=x[1],d1=x[2],d2=x[3],c=c)),axis=1,arr=margins_demand).reshape(-1,1)
    true_share_margins = true_margins + np.concatenate(
      (share_flow_to_a1,
       -share_flow_to_a1),
      axis=1).astype(np.float64)

    true_veto_shortfalls = td_h._process_shortfall_data(true_veto_margins)
    true_share_shortfalls = td_h._process_shortfall_data(true_share_margins)

    assert ((margins - true_margins) == 0).all()
    assert (veto_margins - true_veto_margins == 0).all()
    assert np.isclose(share_margins,true_share_margins).all()
    assert np.isclose(np.array(veto_shortfalls),np.array(true_veto_shortfalls)).all()
    assert np.isclose(np.array(share_shortfalls),np.array(true_share_shortfalls)).all()

  test_sim_time_id = np.array([0,3,4,5,3359,3360,3361,3362])
  true_shortfall_clusters = np.array([1,2,2,2,3,4,4,4])

  test_df = pd.DataFrame({
  "time_id":test_sim_time_id,
  "time_cyclical":test_sim_time_id%td_h.n,
  "m0": - test_sim_time_id - 1})


  assert np.sum(true_shortfall_clusters != np.array(td_h._get_shortfall_clusters(test_df,0)["shortfall_event_id"])) == 0

  test_sim_time_id = np.array([1,3362,3363])
  true_shortfall_clusters = np.array([1,2,2])

  test_df = pd.DataFrame({
  "time_id":test_sim_time_id,
  "time_cyclical":test_sim_time_id%td_h.n,
  "m0": - test_sim_time_id - 1})


  assert np.sum(true_shortfall_clusters != np.array(td_h._get_shortfall_clusters(test_df,0)["shortfall_event_id"])) == 0
