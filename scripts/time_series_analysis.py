import pandas as pd
import dfply
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt

import itertools

import sys
sys.append("scripts/")
from helper_functions import * 

def format_features(df,years):
	df["net_demand"] = np.clip(df["demand"] - df["wind"],a_min=0,a_max=np.Inf)
	df = df.query("period in {y}".format(y=years))# >> mutate(next_net_demand = lead(X.net_demand,1))
	df = df.dropna()
	#df["christmas_time"] = pd.to_datetime(["{y}-12-25 00:00:00+00:00".format(y=str(x)) for x in df["time"].dt.year])
	#df["new_year_time"] = pd.to_datetime(["{y}-01-01 00:00:00+00:00".format(y=str(x)) for x in df["time"].dt.year])
	#df >>= mutate(delta_l1 = X.net_demand_l1 - X.net_demand)
	y = np.array(df["net_demand"]) #variable to predict will be value of net demand
	# one hot encoding for categorical features
	x = pd.concat(
		[pd.get_dummies(df["dow"]),
		 pd.get_dummies(df["hour"]),
		 pd.get_dummies(df["is_working_day"]),
		 df[["sunlight_prop","sun_position"]]
		],axis=1)
	#
	#x = df[["dow","hour","is_working_day","sunlight_prop","sun_position","net_demand"]]
	return np.array(x), y

gb_data = data_with_ts_features("gb") >> rename(demand = X.gbdem_r, wind = X.gbwind_r)
irl_data = data_with_ts_features("ireland") >> rename(demand = X.idem_r, wind = X.iwind_r)
years = [2007,2008,2009]

gb_x, gb_y = format_features(gb_data,years)
irl_x, irl_y = format_features(irl_data,years)

def cv_adaboost(x,y,cv_folds=10,seed=1,n_estimators = [50,100,200,400],max_tree_depths = [3,5,7,9]):
  # train regression adaboost model with cross validated number of trees
  np.random.seed(seed)
  cv_labels = np.random.choice(range(cv_folds),x.shape[0])
  n_estimators_list = []
  max_tree_depths_list = []
  rmse = []
  models = []
  for par_tuple in itertools.product(n_estimators,max_tree_depths):
    n_est, mtd = par_tuple
    n_estimators_list.append(n_est)
    max_tree_depths_list.append(mtd)
    se = []
    error = 0
    for fold in range(cv_folds):
      train_labels = cv_labels != fold
      Y = y[train_labels]
      X = x[train_labels,:]
      #
      X_val = x[np.logical_not(train_labels),:]
      Y_val = y[np.logical_not(train_labels)]
      model = AdaBoostRegressor(random_state=0, n_estimators=n_est, base_estimator= DecisionTreeRegressor(max_depth=mtd), loss='exponential').fit(X,Y)
      se.append(np.sum((model.predict(X_val) - Y_val)**2))
      models.append(model)
        #
    rmse.append(np.sqrt(np.sum(se)/df.shape[0]))
  #
  print("minimum rmse: {x}".format(x=np.min(rmse)))
  the_model = models[np.argmin(rmse)]
  sns.lineplot(n_estimators,rmse)
  plt.show()
  #
  return the_model

gb_model = cv_adaboost(gb_x,gb_y)

#residuals
z = gb_y - gb_model.predict(gb_x)

sns.lineplot(range(len(z)),z)
plt.show()