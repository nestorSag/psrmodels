# psrmodels: a package for adequacy of supply analysis in power systems

This packages implements some useful functions to do adequacy of supply analysis in single area and 2-area power systems. The focus is on time-collapsed models, but efficient simulation functionality is also implemented for time-sequential analysis. In the case of time-collapse models, some semi-parametric extreme value models are available for analysing both net demand (demand minus renewables) and power margins.

## Installation

The package is on PyPi, so 

```
pip install psrmodels 
```

is enough to install the package. It runs on Python `>=3.6`

## Usage

You can see the full documentation [here](https://nestorsag.github.io/psrmodels/index.html#package). Below some toy models are created for time-collapsed and time-sequential analysis

### toy bivariate time-collapsed and time-sequential models

```py
import numpy as np
import pandas as pd

from psrmodels.time_collapsed import BivariateHindcastMargin as tc_model
from psrmodels.time_collapsed import ConvGenDistribution as tc_convgen

from psrmodels.time_dependent import BivariateHindcastMargin as td_model
from psrmodels.time_dependent import ConvGenDistribution as td_convgen

# create toy demand and wind data with 100 observations
np.random.seed(1)

demand = np.random.normal(loc=1000,scale=50,size=(100,2))

wind = np.random.normal(loc=250,scale=50,size=(100,2))

# create toy generator data

gens =  pd.DataFrame({"Capacity": 50*np.ones(15), "Availability": 0.95*np.ones(15)})

# create 2 (identical) conventional generation distributions from generator data

convgen_dists = [tc_convgen(gens),tc_convgen(gens)]

# create time-collapsed bivariate hindcast model

model1 = tc_model(demand=demand,renewables=wind,gen_dists=convgen_dists)

## compute LOLE for area 1 under a 'veto' policy an 1000MW interconnection capacity
model1.lole(c=1000,policy="veto",axis=0)

# now, create a time-sequential model

# first, create time-sequential generators. We need to add a 'TTR' (time to repair) column to our generator data
gens["TTR"] = 50 #50 hours to repair on average
td_convgen_dists = [td_convgen(gens),td_convgen(gens)]

# create time-sequential model
model2 = td_model(demand=demand,renewables=wind,gen_dists=td_convgen_dists)

# simulate post-interconnection sequential power margin values under a veto policy and 1000MW interconnection
sim_data = model2.simulate_post_itc(n_sim=1000,c=1000,policy="veto")
```
