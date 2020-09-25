from dfply import * 
from datetime import datetime, timedelta
from helper_functions import * 
from pysolar.solar import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import holidays, pytz




#instantiate holiday list objects 
uk_holidays = holidays.UnitedKingdom()
ireland_holidays = holidays.Ireland()

#read and transform data
df = pd.read_csv("../../../data/energy/uk_ireland/InterconnectionData.txt",sep=" ") 

#London coordinates
LONDON_LAT = 51.507372
LONDON_LON = -0.127557

#get features
df >>= mutate(time = X.date + " " +  X.time) >>\
  mutate(time = X.time.apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc))) >>\
  drop(["date"]) >>\
  mutate(\
  	year = X.time.apply(lambda x: x.year),\
  	month = X.time.apply(lambda x: x.month),\
  	day = X.time.apply(lambda x: x.day),\
  	hour = X.time.apply(lambda x: x.hour),\
  	hour_of_year = X.time.apply(lambda x: hour_of_year(x)),\
  	day_of_week = X.time.apply(lambda x: x.date().weekday()),\
  	uk_workday = X.time.apply(lambda x: x.date() not in uk_holidays and not weekend(x)),\
  	ireland_workday = X.time.apply(lambda x: x.date() not in ireland_holidays and not weekend(x)),\
    solar_alt = X.time.apply(lambda x: get_altitude(LONDON_LAT,LONDON_LON,x + timedelta(minutes=30))),\
    solar_azi = X.time.apply(lambda x: get_azimuth(LONDON_LAT,LONDON_LON,x + timedelta(minutes=30))))

#data plots
sns.set(style="darkgrid")

#plot Irish demand
sns.lineplot(x="time",y="Idem",data=df,hue="year")
plt.show()

sns.lineplot(x="hour_of_year",y="Idem",data=df,hue="year")
plt.show()

#demand of night hours. Shouldn't vary much, shouldn't show weekend effects
sns.lineplot(x="time",y="Idem",data=df.query("year == 2012 & hour == 3"))
plt.show()

#difference between midday and midnight
aux = df\
  .query("hour in [0,12] & day_of_week in [0,1,2,3,4] & ireland_workday")[["year","month","day","hour","Idem"]]\
  .groupby(["year","month","day"])\
  .apply(lambda x: pd.Series({"diff":x["Idem"].max() - x["Idem"].min()}))\
  .reset_index()

aux["day"] = range(aux.shape[0])

sns.lineplot(x="day",y="diff",data=aux)
plt.show()


#ratio between midday and midnight
aux = df\
  .query("hour in [0,12] & day_of_week in [0,1,2,3,4] & ireland_workday")[["year","month","day","hour","Idem"]]\
  .groupby(["year","month","day"])\
  .apply(lambda x: pd.Series({"diff":x["Idem"].max() / (1 + x["Idem"].min())}))\
  .reset_index()

aux["day"] = range(aux.shape[0])

sns.lineplot(x="day",y="diff",data=aux)
plt.show()

# check how the histogram of demand looks like
# clusters should be visible since it is clearly dependent on time of year
sns.distplot(df.query("hour == 12 & day_of_week == 1 & ireland_workday")["Idem"].dropna())
plt.show()

sns.distplot(df.query("year == 2011")["Iwind"].dropna(),color="skyblue",label="2011")
sns.distplot(df.query("year == 2012")["Iwind"].dropna(),color="purple",label="2012")
sns.distplot(df.query("year == 2013")["Iwind"].dropna(),color="orange",label="2013")
plt.show()

#plot sun altitude in london throughout the year
sns.lineplot(x="hour_of_year",y="solar_alt",data=df.query("hour_of_year <= 100"),hue="year")
plt.show()

#boxplot of demand by hour (working vs nonworking days)
df["weekend"] = df["day_of_week"] >= 5
sns.boxplot(x="hour",y="GBdem",data=df,hue="weekend")
plt.show()

# sunday vs saturday
sns.boxplot(x="hour",y="GBdem",data=df.query("day_of_week in [5,6]"),hue="day_of_week")
plt.show()

# weekday comparison
sns.boxplot(x="hour",y="GBdem",data=df.query("day_of_week in [0,1,2,3,4] & uk_workday"),hue="day_of_week")
plt.show()


#sun trajectory throughout the year
sns.scatterplot(x="solar_azi",y="solar_alt",data=df.query("year == 2012"),hue="hour_of_year", edgecolor=None)
plt.show()

#demand vs azimuth
sns.scatterplot(x="solar_azi",y="Idem",data=df.query("year == 2012"))
plt.show()

#demand vs sun altitude
sns.scatterplot(x="solar_alt",y="Idem",data=df.query("year == 2012"))
plt.show()


#plot irish wind
sns.lineplot(x="time",y="Iwind",data=df,hue="year")
plt.show()

sns.lineplot(x="hour_of_year",y="Iwind",data=df,hue="year")
plt.show()
