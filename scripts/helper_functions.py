from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.patches as mpatches	
import matplotlib.pyplot as plt
from matplotlib import rc

from calendar import monthrange
import pandas as pd
from dfply import *

from datetime import datetime as dt
import pytz, os, math, pytz

import ephem
from workalendar.europe import UnitedKingdom, Ireland
from pysolar.solar import get_altitude

import os

#elementwise functions
def days_in_month(date):
  return monthrange(date.year, date.month)[1]

def hour_of_year(date):
  return date.hour + 24 * (date.timetuple().tm_yday-1)

def weekend(date):
  return True if date.date().weekday() >= 5 else False

def winter_period(dt):
  if dt.month >= 10:
    return dt.year
  else:
    return dt.year - 1

# read UK-Ireland data and add yearly term to each row
def read_data(data_path="../../../data/energy/uk_ireland/InterconnectionData_Rescaled.txt"):
  #print("current wd: " + os.getcwd())
  df = pd.read_csv(data_path,sep=" ") 
  
  df.columns = [x.lower() for x in df.columns]
  #
  df >>= mutate(time = X.date + " " +  X.time) >>\
  mutate(time = X.time.apply(lambda x: dt.strptime(x,"%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc))) >>\
  drop(["date"]) >>\
  mutate(period = X.time.apply(lambda x: winter_period(x)))
  #
  return df

# plot different regions in the plane with their own legends (Margins plane)
def plot_regions(regions,fillings,legends,title,outfile,frame_lims=None,c=None,space="margin",color_fill=True,**kwargs):

  def reverse_axis(tpl):
    return tuple(reversed(tpl))

  n = len(regions)
  
  rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
  ## for Palatino and other serif fonts use:
  #rc('font',**{'family':'serif','serif':['Palatino']})
  rc('text', usetex=True)
  pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
  plt.rcParams.update(pgf_with_rc_fonts)

  
  xmax = max([max([x[0] for x in region]) for region in regions])
  xmin = min([min([x[0] for x in region]) for region in regions])
  ymax = max([max([x[1] for x in region]) for region in regions])
  ymin = min([min([x[1] for x in region]) for region in regions])

  if color_fill:
    polygons = [plt.Polygon(regions[i],True,facecolor=fillings[i],edgecolor=fillings[i],lw=3) for i in range(n)]
    patches = PatchCollection(polygons,match_original=True)
  else:
    polygons = [plt.Polygon(regions[i],hatch=fillings[i], color='black', fill=False) for i in range(n)]
    

  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(111)

  if color_fill:
    ax.add_collection(patches)
  else:
    for polygon in polygons:
      ax.add_patch(polygon)
  
  # add axis lines
  ax.axvline(x=0,ymin=ymin,ymax=ymax,color="grey",linestyle="--")
  ax.axhline(y=0,xmin=xmin,xmax=xmax,color="grey",linestyle="--")

  ax.relim()
  ax.autoscale()
  
  if frame_lims is None:
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
  else:
    ax.set_xlim(frame_lims["x_lims"])
    ax.set_ylim(frame_lims["y_lims"])

  ax.legend(handles=legends,prop={'size': 20})
  
  ax.set_title(title,fontsize=20)
  ax.set_xticks([0])
  ax.set_yticks([0])

  if space == "margin":
    ax.set_xlabel("$m_1$",fontsize=30)
    ax.set_ylabel("$m_2$",fontsize=30)
  else:
    ax.set_xlabel("$x_1$",fontsize=30)
    ax.set_ylabel("$x_2$",fontsize=30)

  ax.tick_params(axis='both', which='major', labelsize=30)

  if c is not None:
    ax.annotate('(-c,c)', xy=(-c, c), xytext=(-c, c+0.5),size=20)
    c_lines = LineCollection([[(-c,0),(-c,c)],[(0,c),(-c,c)]],colors = "black", linewidth=2, linestyle = "--")
    ax.add_collection(c_lines)

  if space == "convgen_veto":
    ax.annotate(r'$(d_{1} - w_{1},d_{2} - w_{2})$',xy=(2*c,2*c), xytext=(2*c,2*c + c),size=23, arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate(r'$(d_{1} - w_{1} + c,d_{2} - w_{2} - c)$', xy=(3*c,c), xytext=(2.1*c,c + c),size=23, arrowprops=dict(facecolor='black', shrink=0.05))


  if space == "convgen_share":
    ax.annotate(r'$(d_{1} - w_{1}-c,d_{2} - w_{2} + c)$',xy=(c,3*c), xytext=(2*c,2*c + c),size=23, arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate(r'$(d_{1} - w_{1} + c,d_{2} - w_{2} - c)$', xy=(3*c,c), xytext=(2.1*c,c + c),size=23, arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate(r'$(0,\frac{d_{2}}{d_{1}}(w_{1} - d_{1} + c) + d_{2} - w_{2} + c)$',xy=(0,2*c), xytext=(1.5*c,2.5*c),size=23, arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate(r'$(\frac{d_{2}}{d_{1}}(w_{1} - d_{1} - c) + d_{2} - w_{2} - c,0)$',xy=(2*c,0), xytext=(0.3*c,1.5*c),size=23, arrowprops=dict(facecolor='black', shrink=0.05))

  if not os.path.exists("figures/"):
    os.mkdir("figures/")
    
  #plt.figure(figsize=(12,8))
  plt.savefig("figures/" + outfile)
  #plt.show()


def create_ts_features(holidays_obj,sun_position_lat,sun_position_lon):
  # read data and add the following features that will be used for detrending
  # sunlight_prop: proportion of sunlight that insides over the northern hemisphere at a given time
  #   used as a proxy for month, without adding multiple categorical levels
  # sun_position: position of the sun with respect to the horizon at a given time
  #   can be used as a proxiy for time of day without adding multiple categorical levels
  # working_day: whether a give ntime belongs to a working day
  #   can be used to take national holidays into account
  df = read_data()

  def sunlight_proportion(theta):
    alpha = 23.5/180.0*math.pi
    return 0.5 + 0.5*math.sin(theta)*math.sin(alpha)*math.sqrt(math.sin(alpha)**2 * math.sin(theta)**2 + math.cos(alpha)**2)

  def get_equinox_dates(current_day):
    y = current_day.year
    year_equinox_date = dt.strptime(str(ephem.next_equinox(str(y))),"%Y/%m/%d %H:%M:%S").replace(tzinfo=pytz.utc)
    
    if current_day > year_equinox_date:
      last_equinox_date = year_equinox_date
      next_equinox_date = dt.strptime(str(ephem.next_equinox(str(y+1))),"%Y/%m/%d %H:%M:%S").replace(tzinfo=pytz.utc)
    else:
      next_equinox_date = year_equinox_date
      last_equinox_date = dt.strptime(str(ephem.next_equinox(str(y-1))),"%Y/%m/%d %H:%M:%S").replace(tzinfo=pytz.utc)
      
    return last_equinox_date, next_equinox_date

  def calculate_sunlight_prop(time):
    last_eq, next_eq = get_equinox_dates(time)
    equinox_diff_sec = (next_eq - last_eq).total_seconds()
    offset_sec = (time - last_eq).total_seconds()

    rads = 2*math.pi*offset_sec/equinox_diff_sec

    return sunlight_proportion(rads)    
  
  # earth's tilt in radians
  alpha = 23.5/180*math.pi
  df >>= mutate(month = X.time.apply(lambda x: x.month),
                dow = X.time.apply(lambda x: x.weekday()),
                hour = X.time.apply(lambda x: x.hour),
                sunlight_prop = X.time.apply(calculate_sunlight_prop),
                sun_position = X.time.apply(lambda x: get_altitude(sun_position_lat,sun_position_lon,x.replace(tzinfo=pytz.utc))),
                is_working_day = X.time.apply(lambda x: int(holidays_obj.is_working_day(x))))
  

  return df

def data_with_ts_features(country_as_str):

  if country_as_str.lower() == "gb":
    holiday_obj = UnitedKingdom()
    sun_position_lon = 0.1278
    sun_position_lat = 51.5074
  elif country_as_str.lower() == "ireland":
    holiday_obj = Ireland()
    sun_position_lat = 53.3498
    sun_position_lon = 6.2603
    
  else:
    raise Exception("country not recognised")
  
  return create_ts_features(holiday_obj,sun_position_lat,sun_position_lon)


def plot_contours(outfile,f,xlim,ylim,**kwargs):
  # plot a function f contour lines in R2 using delimiters xlim, ylim. The function needs to operate on matrices
  fig = plt.figure(figsize=(10,10))

  xlist = np.linspace(-xlim, xlim, 200)
  ylist = np.linspace(-ylim, ylim, 200)
  X, Y = np.meshgrid(xlist, ylist)
  Z = f(X,Y)

  ax = fig.add_subplot(111)
  ax.set_xticks([0])
  ax.set_yticks([0])
  ax.set_xlabel("$M_1$",fontsize=30)
  ax.set_ylabel("$M_2$",fontsize=30)

  contour = plt.contour(X, Y, Z, colors='k',**kwargs)
  plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
  contour_filled = plt.contourf(X, Y, Z,**kwargs)
  plt.gca().set_aspect('equal', adjustable='box')


  #plt.get_legend().remove()
  if not os.path.exists("figures/"):
    os.mkdir("figures/")
  
  #plt.figure(figsize=(12,8))
  plt.savefig("figures/" + outfile,bbox_inches="tight")
