from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches	
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

import sys
from helper_functions import * 

def main():
  xlim = 10
  ylim = 10
  c = 5
  def veto_flow(x,y,c):
    z = np.zeros(x.shape)
    idx = (x<0)*(y > 0)
    z[idx] = np.clip(np.minimum(-x[idx],y[idx]),a_max=c,a_min=-c)

    idx = (x>0)*(y < 0)
    z[idx] = -np.clip(np.minimum(x[idx],-y[idx]),a_max=c,a_min=-c)

    return z

  def share_flow(x,y,c):
    d_ratio = 5.0/6 ## approximate ratio of GB to Ireland 
    z = veto_flow(x,y,c)
    idx = (x<c)*(y < c)*(x + y < 0)
    z[idx] = np.clip(d_ratio*y[idx] - (1-d_ratio)*x[idx],a_max=c,a_min=-c)

    return z

  #levels = np.arange(-c*1.1,c*1.1,0.01)
  levels = range(-6,6)
  plot_contours("veto_flow_contours.png",lambda x,y: veto_flow(x,y,c),xlim,ylim,levels=levels)
  plot_contours("share_flow_contours.png",lambda x,y: share_flow(x,y,c),xlim,ylim,levels=levels)
 
if __name__=="__main__":
   main()