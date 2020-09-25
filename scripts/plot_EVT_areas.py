from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches	
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

import sys
from helper_functions import *

#plot plane regions in which each EVT model would be valid
def main():
  l= 10
  u = 5
  xmin = -l
  xmax = l
  ymin = -l
  ymax = l

  joint_tails = [(u,l),(u,u),(l,u),(l,l)]
  conditional_one = [(0,u),(u,u),(u,l),(0,l)]
  conditional_two = [(u,0),(l,0),(l,u),(u,u)]
  empirical = [(0,0),(u,0),(u,u),(0,u)]

  joint_legend = mpatches.Patch(color='#ff9933', label=r'Classical EVT')
  conditional_legend = mpatches.Patch(color='#ffcc00', label=r'Conditional EVT')
  empirical_legend = mpatches.Patch(color='#99cc00', label=r'Emprirical distribution')

  
  regions = [joint_tails,conditional_one,conditional_two,empirical]
  colors =["#ff9933","#ffcc00","#ffcc00","#99cc00"]
  legends = [joint_legend,conditional_legend,empirical_legend]
  title = "Models for demant-net-of-wind"
  outfile = "EVT_areas.png"
  
  plot_regions(regions,colors,legends,title,outfile)
 
if __name__=="__main__":
   main()
