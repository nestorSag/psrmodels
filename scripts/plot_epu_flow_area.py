from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches	
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

import sys
from helper_functions import *

# flow regions under a share policy shortfall
def main():
  l= 10
  c = 5
  xmin = -l
  xmax = l
  ymin = -l
  ymax = l

  share_flow = [(-l,-l),(c,-l),(c,-c),(-c,c),(-l,c)]
  positive_flow = [(-l,l),(-l,c),(-c,c),(-c,l)]

  share_legend = mpatches.Patch(color='#ff9900', label=r'$\max \left( \min \left( \frac{D_1}{D_1 + D_2} (X_2 - V_2) - \frac{D_2}{D_1 + D_2} (X_1 - V_1), c\right),-c\right)$')
  positive_legend = mpatches.Patch(color='#008ae6', label=r'$ c $')
  
  regions = [share_flow,positive_flow]
  colors =["#ff9900","#008ae6"]
  legends = [share_legend,positive_legend]
  title = "flow $\Delta$ given that PU is non-zero"
  outfile = "epu_flow_regions.png"
  lims = {"x_lims":[-l,l+1],"y_lims":[-l,l]}
  
  plot_regions(regions,colors,legends,title,outfile,lims)

if __name__=="__main__":
   main()
