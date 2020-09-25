from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches	
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc


import sys
from helper_functions import *

def main():
  l= 10
  c = 5
  xmin = -l
  xmax = l
  ymin = -l
  ymax = l

  zero_flow = [(0,0),(l,0),(l,l),(0,l)]
  share_flow = [(-l,-l),(c,-l),(c,-c),(-c,c),(-l,c)]
  positive_flow = [(0,0),(-c,c),(-l,c),(-l,l),(0,l)]
  negative_flow = [(0,0),(c,-c),(c,-l),(l,-l),(l,0)]

  zero_flow_legend = mpatches.Patch(color='#cc3399', label=r'$0$')
  share_legend = mpatches.Patch(color='#ff9900', label=r'$\max \left( \min \left( \frac{D_1}{D_1 + D_2} (X_2 - V_2) - \frac{D_2}{D_1 + D_2} (X_1 - V_1), c\right),-c\right)$')
  positive_legend = mpatches.Patch(color='#008ae6', label=r'$ \min \{ V_1 - X_1, X_2 - V_2 , c\}$')
  negative_legend = mpatches.Patch(color='#00b3b3', label=r'$- \min \{ X_1 - V_1, V_2 - X_2, c\}$')
  
  regions = [zero_flow,share_flow,positive_flow,negative_flow]
  colors =["#cc3399","#ff9900","#008ae6","#00b3b3"]
  legends = [zero_flow_legend,share_legend,positive_legend,negative_legend]
  title = r"flow $\Delta^{share}$"
  outfile = "flow_regions.png"

  plot_regions(regions,colors,legends,title,outfile)
  
if __name__=="__main__":
   main()
