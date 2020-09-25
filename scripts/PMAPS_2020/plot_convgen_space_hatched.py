from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches	
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

import sys
from helper_functions import * 

def main():
  c = 2.5
  l= 10
  xmin = 0
  xmax = l
  ymin = 0
  ymax = l


  ##### plot veto regions in convgen
  x1 = c
  x2 = x1 + c
  x3 = x2 + c

  y1 = c
  y2 = y1 + c
  y3 = y2 + c

  no_imports = [(0,0),(x2,0),(x2,y2),(0,y2)]
  non_saturated = [(x2,0),(x3,0),(x3,y1),(x2,y2)]
  saturated = [(x3,0),(l,0),(l,y1),(x3,y1)]

  hatches = ["//","\\\\","--"]
  
  l1 = mpatches.Patch(hatch=hatches[0], label=r'No imports available for $A_i$',fill=False)
  l2 = mpatches.Patch(hatch=hatches[1], label=r'Non-saturated flow to $A_i$',fill=False)
  l3 = mpatches.Patch(hatch=hatches[2], label=r'Saturated flow to $A_i$',fill=False)
  
  regions = [no_imports,non_saturated,saturated]
  legends = [l1,l2,l3]
  title=""
  outfile = "veto_convgen_regions.png"
  lims = {"y_lims":[0,l],"x_lims":[0,l]}
  plot_regions(regions,hatches,legends,title,outfile,lims,alpha=0.6,c=c,space="convgen_veto",color_fill=False)



  saturated_exports = [(0,y2),(x1,y3),(0,y3)]
  non_saturated = [(0,y2),(x1,y3),(x3,y1),(x2,0),(0,0)]
  saturated_imports = [(x2,0),(l,0),(l,y1),(x3,y1)]
  
  l1 = mpatches.Patch(hatch=hatches[0], label=r'saturated exports from $A_i$',fill=False)
  l2 = mpatches.Patch(hatch=hatches[1], label=r'non-saturated flow',fill=False)
  l3 = mpatches.Patch(hatch=hatches[2], label=r'Saturated imports to $A_i$',fill=False)
  
  regions = [saturated_exports,non_saturated,saturated_imports]
  legends = [l1,l2,l3]
  title=""
  outfile = "share_convgen_regions.png"
  lims = {"y_lims":[0,l],"x_lims":[0,l]}
  plot_regions(regions,hatches,legends,title,outfile,lims,alpha=0.6,c=c,space="convgen_share",color_fill=False)

 
if __name__=="__main__":
   main()
