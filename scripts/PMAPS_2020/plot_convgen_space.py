from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches	
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

import sys
from helper_functions import * 

def main():

  # this function is to make the plots with the X1 axis as the x-axis. originally it was in the y axis but it was confusing for some
  def reverse_axis(coord_list):
    return [(y,x) for (x,y) in coord_list]
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

  
  no_imports = reverse_axis([(0,0),(x2,0),(x2,y2),(0,y2)])
  non_saturated = reverse_axis([(x2,0),(x3,0),(x3,y1),(x2,y2)])
  saturated = reverse_axis([(x3,0),(l,0),(l,y1),(x3,y1)])

  colors =['#238A8DFF','#FDE725FF','#481567FF']
  
  l1 = mpatches.Patch(color=colors[0], label=r'No imports available for $A_1$')
  l2 = mpatches.Patch(color=colors[1], label=r'Non-saturated flow to $A_1$')
  l3 = mpatches.Patch(color=colors[2], label=r'Saturated flow to $A_1$')
  
  regions = [no_imports,non_saturated,saturated]
  legends = [l1,l2,l3]
  title=""
  outfile = "veto_convgen_regions.png"
  lims = {"y_lims":[0,l],"x_lims":[0,l]}
  plot_regions(regions,colors,legends,title,outfile,lims,alpha=0.6,c=c,space="convgen_veto")



  saturated_exports = reverse_axis([(0,y2),(x1,y3),(0,y3)])
  non_saturated = reverse_axis([(0,y2),(x1,y3),(x3,y1),(x2,0),(0,0)])
  saturated_imports = reverse_axis([(x2,0),(l,0),(l,y1),(x3,y1)])
  
  l1 = mpatches.Patch(color=colors[0], label=r'saturated exports from $A_1$')
  l2 = mpatches.Patch(color=colors[1], label=r'non-saturated flow')
  l3 = mpatches.Patch(color=colors[2], label=r'Saturated imports to $A_1$')
  
  regions = [saturated_exports,non_saturated,saturated_imports]
  legends = [l1,l2,l3]
  title=""
  outfile = "share_convgen_regions.png"
  lims = {"y_lims":[0,l],"x_lims":[0,l]}
  plot_regions(regions,colors,legends,title,outfile,lims,alpha=0.6,c=c,space="convgen_share")

 
if __name__=="__main__":
   main()
