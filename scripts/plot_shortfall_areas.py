from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches	
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

import sys

from helper_functions import * 

def main():
  c = 5
  l= 10
  xmin = -l
  xmax = l
  ymin = -l
  ymax = l

  h1 = [(-l,l),(-l,-l),(-c,-l),(-c,l)]
  h2 = [(-c,-l),(0,-l),(0,0),(-c,c)]
  h3 = [(0,-l),(c,-l),(c,-c),(0,0)]
  
  l1 = mpatches.Patch(color='#cc0066', label=r'$h_1$')
  l2 = mpatches.Patch(color='#ff8000', label=r'$h_2$')
  l3 = mpatches.Patch(color='#ffff00', label=r'$h_3$')
  
  regions = [h1,h2,h3]
  colors =['#cc0066','#ff8000','#ffff00']
  legends = [l1,l2,l3]
  title = "Shortfall regions"
  outfile = "shortfall_areas.png"
  lims = {"y_lims":[-l,l],"x_lims":[-l,l+1]}
  plot_regions(regions,colors,legends,title,outfile,lims)
 
if __name__=="__main__":
   main()
