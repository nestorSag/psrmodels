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

  h1 = [(-l,l),(-l,-l),(l,-l)]
  
  l1 = mpatches.Patch(color='#009999', label=r'Shortfall region')
  
  regions = [h1]
  colors =['#009999']
  legends = [l1]
  title = "Shortfall region for an infinite interconnector"
  outfile = "shortfall_region_infty.png"
  plot_regions(regions,colors,legends,title,outfile)
 
if __name__=="__main__":
   main()
