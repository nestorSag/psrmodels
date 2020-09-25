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

  h1 = [(-l,l),(-c,l),(-c,c),(0,0),(0,-l),(-l,-l)] #VETO shortfall region for area 1
  h2 = [(-l,0),(0,0),(c,-c),(l,-c),(l,-l),(-l,-l)]  #VETO shorfall region for area 2
  
  l1 = mpatches.Patch(color='#817ef8', label=r'Area 1 shortfall region')
  l2 = mpatches.Patch(color='#be407f', label=r'Area 2 shortfall region')
  
  regions = [h1,h2]
  colors =['#817ef8','#be407f']
  legends = [l1,l2]
  title = ""
  outfile = "veto_shortfall_regions.png"
  lims = {"y_lims":[-l,l],"x_lims":[-l,l]}
  plot_regions(regions,colors,legends,title,outfile,lims,alpha=0.6,c=5)

  h1 = [(-l,l),(-c,l),(-c,c),(c,-c),(c,-l),(-l,-l)] #share shortfall region for area 1
  h2 = [(-l,c),(-c,c),(c,-c),(l,-c),(l,-l),(-l,-l)]  #share shorfall region for area 2
  
  regions = [h1,h2]
  outfile = "share_shortfall_regions.png"
  plot_regions(regions,colors,legends,title,outfile,lims,alpha=0.6,c=5)
 
if __name__=="__main__":
   main()