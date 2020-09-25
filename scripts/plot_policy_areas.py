from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches	
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc


import sys
from helper_functions import *

# plot areas of zero margin and areas with strictly negative margin
def main():
  c = 5
  l= 10
  xmin = -l
  xmax = l
  ymin = -l
  ymax = l

  share_leq = [(-l,l),(-l,-l),(c,-l),(c,-c),(-c,c),(-c,l)]
  share_l = [(-l,l),(-l,-l),(c-1,-l),(c-1,-c),(-c-1,c),(-c-1,l)]

  veto_leq = [(-l,l),(-l,-l),(c,-l),(c,-c),(-c,c),(-c,l)]
  veto_l = [(-l,l),(-l,-l),(-1,-l),(-1,0),(-c-1,c),(-c-1,l)]

  share_leq_legend = mpatches.Patch(color='#817ef8', label=r'share policy')
  share_l_legend = mpatches.Patch(color='#be407f', label=r'veto policy')

  regions = [share_leq,share_l,veto_leq,veto_l]
  colors =['#817ef8','#be407f','#817ef8','#be407f']
  legends = [share_leq_legend,share_l_legend]
  title = "Shortfall regions by policy"
  outfile = "policy_areas.png"
  
  plot_regions(regions,colors,legends,title,outfile,{"x_lims":[-10,10],"y_lims":[-10,10]})

if __name__=="__main__":
   main()
