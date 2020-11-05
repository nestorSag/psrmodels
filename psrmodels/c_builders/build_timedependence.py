from cffi import FFI
import os

ffibuilder = FFI()

ffibuilder.cdef(""" 
void calculate_post_itc_share_margins_py_interface(
  double* margin_series,
  double* dem_series,
  int period_length,
  int series_length,
  double c);

void calculate_post_itc_veto_margins_py_interface(
  double* margin_series,
  int series_length,
  double c);


void calculate_pre_itc_margins_py_interface(
  double* gen_series,
  double* netdem_series,
  int period_length,
  int series_length);

void simulate_mc_power_grid_py_interface(
    double *output, 
    double *transition_probs,
    double *states,
    double *initial_values,
    int n_generators,
    int n_simulations, 
    int n_timesteps, 
    int n_states,
    int random_seed,
    int simulate_streaks);


	""")

# with open('psrmodels/_c/libtimedependence.h','r') as f:
# 	ffibuilder.cdef(f.read())

header = "#include \"" + os.path.dirname(os.path.abspath(__file__)) + "/../c/libtimedependence.h\""


ffibuilder.set_source("_c_ext_timedependence",  # name of the output C extension
    # """
    # #include "../../psrmodels/_c/libtimedependence.h"
    # """,
    header,
    sources=['psrmodels/c/libtimedependence.c','psrmodels/c/mtwist-1.5/mtwist.c'],
    libraries=['m'])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)