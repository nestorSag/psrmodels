from cffi import FFI
import os

ffibuilder = FFI()

ffibuilder.cdef(""" 
	void simulate_mc_power_grid(
		double *output, 
		double *transition_probs,
		double *states,
		double *initial_values,
		long n_generators,
		long n_simulations, 
		long n_timesteps, 
		long n_states,
		int random_seed,
		int simulate_streaks);

	void calculate_pre_itc_margins(
		double* gen_series,
		double* netdem_series,
		long period_length,
		long series_length,
		long n_areas);

	void calculate_post_itc_veto_margins(
		double* margin_series,
		long series_length,
		long n_areas,
		double c);

	void calculate_post_itc_share_margins(
		double* margin_series,
		double* dem_series,
		long period_length,
		long series_length,
		long n_areas,
		double c);

	""")

# with open('psrmodels/_c/libtimedependence.h','r') as f:
# 	ffibuilder.cdef(f.read())

header = "#include \"" + os.path.dirname(os.path.abspath(__file__)) + "/../_c/libtimedependence.h\""


ffibuilder.set_source("_c_ext_timedependence",  # name of the output C extension
    # """
    # #include "../../psrmodels/_c/libtimedependence.h"
    # """,
    header,
    sources=['psrmodels/_c/libtimedependence.c','psrmodels/_c/mtwist-1.5/mtwist.c'],
    libraries=['m'])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)