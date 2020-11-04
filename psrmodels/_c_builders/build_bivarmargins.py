from cffi import FFI
import os

ffibuilder = FFI()

ffibuilder.cdef(""" 

	double triangle_prob_py_interface(
	    int origin_x,
	    int origin_y,
	    int triangle_length,
	    int min_gen1,
	    int min_gen2,
	    int max_gen1,
	    int max_gen2,
	    double* gen1_cdf_array,
	    double* gen2_cdf_array);

	double cond_eeu_veto_py_interface(
	  int v1,
	  int v2,
	  int c,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array,
	  double* gen1_expectation);

	double cond_eeu_share_py_interface(
	  int d1, 
	  int d2,
	  int v1,
	  int v2,
	  int c,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array,
	  double* gen1_expectation);

	double trapezoid_prob_py_interface(
	  int ul_x,
	  int ul_y,
	  int width,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array);

	void region_simulation_py_interface(
	  int n,
	  int* simulations,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array,
	  int* net_demand,
	  int* demand,
	  int* row_weights,
	  int n_rows,
	  int m1,
	  int m2,
	  int c,
	  int seed,
	  int intersection,
	  int share_policy);

	void conditioned_simulation_py_interface(
	  int n,
	  int* simulations,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array,
	  int* net_demand,
	  int* demand,
	  int* row_weights,
	  int n_rows,
	  int m1,
	  int c,
	  int seed,
	  int share_policy);

	void bivar_ecdf_py_interface(
	  double* ecdf,
	  double* X,
	  int n);

	""")

# with open('psrmodels/_c/libbivarmargins.h','r') as f:
# 	ffibuilder.cdef(f.read())


header = "#include \"" + os.path.dirname(os.path.abspath(__file__)) + "/../_c/libbivarmargins.h\""

ffibuilder.set_source("_c_ext_bivarmargins",  # name of the output C extension
    # """
    # #include "../../psrmodels/_c/libbivarmargins.h"
    # """,
    header,
    sources=['psrmodels/_c/libbivarmargins.c','psrmodels/_c/libunivarmargins.c','psrmodels/_c/mtwist-1.5/mtwist.c'],
    libraries=['m'])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)