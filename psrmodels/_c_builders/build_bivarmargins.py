from cffi import FFI
import os

ffibuilder = FFI()

ffibuilder.cdef(""" 

	double triangle_prob(
		long origin_x,
		long origin_y,
		long triangle_length,
		long min_gen1,
		long min_gen2,
		long max_gen1,
		long max_gen2,
		double* gen1_cdf_array,
		double* gen2_cdf_array);

	double cond_epu_veto(
		long v1,
		long v2,
		long c,
		long min_gen1,
		long min_gen2,
		long max_gen1,
		long max_gen2,
		double* gen1_cdf_array,
		double* gen2_cdf_array,
		double* gen1_expectation);

	double cond_epu_share(
		long d1, 
		long d2,
		long v1,
		long v2,
		long c,
		long min_gen1,
		long min_gen2,
		long max_gen1,
		long max_gen2,
		double* gen1_cdf_array,
		double* gen2_cdf_array,
		double* gen1_expectation);

	double trapezoid_prob(
		long ul_x,
		long ul_y,
		long width,
		long min_gen1,
		long min_gen2,
		long max_gen1,
		long max_gen2,
		double* gen1_cdf_array,
		double* gen2_cdf_array);

	double get_cond_cdf(
		long min_gen1,
		long min_gen2,
		long max_gen1,
		long max_gen2,
		double* gen1_cdf_array,
		double* gen2_cdf_array,
		long m1,
		long m2,
		long v1,
		long v2,
		long d1,
		long d2,
		long c,
		int share_policy);

	void region_simulation(
		long n,
		long* simulations,
		long min_gen1,
		long min_gen2,
		long max_gen1,
		long max_gen2,
		double* gen1_cdf_array,
		double* gen2_cdf_array,
		long* net_demand,
		long* demand,
		long* row_weights,
		long n_rows,
		long m1,
		long m2,
		long c,
		int seed,
		int intersection,
		int share_policy);

	void conditioned_simulation(
		long n,
		long* simulations,
		long min_gen1,
		long min_gen2,
		long max_gen1,
		long max_gen2,
		double* gen1_cdf_array,
		double* gen2_cdf_array,
		long* net_demand,
		long* demand,
		long* row_weights,
		long n_rows,
		long m1,
		long c,
		int seed,
		int share_policy);

	void bivar_ecdf(
		double* ecdf,
		double* X,
		long n);

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