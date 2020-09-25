from cffi import FFI
import os

ffibuilder = FFI()

ffibuilder.cdef(""" 
	double h_margin_cdf(
		long x, 
		long nd_length,
		long gen_min,
		long gen_max,
		long* nd_vals, 
		double* gen_cdf);

	double h_nd_cdf(
		double x,
		long nd_length,
		long* nd_vals);

	double ev_margin_cdf(
		long m,
		double u,
		double p,
		double sigma,
		double xi,
		long nd_length,
		long gen_min,
		long gen_max,
		long* nd_vals,
		double* gen_cdf
		);

	double bev_margin_cdf(
		long m,
		double u,
		double p,
		long n_posterior,
		double* sigma,
		double* xi,
		long nd_length,
		long gen_min,
		long gen_max,
		long* nd_vals,
		double* gen_cdf
		);

	double h_epu(
		long nd_length,
		long gen_min,
		long gen_max,
		long* nd_vals, 
		double* gen_cdf,
		double* gen_expectation);

	double ev_epu(
		double u,
		double p,
		double sigma,
		double xi,
		long nd_length,
		long gen_min,
		long gen_max,
		long* nd_vals,
		double* gen_cdf,
		double* gen_expectation);

	double bev_epu(
		double u,
		double p,
		long n_posterior,
		double *sigma,
		double *xi,
		long nd_length,
		long gen_min,
		long gen_max,
		long* nd_vals,
		double* gen_cdf,
		double* gen_expectation);

	""")

header = "#include \"" + os.path.dirname(os.path.abspath(__file__)) + "/../_c/libunivarmargins.h\""

ffibuilder.set_source("_c_ext_univarmargins",  # name of the output C extension
    # """
    # #include "../../psrmodels/_c/libunivarmargins.h"
    # """,
    header,
    sources=['psrmodels/_c/libunivarmargins.c'],
    libraries=['m'])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)