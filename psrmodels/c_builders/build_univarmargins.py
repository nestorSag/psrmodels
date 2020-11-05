from cffi import FFI
import os

ffibuilder = FFI()

ffibuilder.cdef(""" 
	double empirical_power_margin_cdf_py_interface(
  int x, 
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals, 
  double* gen_cdf);

	double empirical_net_demand_cdf_py_interface(
	  double x,
	  int nd_length,
	  int* nd_vals);

	double semiparametric_power_margin_cdf_py_interface(
  int x,
  double u,
  double p,
  double sigma,
  double xi,
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals,
  double* gen_cdf
  );

  double bayesian_semiparametric_power_margin_cdf_py_interface(
    int m,
    double u,
    double p,
    int n_posterior,
    double* sigma,
    double* xi,
    int nd_length,
    int gen_min,
    int gen_max,
    int* nd_vals,
    double* gen_cdf
    );


	double empirical_eeu_py_interface(
	  int nd_length,
	  int gen_min,
	  int gen_max,
	  int* nd_vals, 
	  double* gen_cdf,
	  double* gen_expectation);

	double semiparametric_eeu_py_interface(
	  double u,
	  double p,
	  double sigma,
	  double xi,
	  int nd_length,
	  int gen_min,
	  int gen_max,
	  int* nd_vals,
	  double* gen_cdf,
	  double* gen_expectation);

	double bayesian_semiparametric_eeu_py_interface(
	  double u,
	  double p,
	  int n_posterior,
	  double *sigma,
	  double *xi,
	  int nd_length,
	  int gen_min,
	  int gen_max,
	  int* nd_vals,
	  double* gen_cdf,
	  double* gen_expectation);
	""")

header = "#include \"" + os.path.dirname(os.path.abspath(__file__)) + "/../c/libunivarmargins.h\""

ffibuilder.set_source("_c_ext_univarmargins",  # name of the output C extension
    # """
    # #include "../../psrmodels/_c/libunivarmargins.h"
    # """,
    header,
    sources=['psrmodels/c/libunivarmargins.c'],
    libraries=['m'])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)