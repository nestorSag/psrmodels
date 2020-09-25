//#ifndef UNIVAR_MARGINS_H_INCLUDED
//#define UNIVAR_MARGINS_H_INCLUDED

/**
 * @brief calculate maximum of two numbers
 */

long max(long num1, long num2);

/**
 * @brief Evaluate available conventional generation CDF or expectation
 *
 * @param y CDF argument
 * @param cdf_array pre-computed array of CDF values
 * @param gen_min minimum conventional generation capacity; can take a negative value under some circumstances
 * @param gen_max maximum conventional generation capacity
 */

double get_gen_array_val(
	long y, 
	double* cdf_array, 
	long gen_min,
	long gen_max);

/**
 * @brief calculate Hindcast power margin CDF
 *
 * @param x CDF argument
 * @param nd_length number of net demand data points 
 * @param gen_min minimum conventional generation capacity; can take a negative value under some circumstances
 * @param gen_max maximum conventional generation capacity
 * @param nd_vals array of net demand values
 * @param gen_cdf array of conventional generation CDF values
 */
double h_margin_cdf(
	long x, 
	long nd_length,
	long gen_min,
	long gen_max,
	long* nd_vals, 
	double* gen_cdf);

/**
 * @brief calculate EPU under hindcast model
 *
 * @param nd_length number of net demand data points 
 * @param gen_min minimum conventional generation capacity; can take a negative value under some circumstances
 * @param gen_max maximum conventional generation capacity
 * @param nd_vals array of net demand values
 * @param gen_cdf array of conventional generation CDF values
 * @param gen_expectation array of partial expectation values for conventional generation CDF
 */
double h_epu(
	long nd_length,
	long gen_min,
	long gen_max,
	long* nd_vals, 
	double* gen_cdf,
	double* gen_expectation);

/**
 * @brief evaluate a Generalized Pareto CDF function
 *
 * @param x CDF argument
 * @param u location parameter
 * @param sigma scale parameter
 * @param xi shape parameter
 */
double gpdist_cdf(
	double x,
	double u,
	double sigma,
	double xi);

/**
 * @brief evaluate a exponential distribution CDF function
 *
 * @param x CDF argument
 * @param u location parameter
 * @param sigma scale parameter
 */
double expdist_cdf(
	double x,
	double u,
	double sigma);

/**
 * @brief evaluate CDF of the GP approximation of the tail of net demand under an extreme values approach
 *
 * @param x CDF argument
 * @param u location parameter
 * @param sigma scale parameter
 * @param xi shape parameter
 */

double ev_nd_tail_cdf(
	long x,
	double u,
	double sigma,
	double xi);

/**
 * @brief evaluate CDF of the GP approximation of the tail of net demand under a Bayesian extreme values approach
 *
 * @param x CDF argument
 * @param u location parameter
 * @param sigma array of scale parameters
 * @param xi array of shape parameters
 */

double bev_nd_tail_cdf(
	long x,
	double u,
	long n_posterior,
	double* sigma,
	double* xi);

/**
 * @brief evaluate CDF of net demand under a Hindcast approach
 *
 * @param x CDF argument
 * @param nd_length number of net demand values
 * @param nd_vals array of net demand values
 */

double h_nd_cdf(
	double x,
	long nd_length,
	long* nd_vals);

/**
 * @brief evaluate CDF of net demand under an extreme values approach
 *
 * @param x CDF argument
 * @param u model threshold of GP approximation
 * @param p quantile of model threshold
 * @param sigma scale parameter of GP approximation
 * @param xi shape parameter of GP approximation
 * @param nd_length number of net demand values
 * @param nd_vals array of net demand values

 */
double ev_nd_cdf(
	long x,
	double u,
	double p,
	double sigma,
	double xi,
	long nd_length,
	long* nd_vals);

/**
 * @brief evaluate PDF of net demand under an extreme values approach
 *
 * @param x PDF argument
 * @param u model threshold of GP approximation
 * @param p quantile of model threshold
 * @param sigma scale parameter of GP approximation
 * @param xi shape parameter of GP approximation
 * @param nd_length number of net demand values
 * @param nd_vals array of net demand values

 */
double ev_nd_pdf(
	long x,
	double u,
	double p,
	double sigma,
	double xi,
	long nd_length,
	long* nd_vals);

/**
 * @brief evaluate CDF of net demand under a Bayesian extreme values approach
 *
 * @param x CDF argument
 * @param u model threshold of GP approximation
 * @param p quantile of model threshold
 * @param n_posterior number of posterior parameter simulations
 * @param sigma posterior samples for scale parameter
 * @param xi posterior samples for shape parameter
 * @param nd_length number of net demand values
 * @param nd_vals array of net demand values

 */
double bev_nd_cdf(
	long x,
	double u,
	double p,
	long n_posterior,
	double* sigma,
	double* xi,
	long nd_length,
	long* nd_vals);

/**
 * @brief evaluate PDF of net demand under a Bayesian extreme values approach
 *
 * @param x PDF argument
 * @param u model threshold of GP approximation
 * @param p quantile of model threshold
 * @param n_posterior number of posterior parameter simulations
 * @param sigma posterior samples for scale parameter
 * @param xi posterior samples for shape parameter
 * @param nd_length number of net demand values
 * @param nd_vals array of net demand values

 */

double bev_nd_pdf(
	long x,
	double u,
	double p,
	long n_posterior,
	double* sigma,
	double* xi,
	long nd_length,
	long* nd_vals);

/**
 * @brief evaluate CDF of power margins under an extreme values approach
 *
 * @param m CDF argument
 * @param u model threshold of GP net demand approximation
 * @param p quantile of model threshold
 * @param sigma location parameter of GP net demand tail approximation
 * @param xi shape parameter of GP net demand tail approximation
 * @param nd_length number of net demand values
 * @param gen_min minimum conventional generation capacity; can take a negative value under some circumstances
 * @param gen_max maximum conventional generation capacity
 * @param nd_vals array of net demand values
 * @param gen_cdf array of available conventional generation CDF

 */

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

/**
 * @brief evaluate CDF of power margins under a Bayesian extreme values approach
 *
 * @param m CDF argument
 * @param u model threshold of GP net demand approximation
 * @param p quantile of model threshold
 * @param n_posterior number of posterior parameter samples
 * @param sigma posterior samples of scale parameter for net demand GP approximation
 * @param xi posterior samples of shape parameter for net demand GP approximation
 * @param nd_length number of net demand values
 * @param gen_min minimum conventional generation capacity; can take a negative value under some circumstances
 * @param gen_max maximum conventional generation capacity
 * @param nd_vals array of net demand values
 * @param gen_cdf array of available conventional generation CDF

 */

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

/**
 * @brief calculate EPU for an EV model of net demand
 *
 * @param u model threshold of GP net demand approximation
 * @param p quantile of model threshold
 * @param sigma posterior samples of scale parameter for net demand GP approximation
 * @param xi posterior samples of shape parameter for net demand GP approximation
 * @param nd_length number of net demand values
 * @param gen_min minimum conventional generation capacity; can take a negative value under some circumstances
 * @param gen_max maximum conventional generation capacity
 * @param nd_vals array of net demand values
 * @param gen_cdf array of available conventional generation CDF values
 * @param gen_expectation array of available conventional generation's partial expectation values

 */
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


/**
 * @brief calculate EPU for a Bayesian EV model of net demand
 *
 * @param u model threshold of GP net demand approximation
 * @param p quantile of model threshold
 * @param n_posterior number of posterior parameter samples
 * @param sigma posterior samples of scale parameter for net demand GP approximation
 * @param xi posterior samples of shape parameter for net demand GP approximation
 * @param nd_length number of net demand values
 * @param gen_min minimum conventional generation capacity; can take a negative value under some circumstances
 * @param gen_max maximum conventional generation capacity
 * @param nd_vals array of net demand values
 * @param gen_cdf array of available conventional generation CDF values
 * @param gen_expectation array of available conventional generation's partial expectation values

 */
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


//#endif

