//#ifndef UNIVAR_MARGINS_H_INCLUDED
//#define UNIVAR_MARGINS_H_INCLUDED


/**
 * @brief This object represents the discrete probability distribution of available conventional generation
 *
 * @param min minimum possible value of available generation; can be non-zero if firm capacity is added or subtracted
 * @param max maximum possible value of available generation
 * @param cdf array with CDF values
 * @param expectation array with cumulative expectation values: expectation[k] = p_min*min + ... + p_k * k
 */

typedef struct DiscreteDistribution{
	int min;
	int max;
	double* cdf;
	double* expectation;
} DiscreteDistribution;

/**
 * @brief This object represents a vector of double observations
 *
 * @param value data
 * @param size data size (length)
 */
typedef struct IntVector{
  int* value;
  int size;
} IntVector;


/**
 * @brief This object represents a fitted Generalised Pareto tail model
 *
 * @param xi fitted shape parameter
 * @param sigma fitted scale parameter
 * @param u model threshold
 * @param p model threshold's quantile
 */
typedef struct GPModel{
	double xi;
	double sigma;
	double u;
	double p;
} GPModel;

/**
 * @brief This object represents a fitted Bayesian Generalised Pareto tail model
 *
 * @param u model threshold
 * @param p model threshold's quantile
 * @param xi posterior trace of shape parameter
 * @param sigma posterior trace of scale parameter
 * @param size trace length
 */

typedef struct PosteriorGPTrace{
	double u;
	double p;
	double* xi;
	double* sigma;
	int size;
} PosteriorGPTrace;

/**
 * @brief Wrapper that gets a value from the specified DiscreteDistribution's arrays
 *
 * @param F DiscreteDistribution
 * @param array array in the same DiscreteDistribution
 * @param x entry to fetch
 */

double cumulative_value(DiscreteDistribution* F, double* array, int x);

/**
 * @brief Returns CDF evaluated at x
 *
 */
double gen_cdf(DiscreteDistribution* F, int x);

/**
 * @brief Returns PDF evaluated at x
 *
 */
double gen_pdf(DiscreteDistribution* F, int x);

/**
 * @brief Returns cumulative expectation evaluated at x
 *
 */
double cumulative_expectation(DiscreteDistribution* F, int x);

double max(double num1, double num2);

double min(double num1, double num2);
/**
 * @brief Returns empirical CDF estimate from an available generation probability model, a net demand sample, and a value x to evaluate at
 *
 */

double empirical_power_margin_cdf(DiscreteDistribution* F, IntVector* net_demand, int x);

/**
 * @brief Returns expected energy unserved estimate from an available generation probability model and a net demand sample
 *
 */
double empirical_eeu(DiscreteDistribution* F, IntVector* net_demand);

/**
 * @brief Returns a Generalised Pareto CDF evaluated at x
 *
 */

double gpdist_cdf(GPModel* gp, double x);
/**
 * @brief Returns an exponential CDF evaluated at x
 *
 */
double expdist_cdf(GPModel* gp, double x);

/**
 * @brief Returns the CDF of the fitted tail model evaluated at x; wraps gp_cdf and expdist_cdf
 *
 */
double tail_model_cdf(GPModel* gp, double x);

/**
 * @brief Returns the CDF of the fitted posterior Bayesian tail model evaluated at x
 *
 */
double bayesian_tail_model_cdf(PosteriorGPTrace* gpt, double x);

/**
 * @brief Returns the empirical CDF of a net demand sample evaluated at x 
 *
 */
double empirical_net_demand_cdf(IntVector* net_demand, double x);

/**
 * @brief Returns the CDF of a net demand model with a GP tail component evaluated at x 
 *
 */
double semiparametric_net_demand_cdf(GPModel* gp, IntVector* net_demand, double x);

/**
 * @brief Returns the PDF of a net demand model with a GP tail component evaluated at x 
 *
 */
double semiparametric_net_demand_pdf(GPModel* gp, IntVector* net_demand, double x);

/**
 * @brief Returns the CDF of a net demand model with a Bayesian GP tail component evaluated at x 
 *
 */
double bayesian_semiparametric_net_demand_cdf(PosteriorGPTrace* gpt, IntVector* net_demand, double x);

/**
 * @brief Returns the PDF of a net demand model with a Bayesian GP tail component evaluated at x 
 *
 */
double bayesian_semiparametric_net_demand_pdf(PosteriorGPTrace* gpt, IntVector* net_demand, double x);

/**
 * @brief Returns the CDF of a power margin model whose net demand has a GP tail component, evaluated at x 
 *
 */
double semiparametric_power_margin_cdf(GPModel* gp, IntVector* net_demand, DiscreteDistribution* F, double x);

/**
 * @brief Returns the CDF of a power margin model whose net demand has a Bayesian GP tail component, evaluated at x 
 *
 */
double bayesian_semiparametric_power_margin_cdf(PosteriorGPTrace* gpt, IntVector* net_demand, DiscreteDistribution* F, double x);
//double bayesian_semiparametric_power_margin_cdf(PosteriorGPTrace* gpt, DiscreteDistribution* F, double x);

/**
 * @brief Returns the EEU estimate of a power margin model whose net demand has GP tail component
 *
 */
double semiparametric_eeu(GPModel* gp, IntVector* net_demand, DiscreteDistribution* F);

/**
 * @brief Returns the EEU estimate of a power margin model whose net demand has a Bayesian GP tail component
 *
 */
double bayesian_semiparametric_eeu(PosteriorGPTrace* gpt, IntVector* net_demand, DiscreteDistribution* F);



// python interfaces


void get_discrete_dist_from_py_objs(DiscreteDistribution* F, double* cdf, double* expectation, int min, int max);

void get_int_vector_from_py_objs(IntVector* vector, int* value, int size);

void get_gp_from_py_objs(GPModel* gp, double xi, double sigma, double u, double p);

void get_gpt_from_py_objs(PosteriorGPTrace* gpt, double* xi, double* sigma, double u, double p, int size);

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
  int x,
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