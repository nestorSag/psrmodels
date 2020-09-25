//#ifndef BIVAR_MARGINS_H_INCLUDED
//#define BIVAR_MARGINS_H_INCLUDED

double sign(double x);

long min(long num1, long num2);

/**
 * @brief Calculate the probability mass of a straight triangular lattice
 *
 * @param origin_x x coordinate of lower left corner
 * @param origin_y y coordinate of lower left corner
 * @param triangle_length of cathethuses
 * @param max_gen1 maximum available generation in area 1
 * @param max_gen2 maximum available generation in area 2
 * @param gen1_cdf_array array of CDF values for area 1 generation
 * @param gen2_cdf_array array of CDF values for area 2 generation
 */

double triangle_prob(
	long origin_x,
	long origin_y,
	long triangle_length,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array);

/**
 * @brief Calculates the probability mass of a vertical stripe segment in bivariate conv. gen. space
 *
 * @param x1 x coordinate
 * @param x2 y coordinate
 * @param max_gen1 maximum available generation in area 1
 * @param max_gen2 maximum available generation in area 2
 * @param gen1_cdf_array array of CDF values for area 1 generation
 * @param gen2_cdf_array array of CDF values for area 2 generation
 */

double x1_stripe_pdf(
	long x1,
	long x2,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array);

/**
 * @brief Calculate conditional EPU given demand and net demand values, under a share policy for a 2-area system
 *
 * @param d1 demand in area 1
 * @param d1 demand in area 2
 * @param v1 net demand in area 1
 * @param v2 net demand in area 2
 * @param c interconnector capacity
 * @param max_gen1 maximum available generation in area 1
 * @param max_gen2 maximum available generation in area 2
 * @param gen1_cdf_array array of CDF values for area 1 generation
 * @param gen2_cdf_array array of CDF values for area 2 generation
 * @param gen1_expectation array of expectation values for area 2 generation
 */
double cond_epu_share(
	long d1, 
	long d2,
	long v1,
	long v2,
	long c,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array,
	double* gen1_expectation);

/**
 * @brief Calculate conditional EPU given demand and net demand values, under a veto policy for a 2-area system
 *
 * @param v1 net demand in area 1
 * @param v2 net demand in area 2
 * @param c interconnector capacity
 * @param max_gen1 maximum available generation in area 1
 * @param max_gen2 maximum available generation in area 2
 * @param gen1_cdf_array array of CDF values for area 1 generation
 * @param gen2_cdf_array array of CDF values for area 2 generation
 * @param gen1_expectation array of expectation values for area 2 generation
 */

double cond_epu_veto(
	long v1,
	long v2,
	long c,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array,
	double* gen1_expectation);

/**
 * @brief evaluate the bivariate probability distribution of conventional generation in a specified point
 *
 * @param x1 conventional generation in area 1
 * @param x2 conventional generation in area 1
 * @param max_gen1 maximum generation in area 1
 * @param max_gen2 maximum generation in area 2
  * @param gen1_cdf_array array of CDF values for generation in area 1
 * @param gen2_cdf_array array of CDF values for generation in area 2
 */

double bigen_cdf(
	long x1,
	long x2,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array);

/**
 * @brief Calculates the probability of a trapezoidal region in conventional generation space.
 * The region is formed by stacking a right triangle where the hypotenuse faces to the right, on
 * top of a rectangle
 *
 * @param ul_x x coordinate of upper left corner
 * @param ul_y y coordinate of upper y corner
 * @param width width of trapezoid
 * @param max_gen1 maximum generation in area 1
 * @param max_gen2 maximum generation in area 2
 * @param gen1_cdf_array array of CDF values for generation in area 1
 * @param gen2_cdf_array array of CDF values for generation in area 2
 */

double trapezoid_prob(
	long ul_x,
	long ul_y,
	long width,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array);

/**
 * @brief Get points characterising the polygon in conventional generation space for a given margin in the given area under a share policy
 *
 * @param p_array array where the points are stored
 * @param m margin value
 * @param v1 net demand value in area 1
 * @param v2 net demand value in area 2
 * @param d1 demand in area 1
 * @param d2 demand in area 2
 * @param c interconnection capacity
 * @param area index (0 for area 1)
 */

void get_share_polygon_points(
	long* p_array,
	long m,
	long v1,
	long v2,
	long d1,
	long d2,
	long c,
	int i);

/**
 * @brief Get points characterising the polygon in conventional generation space for a given margin in the given area under a veto policy
 *
 * @param p_array array where the points are stored
 * @param m margin value
 * @param v1 net demand value in area 1
 * @param v2 net demand value in area 2
 * @param c interconnection capacity
 * @param area index (0 for area 1)
 */

void get_veto_polygon_points(
	long* p_array,
	long m,
	long v1,
	long v2,
	long c,
	int i);

/**
 * @brief Get the power that is exchanged, given initial power margins and interconnection capacity, under a veto policy
 *
 * @param max_gen1 maximum available generation in area 1
 * @param max_gen2 maximum available generation in area 2
 * @param gen1_cdf_array array of CDF values for area 1 generation
 * @param gen2_cdf_array array of CDF values for area 2 generation
 * @param m1 margin in area 1
 * @param m2 margin in area 2
 * @param v1 net demand in area 1
 * @param v2 net demand in area 2
 * @param d1 demand in area 2
 * @param d2 demand in area 2
 * @param c interconnection capacity
 * @param share_policy integer that determines if a veto or a share policy is simulated
 */


double get_cond_cdf(
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

/**
 * @brief Get the power that is exchanged, given initial power margins and interconnection capacity, under a share policy
 *
 * @param m1 margin in area 1
 * @param m2 margin in area 2
 * @param d1 demand in area 1
 * @param d2 demand in area 2
 * @param c interconnection capacity
 */

double share_flow(
	long m1,
	long m2,
	long d1,
	long d2,
	long c);

/**
 * @brief Get the power that is exchanged, given initial power margins and interconnection capacity, under a veto policy
 *
 * @param m1 margin in area 1
 * @param m2 margin in area 2
 * @param c interconnection capacity
 */

long veto_flow(
	long m1,
	long m2,
	long c);

/**
 * @brief get polygon perimeter value at a given x coordinate
 *
 * @param x x coordinate in which to evaluate
 * @param P1 points from the slopped segment that characterises the polygon
 */

long axis1_polygon_upper_bound(
	long x,
	long* P1);

/**
 * @brief get polygon perimeter value at a given x coordinate
 *
 * @param x x coordinate in which to evaluate
 * @param P2 points from the slopped segment that characterises the polygon
 */

long axis2_polygon_upper_bound(
	long x,
	long* P2);

/**
 * @brief Simulate conventional generation values conditioned to being lower than a given threshold
 *
 * @param u uniform random number in [0,1]
 * @param upper_bound upper generation bound
 * @param max_gen maximum generation capacity
 * @param gen_cdf array with CDF values for conventional generation
 */

long boxed_gen_simulation(
	double u,
	long upper_bound,
	long max_gen,
	double* gen_cdf);

/**
 * @brief Simulate values for conventional generation in both areas under a share policy, 
 * such that bivariate post interconnector margins fall into the region specified by m1, m2
 *
 * @param n number of simulated samples
 * @param simulations array where simulations are stored
 * @param max_gen1 maximum generation capacity at area 1
 * @param max_gen2 maximum generation capacity at area 2
 * @param gen1_cdf_array array with CDF values for conventional generation in area 1
 * @param gen1_cdf_array array with CDF values for conventional generation in area 1
 * @param net_demand array of net demand values, of shape (n x 2)
 * @param demand array of demand values, of shape (n x 2)
 * @param row_weights array with number of samples to get from each of the rows being passed
 * @param n_rows number of net demand data observations
 * @param m1 margin upper bound at area 1
 * @param m1 margin upper bound at area 2
 * @param c interconnection capacity
 * @param seed random seed
 * @param intersection integer that determines whether the simulated area is a union or intersection of inequalities
 * @param share_policy integer that determines whether a share or a veto policy is simulated
 */

void region_simulation(
	long n,
	long* simulations,
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

/**
 * @brief Calculate CDF for a bivariate hindcast margin model, given polygons induced by demand and net demand
 *
 * @param max_gen1 maximum generation capacity of area 1
 * @param max_gen1 maximum generation capacity of area 2
 * @param gen1_cdf_array array with CDF values for conventional generation in area 1
 * @param gen1_cdf_array array with CDF values for conventional generation in area 1
 * @param p_array_a1 point of -1 slope segment that characterises polygon of area 1
 * @param p_array_a2 point of -1 slope segment that characterises polygon of area 2
 * @param c1 horizontal length of sloped segment of plygon of area 1
 * @param c2 horizontal length of sloped segment of plygon of area 2
 */
double cond_cdf(
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array,
	long* p_array_a1,
	long* p_array_a2,
	long c1,
	long c2);

/**
 * @brief Simulates values in one axis conditioned on margin values on the other axis
 *
 * @param n number of simulated samples
 * @param simulations array where simulations are stored
 * @param max_gen1 maximum generation capacity at area 1
 * @param max_gen2 maximum generation capacity at area 2
 * @param gen1_cdf_array array with CDF values for conventional generation in area 1
 * @param gen1_cdf_array array with CDF values for conventional generation in area 1
 * @param net_demand array of net demand values, of shape (n x 2)
 * @param demand array of demand values, of shape (n x 2)
 * @param row_weights array with number of samples to get from each of the rows being passed
 * @param n_rows number of net demand data observations
 * @param m1 margin upper bound at area 1
 * @param c interconnection capacity
 * @param seed random seed
 * @param share_policy integer that determines whether a share or veto policy is simulated
 */
void conditioned_simulation(
	long n,
	long* simulations,
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


/**
 * @brief Computes empirical CDF values of bivariate data; divides by n+1 to avoid having probability 1 in some observation
 *
 * @param ecdf ECDF placeholder
 * @param x first component
 * @param y second component
 * @param n number of observations
 */

void bivar_ecdf(
	double* ecdf,
	double* X,
	long n);

/**
 * @brief get the rightmost X coordinate of the intersection or union of both polygons
 *
 * @param P1 array that characterises polygon for area 1
 * @param P2 array that characterises polygon for area 2
 * @param max_gen1 maximum generation for area 1
 * @param intersection whether to apply union or intersection to both polygons
 */

long get_joint_polygon_x_bound(
	long* P1,
	long* P2,
	long max_gen1,
	int intersection);

/**
 * @brief get the highest Y coordinate of the intersection or union of both polygons at a particular X coordinate
 *
 * @param x X coordinate
 * @param P1 array that characterises polygon for area 1
 * @param P2 array that characterises polygon for area 2
 * @param max_gen2 maximum generation for area 2
 * @param intersection whether to apply union or intersection to both polygons
 */

long get_joint_polygon_y_bound_given_x(
	long x,
	long* P1,
	long* P2,
	long max_gen2,
	int intersection);


//#endif
