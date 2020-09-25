//#ifndef TIMEDEPENDENCE_H_INCLUDED
//#define TIMEDEPENDENCE_H_INCLUDED

/**
 * @brief simulate a time series of availability states for a single generator, simulating each step separately
 *
 * @param output 1D array where time series is going to be stored
 * @param states 1D array with state set of given generator
 * @param transition_probs  2D array with transition probabilities for given generator
 * @param x0 initial state. It has to be a valid state for the given generator
 * @param n_timesteps number of time steps to simulate
 * @param n_states number of possible states
 * @return @c void
 */
void simulate_mc_generator_steps(
	double *output, 
	double *states, 
	double *transition_probs,
	double x0,
	long n_timesteps, 
	long n_states);


long simulate_geometric_dist(
	double p);

double min(double num1, double num2) ;

double max(double num1, double num2);

/**
 * @brief simulate a time series of availability states for a single generator, simulating escape time: the time it takes for the generator to switch to a different state. 
 *
 * @param output 1D array where time series is going to be stored
 * @param states 1D array with state set of given generator
 * @param transition_probs  2D array with transition probabilities for given generator
 * @param x0 initial state. It has to be a valid state for the given generator
 * @param n_timesteps number of time steps to simulate
 * @param n_states number of possible states
 * @return @c void
 */
void simulate_mc_generator_streaks(
	double *output, 
	double *states, 
	double *transition_probs,
	double x0, 
	long n_timesteps, 
	long n_states);

long get_next_state_idx(
	double* prob_row, long current_state_idx);
/**
 * @brief simulate a time series of aggregated availabilities for a set of generators
 *
 * @param output 1D array where each time series simulation is going to be stores
 * @param transition_probs 1D array of transition probability matrices for each generator
 * @param states 1D array of state sets for each generator
 * @param initial_values 1D array of initial values for each generator
 * @param n_generators number of generators in given power grid
 * @param n_simulations number of time series to simulate
 * @param n_timesteps number of time steps to simulate
 * @param n_states number of possible states for each generator
 * @param random_seed random seed
 * @param simulate_streaks simulate loop lengths instead of separate steps. If there is a state with a large stationary probability (>0.5) this might be faster
 * @return @c void
 */
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

/**
 * @brief Calculate bivariate pre-interconnection margins
 *
 * @param gen_series available conventional generation time series matrix in row major order
 * @param netdem_series net demand time series matrix in row major order
 * @param period_length length of net demand time series
 * @param series_length length of generation series
 * @param n_areas number of areas
 */

void calculate_pre_itc_margins(
	double* gen_series,
	double* netdem_series,
	long period_length,
	long series_length,
	long n_areas);

// gets the minimum between 3 values

double min3(double a, double b, double c);

/**
 * @brief get power flow to area 1 under a veto policy
 *
 * @param m1 margin of area 1
 * @param m2 margin of area2
 * @param c interconnection capacity
 */

double get_veto_flow(double m1, double m2, double c);


/**
 * @brief get power flow to area 1 under a share policy 
 *
 * @param m1 margin of area 1
 * @param m2 margin of area2
 * @param d1 demand at area 1
 * @param d2 demand at area 2
 * @param c interconnection capacity
 */

double get_share_flow(
	double m1,
	double m2,
	double d1,
	double d2,
	double c);

/**
 * @brief calculate post-interconnection margins under a veto policy
 *
 * @param margin_series1 time series of margins matrix in row major order
 * @param series_length margin time series length
 * @param n_areas number of areas
 * @param c interconnection capacity
 */

void calculate_post_itc_veto_margins(
	double* margin_series,
	long series_length,
	long n_areas,
	double c);


/**
 * @brief calculate post-interconnection margins under a share policy
 *
 * @param margin_series time series of margins matrix in row major order
 * @param dem_series time series of demands matrix in row major order
 * @param period_length length of demand time series
 * @param series_length length of margin time series
 * @param n_areas number of areas
 * @param c interconnection capacity
 */

void calculate_post_itc_share_margins(
	double* margin_series,
	double* dem_series,
	long period_length,
	long series_length,
	long n_areas,
	double c);

// *
//  * @brief Filter out non-outages for area 1 from a margin time series
//  *
//  * @param post_itc_series1 post interconnection margin time series for area 1
//  * @param post_itc_series2 post interconnection margin time series for area 2
//  * @param series_length length of margin time series
//  * @return @c outages in area 1
 

// double* get_outages(
// 	double* post_itc_series1,
// 	double* post_itc_series2,
// 	long series_length);

//#endif
