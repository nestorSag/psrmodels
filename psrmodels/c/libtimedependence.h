//#ifndef TIMEDEPENDENCE_H_INCLUDED
//#define TIMEDEPENDENCE_H_INCLUDED


/**
 * @brief Wrapper around a 2-dimensional double array
 *
 */

typedef struct DoubleMatrix{
  double* value;
  int n_rows;
  int n_cols;
} DoubleMatrix;

/**
 * @brief Markov chain wrapper
 *
 * @param states pointer to array of states
 * @param transition_probs pointer to array of transition probabilities
 * @param initial_state state at time t=0
 * @param n_states number of states
 */
typedef struct MarkovChain{
	double* states;
	double* transition_probs;
	double initial_state;
	int n_states;
} MarkovChain;

/**
 * @brief This object represents a bivariate discrete distribution with independent components
 *
 * @param x first component
 * @param y second component
 */
typedef struct MarkovChainArray{
	MarkovChain* chains;
	int size;
} MarkovChainArray;

/**
 * @brief This object represents a bivariate discrete distribution with independent components
 *
 * @param x first component
 * @param y second component
 */
typedef struct TimeSimulationParameters{
	int n_simulations;
	int n_timesteps;
	int seed;
	int simulate_streaks;

} TimeSimulationParameters;

/**
 * @brief simulate a time series of availability states for a single generator, simulating each step separately
 *
 * @param output 1D array where time series is going to be stored
 * @param chain wrapper for a Markov Chain data
 * @param n_timesteps number of time steps to simulate
 * @return @c void
 */
void simulate_mc_generator_steps(double *output, MarkovChain* chain, int n_timesteps);

double get_element(DoubleMatrix* m, int i, int j);

void set_element(DoubleMatrix* m, int i, int j, double x);

int simulate_geometric_dist(double p);

double min(double num1, double num2) ;

double max(double num1, double num2);

/**
 * @brief simulate a time series of availability states for a single generator, simulating escape time: the time it takes for the generator to switch to a different state. 
 *
 * @param output 1D array where time series is going to be stored
 * @param chain wrapper for a Markov Chain data
 * @param n_timesteps number of time steps to simulate
 * @return @c void
 */
void simulate_mc_generator_streaks(double* output, MarkovChain* chain, int n_timesteps);

int get_next_state_idx(
	double* prob_row, int current_state_idx);
/**
 * @brief simulate a time series of aggregated availabilities for a set of generators
 *
 * @param output 1D array where each time series simulation is going to be stores
 * @param mkv_chains Pointer to MarkovChainArray object
 * @param pars Wrapper of simulation parameter values
 * @return @c void
 */
void simulate_mc_power_grid(DoubleMatrix* output, MarkovChainArray* mkv_chains, TimeSimulationParameters* pars);

/**
 * @brief Calculate bivariate pre-interconnection margins
 *
 * @param gen_series struct that wraps a generation time series simulation
 * @param netdem_series struct that wraps net demand time series
 */

void calculate_pre_itc_margins(DoubleMatrix* gen_series, DoubleMatrix* netdem_series);

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
 * @param power_margin_matrix struct for power margin matrix values
 * @param c interconnection capacity
 */

void calculate_post_itc_veto_margins(DoubleMatrix* power_margin_matrix, double c);


/**
 * @brief calculate post-interconnection margins under a share policy
 *
 * @param power_margin_matrix struct for power margin matrix values
 * @param power_margin_matrix struct for demand matrix values
 * @param c interconnection capacity
 */

void calculate_post_itc_share_margins(DoubleMatrix* power_margin_matrix, DoubleMatrix* demand_matrix, int c);

void get_double_matrix_from_py_objs(DoubleMatrix* m, double* value, int n_rows, int n_cols);

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

//#endif
