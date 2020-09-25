#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mtwist-1.5/mtwist.h"
#include "libtimedependence.h"

// documentation is in .h files 

void simulate_mc_generator_steps(
	double *output, 
	double *states, 
	double *transition_probs,
	double x0,
	long n_timesteps, 
	long n_states){
	// simulate each step in a power availability time series

	long current_state_idx = 0, current_timestep = 0, k = 0;
	double cdf, u;

	//find index of initial state to get initial transition probability row
	while(states[current_state_idx] != x0){
		++current_state_idx;
	}
	output[current_timestep] = states[current_state_idx]; //initial state output

	double *prob_row = &transition_probs[n_states*current_state_idx];


	// simulate n hours and save inplace in output
	for(current_timestep = 1; current_timestep <= n_timesteps; ++current_timestep){
		cdf = 0;
		k = 0;
		u = mt_drand();

		while(cdf < u){
			cdf += prob_row[k];
			++k;
		}


		prob_row = &transition_probs[n_states*(k-1)];
		output[current_timestep] = states[k-1];

	}

}

long simulate_geometric_dist(
	double p){
	// p = probability of looping back to same state
	double u = mt_drand();
	long x;

	//printf("p: %f, u: %f\n",p,u);

	if(u<=p){
		return 0;
	}else{
		x = (long) ceil(log(1.0-u)/log(1.0-p)-1.0);
		return x;
	}
}

long get_next_state_idx(
	double* prob_row, long current_state_idx){

	double u, cdf = 0.0, escape_prob = 1.0 - prob_row[current_state_idx]; //total_prob = probabiliti mass f(x)

	long j = 0;

	u = mt_drand();
	while(cdf < u){
		if(j != current_state_idx){
			cdf += prob_row[j]/escape_prob;
		}
		++j;
	}

	return j-1;

}

/**
 * Find minimum between two numbers.
 */
double min(double num1, double num2) 
{
    return (num1 > num2 ) ? num2 : num1;
}

double max(double num1, double num2) 
{
    return (num1 > num2 ) ? num1 : num2;
}


void simulate_mc_generator_streaks(
	double *output, 
	double *states, 
	double *transition_probs,
	double x0, 
	long n_timesteps, 
	long n_states){
	// simulate 'escape time': number of steps before leaving a state
	// if one state has a large stationary probability (> 0.5) this method might be faster
	// than simply simulating each step

	//find index of initial state to get initial transition probability row
	long current_state_idx = 0, current_timestep = 0, k = 0, streak, next_state_idx;
	double current_state;

	while(states[current_state_idx] != x0){
		++current_state_idx;
	}

	// get initial row and loop probability
	double *prob_row = &transition_probs[n_states*current_state_idx];

	double prob_loop = prob_row[current_state_idx];

	while(current_timestep <= n_timesteps){
		current_state = states[current_state_idx];
		//printf("current timestep: %d\n",current_timestep);
		//printf("current state id: %d\n",current_state_idx);
		streak = min(simulate_geometric_dist(1.0-prob_loop), n_timesteps - current_timestep);
		//printf("streak: %d\n",streak);
		next_state_idx = get_next_state_idx(prob_row,current_state_idx);
		//printf("next state id: %d\n",next_state_idx);

		for(k=current_timestep;k<current_timestep+streak+1;++k){
			output[k] = current_state;
		}

		// update objects
		current_timestep += streak+1;
		current_state_idx = next_state_idx;
		prob_row = &transition_probs[n_states*current_state_idx];
		prob_loop = prob_row[current_state_idx];

	}

}


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
	int simulate_streaks){

	mt_seed32(random_seed);

	long i = 0, j = 0, k = 0;

	long output_length = n_timesteps+1; //number of simulated timesteps + initial state

	double aggregate[output_length];

	for(i=0;i<n_simulations;++i){

		// initialise auxiliary aggregator

		for(k=0;k<output_length;++k){
			aggregate[k] = 0;
		}

		for(j=0;j<n_generators;++j){
			// get generators' output
			double gen_output[n_timesteps];
			
			if(simulate_streaks > 0){

				simulate_mc_generator_streaks(
					gen_output,
					&states[n_states*j],
					&transition_probs[n_states*n_states*j],
					initial_values[j],
					n_timesteps,
					n_states);

			}else{

				simulate_mc_generator_steps(
					gen_output,
					&states[n_states*j],
					&transition_probs[n_states*n_states*j],
					initial_values[j],
					n_timesteps,
					n_states);
			}

			for(k = 0; k < output_length; ++k){
				aggregate[k] += gen_output[k];
			}

		}

		for(k = 0; k < output_length; ++k){
			output[i*output_length+k] = aggregate[k];
		}

	}

}

void calculate_pre_itc_margins(
	double* gen_series,
	double* netdem_series,
	long period_length,
	long series_length,
	long n_areas){

	// Assuming row-major order
	long i = 0, j=0, k=0;
	for(k=0;k<n_areas;++k){
		for(i = 0; i < series_length; ++i){
			gen_series[k + n_areas*i] += (-netdem_series[k + n_areas*j]);
			// this is to avoid using integer remainder operators, which is expensive
			j += 1;
			if(j==period_length){
				j = 0;
			}
		}
	}

}


double min3(double a, double b, double c){
	if(a < b && a < c){
		return a;
	}else{
		if(b < c){
			return b;
		}else{
			return c;
		}
	}
}

double get_veto_flow(double m1, double m2, double c){
	double transfer = 0;
	if(m1 < 0 && m2 > 0){
		transfer = min3(-m1,m2,c);
	}else{
		if(m1 > 0 && m2 < 0){
			transfer = -min3(m1,-m2,c);
		}
	}

	return transfer;
}

double get_share_flow(
	double m1,
	double m2,
	double d1,
	double d2,
	double c){

	double alpha = d1/(d1+d2);
	double unbounded_flow = alpha*m2 - (1-alpha)*m1;

	if (m1+m2 < 0 && m1 < c && m2 < c){
		return min(max( unbounded_flow,-c),c);
	}else{
		return get_veto_flow(m1,m2,c);
	}
}

void calculate_post_itc_veto_margins(
	double* margin_series,
	long series_length,
	long n_areas,
	double c){

	long i;
	double flow;
	for(i=0;i<series_length;++i){
		flow = get_veto_flow(margin_series[i*n_areas],margin_series[i*n_areas+1],c);
		margin_series[i*n_areas] += flow;
		margin_series[i*n_areas+1] += (-flow);
	}
}

void calculate_post_itc_share_margins(
	double* margin_series,
	double* dem_series,
	long period_length,
	long series_length,
	long n_areas,
	double c){

	double flow;
	long i, j=0;
	for(i=0;i<series_length;++i){
		flow = get_share_flow(
			margin_series[i*n_areas],
			margin_series[i*n_areas+1],
			dem_series[j*n_areas],
			dem_series[j*n_areas+1],
			c);
		margin_series[i*n_areas] += flow;
		margin_series[i*n_areas+1] += (-flow);
		// this is to avoid using integer remainder operators, which is expensive
		j += 1;
		if(j==period_length){
			j = 0;
		}

	}
}


// double* get_outages(
// 	double* post_itc_series1,
// 	double* post_itc_series2,
// 	long series_length){

// 	long n_outages = 0;
// 	long i;

// 	for(i =0; i < series_length; ++i){
// 		if(post_itc_series1[i] < 0){
// 			n_outages += 1;
// 		}
// 	}

// 	double* outages_netdem = malloc(2*n_outages*sizeof(long));

// 	long j = 0;
// 	for(i =0; i < series_length; ++i){
// 		if(post_itc_series1[i] < 0){
// 			outages_netdem[j] = post_itc_series1[i];
// 			outages_netdem[n_outages + j] = post_itc_series2[i];
// 			j += 1;
// 		}
// 	}

// 	return outages_netdem
// }
