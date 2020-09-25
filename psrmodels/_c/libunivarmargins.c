#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "libunivarmargins.h"

// documentation is in .h files 

long max(long num1, long num2){
    return (num1 > num2 ) ? num1 : num2;
}

double get_gen_array_val(
	long y, 
	double* array,
	long gen_min, 
	long gen_max){

	if(y<gen_min){
		return 0;
	}else{
		if(y>=gen_max){
			return array[gen_max];
		}else{
			return array[y-gen_min];
		}
	}
}

// // same as function above but with double signature in first argument
// double get_gen_array_val_d(
// 	double y, 
// 	double* array,
// 	long gen_min, 
// 	long gen_max){

// 	if(y<gen_min){
// 		return 0;
// 	}else{
// 		if(y>=gen_max){
// 			return array[gen_max];
// 		}else{
// 			return array[(long y)-gen_min];
// 		}
// 	}
// }

// double get_gen_array_val(
// 	long y, 
// 	double* array,
// 	long gen_max){

// 	return get_gen_array_val(y,array,0,gen_max);
// }

double h_margin_cdf(
	long x, 
	long nd_length,
	long gen_min,
	long gen_max,
	long* nd_vals, 
	double* gen_cdf){

	double cdf = 0;

	long nd_idx;

	for(nd_idx=0;nd_idx<nd_length;++nd_idx){
		cdf += get_gen_array_val(nd_vals[nd_idx]+x,gen_cdf,gen_min,gen_max)/nd_length;
	}

	return cdf;
}

double h_epu(
	long nd_length,
	long gen_min,
	long gen_max,
	long* nd_vals, 
	double* gen_cdf,
	double* gen_expectation){

	double epu = 0;

	long nd_idx;

	for(nd_idx=0;nd_idx<nd_length;++nd_idx){
		epu += (nd_vals[nd_idx]*get_gen_array_val(nd_vals[nd_idx]-1,gen_cdf,gen_min,gen_max)-\
			get_gen_array_val(nd_vals[nd_idx]-1,gen_expectation,gen_min,gen_max))/nd_length;
	}

	return epu;
}

double gpdist_cdf(
	double x,
	double u,
	double sigma,
	double xi){

	double val;

	// assuming that always x > u for improved performance 
	if(xi>=0){
		val = 1.0 - pow(1.0 + xi*(x-u)/sigma,-1.0/xi);
	}else{
		if(x<u - sigma/xi){
			val = 1.0 - pow(1.0 + xi*(x-u)/sigma,-1.0/xi);
		}else{
			val = 1.0;
		}
	}

	return val;
}

double expdist_cdf(
	double x,
	double u,
	double sigma){

	return 1.0 - exp(-(x-u)/sigma);
}

double ev_nd_tail_cdf(
	long x,
	double u,
	double sigma,
	double xi){

	if(xi != 0){
		return gpdist_cdf((double) x,
				u,
				sigma,
				xi);
	}else{
		return expdist_cdf((double) x,
						u,
						sigma);
	}

}

double bev_nd_tail_cdf(
	long x,
	double u,
	long n_posterior,
	double* sigma,
	double* xi){

	double estimator=0;

	long i;


	for(i=0;i<n_posterior;++i){
		
		if(xi[i] != 0){
			estimator += gpdist_cdf(x,
					u,
					sigma[i],
					xi[i]);
		}else{
			estimator += expdist_cdf(x,
							u,
							sigma[i]);
		}
	}

	return estimator/n_posterior;

}

double h_nd_cdf(
	double x,
	long nd_length,
	long* nd_vals){

	long i;

	double nd_below_x = 0;

	for(i=0;i<nd_length;++i){
		if(nd_vals[i]<=x){
			nd_below_x += 1;
		}
	}
	return nd_below_x/nd_length;
}

double ev_nd_cdf(
	long x,
	double u,
	double p,
	double sigma,
	double xi,
	long nd_length,
	long* nd_vals){

	if(x <= u){
		return h_nd_cdf(x,nd_length,nd_vals);
	}else{
		return p + (1.0-p)*ev_nd_tail_cdf(x,u,sigma,xi);
	}
}

double ev_nd_pdf(
	long x,
	double u,
	double p,
	double sigma,
	double xi,
	long nd_length,
	long* nd_vals){

	return ev_nd_cdf(
		x,
		u,
		p,
		sigma,
		xi,
		nd_length,
		nd_vals) - \
	ev_nd_cdf(
		x-1,
		u,
		p,
		sigma,
		xi,
		nd_length,
		nd_vals);
}

double bev_nd_cdf(
	long x,
	double u,
	double p,
	long n_posterior,
	double* sigma,
	double* xi,
	long nd_length,
	long* nd_vals){

	if(x <= u){
		return h_nd_cdf(x,nd_length,nd_vals);
	}else{
		return p + (1.0-p)*bev_nd_tail_cdf(x,u,n_posterior,sigma,xi);
	}
}


double bev_nd_pdf(
	long x,
	double u,
	double p,
	long n_posterior,
	double* sigma,
	double* xi,
	long nd_length,
	long* nd_vals){

	return bev_nd_cdf(
		x,
		u,
		p,
		n_posterior,
		sigma,
		xi,
		nd_length,
		nd_vals) - \
	bev_nd_cdf(
		x-1,
		u,
		p,
		n_posterior,
		sigma,
		xi,
		nd_length,
		nd_vals);
}

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
	){

	long y;

	double cdf = 0;
	double pdf_val, g_cdf;

	for(y=0;y<nd_length;++y){
		if(nd_vals[y] < u){
			cdf += get_gen_array_val(nd_vals[y]+m,gen_cdf,gen_min,gen_max+1)/nd_length;
		}
	}

	for(y=(long) ceil(u);y<gen_max-m+1;++y){
		pdf_val = ev_nd_pdf(y,
							u,
							p,
							sigma,
							xi,
							nd_length,
							nd_vals);

		g_cdf = get_gen_array_val(y+m,gen_cdf,gen_min,gen_max+1);

		cdf += g_cdf*pdf_val;
	}

	cdf += 1.0 - ev_nd_cdf(gen_max-m,
						u,
						p,
						sigma,
						xi,
						nd_length,
						nd_vals);

	return cdf;
}

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
	){

	long y;

	double cdf = 0;

	double pdf_val;
	for(y=max(gen_min,-m);y<gen_max-m+1;++y){
		pdf_val = bev_nd_pdf(y,
							u,
							p,
							n_posterior,
							sigma,
							xi,
							nd_length,
							nd_vals);
		cdf += get_gen_array_val(y+m,gen_cdf,gen_min,gen_max+1)*pdf_val;
	}

	cdf += 1.0 - bev_nd_cdf(gen_max-m,
						u,
						p,
						n_posterior,
						sigma,
						xi,
						nd_length,
						nd_vals);

	return cdf;
}

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
	double* gen_expectation){

	double epu = 0, pdf_val, genpar_expectation;

	long nd_idx, y;

	if(xi>=1){
		epu = -1.0; //infinite expectation
	}else{

		for(nd_idx=0;nd_idx<nd_length;++nd_idx){
			if(nd_vals[nd_idx] < u){
				epu += (nd_vals[nd_idx]*get_gen_array_val(nd_vals[nd_idx]-1,gen_cdf,gen_min,gen_max)-\
				get_gen_array_val(nd_vals[nd_idx]-1,gen_expectation,gen_min,gen_max))/nd_length;
				//if(nd_vals[nd_idx] <= gen_max){
				//	epu -= nd_vals[nd_idx]/nd_length;
				//}
			}
		}

		genpar_expectation = (1-p)*(u + sigma/(1-xi));
		for(y=(long) ceil(u);y<gen_max+1;++y){
			pdf_val = ev_nd_pdf(y,
								u,
								p,
								sigma,
								xi,
								nd_length,
								nd_vals);

			epu += pdf_val*(y*(get_gen_array_val(y-1,gen_cdf,gen_min,gen_max)-1) - get_gen_array_val(y-1,gen_expectation,gen_min,gen_max));
		}

		epu += genpar_expectation - get_gen_array_val(gen_max,gen_expectation,gen_min,gen_max)*(1-ev_nd_cdf(gen_max,
																			u,
																			p,
																			sigma,
																			xi,
																			nd_length,
																			nd_vals));

	}

	return epu;
}


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
	double* gen_expectation){

	double epu = 0, genpar_expectation, pdf_val;

	long nd_idx, y;

	long infinite_expectation = 0;

	for(y=0;y<n_posterior;++y){
		if(xi[y]>=1){
			infinite_expectation = 1;
		}
	}
	if(infinite_expectation==1){
		epu = -1.0;
	}else{

		for(nd_idx=0;nd_idx<nd_length;++nd_idx){
			if(nd_vals[nd_idx] < u){
				epu += (nd_vals[nd_idx]*get_gen_array_val(nd_vals[nd_idx]-1,gen_cdf,gen_min,gen_max)-\
				get_gen_array_val(nd_vals[nd_idx]-1,gen_expectation,gen_min,gen_max))/nd_length;
				//if(nd_vals[nd_idx] <= gen_max){
				//	epu -= nd_vals[nd_idx]/nd_length;
				//}
			}
		}

		genpar_expectation = 0;
		for(y=0;y<n_posterior;++y){
			genpar_expectation += u + sigma[y]/(1-xi[y]);
		}
		genpar_expectation = (1-p)*genpar_expectation/n_posterior;

		for(y=(long) ceil(u);y<gen_max+1;++y){
			pdf_val = bev_nd_pdf(y,
								u,
								p,
								n_posterior,
								sigma,
								xi,
								nd_length,
								nd_vals);

			epu += pdf_val*(y*(get_gen_array_val(y-1,gen_cdf,gen_min,gen_max)-1) - get_gen_array_val(y-1,gen_expectation,gen_min,gen_max));
		}

		epu += genpar_expectation - get_gen_array_val(gen_max,gen_expectation,gen_min,gen_max)*(1-bev_nd_cdf(gen_max,
																			u,
																			p,
																			n_posterior,
																			sigma,
																			xi,
																			nd_length,
																			nd_vals));

	}

	return epu;
}

