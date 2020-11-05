#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "libunivarmargins.h"

double cumulative_value(DiscreteDistribution* F, double* array, int x){
  if(x < F->min){
    return 0.0;
  }else{
    if(x >= F->max - F->min){
      return array[F->max - F->min];
    }else{
      return array[x - F->min];
    }
  } 
}

double gen_cdf(DiscreteDistribution* F, int x){
  return cumulative_value(F,F->cdf,x);
}

double gen_pdf(DiscreteDistribution* F, int x){
  return gen_cdf(F,x) - gen_cdf(F,x-1);
}

double cumulative_expectation(DiscreteDistribution* F, int x){
  return cumulative_value(F,F->expectation,x);
}

// documentation is in .h files 

double max(double num1, double num2){
    return (num1 > num2 ) ? num1 : num2;
}

double min(double num1, double num2){
    return (num1 > num2 ) ? num2 : num1;
}

double empirical_power_margin_cdf(DiscreteDistribution* F, IntVector* net_demand, int x){

  double cdf_val = 0;
  int i=0;

  for(i=0;i<net_demand->size;++i){
    cdf_val += gen_cdf(F,(int) net_demand->value[i]+x);
  }

  return cdf_val/net_demand->size;
}

double empirical_eeu(DiscreteDistribution* F, IntVector* net_demand){
  double eeu = 0;
  int i, current;

  for(i=0;i<net_demand->size;++i){
    current = (int) net_demand->value[i];
    eeu += current*gen_cdf(F,current-1) - cumulative_expectation(F,current-1);
  }

  return eeu/net_demand->size;
}

double gpdist_cdf(GPModel* gp, double x){

  double xi = gp->xi, u = gp->u, sigma = gp->sigma, val = 0;

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

double expdist_cdf(GPModel* gp, double x){
  return 1.0 - exp(-(x-gp->u)/gp->sigma); 
}


double tail_model_cdf(GPModel* gp, double x){
  if(gp->xi != 0){
    return gpdist_cdf(gp, x);
  }else{
    return expdist_cdf(gp, x);
  }
}


double bayesian_tail_model_cdf(PosteriorGPTrace* gpt, double x){

  double estimator=0;

  int i;

  GPModel current;

  for(i=0;i<gpt->size;++i){
    
    current.xi = gpt->xi[i];
    current.u = gpt->u;
    current.p = 1; //irrelevant
    current.sigma = gpt->sigma[i];

    estimator += tail_model_cdf(&current, x);

    /*if(gpt->xi[i] != 0){
      estimator += gpdist_cdf(current,x);
    }else{
      estimator += expdist_cdf(current,x);
    }*/
  }

  return estimator/gpt->size;

}

double empirical_net_demand_cdf(IntVector* net_demand, double x){

  int i;

  double nd_below_x = 0;

  for(i=0;i<net_demand->size;++i){
    if(net_demand->value[i]<=x){
      nd_below_x += 1;
    }
  }
  return nd_below_x/net_demand->size;
}


double semiparametric_net_demand_cdf(GPModel* gp, IntVector* net_demand, double x){

  if(x <= gp->u){
    return empirical_net_demand_cdf(net_demand,x);
  }else{
    return gp->p + (1.0-gp->p)*tail_model_cdf(gp, x);
  }
}


double semiparametric_net_demand_pdf(GPModel* gp, IntVector* net_demand, double x){

  return semiparametric_net_demand_cdf(gp,net_demand,x) - semiparametric_net_demand_cdf(gp,net_demand,x-1);
}

double bayesian_semiparametric_net_demand_cdf(PosteriorGPTrace* gpt, IntVector* net_demand, double x){

  if(x <= gpt->u){
    return empirical_net_demand_cdf(net_demand,x);
  }else{
    return gpt->p + (1.0-gpt->p)*bayesian_tail_model_cdf(gpt,x);
  }
}


double bayesian_semiparametric_net_demand_pdf(PosteriorGPTrace* gpt, IntVector* net_demand, double x){

  return bayesian_semiparametric_net_demand_cdf(gpt,net_demand,x) - bayesian_semiparametric_net_demand_cdf(gpt,net_demand,x-1);
}

double semiparametric_power_margin_cdf(GPModel* gp, IntVector* net_demand, DiscreteDistribution* F, double x){

  int y;

  double cdf_val = 0;
  double pdf_val, g_cdf;

  for(y=0;y<net_demand->size;++y){
    if(net_demand->value[y] < gp->u){
      //cdf += get_gen_array_val(nd_vals[y]+x,gen_cdf,gen_min,gen_max+1)/nd_length;
      cdf_val += gen_cdf(F, (int) net_demand->value[y]+x);
    }
  }
  cdf_val /=net_demand->size;

  for(y=(int) ceil(gp->u);y<F->max-x+1;++y){
    pdf_val = semiparametric_net_demand_pdf(gp,net_demand,y);
    g_cdf = gen_cdf(F,(int) (y+x));
    cdf_val += g_cdf*pdf_val;
  }

  cdf_val += 1.0 - semiparametric_net_demand_cdf(gp,net_demand,F->max-x);

  return cdf_val;
}

double bayesian_semiparametric_power_margin_cdf(PosteriorGPTrace* gpt, IntVector* net_demand, DiscreteDistribution* F, double x) {

  int y;

  double cdf_val = 0;

  double pdf_val;
  for(y=max((double) F->min,-x);y<F->max-x+1;++y){

    pdf_val = bayesian_semiparametric_net_demand_pdf(gpt,net_demand,y);
    cdf_val += gen_cdf(F,(int) (y+x))*pdf_val;
  }

  cdf_val += 1.0 - bayesian_semiparametric_net_demand_cdf(gpt,net_demand,F->max - x);

  return cdf_val;
}


double semiparametric_eeu(GPModel* gp, IntVector* net_demand, DiscreteDistribution* F){

  double eeu = 0, semiparametric_eeu = 0, pdf_val, remainder;

  int i, y, current;

  if(gp->xi>=1){
    eeu = -1.0; //infinite expectation
  }else{

    for(i=0;i<net_demand->size;++i){
      current = (int) net_demand->value[i];
      if(current < gp->u){
        eeu += current*gen_cdf(F,current-1) - cumulative_expectation(F,current-1);
        //printf("current: %d, eeu: %f\n",current,eeu);
      }
    }
    eeu /= net_demand->size;

    //genpar_expectation = (1-gp->p)*(gp->u + gp->sigma/(1-gp->xi));

    for(y=(int) ceil(gp->u);y<F->max+1;++y){

      pdf_val = semiparametric_net_demand_pdf(gp,net_demand,y);

      semiparametric_eeu += pdf_val*(y*gen_cdf(F,y-1) - cumulative_expectation(F,y-1));
    }

    remainder = (1 - semiparametric_net_demand_cdf(gp,net_demand,F->max))*(F->max + gp->sigma/(1-gp->xi));

    eeu += semiparametric_eeu + remainder - cumulative_expectation(F,F->max)*(1-semiparametric_net_demand_cdf(gp,net_demand,F->max));

  }

  return eeu;
}

double bayesian_semiparametric_eeu(PosteriorGPTrace* gpt, IntVector* net_demand, DiscreteDistribution* F){

  double eeu = 0, semiparametric_eeu = 0, remainder =0, pdf_val;

  int i, y, current;

  int infinite_expectation = 0;

  GPModel current_gp;
  current_gp.u = gpt->u;
  current_gp.p = gpt->p;

  for(y=0;y<gpt->size;++y){
    if(gpt->xi[y]>=1){
      infinite_expectation = 1;
    }
  }
  if(infinite_expectation==1){
    eeu = -1.0;
  }else{

    for(i=0;i<net_demand->size;++i){
      current = (int) net_demand->value[i];
      if(current < gpt->u){
        eeu += current*gen_cdf(F,current-1) - cumulative_expectation(F,current-1);
        //printf("current: %d, eeu: %f\n",current,eeu);
      }
    }
    eeu /= net_demand->size;

    for(y=(int) ceil(gpt->u);y<F->max+1;++y){

      pdf_val = bayesian_semiparametric_net_demand_pdf(gpt,net_demand,y);

      semiparametric_eeu += pdf_val*(y*gen_cdf(F,y-1) - cumulative_expectation(F,y-1));
    }

    for(i=0;i<gpt->size;++i){
      current_gp.xi = gpt->xi[i];
      current_gp.sigma = gpt->sigma[i]; 
      remainder += (1.0 - semiparametric_net_demand_cdf(&current_gp,net_demand,F->max))*(F->max + current_gp.sigma/(1.0-current_gp.xi));
      //genpar_expectation += gpt->u + gpt->sigma[i]/(1-gpt->xi[i]);
    }
    //genpar_expectation = (1-gpt->p)*genpar_expectation/gpt->size;
    remainder /= gpt->size;

    eeu += semiparametric_eeu + remainder - cumulative_expectation(F,F->max)*(1-bayesian_semiparametric_net_demand_cdf(gpt,net_demand,F->max));

  }

  return eeu;
}
















// ************** Python interfaces

void get_discrete_dist_from_py_objs(DiscreteDistribution* F, double* cdf, double* expectation, int min, int max){
  F->cdf = cdf;
  F->expectation = expectation;
  F->max = max;
  F->min = min;
}

void get_int_vector_from_py_objs(IntVector* vector, int* value, int size){

  vector -> value = value;
  vector ->size = size;
}

void get_gp_from_py_objs(GPModel* gp, double xi, double sigma, double u, double p){
  gp->sigma = sigma;
  gp->u = u;
  gp->xi = xi;
  gp->p = p;
}

void get_gpt_from_py_objs(PosteriorGPTrace* gpt, double* xi, double* sigma, double u, double p, int size){
  gpt->sigma = sigma;
  gpt->u = u;
  gpt->xi = xi;
  gpt->p = p;
  gpt->size = size;
}

double empirical_power_margin_cdf_py_interface(
  int x, 
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals, 
  double* gen_cdf){

  DiscreteDistribution F;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_cdf, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);

  return empirical_power_margin_cdf(&F,&net_demand,x);

}

double empirical_net_demand_cdf_py_interface(
  double x,
  int nd_length,
  int* nd_vals){

  IntVector net_demand;

  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);

  return empirical_net_demand_cdf(&net_demand,x);
}

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
  ){

  DiscreteDistribution F;
  GPModel gp;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_cdf, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);
  get_gp_from_py_objs(&gp,xi,sigma,u,p);

  return semiparametric_power_margin_cdf(&gp, &net_demand, &F, x);
}

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
  ){

  PosteriorGPTrace gpt;
  DiscreteDistribution F;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_cdf, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);
  get_gpt_from_py_objs(&gpt, xi, sigma, u, p, n_posterior);

  return bayesian_semiparametric_power_margin_cdf(&gpt, &net_demand, &F, x);
}

double empirical_eeu_py_interface(
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals, 
  double* gen_cdf,
  double* gen_expectation){

  DiscreteDistribution F;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_expectation, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);

  return empirical_eeu(&F,&net_demand);

}

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
  double* gen_expectation){

  DiscreteDistribution F;
  GPModel gp;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_expectation, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);
  get_gp_from_py_objs(&gp,xi,sigma,u,p);

  return semiparametric_eeu(&gp, &net_demand, &F);
}

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
  double* gen_expectation){

  PosteriorGPTrace gpt;
  DiscreteDistribution F;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_expectation, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);
  get_gpt_from_py_objs(&gpt, xi, sigma, u, p, n_posterior);

  return bayesian_semiparametric_eeu(&gpt, &net_demand, &F);

}