#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "mtwist-1.5/mtwist.h"
#include "libunivarmargins.h"

// documentation is in .h files 

double sign(double x){
	if(x > 0){
		return 1.0;
	}else if(x < 0){
		return -1.0;
	}else{
		return 0.0;
	}
}

long min(long num1, long num2){
    return (num1 > num2 ) ? num2 : num1;
}

double x1_stripe_pdf(
	long x1,
	long x2,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array){

	double prob = (get_gen_array_val(x1,gen1_cdf_array,0,max_gen1) - \
		get_gen_array_val(x1-1,gen1_cdf_array,0,max_gen1)) * \
		get_gen_array_val(x2,gen2_cdf_array,0,max_gen2);

	return prob;
}

double bigen_cdf(
	long x1,
	long x2,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array){

	double prob = get_gen_array_val(x1,gen1_cdf_array,0,max_gen1) * \
	get_gen_array_val(x2,gen2_cdf_array,0,max_gen2);

	return prob;
}

double bigen_pdf(
	long x1,
	long x2,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array){

	double r;
	r = bigen_cdf(
		x1,
		x2,
		max_gen1,
		max_gen2,
		gen1_cdf_array,
		gen2_cdf_array) -\
	bigen_cdf(
		x1-1,
		x2,
		max_gen1,
		max_gen2,
		gen1_cdf_array,
		gen2_cdf_array) -\
	bigen_cdf(
		x1,
		x2-1,
		max_gen1,
		max_gen2,
		gen1_cdf_array,
		gen2_cdf_array) +\
	bigen_cdf(
		x1-1,
		x2-1,
		max_gen1,
		max_gen2,
		gen1_cdf_array,
		gen2_cdf_array);

	return r;
}

double triangle_prob(
	long origin_x,
	long origin_y,
	long triangle_length,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array){

	// by interior lattice below, I mean points not in any triangle side
	// except maybe the hypotenuse

	double val, gen1_pdf, gen2_pdf, square_prob;

	long l = (long) floor(triangle_length/2);

	long displaces_righthand_x = origin_x + l, displaced_righthand_y = origin_y;
	long displaces_upper_x = origin_x, displaced_upper_y = origin_y + l;
	//#long displaced_triangle_origin_h[2] = {origin_x + l, origin_y};
	//long displaced_triangle_origin_v[2] = {origin_x, origin_y + l};
	
	if(triangle_length <= 1){
		// interior lattice of length-1 triangle is empty
		val = 0.0;
	}else{
		// interior lattice of length 2 triangle consist in a single point
		if(triangle_length == 2){
			gen1_pdf = get_gen_array_val(origin_x+1,gen1_cdf_array,0,max_gen1) - \
			get_gen_array_val(origin_x,gen1_cdf_array,0,max_gen1);

			gen2_pdf = get_gen_array_val(origin_y+1,gen2_cdf_array,0,max_gen2) - \
			get_gen_array_val(origin_y,gen2_cdf_array,0,max_gen2);

			val = gen1_pdf * gen2_pdf;
		}else{

			square_prob = bigen_cdf(
				origin_x+l,
				origin_y+l,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) -
				bigen_cdf(
				origin_x,
				origin_y+l,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) -\
				bigen_cdf(
				origin_x+l,
				origin_y,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) +\
				bigen_cdf(
				origin_x,
				origin_y,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);


			val = square_prob + triangle_prob(
				displaces_upper_x,
				displaced_upper_y,
				triangle_length - l,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) + \
				triangle_prob(
				displaces_righthand_x,
				displaced_righthand_y,
				triangle_length - l,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);

		}
	}

	return val;
}

double trapezoid_prob(
	long ul_x,
	long ul_y,
	long width,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array){
	// ulc = upper left
	long ur_x = ul_x + width, ur_y = ul_y - width;

	double prob = 0;

	prob = \
	bigen_cdf(
		ur_x,
		ur_y,
		max_gen1,
		max_gen2,
		gen1_cdf_array,
		gen2_cdf_array) -\
	bigen_cdf(
		ul_x,
		ur_y,
		max_gen1,
		max_gen2,
		gen1_cdf_array,
		gen2_cdf_array) +\
	triangle_prob(
		ul_x,
		ur_y,
		width,
		max_gen1,
		max_gen2,
		gen1_cdf_array,
		gen2_cdf_array);

	return prob;

}

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
	double* gen1_expectation){

	double epu = 0;
	double d1_div_d2 = ((double)d1)/d2;
	double beta0 =  (double) v1 - d1_div_d2*v2 + ((double)d1+d2)/d2*c;
	double alpha0 = (double) v1 - d1_div_d2*v2 - ((double)d1+d2)/d2*c;
    double r = ((double) d1)/(d1+d2);

    long x2, beta, alpha ;
    double gen2_pdf;

    for(x2=0;x2<v2+c;++x2){

    	//EPU += -r*FX2.pdf(x2)*((x2-v1-v2)*(FX1.cdf(beta)-FX1.cdf(alpha-1)) + FX1.expectation(fro=alpha,to=beta))
    	
    	beta = (long) min(beta0 + d1_div_d2*x2,v1+v2-x2);
    	alpha = (long) ceil(alpha0 + d1_div_d2*x2);
    	gen2_pdf = (get_gen_array_val(x2,gen2_cdf_array,0,max_gen2) - get_gen_array_val(x2-1,gen2_cdf_array,0,max_gen2));
    	epu += -r*gen2_pdf*\
    	   ((x2-v1-v2)*(get_gen_array_val(beta,gen1_cdf_array,0,max_gen1) - get_gen_array_val(alpha-1,gen1_cdf_array,0,max_gen1)) + \
    	   (get_gen_array_val(beta,gen1_expectation,0,max_gen1) - get_gen_array_val(alpha-1,gen1_expectation,0,max_gen1)));
    }

    for(x2=0;x2<v2-c;++x2){

    	beta = (long) floor(beta0 + d1_div_d2*x2);

    	//EPU += FX2.pdf(x2)*((v1+c)*(FX1.cdf(v1+c)-FX1.cdf(beta))-FX1.expectation(fro=beta+1,to=v1+c))
    	gen2_pdf = (get_gen_array_val(x2,gen2_cdf_array,0,max_gen2) - get_gen_array_val(x2-1,gen2_cdf_array,0,max_gen2));
    	epu += gen2_pdf*\
    	   ((v1+c)*(get_gen_array_val(v1+c,gen1_cdf_array,0,max_gen1) - get_gen_array_val(beta,gen1_cdf_array,0,max_gen1)) - \
    	   (get_gen_array_val(v1+c,gen1_expectation,0,max_gen1) - get_gen_array_val(beta,gen1_expectation,0,max_gen1)));
    }

    for(x2=(long) ceil(v2+c-((double)d2)/d1*(v1-c));x2<v2+c;++x2){
    	alpha = (long) ceil(alpha0 + d1_div_d2*x2);

    	//EPU += FX2.pdf(x2)*((v1-c)*FX1.cdf(alpha-1)-FX1.expectation(to=alpha-1))

    	gen2_pdf = (get_gen_array_val(x2,gen2_cdf_array,0,max_gen2) - get_gen_array_val(x2-1,gen2_cdf_array,0,max_gen2));
    	epu += gen2_pdf*\
    	   ((v1-c)*get_gen_array_val(alpha-1,gen1_cdf_array,0,max_gen1) - get_gen_array_val(alpha-1,gen1_expectation,0,max_gen1));
    }

    if(v1-c >= 0){
    	//EPU += (1-FX2.cdf(v2+c-1))*((v1-c)*FX1.cdf(v1-c)-FX1.expectation(to=v1-c))
    	epu += (1-get_gen_array_val(v2+c-1,gen2_cdf_array,0,max_gen2))*((v1-c)*get_gen_array_val(v1-c,gen1_cdf_array,0,max_gen1)-get_gen_array_val(v1-c,gen1_expectation,0,max_gen1));
    }

    return epu;
}

double cond_epu_veto(
	long v1,
	long v2,
	long c,
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array,
	double* gen1_expectation){

	double epu = 0;
	long x2;
    double gen2_pdf;

    //EPU = (1 - FX2.cdf(v2+c))*((v1-c)*FX1.cdf(v1-c) - FX1.expectation(to=v1-c)) + FX2.cdf(v2-1)*(v1*FX1.cdf(v1) - FX1.expectation(to=v1))

    epu = (1 - get_gen_array_val(v2+c,gen2_cdf_array,0,max_gen2))*\
    ((v1-c)*get_gen_array_val(v1-c,gen1_cdf_array,0,max_gen1) - get_gen_array_val(v1-c,gen1_expectation,0,max_gen1)) + \
    get_gen_array_val(v2-1,gen2_cdf_array,0,max_gen2)*(v1*get_gen_array_val(v1,gen1_cdf_array,0,max_gen1)-get_gen_array_val(v1,gen1_expectation,0,max_gen1));

    for(x2=v2;x2<v2+c+1;++x2){

    	//EPU += FX2.pdf(x2) * ((v1+v2-x2)*FX1.cdf(v1+v2-x2) - FX1.expectation(to=v1+v2-x2))
    	gen2_pdf = (get_gen_array_val(x2,gen2_cdf_array,0,max_gen2) - get_gen_array_val(x2-1,gen2_cdf_array,0,max_gen2));
    	epu += gen2_pdf*\
    	   ((v1+v2-x2)*get_gen_array_val(v1+v2-x2,gen1_cdf_array,0,max_gen1) - get_gen_array_val(v1+v2-x2,gen1_expectation,0,max_gen1));
    }

    return epu;
}

void get_share_polygon_points(
	long* p_array,
	long m,
	long v1,
	long v2,
	long d1,
	long d2,
	long c,
	int i){

	long nd1, nd2, q1, q2;
	long p11, p12, p11_r, p12_r;
	long delta;
	double r, res;

	if(i==1){
		nd1 = v2;
		nd2 = v1;
		q1 = d2;
		q2 = d1;
	}else{
		nd1 = v1;
		nd2 = v2;
		q1 = d1;
		q2 = d2;
	}

	if(m >= 0){
		p11 = nd1 + m;
		p12 = nd2;
		delta = c;
	}else{
		p11 = nd1 - c + m;
		// this commented line fix a woird behaviour due to rounding to 1MW:
		// as c -> inf, the risk in the two areas do not fully converge because 
		// rounding causes that some cases go unacounted for, for example: if the systemwide
		// shortfall is 1MW, a fractional portion is going to be beared by each area,
		// but rounding makes this fractional part disappear and become zero.
		

		// floor function is necessary to prune lattice correctly
		res = ((double) q2)/q1*m;
		p12 = (long) (nd2 + c + res);
		delta = 2*c;
	}
	p11_r = p11 + delta;
	p12_r = p12 - delta;
	if(i==1){
		p_array[0] = p12_r;
		p_array[1] = p11_r;
		p_array[2] = p12;
		p_array[3] = p11;
	}else{
		p_array[0] = p11;
		p_array[1] = p12;
		p_array[2] = p11_r;
		p_array[3] = p12_r;
	}
}

void get_veto_polygon_points(
	long* p_array,
	long m,
	long v1,
	long v2,
	long c,
	int i){

	long nd1, nd2;
	long p11, p12, p11_r, p12_r;

	if(i==1){
		nd1 = v2;
		nd2 = v1;
	}else{
		nd1 = v1;
		nd2 = v2;
	}

	if(m >= 0){
		p11 = nd1 + m;
		p12 = nd2;
	}else{
		p11 = nd1 - c + m;
		p12 = nd2 + c;
	}
	p11_r = p11 + c;
	p12_r = p12 - c;

	if(i==1){
		p_array[0] = p12_r;
		p_array[1] = p11_r;
		p_array[2] = p12;
		p_array[3] = p11;
	}else{
		p_array[0] = p11;
		p_array[1] = p12;
		p_array[2] = p11_r;
		p_array[3] = p12_r;
	}
}

long axis1_polygon_upper_bound(
	long x,
	long* P1){

	long res;
	if(x <= P1[0]){
		res = (long) (LONG_MAX/2);
	}else if(x <= P1[2]){
		res = P1[1] - (x - P1[0]);
	}else{
		res = (long) (-LONG_MAX/2);
	}
	return res;
}

long axis2_polygon_upper_bound(
	long x,
	long* P2){

	long res;
	if(x <= P2[0]){
		res = P2[1];
	}else if(x <= P2[2]){
		res = P2[1] - (x - P2[0]);
	}else{
		res = P2[3];
	}
	return res;
}

long boxed_gen_simulation(
	double u,
	long upper_bound,
	long max_gen,
	double* gen_cdf_array){

	long lb = 0, i=0;
	long ub = (long) min(upper_bound,max_gen);

	double p_lb = get_gen_array_val(lb-1,gen_cdf_array,0,max_gen);
	double box_prob = get_gen_array_val(ub,gen_cdf_array,0,max_gen) - p_lb;

	while(u > (get_gen_array_val(lb+i,gen_cdf_array,0,max_gen) - p_lb)/box_prob) {
		i +=1;
	}

	return lb+i;
}

long veto_flow(
	long m1,
	long m2,
	long c){

	long res;
	if(m1 > 0 && m2 < 0){
      res = -min(c,min(m1,-m2));
	}else if(m1 < 0 && m2 > 0){
      res = min(c,min(m2,-m1));
	}else{
      res = 0;
	}
	return res;
}

double share_flow(
	long m1,
	long m2,
	long d1,
	long d2,
	long c){

	double flow;
	if(m1+m2 < 0 && m1 < c && m2 < c){
		//res = min(c,max(-c,((float) d1)/(d1+d2)*m2 - ((float) d2)/(d1+d2)*m1));
		flow = ((float) d1)/(d1+d2)*m2 - ((float) d2)/(d1+d2)*m1;
		flow = (flow >= c ? c : (flow <= -c ? -c: flow));
	}else{
		flow = (double) veto_flow(m1,m2,c);
	}
	return flow;
}

double cond_cdf(
	long max_gen1,
	long max_gen2,
	double* gen1_cdf_array,
	double* gen2_cdf_array,
	long* p_array_a1,
	long* p_array_a2,
	long c1,
	long c2) { 
	 
	long p11 = p_array_a1[0];
	long p12 = p_array_a1[1];
	long p11_r = p_array_a1[2];
	long p12_r = p_array_a1[3];

	long p21 = p_array_a2[0];
	long p22 = p_array_a2[1];
	long p21_r = p_array_a2[2];
	long p22_r = p_array_a2[3];

	double cdf = 0;
	long diff = 0;
	//#if P2 segment is inside marginal polygon of area 1
	if(p21_r <= p11 || (p21_r <= p11_r && p22_r <= p12 + p11 - p21_r)) {

		cdf = \
		bigen_cdf(
			p21,
			p22,
			max_gen1,
			max_gen2,
			gen1_cdf_array,
			gen2_cdf_array) +\
		trapezoid_prob(
			p21,
			p22,
			c2,
			max_gen1,
			max_gen2,
			gen1_cdf_array,
			gen2_cdf_array);
		//# if segment is inside lower rectangle
		if(p22_r <= p12_r){

			cdf +=\
			bigen_cdf(
				p11_r,
				p22_r,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) -\
			bigen_cdf(
				p21_r,
				p22_r,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);

		//# if it's in middle trapezoidal section
		}else if(p22_r <= p12){

			diff = p12 - p22_r;

			cdf +=\
			bigen_cdf(
				p11 + diff,
				p22_r,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) -\
			bigen_cdf(
				p21_r,
				p22_r,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) +\
			trapezoid_prob(
				p11 + diff,
				p12 - diff,
				c1 - diff,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);


		}else{
			//# if it's in uppermost rectangle
			cdf +=\
			bigen_cdf(
				p11,
				p22_r,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) -\
			bigen_cdf(
				p21_r,
				p22_r,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) +\
			trapezoid_prob(
				p11,
				p12,
				c1,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);

		}
	//#if P2 segment is completely outside of marginal polygon of area 1
	}else if(p21 >= p11_r || (p21 >= p11 && p22 >= p12 + p11 - p21)){
		
		if(p22 <= p12_r){
			//# if it's not higher than lowermost quadrant
			cdf +=\
			bigen_cdf(
				p11_r,
				p22,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);

		}else if(p22 <= p12){
			//# if it's no higher than middle trapezoid
			diff = p12 - p22;

			cdf +=\
			bigen_cdf(
				p11+diff,
				p22,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) +\
			trapezoid_prob(
				p11 + diff,
				p22,
				c1 - diff,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);

		}else{
			// if it's higher than trapezoid
			cdf +=\
			bigen_cdf(
				p11,
				p22,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) +\
			trapezoid_prob(
				p11,
				p12,
				c1,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);
		}
		
	}else{

		if(p21 <= p11){
			//# if the crossing is at the upper rectangle
			cdf +=\
			bigen_cdf(
				p21,
				p22,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) +\
			trapezoid_prob(
				p21,
				p22,
				p11-p21,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) +\
			trapezoid_prob(
				p11,
				p12,
				c1,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);
		}else{
			// if crossing is at the bottom rectangle
			cdf +=\
			bigen_cdf(
				p21,
				p22,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array) +\
			trapezoid_prob(
				p21,
				p22,
				p11_r-p21,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);
		}
	}

	return cdf;
}

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
	int share_policy){

	long P1[4];
	long P2[4];
	double cdf;
	long c1, c2;

	if(share_policy>0){
		get_share_polygon_points(P1,m1,v1,v2,d1,d2,c,0);

		get_share_polygon_points(P2,m2,v1,v2,d1,d2,c,1);

		if(m1 < 0){
			c1 = 2*c;
		}else{
			c1 = c;
		}

		if(m2 < 0){
			c2 = 2*c;
		}else{
			c2 = c;
		}

	}else{

		get_veto_polygon_points(P1,m1,v1,v2,c,0);

		get_veto_polygon_points(P2,m2,v1,v2,c,1);

		c1 = c;
		c2 = c;
	}

	cdf = cond_cdf(
		max_gen1,
		max_gen2,
		gen1_cdf_array,
		gen2_cdf_array,
		P1,
		P2,
		c1,
		c2);

	// rounding errors can make calculations negative in the order of 1e-60
	return (cdf >= 0.0 ? cdf : 0.0);
}

long get_joint_polygon_x_bound(
	long* P1,
	long* P2,
	long max_gen1,
	int intersection){

	// find where the polygon derived from P2 crosses the x axis
	long compared;
	long p2_crossing, p1_crossing;
	if(P2[3] > 0){
		p2_crossing = LONG_MAX/2;
	}else if(P2[1] <= 0){
		p2_crossing = P2[0];
	}else{
		p2_crossing = P2[2] + P2[3];
	}

	// find where the polygon derived from P1 crosses the x axis
	if(P1[3] >= 0){
		p1_crossing = P1[2];
	}else if(P1[1] <= 0){
		p1_crossing = P1[0];
	}else{
		p1_crossing = P1[2] + P1[3];
	}

	compared = intersection > 0 ? min(p1_crossing,p2_crossing) : max(p1_crossing,p2_crossing);
	return(min(max_gen1,compared));

}

long get_joint_polygon_y_bound_given_x(
	long x,
	long* P1,
	long* P2,
	long max_gen2,
	int intersection){

	long compared, p1_y_bound, p2_y_bound;

	p1_y_bound = axis1_polygon_upper_bound(x,P1);
	p2_y_bound = axis2_polygon_upper_bound(x,P2);

	compared = intersection > 0 ? min(p1_y_bound,p2_y_bound) : max(p1_y_bound,p2_y_bound);
	
	return(min(max_gen2,compared));

}

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
	int share_policy){

	mt_seed32(seed);

	long row, nd1, nd2, d1, d2, x1, x2, x1_ub, x2_ub, v_flow, m1_s, m2_s;
	double u, s_flow;

	// these 2 arrays contain the points that characterise polygons that need to be integrated
	// as (P_11,P_12), (P_21,P_22) => [P_11,P_12,P_21,P_22]
	long P1[4];
	long P2[4];

	double x1_cond_cdf_array[max_gen1+1]; //not all entries will be filled, but it works

	long n_sim = 0, i, j, mx1;

	// iterate over the provided rows
	for(row=0;row<n_rows;row++){

		// initialize cond prob array
		for(i=0;i<=max_gen1;i++){
			x1_cond_cdf_array[i] = 1;
		}

		nd1 = net_demand[2*row];
		nd2 = net_demand[2*row+1];
		d1 = demand[2*row];
		d2 = demand[2*row+1];

		if(share_policy > 0){
			d1 = demand[2*row];
			d2 = demand[2*row+1];
			get_share_polygon_points(P1,m1,nd1,nd2,d1,d2,c,0);
			get_share_polygon_points(P2,m2,nd1,nd2,d1,d2,c,1);
		}else{
			get_veto_polygon_points(P1,m1,nd1,nd2,c,0);
			get_veto_polygon_points(P2,m2,nd1,nd2,c,1);
		}

		x1_ub = get_joint_polygon_x_bound(P1,P2,max_gen1,intersection);

		for(x1=0;x1<=x1_ub;x1++){

			x2_ub = get_joint_polygon_y_bound_given_x(x1,P1,P2,max_gen2,intersection);

			x1_cond_cdf_array[x1] = x1_stripe_pdf(
				x1,
				x2_ub,
				max_gen1,
				max_gen2,
				gen1_cdf_array,
				gen2_cdf_array);

			if(x1 > 0){
				x1_cond_cdf_array[x1] += x1_cond_cdf_array[x1-1];
			}
		}

		for(j=0;j<row_weights[row];j++){

			u = mt_drand();

			x1 = boxed_gen_simulation(u,x1_ub,max_gen1,x1_cond_cdf_array);

			x2_ub = get_joint_polygon_y_bound_given_x(x1,P1,P2,max_gen2,intersection);

			u = mt_drand();

			x2 = boxed_gen_simulation(u,x2_ub,max_gen2,gen2_cdf_array);
			
			m1_s = x1 - nd1;
			m2_s = x2 - nd2;

			if(share_policy>0){
				s_flow = share_flow(m1_s, m2_s, d1, d2, c);
				simulations[2*n_sim] = (long) (m1_s + s_flow);
				simulations[2*n_sim+1] = (long) (m2_s - s_flow);

			}else{
				v_flow = veto_flow(m1_s, m2_s, c);
				simulations[2*n_sim] = m1_s + v_flow;
				simulations[2*n_sim+1] = m2_s - v_flow;
			}

			n_sim++;

		}
	}
}

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
	int share_policy){

	mt_seed32(seed);

	long row, nd1, nd2, d1, d2, x1, x2, v_flow, m1_s, m2_s;
	double u, s_flow;

	long P1[4];
	double cond_x2_cdf[max_gen2+1]; //not all entries will be filled, but it works

	long n_sim = 0, j;

	// if(share_policy > 0){
	// 	printf("share policy");
	// }

	// iterate over the provided rows
	for(row=0;row<n_rows;row++){

		nd1 = net_demand[2*row];
		nd2 = net_demand[2*row+1];

		if(share_policy > 0){
			d1 = demand[2*row];
			d2 = demand[2*row+1];
			get_share_polygon_points(P1,m1,nd1,nd2,d1,d2,c,0);
		}else{
			get_veto_polygon_points(P1,m1,nd1,nd2,c,0);
		}

		// initialize cond prob array
		for(x2=0;x2<=max_gen2;x2++){
			if(x2 < P1[3]){
				cond_x2_cdf[x2] = bigen_pdf(
					P1[2],
					x2,
					max_gen1,
					max_gen2,
					gen1_cdf_array,
					gen2_cdf_array);
			}else if(x2 <= P1[1]){
				cond_x2_cdf[x2] = bigen_pdf(
					P1[2] - (x2 - P1[3]),
					x2,
					max_gen1,
					max_gen2,
					gen1_cdf_array,
					gen2_cdf_array);
			}else{
				cond_x2_cdf[x2] = bigen_pdf(
					P1[0],
					x2,
					max_gen1,
					max_gen2,
					gen1_cdf_array,
					gen2_cdf_array);
			}
			if(x2>0){
				cond_x2_cdf[x2] += cond_x2_cdf[x2-1];
			}
		}
		
		// normalize conditional cdf
		/*for(x2=0;x2<=max_gen2;x2++){
			cond_x2_cdf[x2] /= cond_x2_cdf[max_gen2];
		}*/

		for(j=0;j<row_weights[row];j++){

			u = mt_drand();
			x2 = boxed_gen_simulation(u,max_gen2,max_gen2,cond_x2_cdf);

			if(x2 >= P1[1]){
				x1 = P1[0];
			}else if(x2 <= P1[3]){
				x1 = P1[2];
			}else{
				x1 = P1[0] + (P1[1] - x2);
			}

			m1_s = x1 - nd1;
			m2_s = x2 - nd2;

			if(share_policy>0){
				s_flow = share_flow(m1_s, m2_s, d1, d2, c);
				simulations[2*n_sim] = (long) (m1_s + s_flow);
				simulations[2*n_sim+1] = (long) (m2_s - s_flow);

			}else{
				v_flow = veto_flow(m1_s, m2_s, c);
				simulations[2*n_sim] = m1_s + v_flow;
				simulations[2*n_sim+1] = m2_s - v_flow;
			}

			n_sim++;
		}
	}
}


void bivar_ecdf(
	double* ecdf,
	double* X,
	long n){

	int i,j;
	double iter_ecdf;
	for(i=0;i<n;i++){
		iter_ecdf = 0.0;
		//printf("(%f,%f) larger than: \n", x[i],y[i]);
		for(j=0;j<n;j++){
			if(X[2*j] <= X[2*i] && X[2*j+1] <= X[2*i+1]){
				//printf("(%f,%f)\n", x[j],y[j]);
				iter_ecdf ++;
			}
		}
		//printf("ECDF value: %f \n", iter_ecdf);
		ecdf[i] = iter_ecdf/(n+1);
	}

}