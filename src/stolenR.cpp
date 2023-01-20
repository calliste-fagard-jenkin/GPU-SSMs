#include <cfloat>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../include/stolenR.h"

// List of the 'useful' R functions this file includes:
// set_seed(unsigned int, unisgned int)
// get_seed(unsigned int, unsigned int)
// runif(double, double)
// rnorm(double, double)
// rbinom(int n, double pp)
// rgamma(double a, double scale)
// rbeta(double aa, double bb)
// rpois(double mu)
// rnbinom(int size, double prob)
// dnorm(double x, double mu, double sigma, int give_log)
// dpois(double x, double lambda, int give_log)
// dgamma(double x, double shape, double scale, int give_log
// dbinom(double x, double n, double p, double q, int give_log)
// dbeta(double x, double a, double b, int give_log)

#define repeat for(;;)
#define expmax	(DBL_MAX_EXP * M_LN2)/* = log(DBL_MAX) */
#define one_7	0.1428571428571428571
#define one_12	0.0833333333333333333
#define one_24	0.0416666666666666667
#define M_LN_SQRT_2PI	0.918938533204672741780329736406
#define M_1_SQRT_2PI	0.398942280401432677939946059934
#define M_LN_SQRT_PId2	0.225791352644727432363097614947
#define M_LN_2PI	1.837877066409345483560659472811	
#define M_2PI		6.283185307179586476925286766559
#define M_PI 3.14159265358979323846

#define R_D_exp(x)	(give_log	?  (x)	 : exp(x))
#define R_D_fexp(f,x)     (give_log ? -0.5*log(f)+(x) : exp(x)/sqrt(f))
#define R_D_val(x)	(give_log	? log(x) : (x))	
#define CAL_CHECK(x) (give_log ? -DBL_MAX : 0); // for a probability of 0 for an edge case

# ifndef __STDC_WANT_IEC_60559_FUNCS_EXT__
#  define __STDC_WANT_IEC_60559_FUNCS_EXT__ 1
# endif

static unsigned int I1 = 1234, I2 = 5678;
static double BM_norm_keep = 0.0;

double fmin2(double x, double y) {
	return (x < y) ? x : y;
}

double fmax2(double x, double y) {
	return (x < y) ? y : x;
}

int imax2(int x, int y) {
	return (x < y) ? y : x;
}

int imin2(int x, int y) {
	return (x < y) ? x : y;
}

double fsign(double x, double y) {
	return((y>=0) ? fabs(x) : -fabs(x));
}


double r_sinpi(double x) {
	x = fmod(x, 2.); // sin(pi(x + 2k)) == sin(pi x)  for all integer k
	// map (-2,2) --> (-1,1] :
	if (x <= -1) x += 2.; else if (x > 1.) x -= 2.;
	if (x == 0. || x == 1.) return 0.;
	if (x == 0.5)	return  1.;
	if (x == -0.5)	return -1.;
	// otherwise
	return sin(M_PI * x);
}

void set_seed(unsigned int i1, unsigned int i2)
{
    I1 = i1; I2 = i2;
}

void get_seed(unsigned int* i1, unsigned int* i2)
{
    *i1 = I1; *i2 = I2;
}


double unif_rand(void)
{
    I1 = 36969 * (I1 & 0177777) + (I1 >> 16);
    I2 = 18000 * (I2 & 0177777) + (I2 >> 16);
    return ((I1 << 16) ^ (I2 & 0177777)) * 2.328306437080797e-10; // in [0,1)
}

double runif(double a, double b)
{
    if (a == b) return a;

    else {
        double u;
        // This is true of all builtin generators, but protect against
        // user-supplied ones
        do { u = unif_rand(); } while (u <= 0 || u >= 1);
        return a + (b - a) * u;
    }
}

double norm_rand(void)
{
    double s, theta, R;

    if (BM_norm_keep != 0.0) { /* An exact test is intentional */
        s = BM_norm_keep;
        BM_norm_keep = 0.0;
        return s;
    }

    theta = 2 * M_PI * unif_rand();
    R = sqrt(-2 * log(unif_rand())) + 10 * DBL_MIN; /* ensure non-zero */
    BM_norm_keep = R * sin(theta);
    return R * cos(theta);
}

double rnorm(double mu, double sigma)
{
    return mu + sigma * norm_rand();
}

double rbinom(int n, double pp)
{
	static double c, fm, npq, p1, p2, p3, p4, qn;
	static double xl, xll, xlr, xm, xr;
	static int m;

	double f, f1, f2, u, v, w, w2, x, x1, x2, z, z2;
	double p, q, np, g, r, al, alv, amaxp, ffm, ynorm;
	int i, ix, k;

	if (n == 0 || pp == 0.) return 0;
	if (pp == 1.) return n;

	p = fmin2(pp, 1. - pp);
	q = 1. - p;
	np = n * p;
	r = p / q;
	g = r * (n + 1);

	if (np < 30.0) {
		/* inverse cdf logic for mean less than 30 */
		qn = pow(q, n);
		goto L_np_small;
	}
	
	else {
		ffm = np + p;
		m = (int)ffm;
		fm = m;
		npq = np * q;
		p1 = (int)(2.195 * sqrt(npq) - 4.6 * q) + 0.5;
		xm = fm + 0.5;
		xl = xm - p1;
		xr = xm + p1;
		c = 0.134 + 20.5 / (15.3 + fm);
		al = (ffm - xl) / (ffm - xl * p);
		xll = al * (1.0 + 0.5 * al);
		al = (xr - ffm) / (xr * q);
		xlr = al * (1.0 + 0.5 * al);
		p2 = p1 * (1.0 + c + c);
		p3 = p2 + c / xll;
		p4 = p3 + c / xlr;
	}

	/*-------------------------- np = n*p >= 30 : ------------------- */
	repeat{
        u = unif_rand() * p4;
		v = unif_rand();
		
		/* triangular region */
		if (u <= p1) {
			ix = (int)(xm - p1 * v + u);
			goto finis;
		}

		/* parallelogram region */
		if (u <= p2) {
			x = xl + (u - p1) / c;
			v = v * c + 1.0 - fabs(xm - x) / p1;
		
			if (v > 1.0 || v <= 0.) continue;
			ix = (int)x;
		}

		else {
			if (u > p3) {	/* right tail */
				ix = (int)(xr - log(v) / xlr);
				
				if (ix > n) continue;
				v = v * (u - p3) * xlr;
			}
		
			else {/* left tail */
				ix = (int)(xl + log(v) / xll);
				if (ix < 0) continue;
				v = v * (u - p2) * xll;
			}
		}
	    
		/* determine appropriate way to perform accept/reject test */
		k = abs(ix - m);
		if (k <= 20 || k >= npq / 2 - 1) {
			/* explicit evaluation */
			f = 1.0;
			if (m < ix) for (i = m + 1; i <= ix; i++) f *= (g / i - r);
			else if (m != ix) for (i = ix + 1; i <= m; i++) f /= (g / i - r);
			if (v <= f) goto finis;
		}

		else {
			/* squeezing using upper and lower bounds on log(f(x)) */
			amaxp = (k / npq) * ((k * (k / 3. + 0.625) + 0.1666666666666) / npq + 0.5);
			ynorm = -k * k / (2.0 * npq);
			alv = log(v);
			if (alv < ynorm - amaxp) goto finis;
			
			if (alv <= ynorm + amaxp) {
				/* stirling's formula to machine accuracy */
				/* for the final acceptance/rejection test */
				x1 = ix + 1;
				f1 = fm + 1.0;
				z = n + 1 - fm;
				w = n - ix + 1.0;
				z2 = z * z;
				x2 = x1 * x1;
				f2 = f1 * f1;
				w2 = w * w;
				if (alv <= xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) + (ix - m) * log(w * p / (x1 * q)) + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.)
				goto finis;
			}
		}
	}

	L_np_small:
	/*---------------------- np = n*p < 30 : ------------------------- */

	repeat{
		ix = 0;
	    f = qn;
	    u = unif_rand();
	    repeat {
			if (u < f) goto finis;
			if (ix > 110) break;
			u -= f;
			ix++;
			f *= (g / ix - r);
	    }
	}
	
	finis:
	if (pp > 0.5) ix = n - ix;
	return (double)ix;
}

double exp_rand(void)
{
	/* q[k-1] = sum(log(2)^k / k!)  k=1,..,n, */
	/* The highest n (here 16) is determined by q[n-1] = 1.0 */
	/* within standard precision */
	const static double q[] =
	{
	0.6931471805599453,
	0.9333736875190459,
	0.9888777961838675,
	0.9984959252914960,
	0.9998292811061389,
	0.9999833164100727,
	0.9999985691438767,
	0.9999998906925558,
	0.9999999924734159,
	0.9999999995283275,
	0.9999999999728814,
	0.9999999999985598,
	0.9999999999999289,
	0.9999999999999968,
	0.9999999999999999,
	1.0000000000000000
	};

	double a = 0.;
	double u = unif_rand();    /* precaution if u = 0 is ever returned */
	while (u <= 0. || u >= 1.) u = unif_rand();
	for (;;) {
		u += u;
		if (u > 1.) break;
		a += q[0];
	}
	u -= 1.;

	if (u <= q[0])
		return a + u;

	int i = 0;
	double ustar = unif_rand(), umin = ustar;
	do {
		ustar = unif_rand();
		if (umin > ustar)
			umin = ustar;
		i++;
	} while (u > q[i]);
	return a + umin * q[0];
}

double rgamma(double a, double scale)
{
	// Constants :
	const static double sqrt32 = 5.656854;
	const static double exp_m1 = 0.36787944117144232159;// exp(-1) = 1/e 

	// Coefficients q[k] - for q0 = sum(q[k]*a^(-k))
	// Coefficients a[k] - for q = q0+(t*t/2)*sum(a[k]*v^k)
	// Coefficients e[k] - for exp(q)-1 = sum(e[k]*q^k)
	//
	double q1 = 0.04166669;
	double q2 = 0.02083148;
	double q3 = 0.00801191;
	double q4 = 0.00144121;
	double q5 = -7.388e-5;
	double q6 = 2.4511e-4;
	double q7 = 2.424e-4;

	double a1 = 0.3333333;
	double a2 = -0.250003;
	double a3 = 0.2000062;
	double a4 = -0.1662921;
	double a5 = 0.1423657;
	double a6 = -0.1367177;
	double a7 = 0.1233795;

	// State variables [FIXME for threading!] :
	static double aa = 0.;
	static double aaa = 0.;
	static double s, s2, d;    // no. 1 (step 1) 
	static double q0, b, si, c;// no. 2 (step 4) 

	double e, p, q, r, t, u, v, w, x, ret_val;

	if (a <= 0.0 || scale <= 0.0) return 0.;
	if (a < 1.) { // GS algorithm for parameters a < 1 
		e = 1.0 + exp_m1 * a;
		repeat{
			p = e * unif_rand();
			if (p >= 1.0) {
			x = -log((e - p) / a);
			if (exp_rand() >= (1.0 - a) * log(x))
				break;
			}
	 else {
  x = exp(log(p) / a);
  if (exp_rand() >= x)
	  break;
  }
		}
		return scale * x;
	}

	// --- a >= 1 : GD algorithm --- 

	// Step 1: Recalculations of s2, s, d if a has changed
	if (a != aa) {
		aa = a;
		s2 = a - 0.5;
		s = sqrt(s2);
		d = sqrt32 - s * 12.0;
	}
	// Step 2: t = standard normal deviate,
	//		   x = (s,1/2) -normal deviate.

			   // immediate acceptance (i) 
	t = norm_rand();
	x = s + 0.5 * t;
	ret_val = x * x;
	if (t >= 0.0)
		return scale * ret_val;

	// Step 3: u = 0,1 - uniform sample. squeeze acceptance (s) 
	u = unif_rand();
	if (d * u <= t * t * t)
		return scale * ret_val;

	// Step 4: recalculations of q0, b, si, c if necessary 

	if (a != aaa) {
		aaa = a;
		r = 1.0 / a;
		q0 = ((((((q7 * r + q6) * r + q5) * r + q4) * r + q3) * r
			+ q2) * r + q1) * r;

		// Approximation depending on size of parameter a
		// The constants in the expressions for b, si and c 
		// were established by numerical experiments

		if (a <= 3.686) {
			b = 0.463 + s + 0.178 * s2;
			si = 1.235;
			c = 0.195 / s - 0.079 + 0.16 * s;
		}
		else if (a <= 13.022) {
			b = 1.654 + 0.0076 * s2;
			si = 1.68 / s + 0.275;
			c = 0.062 / s + 0.024;
		}
		else {
			b = 1.77;
			si = 0.75;
			c = 0.1515 / s;
		}
	}
	// Step 5: no quotient test if x not positive

	if (x > 0.0) {
		// Step 6: calculation of v and quotient q
		v = t / (s + s);
		if (fabs(v) <= 0.25)
			q = q0 + 0.5 * t * t * ((((((a7 * v + a6) * v + a5) * v + a4) * v
				+ a3) * v + a2) * v + a1) * v;
		else
			q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);


		// Step 7: quotient acceptance (q)
		if (log(1.0 - u) <= q)
			return scale * ret_val;
	}

	repeat{
		// Step 8: e = standard exponential deviate
		e = exp_rand();
		u = unif_rand();
		u = u + u - 1.0;
		if (u < 0.0)
			t = b - si * e;
		else
			t = b + si * e;
		// Step	 9:  rejection if t < tau(1) = -0.71874483771719 
		if (t >= -0.71874483771719) {
			// Step 10:	 calculation of v and quotient q 
			v = t / (s + s);
			if (fabs(v) <= 0.25)
			q = q0 + 0.5 * t * t *
				((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v
				  + a2) * v + a1) * v;
			else
			q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);
			// Step 11:	 hat acceptance (h) 
			// (if q not positive go to step 8)
			if (q > 0.0) {
			w = expm1(q);
			//  ^^^^^ original code had approximation with rel.err < 2e-7
			// if t is rejected sample again at step 8
			if (c * fabs(u) <= w * exp(e - 0.5 * t * t))
				break;
			}
		}
	} // repeat .. until  `t' is accepted
	x = s + 0.5 * t;
	return scale * x * x;
}


double rbeta(double aa, double bb)
{
	double a, b, alpha;
	double r, s, t, u1, u2, v, w, y, z;
	int qsame;
	/* FIXME:  Keep Globals (properly) for threading */
	/* Uses these GLOBALS to save time when many rv's are generated : */
	static double beta, gamma, delta, k1, k2;
	static double olda = -1.0;
	static double oldb = -1.0;

	/* Test if we need new "initializing" */
	qsame = (olda == aa) && (oldb == bb);
	if (!qsame) { olda = aa; oldb = bb; }

	a = fmin2(aa, bb);
	b = fmax2(aa, bb); /* a <= b */
	alpha = a + b;

#define v_w_from__u1_bet(AA) 			\
	    v = beta * log(u1 / (1.0 - u1));	\
	    if (v <= expmax) {			\
		w = AA * exp(v);		\
	    } else				\
		w = DBL_MAX


	if (a <= 1.0) {	/* --- Algorithm BC --- */

	/* changed notation, now also a <= b (was reversed) */

		if (!qsame) { /* initialize */
			beta = 1.0 / a;
			delta = 1.0 + b - a;
			k1 = delta * (0.0138889 + 0.0416667 * a) / (b * beta - 0.777778);
			k2 = 0.25 + (0.5 + 0.25 / delta) * a;
		}
		/* FIXME: "do { } while()", but not trivially because of "continue"s:*/
		for (;;) {
			u1 = unif_rand();
			u2 = unif_rand();
			if (u1 < 0.5) {
				y = u1 * u2;
				z = u1 * y;
				if (0.25 * u2 + z - y >= k1)
					continue;
			}
			else {
				z = u1 * u1 * u2;
				if (z <= 0.25) {
					v_w_from__u1_bet(b);
					break;
				}
				if (z >= k2)
					continue;
			}

			v_w_from__u1_bet(b);

			if (alpha * (log(alpha / (a + w)) + v) - 1.3862944 >= log(z))
				break;
		}
		return (aa == a) ? a / (a + w) : w / (a + w);

	}
	else {		/* Algorithm BB */

		if (!qsame) { /* initialize */
			beta = sqrt((alpha - 2.0) / (2.0 * a * b - alpha));
			gamma = a + 1.0 / beta;
		}
		do {
			u1 = unif_rand();
			u2 = unif_rand();

			v_w_from__u1_bet(a);

			z = u1 * u1 * u2;
			r = gamma * v - 1.3862944;
			s = a + r - w;
			if (s + 2.609438 >= 5.0 * z)
				break;
			t = log(z);
			if (s > t)
				break;
		} while (r + alpha * log(alpha / (b + w)) < t);

		return (aa != a) ? b / (b + w) : w / (b + w);
	}
}

double rpois(double mu)
{	
	/* Factorial Table (0:9)! */
	const static double fact[10] =
	{
	1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880.
	};

	double a0 = -0.5, a1 = 0.3333333, a2 = -0.2500068, a3 = 0.2000118, a4 = -0.1661269,
		a5 = 0.1421878, a6 = -0.1384794, a7 = 0.1250060;

	/* These are static --- persistent between calls for same mu : */
	static int l, m;
	static double b1, b2, c, c0, c1, c2, c3;
	static double pp[36], p0, p, q, s, d, omega;
	static double big_l;/* integer "w/o overflow" */
	static double muprev = 0., muprev2 = 0.;/*, muold	 = 0.*/

	/* Local Vars  [initialize some for -Wall]: */
	double del, difmuk = 0., E = 0., fk = 0., fx, fy, g, px, py, t, u = 0., v, x;
	double pois = -1.;
	int k, kflag, big_mu, new_big_mu = 0;

	if (mu <= 0.) return 0.;

	big_mu = mu >= 10.;
	if (big_mu) new_big_mu = 0;

	if (!(big_mu && mu == muprev)) {/* maybe compute new persistent par.s */

		if (big_mu) {
			new_big_mu = 0;
			/* Case A. (recalculation of s,d,l	because mu has changed):
			 * The poisson probabilities pk exceed the discrete normal
			 * probabilities fk whenever k >= m(mu).
			 */
			muprev = mu;
			s = sqrt(mu);
			d = 6. * mu * mu;
			big_l = floor(mu - 1.1484);
			/* = an upper bound to m(mu) for all mu >= 10.*/
		}
		else { /* Small mu ( < 10) -- not using normal approx. */

			/* Case B. (start new table and calculate p0 if necessary) */

			/*muprev = 0.;-* such that next time, mu != muprev ..*/
			if (mu != muprev) {
				muprev = mu;
				m = imax2(1, (int)mu);
				l = 0; /* pp[] is already ok up to pp[l] */
				q = p0 = p = exp(-mu);
			}

			repeat{
				/* Step U. uniform sample for inversion method */
				u = unif_rand();
				if (u <= p0)
					return 0.;

				/* Step T. table comparison until the end pp[l] of the
				   pp-table of cumulative poisson probabilities
				   (0.458 > ~= pp[9](= 0.45792971447) for mu=10 ) */
				if (l != 0) {
					for (k = (u <= 0.458) ? 1 : imin2(l, m); k <= l; k++)
					if (u <= pp[k])
						return (double)k;
					if (l == 35) /* u > pp[35] */
					continue;
				}
				/* Step C. creation of new poisson
				   probabilities p[l..] and their cumulatives q =: pp[k] */
				l++;
				for (k = l; k <= 35; k++) {
					p *= mu / k;
					q += p;
					pp[k] = q;
					if (u <= q) {
					l = k;
					return (double)k;
					}
				}
				l = 35;
			} /* end(repeat) */
		}/* mu < 10 */

	} /* end {initialize persistent vars} */

/* Only if mu >= 10 : ----------------------- */

	/* Step N. normal sample */
	g = mu + s * norm_rand();/* norm_rand() ~ N(0,1), standard normal */

	if (g >= 0.) {
		pois = floor(g);
		/* Step I. immediate acceptance if pois is large enough */
		if (pois >= big_l)
			return pois;
		/* Step S. squeeze acceptance */
		fk = pois;
		difmuk = mu - fk;
		u = unif_rand(); /* ~ U(0,1) - sample */
		if (d * u >= difmuk * difmuk * difmuk)
			return pois;
	}

	/* Step P. preparations for steps Q and H.
	   (recalculations of parameters if necessary) */

	if (new_big_mu || mu != muprev2) {
		/* Careful! muprev2 is not always == muprev
	   because one might have exited in step I or S
	   */
		muprev2 = mu;
		omega = M_1_SQRT_2PI / s;
		/* The quantities b1, b2, c3, c2, c1, c0 are for the Hermite
		 * approximations to the discrete normal probabilities fk. */

		b1 = one_24 / mu;
		b2 = 0.3 * b1 * b1;
		c3 = one_7 * b1 * b2;
		c2 = b2 - 15. * c3;
		c1 = b1 - 6. * b2 + 45. * c3;
		c0 = 1. - b1 + 3. * b2 - 15. * c3;
		c = 0.1069 / mu; /* guarantees majorization by the 'hat'-function. */
	}

	if (g >= 0.) {
		/* 'Subroutine' F is called (kflag=0 for correct return) */
		kflag = 0;
		goto Step_F;
	}


	repeat{
		/* Step E. Exponential Sample */

		E = exp_rand();	/* ~ Exp(1) (standard exponential) */

		/*  sample t from the laplace 'hat'
			(if t <= -0.6744 then pk < fk for all mu >= 10.) */
		u = 2 * unif_rand() - 1.;
		t = 1.8 + fsign(E, u);
		if (t > -0.6744) {
			pois = floor(mu + s * t);
			fk = pois;
			difmuk = mu - fk;

			/* 'subroutine' F is called (kflag=1 for correct return) */
			kflag = 1;

		  Step_F: /* 'subroutine' F : calculation of px,py,fx,fy. */

			if (pois < 10) { /* use factorials from table fact[] */
			px = -mu;
			py = pow(mu, pois) / fact[(int)pois];
			}
			else {
				/* Case pois >= 10 uses polynomial approximation
				   a0-a7 for accuracy when advisable */
				del = one_12 / fk;
				del = del * (1. - 4.8 * del * del);
				v = difmuk / fk;
				if (fabs(v) <= 0.25)
					px = fk * v * v * (((((((a7 * v + a6) * v + a5) * v + a4) *
							  v + a3) * v + a2) * v + a1) * v + a0)
					- del;
				else /* |v| > 1/4 */
					px = fk * log(1. + v) - difmuk - del;
				py = M_1_SQRT_2PI / sqrt(fk);
				}
				x = (0.5 - difmuk) / s;
				x *= x;/* x^2 */
				fx = -0.5 * x;
				fy = omega * (((c3 * x + c2) * x + c1) * x + c0);
				if (kflag > 0) {
					/* Step H. Hat acceptance (E is repeated on rejection) */
					if (c * fabs(u) <= py * exp(px + E) - fy * exp(fx + E))
						break;
					}
			 else
					/* Step Q. Quotient acceptance (rare case) */
					if (fy - u * fy <= py * exp(px - fx))
						break;
				}/* t > -.67.. */
	}
	return pois;
}

double rnbinom(int size, double prob)
{
	return (prob == 1) ? 0 : rpois(rgamma(size, (1 - prob) / prob));
}

double dnorm(double x, double mu, double sigma, int give_log)
{
	x = (x - mu) / sigma;
	x = fabs(x);
	if (give_log) return -(M_LN_SQRT_2PI + 0.5 * x * x + log(sigma));
	else return M_1_SQRT_2PI * exp(-0.5 * x * x) / sigma;
}


double chebyshev_eval(double x, const double* a, const int n)
{
	double b0, b1, b2, twox;
	int i;

	twox = x * 2;
	b2 = b1 = 0;
	b0 = 0;
	for (i = 1; i <= n; i++) {
		b2 = b1;
		b1 = b0;
		b0 = twox * b1 - b2 + a[n - i];
	}
	return (b0 - b2) * 0.5;
}

double lgammacor(double x)
{
	const static double algmcs[15] = {
	+.1666389480451863247205729650822e+0,
	-.1384948176067563840732986059135e-4,
	+.9810825646924729426157171547487e-8,
	-.1809129475572494194263306266719e-10,
	+.6221098041892605227126015543416e-13,
	-.3399615005417721944303330599666e-15,
	+.2683181998482698748957538846666e-17,
	-.2868042435334643284144622399999e-19,
	+.3962837061046434803679306666666e-21,
	-.6831888753985766870111999999999e-23,
	+.1429227355942498147573333333333e-24,
	-.3547598158101070547199999999999e-26,
	+.1025680058010470912000000000000e-27,
	-.3401102254316748799999999999999e-29,
	+.1276642195630062933333333333333e-30
	};

	double tmp;

#define xmax  3.745194030963158e306

	if (x < 94906265.62425156) {
		tmp = 10 / x;
		return chebyshev_eval(tmp * tmp * 2 - 1, algmcs, 5) / x;
	}
	return 1 / (x * 12);
}


double lgammafn(double x)
{
	double ans, y, sinpiy;


	/* For IEEE double precision DBL_EPSILON = 2^-52 = 2.220446049250313e-16 :
	   xmax  = DBL_MAX / log(DBL_MAX) = 2^1024 / (1024 * log(2)) = 2^1014 / log(2)
	   dxrel = sqrt(DBL_EPSILON) = 2^-26 = 5^26 * 1e-26 (is *exact* below !)
	 */
#define xmax  2.5327372760800758e+305
#define dxrel 1.490116119384765625e-8

	y = fabs(x);
	if (y < 1e-306) return -log(y); // denormalized range, R change
	if (y <= 10) return log(fabs(gammafn(x)));
	/*
	  ELSE  y = |x| > 10 ---------------------- */

	if (y > xmax) return DBL_MAX;
	if (x > 0) return M_LN_SQRT_2PI + (x - 0.5) * log(x) - x + lgammacor(x);

	sinpiy = fabs(r_sinpi(y));
	ans = M_LN_SQRT_PId2 + (x - 0.5) * log(y) - x - log(sinpiy) - lgammacor(y);
	return ans;
}


double stirlerr(double n)
{

#define S0 0.083333333333333333333       /* 1/12 */
#define S1 0.00277777777777777777778     /* 1/360 */
#define S2 0.00079365079365079365079365  /* 1/1260 */
#define S3 0.000595238095238095238095238 /* 1/1680 */
#define S4 0.0008417508417508417508417508/* 1/1188 */

	/*
	  error for 0, 0.5, 1.0, 1.5, ..., 14.5, 15.0.
	*/
	const static double sferr_halves[31] = {
	0.0, /* n=0 - wrong, place holder only */
	0.1534264097200273452913848,  /* 0.5 */
	0.0810614667953272582196702,  /* 1.0 */
	0.0548141210519176538961390,  /* 1.5 */
	0.0413406959554092940938221,  /* 2.0 */
	0.03316287351993628748511048, /* 2.5 */
	0.02767792568499833914878929, /* 3.0 */
	0.02374616365629749597132920, /* 3.5 */
	0.02079067210376509311152277, /* 4.0 */
	0.01848845053267318523077934, /* 4.5 */
	0.01664469118982119216319487, /* 5.0 */
	0.01513497322191737887351255, /* 5.5 */
	0.01387612882307074799874573, /* 6.0 */
	0.01281046524292022692424986, /* 6.5 */
	0.01189670994589177009505572, /* 7.0 */
	0.01110455975820691732662991, /* 7.5 */
	0.010411265261972096497478567, /* 8.0 */
	0.009799416126158803298389475, /* 8.5 */
	0.009255462182712732917728637, /* 9.0 */
	0.008768700134139385462952823, /* 9.5 */
	0.008330563433362871256469318, /* 10.0 */
	0.007934114564314020547248100, /* 10.5 */
	0.007573675487951840794972024, /* 11.0 */
	0.007244554301320383179543912, /* 11.5 */
	0.006942840107209529865664152, /* 12.0 */
	0.006665247032707682442354394, /* 12.5 */
	0.006408994188004207068439631, /* 13.0 */
	0.006171712263039457647532867, /* 13.5 */
	0.005951370112758847735624416, /* 14.0 */
	0.005746216513010115682023589, /* 14.5 */
	0.005554733551962801371038690  /* 15.0 */
	};
	double nn;

	if (n <= 15.0) {
		nn = n + n;
		if (nn == (int)nn) return(sferr_halves[(int)nn]);
		return(lgammafn(n + 1.) - (n + 0.5) * log(n) + n - M_LN_SQRT_2PI);
	}

	nn = n * n;
	if (n > 500) return((S0 - S1 / nn) / n);
	if (n > 80) return((S0 - (S1 - S2 / nn) / nn) / n);
	if (n > 35) return((S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n);
	/* 15 < n <= 35 : */
	return((S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n);
}


double gammafn(double x)
{
	const static double gamcs[42] = {
	+.8571195590989331421920062399942e-2,
	+.4415381324841006757191315771652e-2,
	+.5685043681599363378632664588789e-1,
	-.4219835396418560501012500186624e-2,
	+.1326808181212460220584006796352e-2,
	-.1893024529798880432523947023886e-3,
	+.3606925327441245256578082217225e-4,
	-.6056761904460864218485548290365e-5,
	+.1055829546302283344731823509093e-5,
	-.1811967365542384048291855891166e-6,
	+.3117724964715322277790254593169e-7,
	-.5354219639019687140874081024347e-8,
	+.9193275519859588946887786825940e-9,
	-.1577941280288339761767423273953e-9,
	+.2707980622934954543266540433089e-10,
	-.4646818653825730144081661058933e-11,
	+.7973350192007419656460767175359e-12,
	-.1368078209830916025799499172309e-12,
	+.2347319486563800657233471771688e-13,
	-.4027432614949066932766570534699e-14,
	+.6910051747372100912138336975257e-15,
	-.1185584500221992907052387126192e-15,
	+.2034148542496373955201026051932e-16,
	-.3490054341717405849274012949108e-17,
	+.5987993856485305567135051066026e-18,
	-.1027378057872228074490069778431e-18,
	+.1762702816060529824942759660748e-19,
	-.3024320653735306260958772112042e-20,
	+.5188914660218397839717833550506e-21,
	-.8902770842456576692449251601066e-22,
	+.1527474068493342602274596891306e-22,
	-.2620731256187362900257328332799e-23,
	+.4496464047830538670331046570666e-24,
	-.7714712731336877911703901525333e-25,
	+.1323635453126044036486572714666e-25,
	-.2270999412942928816702313813333e-26,
	+.3896418998003991449320816639999e-27,
	-.6685198115125953327792127999999e-28,
	+.1146998663140024384347613866666e-28,
	-.1967938586345134677295103999999e-29,
	+.3376448816585338090334890666666e-30,
	-.5793070335782135784625493333333e-31
	};

	int i, n;
	double y;
	double sinpiy, value;

# define ngam 22
# define xmin -170.5674972726612
# define xmax  171.61447887182298
# define xsml 2.2474362225598545e-308
# define dxrel 1.490116119384765696e-8

	y = fabs(x);

	if (y <= 10) {

		/* Compute gamma(x) for -10 <= x <= 10
		 * Reduce the interval and find gamma(1 + y) for 0 <= y < 1
		 * first of all. */

		n = (int)x;
		if (x < 0) --n;
		y = x - n;/* n = floor(x)  ==>	y in [ 0, 1 ) */
		--n;
		value = chebyshev_eval(y * 2 - 1, gamcs, ngam) + .9375;
		if (n == 0)
			return value;/* x = 1.dddd = 1+y */

		if (n < 0) {


			/* The argument is so close to 0 that the result would overflow. */
			if (y < xsml) {
				if (x > 0) return DBL_MAX;
				else return -DBL_MAX;
			}

			n = -n;

			for (i = 0; i < n; i++) {
				value /= (x + i);
			}
			return value;
		}
		else {
			/* gamma(x) for 2 <= x <= 10 */

			for (i = 1; i <= n; i++) {
				value *= (y + i);
			}
			return value;
		}
	}
	else {
		/* gamma(x) for	 y = |x| > 10. */

		if (x > xmax) return DBL_MAX;
		if (x < xmin) return 0.;

		if (y <= 50 && y == (int)y) { /* compute (n - 1)! */
			value = 1.;
			for (i = 2; i < y; i++) value *= i;
		}

		else { /* normal case */
			value = exp((y - 0.5) * log(y) - y + M_LN_SQRT_2PI +
				((2 * y == (int)2 * y) ? stirlerr(y) : lgammacor(y)));
		}

		if (x > 0) return value;

		sinpiy = r_sinpi(y);
		if (sinpiy == 0) return DBL_MAX;
		return -M_PI / (y * sinpiy * value);
	}
}

double bd0(double x, double np)
{
	double ej, s, s1, v;
	int j;

	if (fabs(x - np) < 0.1 * (x + np)) {
		v = (x - np) / (x + np);  // might underflow to 0
		s = (x - np) * v;/* s using v -- change by MM */
		if (fabs(s) < DBL_MIN) return s;
		ej = 2 * x * v;
		v = v * v;
		for (j = 1; j < 1000; j++) { /* Taylor series; 1000: no infinite loop
						as |v| < .1,  v^2000 is "zero" */
			ej *= v;// = v^(2j+1)
			s1 = s + ej / ((j << 1) + 1);
			if (s1 == s) /* last term was effectively 0 */
				return s1;
			s = s1;
		}
	}
	/* else:  | x - np |  is not too small */
	return(x * log(x / np) + np - x);
}

double dpois(double x, double lambda, int give_log)
{
	/*       x >= 0 ; integer for dpois(), but not e.g. for pgamma()!
		lambda >= 0
	*/
	if (x <= lambda * DBL_MIN) return(R_D_exp(-lambda));
	if (lambda < x * DBL_MIN) return(R_D_exp(-lambda + x * log(lambda) - lgammafn(x + 1)));
	return(R_D_fexp(M_2PI * x, -stirlerr(x) - bd0(x, lambda)));
}

double dgamma(double x, double shape, double scale, int give_log)
{
	double pr;
	if (x == 0) return give_log ? -log(scale) : 1 / scale;
	if (shape < 1) {
		pr = dpois(shape, x / scale, give_log);
		return give_log ? pr + log(shape / x) : pr * shape / x;
	}
	/* else  shape >= 1 */
	pr = dpois(shape - 1, x / scale, give_log);
	return give_log ? pr - log(scale) : pr / scale;
}


double dbinom(double x, double n, double p, double q, int give_log)
{
	double lf, lc;

	if (x == 0) {
		lc = (p < 0.1) ? -bd0(n, n * q) - n * p : n * log(q);
		return(R_D_exp(lc));
	}
	if (x == n) {
		lc = (q < 0.1) ? -bd0(n, n * p) - n * q : n * log(p);
		return(R_D_exp(lc));
	}

	if (x < 0 || x > n) return(give_log ? -DBL_MAX : 0);

	/* n*p or n*q can underflow to zero if n and p or q are small.  This
	   used to occur in dbeta, and gives NaN as from R 2.3.0.  */
	lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);

	/* f = (M_2PI*x*(n-x))/n; could overflow or underflow */
	/* Upto R 2.7.1:
	 * lf = log(M_2PI) + log(x) + log(n-x) - log(n);
	 * -- following is much better for  x << n : */
	lf = M_LN_2PI + log(x) + log1p(-x / n);

	return R_D_exp(lc - 0.5 * lf);
}

double lbeta(double a, double b)
{
	double corr, p, q;

	p = q = a;
	if (b < p) p = b;/* := min(a,b) */
	if (b > q) q = b;/* := max(a,b) */

	/* both arguments must be >= 0 */
	if (p == 0) return DBL_MAX;

	if (p >= 10) {
		/* p and q are big. */
		corr = lgammacor(p) + lgammacor(q) - lgammacor(p + q);
		return log(q) * -0.5 + M_LN_SQRT_2PI + corr
			+ (p - 0.5) * log(p / (p + q)) + q * log1p(-p / (p + q));
	}
	else if (q >= 10) {
		/* p is small, but q is big. */
		corr = lgammacor(q) - lgammacor(p + q);
		return lgammafn(p) + corr + p - p * log(p + q)
			+ (q - 0.5) * log1p(-p / (p + q));
	}
	else {
		/* p and q are small: p <= q < 10. */
		/* R change for very small args */
		if (p < 1e-306) return lgamma(p) + (lgamma(q) - lgamma(p + q));
		else return log(gammafn(p) * (gammafn(q) / gammafn(p + q)));
	}
}

double dbeta(double x, double a, double b, int give_log)
{
	if (x == 0) {
		if (a < 1) return(DBL_MAX);
		/* a == 1 : */ return(R_D_val(b));
	}
	if (x == 1) {
		if (b < 1) return(DBL_MAX);
		/* b == 1 : */ return(R_D_val(a));
	}

	double lval;
	if (a <= 2 || b <= 2)
		lval = (a - 1) * log(x) + (b - 1) * log1p(-x) - lbeta(a, b);
	else
		lval = log(a + b - 1) + dbinom(a - 1, a + b - 2, x, 1 - x, 1);

	return R_D_exp(lval);
}
