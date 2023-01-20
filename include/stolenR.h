// For mutual recursion:
double gammafn(double);
double lgammafn(double);

// Max funcs:
double fmin2(double x, double y);
double fmax2(double x, double y);
int imax2(int x, int y);
int imin2(int x, int y);
double fsign(double x, double y);
double r_sinpi(double x);

// Seeds:
void set_seed(unsigned int i1, unsigned int i2);
void get_seed(unsigned int* i1, unsigned int* i2);

// Samplers:
double unif_rand(void);
double runif(double a, double b);
double norm_rand(void);
double rnorm(double mu, double sigma);
double rbinom(int n, double pp);
double exp_rand(void);
double rgamma(double a, double scale);
double rbeta(double aa, double bb);
double rpois(double mu);
double rnbinom(int size, double prob);
double dnorm(double x, double mu, double sigma, int give_log);

// Linear algebra:
double chebyshev_eval(double x, const double* a, const int n);
double lgammacor(double x);
double lgammafn(double x);
double stirlerr(double n);

// Densities:
double gammafn(double x);
double bd0(double x, double np);
double dpois(double x, double lambda, int give_log);
double dgamma(double x, double shape, double scale, int give_log);
double dbinom(double x, double n, double p, double q, int give_log);
double lbeta(double a, double b);
double dbeta(double x, double a, double b, int give_log);
