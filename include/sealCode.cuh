// We list here only the CPP wrappers for the CUDA functions we might want to call in MAIN.

// Small utils:
int computerGPUcount();
void resetAllDevices();

// Used in proposal generating functions:
using namespace Eigen;
typedef Matrix<float, Dynamic, Dynamic> FloatMatrix;

// Various proposal related functions:
struct parameterSpec {

    float (*log_density_function) (float, float, float, float, float);
    float hyper_par_0, hyper_par_1, offset_0, offset_1;

    float evaluateLogDensity(float x) {
        return log_density_function(x, hyper_par_0, hyper_par_1, offset_0, offset_1);
    }
};

bool sealsLegalityChecks(float* theta, int n_theta);
void sealsThetaMapGeneral(float* input, float* output, int R, int n);
void sealsThetaMapCloning(float* input, float* output, int R, int n);
float scaledBetaLogDensity(float x, float alpha, float beta, float offset_0, float offset_1);
float scaledGammaLogDensity(float x, float alpha, float beta, float offset_0, float offset_1); 

// Various MCMC and testing functions:
void sealsMultiRegionMCMC(int n_samples, int n_particles, float* theta_0, FloatMatrix sigma,
    char* filename, char* dbg_filename, parameterSpec* prior_specifications, int n_states,
    int n_theta, int n_theta_region, int n_regions, int tpb, int T, int** y, int* y0,
    int create_csv_header, bool (*legalityChecks) (float*, int),
    void (*thetaMapper) (float*, float*, int, int), float* k0, int n_indep_ests, int G,
    int B, int verbose, int prop_1p, int useR, int* EEI, float M, int L, int C, int ieCV);

void sealsBFinder(int n_samples, int n_particles, float* theta, char* filename,
    int n_states, int n_theta, int n_theta_region, int n_regions, int tpb, int T,
    int** y, int* y0, int create_csv_header, void (*thetaMapper) (float*, float*, int, int),
    float* k0, int n_indep_ests, int* EEI, int G, int B, float incr, int n_incr, float M,
    int L, int append, int ieCV);

void sealsNFinder(int n_samples, int n_particles, float* theta, char* filename,
    int n_states, int n_theta, int n_theta_region, int n_regions, int tpb, int T,
    int** y, int* y0, int create_csv_header, void (*thetaMapper) (float*, float*, int, int),
    float* k0, int n_indep_ests, int* EEI, int G, int B, float incr, int n_incr, float M = -23,
    int L = 1, int append = 0, int ieCV = 27);