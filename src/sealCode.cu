#include <iostream>
#include <chrono>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/CUDA Samples/v10.2/common/inc/helper_cuda.h"
#include "../include/Eigen/Eigen"
#include "../include/stolenR.h"
#include "../include/sealCode.cuh"

///////////////////////////////////////////// UTILS ////////////////////////////////////////////////

void computerGPUCount_private(int* G){
    /*
    purpose : Returns the number of CUDA capabe GPUs on the machine by writing it to the 
              location of an int d
    */
    cudaGetDeviceCount(G);
}

int computerGPUcount(){
    int G;
    computerGPUCount_private(&G);
    return G;
}

void resetAllDevices_private(){
    /*
    purpose : Finds out the number of devices and clears all device memory, primarily as a 
              way of avoiding hanging GPUs at the end of a script.
    */
    int n_GPUs;
    computerGPUCount_private(&n_GPUs);

    for (int dev_id = 0; dev_id < n_GPUs; dev_id++){
        checkCudaErrors(cudaSetDevice(dev_id));
        checkCudaErrors(cudaDeviceReset());
    }
}

void resetAllDevices(){
    /*
    purpose : A wrapper that resets the memory on all GPUs without directly containing
              any CUDA code
    */
    resetAllDevices_private();
}

__device__ float dGammaFixed(float x, float k0){
    /*
    purpose : dGamma function with the expectation that x has had k0 removed from its value already.
              Uses the CV of the original (2008) independent estimate, by holding K1 and K2 fixed, 
              at the values specified in the 2019 Thomas et al paper.
    */
    float X = x - k0;
    return - 122.3281 - X / 2719.37889 + 11.95541 * log(X);
}

__device__ float dGammaCV1(float x, float k0){
    /*
    purpose : dGamma function with 2008 mean and coefficient of variation fixed at 1%
    */
    float X = x - k0;
    return -94693.04 - (X / 3.523067) + 9999 * log(X);
}

__device__ float dGammaCV5(float x, float k0){
    /*
    purpose : dGamma function with 2008 mean and coefficient of variation fixed at 5%
    */
    float X = x - k0;
    return -3785.792 - (X / 88.07667) + 399 * log(X);
}

// In order to be able to pass device functions as arguments to other functions, we must make a
// static pointer for each function:
__device__ float (*p_dGammaFixed) (float, float) = dGammaFixed;
__device__ float (*p_dGammaCV1) (float, float) = dGammaCV1;
__device__ float (*p_dGammaCV5) (float, float) = dGammaCV5;

__global__ void dGammaGeneralGPU(float* d_totals, float* d_output, int n, float k0,
    float (*dGammaGPUfunc) (float, float)){
    /*
    purpose : Behaves as dgammaSeals2008GPU but with k0 as a general parameter
    inputs  : d_totals      - The device side pointer of population totals
              d_output      - The device sound pointer which stores the calculation output
              n             - The number of entries in the above two vectors
              k0            - The independent estimate k0 value
              dGammaGPUfunc - The function which evaluates the Gamma likelihood of the independent
                              estimate
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) d_output[tid] += dGammaGPUfunc(d_totals[tid], k0);
}

template <class CS>
__global__ void setCurandSeeds(CS* d_states, int n, time_t seed) {
    // purpose : Sets up the initial seeds for a collection of states to be used by the curand
    //           device API.
    // inputs  : d_states - Pointer to the vector of states for the RNGs
    //           n        - The number of states to be initialised
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) curand_init(seed, tid, 0, &d_states[tid]);
}

template <class FM>
FM sqrtPmax0(FM vector) {
    // purpose : Returns the vector that would result from calling pmax(vector, 0) followed by
    //           sqrt in R. i.e, performs max(value, 0) for each value in the input vector, then 
    //           returns the sqrt of the result
    // input   : vector - Input vector of values
    // output  : A FloatMatrix object
    int n = vector.rows();
    FM output(n, 1);
    for (int i = 0; i < n; i++) output(i, 0) = (vector(i, 0) >= 0) ? sqrt(vector(i, 0)) : 0;
    return output;
}

template <class FM1, class FM2>
FloatMatrix multiplyRowsByVector(FM1 matrix, FM2 vector) {
    // purpose : Multiplies each row of matrix by the constant in the same numbered entry in
    //           vector. Assumes matrix is symmetric and the vector has the same number of columns
    //           as the dimensionality of matrix.
    int n = matrix.rows();
    FloatMatrix output(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) output(i, j) = vector(i, 0) * matrix(i, j);
    }

    return output;
}

template <class CS>
__device__ float rnormDevice(float mu, float sd, CS* state_address) {
    return curand_normal(state_address) * sd + mu;
}

template <class FM>
FM rmvnormPrep(FM sigma) {
    // purpose : Produces the pre-calculations required for generating multivariate normal deviates
    // inputs  : sigma - Object containing the symmetric variance-covariance matrix
    // Note, we swap the eigen vectors and values ot be in decreasing order to remain consistent
    // with the implementation of rmvnorm in R.
    int n = sigma.rows();
    SelfAdjointEigenSolver<FM> solved_sigma(sigma);
    FloatMatrix eig_vecs(n, n), working_mat(n, n), R(n, n), eig_vals(n, 1), eig_holder(n, 1);

    eig_holder = sqrtPmax0(solved_sigma.eigenvalues());
    for (int i = 0; i < n; i++) eig_vals(i, 0) = eig_holder(n - i - 1, 0);

    working_mat = solved_sigma.eigenvectors();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) eig_vecs(i, j) = -working_mat(i, n - j - 1);
    }

    working_mat = multiplyRowsByVector(eig_vecs.transpose(), eig_vals);
    R = eig_vecs * working_mat;
    R.transposeInPlace();
    return R;
}

template <class FM>
int rmvnorm(float* output, float* means, FM R) {
    // purpose : Produces a sample from a multivariate normal distribution, given pre-computations
    // inputs  : output - The float array where the deviate should be stored
    //           means  - The vector of means for each dimension of the normal
    //           R      - The matrix of pre-computed values produced by calling rmvnormPrep. In 
    //                    many cases (such as during an MCMC) a large number of values from the
    //                    same covariance structure are required. This pre-computation avoids
    //                    having to needlessly calculate the eigenvalues and vectors of the
    //                    covariance matrix every time
    //           n      - The number of dimensions to the normal distribution
    int n = R.rows();
    FloatMatrix deviates(1, n);
    for (int i = 0; i < n; i++) deviates(0, i) = rnorm(0, 1);
    deviates = deviates * R;
    for (int i = 0; i < n; i++) output[i] = deviates(0, i) + means[i];
    return 0;
}

template <class CS>
__device__ int rbinomLowNP(int n, float pp, CS* state_address){
    // Note: There is no need to point to a local state address here, since it is 
    //       expected that this function will only be called by __global__
    //       functions that have created a local state_address already
    // This function is a copy of the R programming language's source code
    // for binomial random deviates in cases where n * p < 30. The expectation is that 
    // it leads to very large amounts of thread divergence, but makes up for this 
    // extra compute time by being far more efficient than generating binomials
    // with a sum of bernoullis.

    int ix;
    float p, r, f, u, q, qn, g;
    
    p = fminf(pp, 1. - pp);
    q = 1 - p;
    r = p / q;
    g = r * (n + 1);
    qn = pow(q, n);
    
    for(;;){
        ix = 0;
        f = qn;
        u = curand_uniform(state_address);
        for(;;){
            if (u < f) goto finish;
            if (ix > 100) break;
            u -= f;
            ix++;
            f *= (g / ix - r);
        }
    }

    finish:
    ix = n - ix;
    return ix;
}

template <class CS>
__device__ int rbinomNormApprox(int n, float p, CS* state_address) {
    // Note: Can lead to warp divergence but typically in our application
    //       threads in a warp will either almost all be using the approx
    //       or none at all
    float np = n * p, q = 1 - p, nq = n * q;
    if (np < 30 | nq < 30) return(rbinomLowNP(n, p, state_address));

    else {
        float normals = curand_normal(state_address);
        int output = round(normals * sqrt(np * q) + np);
        if (output < 0) output = 0;
        if (output > n) output = n;
        return (output < 0) ? 0 : output;
    }
}

template <class CS>
__device__ int rnbinomNormApprox(int n, float p, CS* state_address) {
    // Note this function is not exact, and the aproximation is 
    // only good because in our application n is usually > 2000 and
    // 0.6 < p < 0.9
    int output;
    float normals = curand_normal(state_address);
    float pinv = (1 / p), nqpinv = n * (1 - p) * pinv;
    output = round(normals * sqrt(nqpinv * pinv) + nqpinv);
    return (output < 0) ? 0 : output;
}

__device__ float fabs_device(float x) {
    return (x < 0) ? -x : x;
}

template <class CS>
__global__ void generateCurandNormals(CS* d_states, float* d_results, int n) {
    // purpose : d_states
    // inputs  : d_states  - Pointer to the vector of states for the RNGs
    //           d_results - Pointer to the GPU memory where the results should be stored
    //           n         - The number of numbers to generate
    //           k         - The number of random numbers to be generated per thread
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) d_results[tid] = curand_normal(&d_states[tid]);
}


template <class T>
__global__ void fillWithConst(T* d_vector, T constant, int n) {
    // purpose : Fills a piece of device memory with constants
    // inputs  : d_vector - The vector of inputs to be written over
    //           const    - The value each entry of the vector should be written over with
    //           n        - The largest index in the vector which should be overwritten
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) d_vector[tid] = constant;
}

template <class T>
T Min(T* arg, int n) {
    // purpose : Returns the minimum of a vector
    // inputs  : floats - The vector of ints
    //           n      - The number of numbers to go through
    // output  : A class T result

    float temp = arg[0];
    for (int i = 1; i < n; i++) if (arg[i] < temp) temp = arg[i];
    return(temp);
}

template <class T>
T Max(T* arg, int n) {
    // purpose : Returns the maximum of a vector
    // inputs  : floats - The vector of ints
    //           n      - The number of numbers to go tmaxmaxhrough
    // output  : A class T result

    float temp = arg[0];
    for (int i = 1; i < n; i++) if (arg[i] > temp) temp = arg[i];
    return(temp);
}

template <class T>
__global__ void systematicSamplingGPU(int* d_out, T* d_weights, T* d_ws, int n, double U) {
    // purpose : Selects members of (0, n-1) via systematic sampling, using d_weights as weightings
    //           for each entry. Single threaded GPU version of the C code, simply to avoid memcpys
    //           of the data back to the CPU
    // inputs  : d_out     - Pointer to the GPU memory where the sampled entries should be written
    //           d_weights - Pointer to the memory on the GPU where the sampling weights are stored
    //           d_ws      - A pointer of GPU memory where the sum of the weights is stored in the
    //                       first entry
    //           n         - The number of indices we sample from (in the range [0, 1, ..., n - 1])
    //           U         - The uniform deviate from which we can calculate the offset
    // Note : single threaded application.

    double w = 0;
    for (int i = 0; i < n; i++) w += d_weights[i];

    int counter = 0;
    double total = 0;

    for (int i = 0; i < n; i++) {
        while (total < (double(i) + U) / n) {
            total += d_weights[counter] / w;
            counter++;
        }//end while

        d_out[i] = counter - 1;
    }//end for
}

template <class T>
__global__  void naiveSumGPU(T* d_input, T* d_output, int start, int end){
    /*
    purpose : This kernel is designed to be a single threaded addition 
              function that only deals with summing the remainder
              when the workload has been chopped up into blocks of tpb.
    inputs  : d_input  - The vector of elements to be summed
              d_output - The output variable where the sum should be
                         written
              start    - The first index which should be summed
              end      - The index up until the sum should continue
    Note    : In practice it is hoped that this function will be used 
              very little, since careful consideration of block sizes
              should avoid having remainders at every iteration
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0){
        for (int i = start; i < end; i++) d_output[0] += d_input[i];
    }
}

template <class T>
__global__ void reduceSumGPU(T* d_input, T* d_output){
    /*
    purpose : Performs a sum reduction of a fixed blocksize using sequential addressing
    inputs  :  d_input  - The data to be summed.
               d_output - The location of the output. Can be the same as the input if this
                          does not need to be preserved
    Note    : This function is designed to work with perfect powers of two, and the 
              gridsize / blocksize will determine the behaviour
    Note    : This function dynamically allocated its shared memory allocation, and so
              when the kernel is invoked, sizeof(T) * thread_per_block must be added to 
              the kernel call after the grid and block dimensions
    */
    unsigned int i = threadIdx.x;
    unsigned int tid = i + blockIdx.x * blockDim.x * 2;

    // Ask for the shared memory this thread will use:
    extern __shared__ T sdata[];

    // Do one addition while loading the data into shared mem:    
    sdata[i] = (T) d_input[tid] + d_input[tid + blockDim.x];

    // Perform the reduction:
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if (i < s) sdata[i] += sdata[i + s];   
        __syncthreads();
    }

    // The lead thread in the block writes the output sum from the block:
    if (i == 0) d_output[blockIdx.x] = (T) sdata[0];
}

template <class T>
void sumReductionGPU(T* d_input, T* d_out, int n, int tpb){
    /*
    purpose : Uses the reduceSumGPU kernel which performs a single pass of a sum
              reduction with multiple iterations to find the sum of a vector
              of any length
    inputs  : d_input - The vector to be summed
              d_out   - The location where the final answer should be stored
              n       - The length of the vector
              tpb     - The number of threads per block to be used

    NOTE : With doubles, tpb = 64 is the highest that can be done without causing errors due to 
           insufficient shared memory
    */
    // Determine initial grid size:
    int twotpb = 2 * tpb;
    int blocks = n / twotpb;
    int r = n % twotpb;

    // Create the output vector:
    T* d_output;
    checkCudaErrors(cudaMalloc((void**)&d_output, sizeof(T) * blocks));
    fillWithConst<<<1, 1>>>(d_out, (T) 0, 1);

    // First reduction:
    if (r > 0) naiveSumGPU<<<1, 1>>>(d_input, d_out, blocks * tpb, blocks * tpb + r);
    reduceSumGPU<<<blocks, tpb, sizeof(T) * tpb>>>(d_input, d_output);

    // Recalculate settings:
    n = blocks;
    blocks = n / twotpb;
    r = n % twotpb;

    // All other reductions:
    while (blocks > 0){

        // Perform the reduction:
        if (r > 0) naiveSumGPU<<<1, 1>>>(d_output, d_out, n - r, n);
        reduceSumGPU<<<blocks, tpb, sizeof(T) * tpb>>>(d_output, d_output);

        // Recalculate settings:
        n = blocks;
        blocks = n / twotpb;
        r = n % twotpb;
    }

    // Add the final < tpb numbers:
    if (r > 0) naiveSumGPU<<<1, 1>>>(d_output, d_out, 0, r);

    // Free memory:
    checkCudaErrors(cudaFree(d_output)); 
}

template <class T>
__global__ void maxGPUDriver(T* d_input, T* d_output, int n, int apt){
    /*
    purpose : Performs a very lazy parallel search for the largest value in a vector
    inputs  :  d_input  - The data whose maximum we wish to find
               d_output - The location of the output. Needs to have enough space for 
                          one T written per block
               n        - The number of total items in the vector
               atp      - The number of actions per thread (i.e. the number of elements in
                          the input vector that it'll check)
    Note    : It is assumed this function will always be run so that each thread has
              to look over atp numbers. We require atp * T memory to be assigned to shared
              mem for this kernel to run.
    */
    // Determine thread starting point, and block end point:
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x * apt;
    unsigned int const end = (blockIdx.x + 1) * blockDim.x * apt;

    // Give us shared memory to store the result of each thread:
    extern __shared__ T sdata[];

    // If a thread will do nothing, we first force it to write in the value of the first thread
    // in the block (since this is the only thread guaranteed to be < n, if usless blocks have not
    // been assigned). This prevents 0 being returned as a block's output when all of the weights
    // are below 1, in a block that had some threads that did nothing;
    if (tid >= n) sdata[threadIdx.x] = d_input[blockDim.x * apt * blockIdx.x];
    
    if (tid < n){
        // Get the threads to all go and check atp numbers:
        T maximum = (d_input[tid]);
        while (tid < n && tid < end){
            if (d_input[tid] > maximum) maximum = d_input[tid];
            tid += blockDim.x;
        }//end while
        
        // Write the thread result to shared memory and wait for the rest of the block to catch up:
        sdata[threadIdx.x] = maximum;
        __syncthreads();

        // Get the lead thread to add up the work for the entire block:
        if (threadIdx.x == 0){
            for (int i = 1; i < blockDim.x; i++) if (sdata[i] > maximum) maximum = sdata[i];
            d_output[blockIdx.x] = maximum;
        }//end if threadIdx.x
    }//end if tid
}//end function

template <class T> 
__global__ void singleThreadedMaxGPU(T* d_in, T* d_out, int n){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid == 0){
        T M = d_in[0];
        for (int i = 1; i < n; i++) if (d_in[i] > M) M = d_in[i];
        d_out[0] = M;
    }//end if
}//end function

template <class T> 
void parallelMax(T* d_in, T* d_output, int n, int tpb, int apt){
    /*
    purpose : Uses the maxGPUDriver kernel to find the maximum of a vector of n values using a grid
              of blocks with tpb threads each
    inputs  : d_in  - The input vector
              n     - The number of elements
              tpb   - The number of threads per block
              apt   - The number of elements that each thread will check
    */
    
    // Calculate grid size:
    unsigned int blocks = n / (tpb * apt);
    blocks = blocks * tpb * apt >= n ? blocks : blocks + 1;

    // Allocate memory for one T per block:
    T* d_out;
    checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(T) * blocks));

    // Run the kernel which checks apt numbers per thread, and then uses the shared memory in the 
    // block to do a single threaded check of the work performed by all threads in that block:
    maxGPUDriver<<<blocks, tpb, sizeof(T) * tpb>>>(d_in, d_out, n, apt);
    singleThreadedMaxGPU<<<1, 1>>>(d_out, d_output, blocks);
    checkCudaErrors(cudaFree(d_out));
}//end function

template <class T, class CS>
__global__ void metropolisResampleGPU(T* d_weights, int* d_out, CS* d_states, int n,
    int B, int logw = 0) {
    // purpose : An approximation of multinomial resampling using the Metropolis sampler described 
    //           in Murray, Lee & Jacob (2016). 
    // inputs  : d_weights - The vector of weights which are to be sampled from proportionally
    //           d_out     - The vector where the chosen indices should be stored
    //           n         - The number of weights (and therefore indices)
    //           B         - The number of iterations the Metropolis algorithm before we consider
    //                       convergence to be likely
    //           logw      - If >= 1, tells us that the weights ar ein fact log weights, and that
    //                       we should calculate the metropolis ratio accordingly
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {

        int k = tid;
        curandState local_state = d_states[tid];
        curandState* local_state_address = &local_state;

        // To avoid checking if we have log weights each iteration:
        if (logw < 1){
            for (int i = 0; i < B; i++) {
                float U = curand_uniform(local_state_address);
                int J = curand_uniform(local_state_address) * n;
                if (U < d_weights[J] / d_weights[k]) k = J;
            }//end for
        }// end if logw

        else{
            for (int i = 0; i < B; i++) {
                float U = curand_uniform(local_state_address);
                int J = curand_uniform(local_state_address) * n;
                if (U < exp(d_weights[J] - d_weights[k])) k = J;
            }//end for
        }
        
        d_out[tid] = k;
        d_states[tid] = *local_state_address;
    }
}

template <class T, class CS>
__global__ void rejectionResampleGPU(T* d_weights, int* d_out, CS* d_states, int n,
    T* max_w, int logw = 0) {
    /*
    purpose : A rejection sampler for multinomial sampling 
    inputs  : d_weights - The vector of weights which are to be sampled from proportionally
              d_out     - The vector where the chosen indices should be stored
              n         - The number of weights (and therefore indices)
              max_w     - The largest weight in the set. Usually has to be computed serially
              logw      - If >= 1, tells us that the weights are in fact log weights, and that
                          we should calculate the metropolis ratio accordingly
    */
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        T W = max_w[0];
        curandState local_state = d_states[tid];
        unsigned int j = tid;
        T u = curand_uniform(&local_state);
        
        if (logw > 0){
            while(u > exp(d_weights[j] - W)){
                j = curand_uniform(&local_state) * n;
                u = curand_uniform(&local_state);
            }//end while
        }//end if
        
        else{
            while(u > (d_weights[j] / W)){
                j = curand_uniform(&local_state) * n;
                u = curand_uniform(&local_state);
            }//end while
        }//end else

        d_out[tid] = j;
        d_states[tid] = local_state;
    }//end if tid
}//end function

////////////////////////////////////////////// END OF UTILS ////////////////////////////////////////

/////////////////////////////////////////// SEALS CODE GPU /////////////////////////////////////////
template <class T>
__global__ void dnormSealsGPU(int x, T* d_out, int* d_mu, float* d_theta, int n, int peq = 0) {
	// purpose : Calculates the density of the normal distribution at a given set of data points
	// inputs  : x   - The observation
	//           out - Pointer to the memory where the output should be stored
	//           mu  - The mean of the normal distribution (vector)
	//           sd  - The standard deviation of the noormal distribution (vector)
	//           n   - The number of points to consider in the vector of observations
	//           peq - (plus equals) If this is >= 1, we append log likelihood values to the 
	//                 output instead of overwriting them. This is useful when running this
	//                 function in sequence multiple times when the likelihood of multiple
	//                 independent regions has to be considered.
        float PI = 3.14159265358979323;//8462643383279502884197169399;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < n) {
		float sd = d_mu[tid] / sqrt(d_theta[5]);
		float v = 2 * sd * sd;
		float d = x - d_mu[tid];
		d = -(0.5) * log(PI * v) - (1.0 / v) * d * d;
		(peq > 0) ? d_out[tid] += d : d_out[tid] = d;
	}
}

__global__ void produceIntitialStatesGPU(int* d_states, float* d_theta, int y_0, int n,
    curandState* d_curand_states, int include_omega = 0) {
    // purpose : Samples initial state values given parameter set theta and initial observation y_0
    // inputs  : states - A pointer to the memory where the generated states should be written
    //           theta  - Pointer to a vector of floats of parameter values
    //           y_0    - The initial count of pups in the population
    //           n      - The number of sets of initial state values to generate
    //

    // NOTE: To make life easier since many operations are performed one column at a time, it is
    //       assumed that the state matrix is stored by column and not by row. I.e. 
    //       [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (1,0), ...]. This also has the
    //       advantage of being consistent with the way the original R code stores matrices.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        curandState local_state = d_curand_states[tid];
        float normal, a, b, prob, beta_r;
        int offset;
        
        float omega;
        omega = include_omega > 0 ? d_theta[6] : 2.0f;

        // Pup survival at carrying cap:
        beta_r = 0.5 * d_theta[2] * d_theta[0] * pow(d_theta[1], 5);
        beta_r /= (1 - d_theta[1]);
        beta_r = pow(beta_r - 1, 1 / d_theta[4]);
        beta_r /= d_theta[3];
        float phi_p = d_theta[0] / (1 + pow(beta_r * y_0, d_theta[4]));

        // Reverse the normal observation density:
        normal = fabs_device(rnormDevice(y_0, y_0 / sqrt(d_theta[5]), &local_state));
        a = normal / 1.3, b = normal * 1.3; // hardcoded value from Len's paper
        normal = a + curand_uniform(&local_state) * (b - a);
        d_states[tid] = round(normal);

        // Generate the remaining states with binomial trials of survival numbers:
        for (int column = 1; column < 6; column++) {
            prob = d_theta[1];
            if (column == 1) prob = phi_p / omega;
            offset = column * n + tid; //check these coalesce
            d_states[offset] = rbinomNormApprox(d_states[offset - n], prob, &local_state);
        }

        // Generate the final state column:
        offset = 6 * n;
        int deviate = rnbinomNormApprox(d_states[tid], d_theta[2], &local_state);
        d_states[offset + tid] = deviate + d_states[tid];

        // Update global state:
        d_curand_states[tid] = local_state;
    }
}

__global__ void sampleStatesDevAPI(int* d_states, int* d_new_states, int* d_rows,
    float* d_theta, int n, curandState* d_curand_states, int include_omega = 0, int debug = 0) {
    // purpose : Samples a new set of states given a matrix of old states and a vector of parameter
    //           values for the model. Uses a pointer to a list of indices to avoid having to write
    //           all of the new values to a matrix during resampling, since we only care about 
    //           being able to read the right values.
    // inputs  : states     - Pointer to the matrix of previous states
    //           new_states - Pointer to the piece of memory where new states should be written
    //           rows       - Pointer to a vector of indices which indicate the states row which 
    //                        has been selected as the basis for each new row
    //           theta      - Pointer to the vector of model parameter values
    //           n          - The number of states currently being dealt with
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
           
	float omega = include_omega > 0 ? d_theta[6] : 2.0f;
        curandState local_state = d_curand_states[tid];
        int row = d_rows[tid];
        float beta_r;
        
        beta_r = 0.5 * d_theta[2] * d_theta[0] * pow(d_theta[1], 5);
        beta_r /= (1 - d_theta[1]);
        beta_r = pow(beta_r - 1, 1 / d_theta[4]);
        beta_r /= d_theta[3];
        
        // Pup survival at carrying cap:
        float phi_p = d_theta[0] / (1 + pow(beta_r * d_states[row], d_theta[4]));

        // Adult survival: 
        float prob = d_theta[1];

        // Surviving female pups (age group 2):
        d_new_states[n + tid] = rbinomNormApprox(d_states[row], phi_p / omega, &local_state);

        // Age groups 3 through 6 (used to be a for loop, expanded for readability):
        d_new_states[n * 2 + tid] = rbinomNormApprox(d_states[n * 2 + row - n], prob, &local_state);
        d_new_states[n * 3 + tid] = rbinomNormApprox(d_states[n * 3 + row - n], prob, &local_state);
        d_new_states[n * 4 + tid] = rbinomNormApprox(d_states[n * 4 + row - n], prob, &local_state);
        d_new_states[n * 5 + tid] = rbinomNormApprox(d_states[n * 5 + row - n], prob, &local_state);

        // 6+ years old (age group 7): 
        int deviate = rbinomNormApprox(d_states[n * 5 + row], prob, &local_state);
        int deviate2 = rbinomNormApprox(d_states[n * 6 + row], prob, &local_state);
        d_new_states[n * 6 + tid] = deviate + deviate2;
        
        // New pups:
        d_new_states[tid] = rbinomNormApprox(deviate + deviate2, d_theta[2], &local_state);

        // Update global state:
        d_curand_states[tid] = local_state;
    }
}

float scaledGammaLogDensity(float x, float alpha, float beta, float offset_0, float offset_1) {
    // purpose : Evaluates the density of the generalised gamma distribution using the C code
    //           underpinning R's dgamma function.
    // inputes : x        - The value at which the pdf should be evaluated
    //           alpha    - Parameter of the gamma distribution 
    //           beta     - Parameter of the gamma distribution
    //           offset_0 - The multiplicative link offset i.e B in Y = A + B * X
    //           offset_1 - The additive link offset i.e A in Y = A + B * X
    if (x < 0 || alpha < 0 || beta < 0) return(-1.79e308);
    return dgamma((x - offset_1) / offset_0, alpha, beta, 1) / offset_0;
}

float scaledBetaLogDensity(float x, float alpha, float beta, float offset_0, float offset_1) {
    // purpose : Evaluates the density of the generalised beta distribution using the C code
    //           underpinning R's dbeta function.
    // inputes : x        - The value at which the pdf should be evaluated
    //           alpha    - Parameter of the beta distribution 
    //           beta     - Parameter of the beta distribution
    //           offset_0 - The multiplicative offset i.e B in Y = A + B * X
    //           offset_1 - The additive offset i.e A in Y = A + B * X
    if (x < 0 || alpha < 0 || beta < 0) return(-1.79e308);
    return dbeta((x - offset_1) / offset_0, alpha, beta, 1) / offset_0;
}

float getPriorLogDensity(parameterSpec* priors, float* theta, int n) {
    // purpose : Evaluates the prior density of a given set of parameters
    // inputs  : priors - Pointer to an array of objects which contain prior specifications
    //           theta  - The vector of proposed parameters
    //           n      - The number of parameters for which the prior should be evaluated

    parameterSpec holding_spec;
    float output = 0;

    for (int i = 0; i < n; i++) {
        holding_spec = priors[i];
        float holding = holding_spec.evaluateLogDensity(theta[i]);
        output += holding_spec.evaluateLogDensity(theta[i]);
    }

    return output;
}

bool sealsLegalityChecks(float* theta, int n_theta) {
    // purpose : Checks if the particle filter is able to run with the proposed set of parameter
    //           values
    // inputs  : A pointer to the 10 parameter values
    // Note: This function can be used as the legality check for a single region, or all regions, 
    //       since the order og the first 3 parameters will be the same in these cases, and the same
    //       general restrictions apply to all the others.
    float alpha = theta[2], phi_p = theta[0], phi_a = theta[1];
    if (alpha * phi_p * pow(phi_a, 5) + 2 * phi_a < 2) return false;
    if (Min(theta, n_theta) < 0) return false;
    if (Max(theta, 3) > 1) return false;
   
    // Check the limits put in place for the shifted parameters:
    if (theta[1] < 0.8 || theta[1] > 0.97) return false;
    if (theta[2] < 0.6) return false;
    if (theta[9] < 1.6) return false;
    return true;
}

void sealsThetaMapGeneral(float* input, float* output, int R, int n){
    // purpose : Extracts the region specific parameters from a vector of all parameters for the
    //           multi-region seals model, including the sex-ratio parameter
    // inputs  : input  - Array with parameters for all regions
    //           output - Where to write the region specific parameters
    //           R      - The total number of regions
    //           n      - The region number we are currently fetching for 
    // Input order is : [phi_pmax, phi_a, alpha, rho, phi, all chi, omega]
    // Output order is : [phi_pmax, phi_a, alpha, all chi, rho, phi, omega]
    memcpy(&output[0], &input[0], sizeof(float) * 3); // phi_pmax, phi_a, alpha
    memcpy(&output[3], &input[5 + n], sizeof(float)); // chi
    memcpy(&output[4], &input[3], sizeof(float) * 2); // rho, psi
    memcpy(&output[6], &input[5 + R], sizeof(float)); // omega
}

template <class T>
__global__ void colSumGPU(T* d_pointer, float* d_output, int n_s, int n, int start_col,
    int R = 0){
    /*
    purpose : Takes the matrix d_pointer with n rows and k columns, and adds the sum of the
              k columns to the correct index of the output vector for each row.
    inputs  : d_pointer - Pointer to the vector containing a byColums matrix of values
              d_output  - Pointer to the vector of values to write the output
              n_s       - The number of columns, in our case, the number of states
                          each particle is keeping track of
              n         - The number of rows, in our case, the number of particles
              start_col - The column number where the counting begins
              R         - If == 0, this will reset the output vector, this is to 
                          be used for resetting a total at the start of a loop 
                          where this kernel will be called multiple times to calculate
                          the totals of multiple matrices
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n){
       T x = 0;
       // We start at i = 1 instead of zero because the model doesn't include pups
       // in the independent estimate of total population size:
       for (int i = start_col; i < n_s; i++) x += d_pointer[tid + n * i];
       if (R == 0) d_output[tid] = 0;
       d_output[tid] += (float) x;
    }
}

__global__ void scaleVectorGPU(float* d_vector, float constant, int n){
    /*
    purpose : Multiplies entries of a vector by a constant on the GPU
    inputs  : d_vector - The pointer to the GPU values to be scaled
              constant - The scaling factor for the vector
              n        - The number of entries in the vector to be scaled
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) d_vector[tid] *= constant;
}

template <class T>
__global__ void expDiffGPU(T* d_input, T* d_output, T M, int n){
    /*
    purpose : Exponentiates each element of a vector after adding the constant M to it
    inputs  : d_input  - The input vector
              d_output - The output vector
              M        - The constant to be added to each element
              n        - The number of elements in the vector
    */
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) d_output[tid] = exp(d_input[tid] + M);
}

__global__ void sealsLikUpdate(float* d_weights_sum, float* d_ll, int n_per_GPU, int G, float M,
    int debug = 0){
    /*
    purpose : Updates the log likelihood value given a pre-calculated sum of exp(log(w) - M)
    inputs  : d_weights_sum - Pointer to the precalculated portion of the ll
              d_ll          - Pointer to the running total of the ll where the update should go
              n_per_gpu     - The number of particles being used on each GPU
              G             - The number of GPUs being used for the calculation
              M             - The number that was removed from the weights to help with
                              numerical errors
    */
     unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
     if (tid == 0){
        if (debug == 1) printf("%f\n", (log(d_weights_sum[0]) - log((float) n_per_GPU) + M));//debug
        d_ll[0] += (log(d_weights_sum[0]) - log((float) n_per_GPU) + M) / G;
    }
}

__host__ float bootstrapFilterMultiRegionForMCMC(int T, int** y, int* y0, float* theta_all,
    int n, int n_s, int tpb, float** h_weights, int*** d_old_states, int*** d_new_states,
    int** d_indices, float** d_weights, float** d_pop_totals, float** d_weights_sum,
    float*** d_theta, float** h_theta, curandState** d_curand_states, int n_theta_R, int G, int B,
    int R, int* extra_eval_ind, void (*thetaMap) (float*, float*, int, int), float* k0,
    int n_indep_ests, float (*dGammaChoice) (float, float), float M = -23.0f) {
    // purpose : performs a sequential importance sampling estimate of the log likelihood value
    // inputs  : T               - The number of time steps in the data
    //           y               - Array of pointers to the time series of observations for each 
    //                             region
    //           y0              - Array of initial population sizes for each region
    //           theta           - The parameter values (for the whole model, with all regions)
    //           n               - The number of particles to be used for ALL particle filters in
    //                             TOTAL. i.e. 10 000 particles with G == 2 will lead to 5000
    //                             particles being used by each GPU.
    //           tpb             - The number of GPU threads per block
    //           h_weights       - Pointer to the host pointer to the host-side particle weights *
    //           d_old_states    - Pointer to the host pointer to the gpu-side hidden states     *
    //           d_new_states    - Pointer to the host pointer to the gpu-side hidden states     *
    //           d_indices       - Pointer to the host pointer to the gpu-side sampling indices  *
    //           d_weights       - Pointer to the host pointer to the gpu-side particle weights  *
    //           d_pop_totals    - Pointer to the host pointer to the gpu-side population totals *
    //           d_theta         - Pointer to the host pointer to the gpu-side theta values      *
    //           h_theta         - Host-side theta used as a buffer to copy each region's theta  *
    //                             to the correct GPU
    //           d_curand_states - Pointer to the host pointer to the gpu-side rng seeds
    //           n_theta         - The number of parameters being fitted over the whole model
    //           n_theta_R       - The number of parameters being fitted to a single region (the
    //                             number of parameters required by the whole model, minus the 
    //                             number of parameters which are specific only to other regions
    //           G               - The number of GPUs between which to divide the workload. Defaults
    //                             to using device numbers 0, ..., G - 1 recognised by the system
    //           B               - The number of timesteps to be used with the Metropolis-Resampling 
    //                             of particles. If B < 0, parallel rejection sampling is used. 
    //                             This is unbiased, but considerably slower due to a poor 
    //                             algorithm design for parallelism
    //           R               - The number of regions in the data set, each with a time series
    //           extra_eval_ind  - Optionally, an additional lkelihood can be evaluated for a point
    //                             in the time series, this argument species which time-point
    //                             this applies to
    //           thetaMap        - A function which takes an input float vector of all model 
    //                             parameters, and fills a piece of memory with the parameters
    //                             for a specific region
    //           M               - The maximum expected value for a log weight. This is used to 
    //                             scale the log weights stored as floats so that the majority of 
    //                             numerical errors occur with the smallest weights
    //           dGammaChoice    - A __device__ function which evaluates the gamma llikelihood
    //                             of the independent estimate, for a single population total
    //
    // Note: This function will increment n_particles to be divisible by the number of GPUs used
    // Note: All parameters with a * in the above list only need to be mallocked, but don't need
    //       to contain any specific values. This function is designed to be used by MCMC functions
    //       and so the user should never have to manually produce these mallocked variables.
    n = (n % G == 0) ? n : n + G - (n % G);
    int n_per_gpu = n / G;
    int blocks = ceil(float(n_per_gpu) / tpb);

    float mll = 0;
    float** NLL, **d_buffer, *h_NLL;
    int indep_est_index = 0;

    checkCudaErrors(cudaMallocHost((void**)&NLL, sizeof(float*) * G));
    checkCudaErrors(cudaMallocHost((void**)&h_NLL, sizeof(float) * G));
    checkCudaErrors(cudaMallocHost((void**)&d_buffer, sizeof(float*) * G));

    // Change the first 0 to 1 to include the sex ratio in the forward projections and initial
    // state creation, otherwise sex at birth ratio will be assumed as 1/2:
    int include_omega = n_theta_R == 7 ? 0 : 0;

    for (int dev_id = 0; dev_id < G; dev_id++){
        checkCudaErrors(cudaSetDevice(dev_id));
        checkCudaErrors(cudaMalloc((void**)&NLL[dev_id], sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&d_buffer[dev_id], sizeof(float)));
        fillWithConst<<<1, 1>>>(NLL[dev_id], (float) 0.0, 1);
    }

    for (int region = 0; region < R; region++){
        // Get the theta mapping for each region:
        thetaMap(theta_all, h_theta[region], R, region);
    }

    // Set weights to 1 (so 0 for log weights) on GPUs, and copy over theta:
    for (int dev_id = 0; dev_id < G; dev_id++) {
        checkCudaErrors(cudaSetDevice(dev_id));
        fillWithConst<<<blocks, tpb>>>(d_weights[dev_id], 0.0f, n_per_gpu);
        for (int region = 0; region < R; region++){
            checkCudaErrors(cudaMemcpyAsync(d_theta[dev_id][region],
                h_theta[region], sizeof(float) * n_theta_R,
                cudaMemcpyHostToDevice));
        }//end for region
    }//end for dev_id

    // Produce the initial states at the first time step for the PF:
    for (int dev_id = 0; dev_id < G; dev_id++) {
        checkCudaErrors(cudaSetDevice(dev_id));
        checkCudaErrors(cudaDeviceSynchronize());
        for (int region = 0; region < R; region++){
            produceIntitialStatesGPU<<<blocks, tpb>>>(d_old_states[dev_id][region],
                d_theta[dev_id][region], y0[region], n_per_gpu, d_curand_states[dev_id],
                include_omega);
        }//end for region
    }//end for dev_id

    for (int i = 0; i < T; i++) {
        for (int dev_id = 0; dev_id < G; dev_id++) {

            checkCudaErrors(cudaSetDevice(dev_id));

            if (B <= 0) {
                // We sample the same indices from each region (in essence treating a 
                // a particle as being a combination of all regions at once):
                parallelMax(d_weights[dev_id], d_buffer[dev_id], n_per_gpu, 32, 64);
                rejectionResampleGPU<<<blocks, tpb>>>(d_weights[dev_id], d_indices[dev_id],
                    d_curand_states[dev_id], n_per_gpu, d_buffer[dev_id], 1);
            }//end if

            else {
                metropolisResampleGPU<<<blocks, tpb>>>(d_weights[dev_id], d_indices[dev_id],
                    d_curand_states[dev_id], n_per_gpu, B, 1);
            }//end else

            for (int region = 0; region < R; region++){
                sampleStatesDevAPI<<<blocks, tpb>>>(d_old_states[dev_id][region],
                    d_new_states[dev_id][region], d_indices[dev_id],
                    d_theta[dev_id][region], n_per_gpu,
                    d_curand_states[dev_id], include_omega);
                
                // This will overwrite the d_weights vector for region 0, and then 
                // append the likelihood contribution from each region: 
                if (region == 0)
                    fillWithConst<<<blocks, tpb>>>(d_weights[dev_id], 0.0f, n_per_gpu);

                // We put in a small check so that pieces of data that have been 
                // marked as NA do not contribute to the likelihood, even though
                // we still need the particle filter to project the states
                // 'through' this timestep:
                if (y[region][i] > 0){
                    dnormSealsGPU<<<blocks, tpb>>>(y[region][i], d_weights[dev_id],
                        d_new_states[dev_id][region], d_theta[dev_id][region],
                        n_per_gpu, 1);
                }//end if
            }//end for region
        }//end for dev_id

        for (int dev_id = 0; dev_id < G; dev_id++){
            checkCudaErrors(cudaSetDevice(dev_id));
            // If we're in a year which contains an independent population size
            // estimate, then calculate its density and increment the weights vector 
            // accordingly on each GPU:
            if (n_indep_ests > 0 && indep_est_index < n_indep_ests){
                if (i == extra_eval_ind[indep_est_index]){
                    // Get the total population size by summing rows for each particle:
                    for (int region = 0; region < R; region++){
                        colSumGPU<<<blocks, tpb>>>(d_new_states[dev_id][region],
                            d_pop_totals[dev_id], n_s, n_per_gpu, 1, region);
                    }//end for region

                    // Scale the population totals by omega:
                    scaleVectorGPU<<<blocks, tpb>>>(d_pop_totals[dev_id],
                        h_theta[0][n_theta_R - 1], n_per_gpu);

                    // Get the gamma PDF of the total population size with k0, k1 and k2 
                    // values taken from the supplementary materials of Thomas et al (2019):
                    dGammaGeneralGPU<<<blocks, tpb>>>(d_pop_totals[dev_id], d_weights[dev_id],
                        n_per_gpu, k0[indep_est_index], dGammaChoice);

                    // Increment to the next independent estimate (only for the final GPU):
                    if (dev_id == (G - 1)) indep_est_index++;

                }//end if i == extra_eval_ind[indep_est_index]
            }//end if n_indep_ests > 0
        
            // Try to do as much of the calculation for averaging the weights on the GPU as
            // possible, making a copy so that we can leave the weights as log weights to be 
            // resampled at the next timestep with fewer numerical errors:
            expDiffGPU<<<blocks, tpb>>>(d_weights[dev_id], d_pop_totals[dev_id], -M, n_per_gpu);
            sumReductionGPU(d_pop_totals[dev_id], d_buffer[dev_id], n_per_gpu, 32);
            sealsLikUpdate<<<1, 1>>>(d_buffer[dev_id], NLL[dev_id], n_per_gpu, G, M, 0);

            for (int region = 0; region < R; region++){
                std::swap(d_old_states[dev_id][region], d_new_states[dev_id][region]);
            }//end for region
        }//end for dev_id
    }//end for T
    
    // Copy over the final results
    for (int dev_id = 0; dev_id < G; dev_id++) {
        checkCudaErrors(cudaSetDevice(dev_id));
        checkCudaErrors(cudaMemcpyAsync(&h_NLL[dev_id], NLL[dev_id], sizeof(float),
            cudaMemcpyDeviceToHost));
    }//end for dev_id

    // Ensure memcpy done and sum results from all GPUs:
    for (int dev_id = 0; dev_id < G; dev_id++) {
        checkCudaErrors(cudaSetDevice(dev_id));
        checkCudaErrors(cudaDeviceSynchronize());
        mll += h_NLL[dev_id];
    }//end for dev_id

    for (int dev_id = 0; dev_id < G; dev_id++){
        checkCudaErrors(cudaSetDevice(dev_id));
        checkCudaErrors(cudaFree(d_buffer[dev_id]));
        checkCudaErrors(cudaFree(NLL[dev_id]));
    }

    checkCudaErrors(cudaFreeHost(d_buffer));
    checkCudaErrors(cudaFreeHost(h_NLL));
    checkCudaErrors(cudaFreeHost(NLL));

    //float* d_debug;
    return(mll);
}

__host__ float bootstrapFilterLooper(int T, int** y, int* y0, float* theta_all,
    int n, int n_s, int tpb, float** h_weights, int*** d_old_states, int*** d_new_states,
    int** d_indices, float** d_weights, float** d_pop_totals, float** d_weights_sum,
    float*** d_theta, float** h_theta, curandState** d_curand_states, int n_theta_R, int G, int B,
    int R, int* extra_eval_ind, void (*thetaMap) (float*, float*, int, int), float* k0,
    int n_indep_ests, float (*dGammaChoice) (float, float), float M = -23.0f, int L = 1){
        /*
        purpose : Uses the bootstrapFilterMultiRegionForMCMC function to perform multiple particle
                  filter evaluations to estimate the (float*) dGammaChoice(float, float)value of the log likelihood by taking their
                  mean
        inputs  : L - The number of calls to the particle filter to use to obtain the final 
                      estimate. All other parameters are identical to the bootstrap filter
                      that this function uses
        */
        float output = 0;

        for (unsigned int i = 0; i < L; i++){
            output += bootstrapFilterMultiRegionForMCMC(T, y, y0, theta_all, n, n_s, tpb, 
                h_weights, d_old_states, d_new_states, d_indices, d_weights, d_pop_totals,
                d_weights_sum, d_theta, h_theta, d_curand_states, n_theta_R, G, B, R,
                extra_eval_ind, thetaMap, k0, n_indep_ests, dGammaChoice, M) / L;
        }

        return output;
}

template <class FM>
void sealsMallocsForMCMC_DEV(float*** &d_theta, float** &d_weights, float** &h_weights,
    int*** &d_old_states, int*** &d_new_states, int** &d_indices, float** &d_pop_totals,
    float** &d_weights_sum, curandState** &d_curand_states, float** &h_theta_single_region,
    float* &h_old_theta, float* &h_theta_proposed, float* &accepted_count,
    float* &proposed_count, int G, int R, int n_theta_region, int n_theta, int n_per_gpu,
    float* &sigmas, FM sigma, size_t mat_size, int verbose, int blocks, int tpb){
    /*
    purpose : Given some variables, will free the correct memory for all allocations created for,
              the bootstrap multi-region multi-gpu particle filter. This also initialises the 
              Curand seeds on all devices.
    d_theta               - Pointer responsible for device-side region_specific parameters
    d_weights             - Pointer responsible for device-side weights
    h_weights             - Pointer reponsible for host-side weights
    d_old_states          - Pointer responsible for device-side states
    d_new_states          - Pointer responsible for device-side states
    d_indices             - Pointer responsible for device-side indices
    d_pop_totals          - Pointer responsible for device-side population total counts
    d_weights_sum         - Pointer responsible for device-side weights total
    d_curand_states       - Pointer responsible for device-side curand seeds
    h_theta_single_region - Pointer responsible for host-side all-region parameters
    h_old_theta           - Pointer responsible for host-side old all-region parameters
    h_theta_proposed      - Pointer responsible for host-side indices new all-region parameters
    accepted_count        - Pointer responsible for storing acceptance counts of proposals
    proposed_count        - Pointer responsible for storing proposal counts for each parameter 
    G                     - The number of GPUs the calculation is being run on
    R                     - The number of regions in the study / survey
    n_theta_region        - The number of parameters required to fit the model to a single region
    n_theta               - The number of parameters to fit the model to all regions
    n_per_gpu             - The number of particles to be used on each GPU
    sigmas                - Pointer responsible for proposal standard deviations 
    sigma                 - Pointer responsible for proposal covariance structure
    mat_size              - The size in bytes of the matrix which stores the state of all particles
    verbose               - An integer which indicates if more detailed printouts should be used
    prop1p                - An integer which indicates if parameter proposals are 1 dim at a time
    blocks                - The number of blocks with which to run kernels
    tpb                   - The number of threads per block
    */
    time_t seed; time(&seed);

    /// turn h_theta into an R x n_theta_region float pointer
    /// turn d_theta into a G x R x n_theta_region float pointer
    /// turn initial states into G x R x n_states x n_per_gpu pointer
    /// ditto with old states and new states

    // To allow a general N GPU approach:
    checkCudaErrors(cudaMallocHost((void**)&d_theta, sizeof(float**) * G));
    checkCudaErrors(cudaMallocHost((void**)&d_weights, sizeof(float*) * G));
    checkCudaErrors(cudaMallocHost((void**)&h_weights, sizeof(float*) * G));
    checkCudaErrors(cudaMallocHost((void**)&d_old_states, sizeof(int**) * G));
    checkCudaErrors(cudaMallocHost((void**)&d_new_states, sizeof(int**) * G));
    checkCudaErrors(cudaMallocHost((void**)&d_indices, sizeof(int*) * G));
    checkCudaErrors(cudaMallocHost((void**)&d_pop_totals, sizeof(float*) * G));
    checkCudaErrors(cudaMallocHost((void**)&d_weights_sum, sizeof(float*) * G));
    checkCudaErrors(cudaMallocHost((void**)&d_curand_states, sizeof(int*) * G));
    checkCudaErrors(cudaMallocHost((void**)&h_theta_single_region,
        sizeof(float*) * R));

    // Setup for device curand API:
    for (int dev_id = 0; dev_id < G; dev_id++) {
        cudaSetDevice(dev_id);
        checkCudaErrors(cudaMalloc((void**)&d_curand_states[dev_id],
            sizeof(curandState) * n_per_gpu));
        setCurandSeeds<<<blocks, tpb>>>(d_curand_states[dev_id], n_per_gpu, seed + dev_id);
    }

    // Mallocs for GPUs one by one:
    for (int dev_id = 0; dev_id < G; dev_id++) {
        checkCudaErrors(cudaSetDevice(dev_id));
        checkCudaErrors(cudaMalloc((void**)&d_weights_sum[dev_id], sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&d_weights[dev_id], sizeof(float) * n_per_gpu));
        checkCudaErrors(cudaMallocHost((void**)&h_weights[dev_id], sizeof(float) * n_per_gpu));
        checkCudaErrors(cudaMalloc((void**)&d_indices[dev_id], sizeof(int) * n_per_gpu));
        checkCudaErrors(cudaMalloc((void**)&d_pop_totals[dev_id], sizeof(float) * n_per_gpu));
        checkCudaErrors(cudaMallocHost((void**)&d_theta[dev_id], sizeof(float*) * R));
        checkCudaErrors(cudaMallocHost((void**)&d_new_states[dev_id], sizeof(int*) * R));
        checkCudaErrors(cudaMallocHost((void**)&d_old_states[dev_id], sizeof(int*) * R));
    
        for (int region = 0; region < R; region++){
            // For parameter storage:
            checkCudaErrors(cudaMalloc((void**)&d_theta[dev_id][region],
                sizeof(float) * n_theta_region));
            
            if (dev_id == 0){
                checkCudaErrors(cudaMallocHost((void**)
                    &h_theta_single_region[region],
                    sizeof(float) * n_theta_region));
            }//end if            

            // For state storage:
            checkCudaErrors(cudaMalloc((void**)&d_new_states[dev_id][region], mat_size));
            checkCudaErrors(cudaMalloc((void**)&d_old_states[dev_id][region], mat_size));
            
        }//end for region
    }//end for dev_id

    // Host side arrays for proposed parameters:
    checkCudaErrors(cudaMallocHost((void**)&h_old_theta, sizeof(float) * n_theta));
    checkCudaErrors(cudaMallocHost((void**)&h_theta_proposed, sizeof(float) * n_theta));

    if (verbose) {
        // Setup for detailed acceptance ratio printouts:
        checkCudaErrors(cudaMallocHost((void**)&accepted_count, sizeof(float) * n_theta));
        checkCudaErrors(cudaMallocHost((void**)&proposed_count, sizeof(float) * n_theta));
    }
}

__host__ float bootstrapFilterCloning(int T, int** y, int* y0, float* theta_all,
    int n, int n_s, int tpb, float** h_weights, int*** d_old_states, int*** d_new_states,
    int** d_indices, float** d_weights, float** d_pop_totals, float** d_weights_sum,
    float*** d_theta, float** h_theta, curandState** d_curand_states, int n_theta_R, int G, int B,
    int R, int* extra_eval_ind, void (*thetaMap) (float*, float*, int, int), float* k0,
    int n_indep_ests, float (*dGammaChoice) (float, float) = &dGammaFixed, float M = -23.0f, 
    int L = 1, int C = 1){
        /*
        purpose : Uses the bootstrapFilterMultiRegionForMCMC function to perform multiple particle
                  filter evaluations to estimate the value of the log likelihood by taking their
                  mean
        inputs  : L - The number of calls to the particle filter to use to obtain the final 
                      estimate. All other parameters are identical to the bootstrap filter
                      that this function uses
                  C - The number of times the data set has been cloned. i.e. with 4 regions and
        */
        float output = 0;
        for (unsigned int c = 0; c < C; c++){
            for (unsigned int i = 0; i < L; i++){
                output += bootstrapFilterMultiRegionForMCMC(T, y, y0, theta_all, n, n_s, tpb, 
                    h_weights, d_old_states, d_new_states, d_indices, d_weights, d_pop_totals,
                    d_weights_sum, d_theta, h_theta, d_curand_states, n_theta_R, G, B, R,
                    extra_eval_ind, thetaMap, k0, n_indep_ests, dGammaChoice, M) / L;
            }
        }

        return output;
}


template <class FM>
void sealsMemFreeForMCMC_DEV(float*** &d_theta, float** &d_weights, float** &h_weights,
    int*** &d_old_states, int*** &d_new_states, int** &d_indices, float** &d_pop_totals,
    float** &d_weights_sum, curandState** &d_curand_states, float** &h_theta_single_region,
    float* &h_old_theta, float* &h_theta_proposed, float* &accepted_count,
    float* &proposed_count, int G, int R, int n_theta_region, int n_theta, int n_per_gpu,
    float* &sigmas, FM sigma, size_t mat_size, int verbose, int blocks, int tpb){
    /*
    purpose : Given some variables, will free the correct memory for all allocations created for,
              the bootstrap multi-region multi-gpu particle filter. This also initialises the 
              Curand seeds on all devices.
    d_theta               - Pointer responsible for device-side region_specific parameters
    d_weights             - Pointer responsible for device-side weights
    h_weights             - Pointer reponsible for host-side weights
    d_old_states          - Pointer responsible for device-side states
    d_new_states          - Pointer responsible for device-side states
    d_indices             - Pointer responsible for device-side indices
    d_pop_totals          - Pointer responsible for device-side population total counts
    d_curand_states       - Pointer responsible for device-side curand seeds
    h_theta_single_region - Pointer responsible for host-side all-region parameters
    h_old_theta           - Pointer responsible for host-side old all-region parameters
    h_theta_proposed      - Pointer responsible for host-side indices new all-region parameters
    accepted_count        - Pointer responsible for storing acceptance counts of proposals
    proposed_count        - Pointer responsible for storing proposal counts for each parameter 
    G                     - The number of GPUs the calculation is being run on
    R                     - The number of regions in the study / survey
    n_theta_region        - The number of parameters required to fit the model to a single region
    n_theta               - The number of parameters to fit the model to all regions
    n_per_gpu             - The number of particles to be used on each GPU
    sigmas                - Pointer responsible for proposal standard deviations 
    sigma                 - Pointer responsible for proposal covariance structure
    mat_size              - The size in bytes of the matrix which stores the state of all particles
    verbose               - An integer which indicates if more detailed printouts should be used
    prop1p                - An integer which indicates if parameter proposals are 1 dim at a time
    blocks                - The number of blocks with which to run kernels
    tpb                   - The number of threads per block
    */
    if (verbose) {
        checkCudaErrors(cudaFreeHost(accepted_count));
        checkCudaErrors(cudaFreeHost(proposed_count));    
    }

    checkCudaErrors(cudaFreeHost(h_old_theta));
    checkCudaErrors(cudaFreeHost(h_theta_proposed));

    for (int dev_id = 0; dev_id < G; dev_id++){
        checkCudaErrors(cudaSetDevice(dev_id));
        
        for (int region = 0; region < R; region++){
            // Region specific vectors:
            if (dev_id == 0) checkCudaErrors(cudaFreeHost(h_theta_single_region[region]));
            checkCudaErrors(cudaFree(d_new_states[dev_id][region]));
            checkCudaErrors(cudaFree(d_old_states[dev_id][region]));
            checkCudaErrors(cudaFree(d_theta[dev_id][region]));
        }//end for region
        
        // Host vectors / containers:
        checkCudaErrors(cudaFreeHost(d_new_states[dev_id]));
        checkCudaErrors(cudaFreeHost(d_old_states[dev_id]));
        checkCudaErrors(cudaFreeHost(d_theta[dev_id]));
        checkCudaErrors(cudaFreeHost(h_weights[dev_id]));

        // GPU vectors:
        checkCudaErrors(cudaFree(d_pop_totals[dev_id]));
        checkCudaErrors(cudaFree(d_weights[dev_id]));
        checkCudaErrors(cudaFree(d_indices[dev_id]));
        checkCudaErrors(cudaFree(d_weights_sum[dev_id]));

        // Curand states:
        checkCudaErrors(cudaFree(d_curand_states[dev_id]));
    }//end for dev_id

    // Host-side containers for pointers of pointers:
    checkCudaErrors(cudaFreeHost(d_theta));
    checkCudaErrors(cudaFreeHost(d_weights));
    checkCudaErrors(cudaFreeHost(h_weights));
    checkCudaErrors(cudaFreeHost(d_old_states));
    checkCudaErrors(cudaFreeHost(d_new_states));
    checkCudaErrors(cudaFreeHost(d_indices));
    checkCudaErrors(cudaFreeHost(d_pop_totals));
    checkCudaErrors(cudaFreeHost(d_weights_sum));
    checkCudaErrors(cudaFreeHost(d_curand_states));
    checkCudaErrors(cudaFreeHost(h_theta_single_region));
}

void sealsMultiRegionMCMC(int n_samples, int n_particles, float* theta_0, FloatMatrix sigma,
    char* filename, char* dbg_filename, parameterSpec* prior_specifications, int n_states,
    int n_theta, int n_theta_region, int n_regions, int tpb, int T, int** y, int* y0,
    int create_csv_header, bool (*legalityChecks) (float*, int),
    void (*thetaMapper) (float*, float*, int, int), float* k0, int n_indep_ests, int G,
    int B, int verbose, int useR, int* EEI, float M, int L, int C = 1, 
    int ieCV = 27) {
    /*
    purpose : Performs a single chain of MCMC using particle filters
    Inputs  : n_samples            - The number of MCMC samples to obtain from the chain
              n_particles          - The number of TOTAL particles to be split across all GPUs
              theta_0              - The starting array for model parameters
              sigma                - The covariance matrix for the proposals 
              filename             - The filename where the MCMC results should be written
              dbg_filename         - The filename for writing debugging output
              prior_specifications - The prior specifications struct which specifies how to 
                                     evaluate prior density
              n_states             - The number of hidden states in the model
              n_theta              - The number of parameters being fitted
              n_theta_region       - The number of parameters when fitting a single region
              n_regions            - The number of (independent) regions
              tpb                  - The number of threads per block
              T                    - The number of time steps in the series
              y                    - array of array of observations
              y0                   - The initial observation values for each region for the
                                     generation of initial states
              create_csv_header    - If TRUE, writes a header onto the CSV file
              legalityChecks       - Pointer to a function which performs the parameter legality
                                     checks for instant rejection
              thetaMapper          - A function which can take as input a region number and the 
                                     array of ALL model parameter values to return those specific
                                     to the given region. See sealsThetaMap()
              k0                   - The shift parameter of the shifted gamma distribution for
                                     the population total
              G                    - The number of graphics cards which should be used to 
                                     distribute the workload. Uses card numbers 0, ..., G - 1 
                                     by default
              B                    - The number of iterations of Metropolis Sampling to be used
                                     for resampling weight indices at every time step for the 
                                     particle filter
              verbose              - If != 0, will print 20 updates throughout the MCMC, with 
                                     average acceptance ratios, which are given parameter by
                                     parameter for the one-by-one proposal.
              useR                 - If >= 1, the function will assume that the SD matrix is 
                                     to be considered as the precalculated matrix for
                                     multivariate normals. This is useful when starting a chain
                                     initially, when a sparse matrix may have numerical errors.
              EEI                  - Extra Eval Ind, indicates the timestep number (from 
                                     0, 1, ..., T - 1, T
              M                    - To aid numerical stability, a constant M is removed from 
                                     all of the weights before they are exponentiated. It helps
                                     when this value is as close as possible to the largest 
                                     log(weight), which in our case is usually on the order of
                                     -23.
              L                    - The number of times to run the filter to get an estimate, by 
                                     averaging through all results
              C                    - Allows the likelihood to be called multiple times, as if the
                                     data set contained C identical copies of all the regions. This
                                     can be used to run Data Cloning MCMC chains to determine the
                                     identifiability of parameters in certain conditions
              ieCV                 - The choice for the CV of the independent estimate. Available 
                                     options are 1% (1), 5% (5), and approx 27% (any other value), 
                                     where the final choice is the CV of the original independent
                                     estimate used in the 2019 Thomas et al paper.
    */

    // Hard-coded tuning values:
    int n_printouts = 20;
    int sample_min = n_printouts;

    // Set to something high to turn off the feature (which re-estimates the llikelihood with every
    // proposal, if proposals are rejected rejection_limit times):
    int rejection_limit = n_samples;

    // Create output file:
    FILE* file = fopen(filename, "w+");
    FILE* debug_file = fopen(dbg_filename, "w+");

    // Add named columns if required:
    if (create_csv_header) {
        for (int i = 0; i < n_theta; i++) fprintf(file, "theta%i,", i);
        fprintf(file, "log_target\n");
    }

    if (n_samples < sample_min) {
        printf("Warning: n_samples has been increased to the minimum of %i\n", sample_min);
        n_samples = sample_min;
    }

    // Select the chosen dGamma function:
    float (*h_dGamma)(float, float);
    
    switch(ieCV){
        // Note: the static pointers for these functions are definied near the top of this same
        //       file.
        case 1:
            cudaMemcpyFromSymbol(&h_dGamma, p_dGammaCV1, sizeof(float (*)(float, float)));
            break;
        case 5:
            cudaMemcpyFromSymbol(&h_dGamma, p_dGammaCV5, sizeof(float (*)(float, float)));
            break;
        default:
            cudaMemcpyFromSymbol(&h_dGamma, p_dGammaFixed, sizeof(float (*)(float, float)));
            break;
    }

    // Grid dimension setup:
    n_particles = (n_particles % G == 0) ? n_particles : n_particles + G - (n_particles % G);
    int n_per_gpu = n_particles / G;
    int blocks = ceil(float(n_per_gpu) / tpb);
    size_t mat_size = sizeof(int) * n_per_gpu * n_states;

    // Host and device-side pointers for particle filters:
    curandState** d_curand_states;
    int*** d_old_states, *** d_new_states, ** d_indices;
    float*** d_theta, ** d_weights, ** h_weights, ** h_theta_single_region,
        ** d_pop_totals, ** d_weights_sum;

    // Other required variables:
    int modified_par = 0, rejection_count = 0;
    float* h_old_theta, * h_theta_proposed, * accepted_count, * proposed_count,
        * sigmas, previous_ltarget = 0, current_ltarget = 0;
    FloatMatrix R(n_theta, n_theta);

    // All mallocs, GPU-side and host-side are performed by this function:
    sealsMallocsForMCMC_DEV(d_theta, d_weights, h_weights, d_old_states, d_new_states,
        d_indices, d_pop_totals, d_weights_sum, d_curand_states, h_theta_single_region,
        h_old_theta, h_theta_proposed, accepted_count, proposed_count, G, n_regions,
        n_theta_region, n_theta, n_per_gpu, sigmas, sigma, mat_size, verbose,
        blocks, tpb);

    // Copy over starting parameter values:
    checkCudaErrors(cudaMemcpy(h_old_theta, theta_0, sizeof(float) * n_theta,
        cudaMemcpyHostToHost));

    // Proposal covariance structure setup:
    R = (useR ? sigma : rmvnormPrep(sigma));

    // Evaluate density at initial proposed parameters:
    previous_ltarget += getPriorLogDensity(prior_specifications, h_old_theta, n_theta);
    previous_ltarget += bootstrapFilterCloning(T, y, y0, h_old_theta, n_per_gpu, n_states, tpb,
        h_weights, d_old_states, d_new_states, d_indices, d_weights, d_pop_totals, d_weights_sum,
        d_theta, h_theta_single_region, d_curand_states, n_theta_region, G, B, n_regions, EEI,
        thetaMapper, k0, n_indep_ests,h_dGamma, M, L, C);

    // Finally, it's MCMC time:
    for (int i = 0; i < n_samples; i++) {

        // Print an update to the user:
        if ((i % (n_samples / n_printouts) == (n_samples / n_printouts) - 1) && verbose) {
            printf("Finished iteration %i with mean acceptances of ", i + 1);
            printf("%f ", accepted_count[0] / (proposed_count[0] + 1));
            printf("\n");
        }

        // Propose new value for theta:
        modified_par = rmvnorm(h_theta_proposed, h_old_theta, R);
        if (verbose) proposed_count[modified_par] += 1;

        // Reject the proposal instantly if prior says NO:
        if (legalityChecks(h_theta_proposed, n_theta)) {
            // Get prior density of proposed point:
            // TODO: implement a rejection straight away if this is below some threshold?
            current_ltarget = getPriorLogDensity(prior_specifications, h_theta_proposed, n_theta);
            
            // Print unexpected prior error parameter sets to the debug_file:
            if (isnan(current_ltarget)){
                for (int q = 0; q < n_theta - 1; q++){
                    fprintf(debug_file, "%f,", h_theta_proposed[q]);
                }

                fprintf(debug_file, "%f\n", h_theta_proposed[n_theta - 1]);
            }

            // Only bother evaluating the likelihood if the priors return a finite answer:
            if (!isnan(current_ltarget)){
                // Get estimate of likelihood:
                current_ltarget += bootstrapFilterCloning(T, y, y0, h_theta_proposed, n_particles,
                    n_states, tpb, h_weights, d_old_states, d_new_states, d_indices, d_weights,
                    d_pop_totals, d_weights_sum, d_theta, h_theta_single_region, d_curand_states,
                    n_theta_region, G, B, n_regions, EEI, thetaMapper, k0, n_indep_ests,
                    h_dGamma, M, L, C);
                
                // If we've rejected enough times, re-estimate the llik of the old parameters, to
                // avoid getting stuck:
                if (rejection_count >= rejection_limit) {
                    previous_ltarget = getPriorLogDensity(prior_specifications,h_old_theta,n_theta);
                    previous_ltarget = bootstrapFilterCloning(T, y, y0,
                        h_old_theta, n_particles, n_states, tpb, h_weights, d_old_states,
                        d_new_states, d_indices, d_weights, d_pop_totals,  d_weights_sum, d_theta,
                        h_theta_single_region, d_curand_states, n_theta_region, G, B, n_regions,
                        EEI, thetaMapper, k0, n_indep_ests, h_dGamma, M, L, C);
                }//if
            
                // Determine accept / reject:
                if (runif(0, 1) < exp(current_ltarget - previous_ltarget)) {
                    rejection_count = 0;
                    if (verbose) accepted_count[modified_par] += 1;
                    previous_ltarget = current_ltarget;
                    checkCudaErrors(cudaMemcpy(h_old_theta, h_theta_proposed,
                        sizeof(float) * n_theta, cudaMemcpyHostToHost));
                }//if runif

                else rejection_count += 1; // rejected by metropolis ratio
            }//if !isnan
          
            else rejection_count += 1;     // rejected by isnan target
        }// if legalityChecks

        else rejection_count += 1;         // rejected by isnan prior
	
        // Write result to file:
        for (int j = 0; j < n_theta; j++) fprintf(file, "%f,", h_old_theta[j]);
        fprintf(file, "%f\n", previous_ltarget);

    }//for

    // Frees all allocated memory:
    sealsMemFreeForMCMC_DEV(d_theta, d_weights, h_weights, d_old_states, d_new_states,
        d_indices, d_pop_totals, d_weights_sum, d_curand_states, h_theta_single_region,
        h_old_theta, h_theta_proposed, accepted_count, proposed_count, G, n_regions,
        n_theta_region, n_theta, n_per_gpu, sigmas, sigma, mat_size, verbose,
        blocks, tpb);

    // reset devices :
    for (int dev_id = 0; dev_id < G; dev_id++) {
        checkCudaErrors(cudaSetDevice(dev_id));
        checkCudaErrors(cudaDeviceReset());
    }

    fclose(file);
    fclose(debug_file);
}

void debugRoutine(){
}