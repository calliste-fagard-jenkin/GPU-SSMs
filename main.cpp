// Include C++ header files.
#include <iostream>
#include <chrono>

// Include local CUDA header files.
#include "include/obtainDeviceInfo.cuh"
#include "include/stolenR.h"
#include "include/Eigen/Eigen"
#include "include/sealCode.cuh"

int main() {
    ///////////////////////////////////// MCMC SETTINGS ////////////////////////////////////////////
    int P = 16,
    n_particles = 1 << P,        // The number of particles to use (total, not per GPU)
    n_samples = 1e5,             // The number of MCMC samples to obtain during the run
    n_states = 7,                // Number of age groups where a population count is saved
    n_theta = 10,                // Number of total parameters
    n_theta_single_region = 7,   // Number of parameters needed to evaluate a single region
    n_regions = 4,               // The number of independent survey regions 
    verbose = 1,                 // >= 1 for extra output during MCMC
    oneby1 = 0,                  // < 1 to ensure all parameters are updated with each proposal
    tpb = 512,                   // Threads per block of execution on the GPU(s)
    useR = 1,                    // >= 1 to use a pre-calculation proposal covariance structures
    EEI = 23,                    // Extra Eval Index - The timestep of the independent estimate 
    no_EEI = 100,                // A very high EEI, to exclude the independent est
    B = 0,                       // The number of metropolis iterations per resample. 0 = rejection.
    T = 26,                      // The number of timesteps in the timeseries
    G = 2,                       // The number of GPUs to use for the calculation
    create_csv_header = 1,       // >= 1 To create column names in our output file
    n_chains = 2,                // The number of MCMC chains to run
    L = 3,                       // The number of filters averaged together to estimate the llik
    ieCV = 27;                   // The coef of var of the independent estimate. Could be 1 or 5 too
    float M = -23.5;             // Helps in minimising numerical errors, application specific
    
    // Output file names:
    char debug_filename[] = "debug_file.csv";
    char mcmc_basename[] = "paper_chain";
    char mcmc_endname[] = ".csv";
    char* mcmc_name;

    mcmc_name =  (char*) malloc(strlen(mcmc_basename) + strlen(mcmc_endname) + 100);
    
    //////////////////////////////////// HARDCODED DATA-SETS ///////////////////////////////////////

    // Pup production estimates for all years of the survey:
    int IH_paper[] = {1190, 1711, 2002, 1960, 1956, 2032, 2411, 2816, 2923, 2719, 3050, 3117, 3076,
    3087, 2787, 3223, 3032, 3096, 3386, 3385, 3427, 3470, 3118, 3317, -1, 3108};
    
    int OH_paper[] = {8165, 8455, 8777, 8689, 9275, 9801, 10617, 12215, 11915, 12054, 12713, 13176,
    11946, 12434, 11759, 13472, 12427, 11248, 12741, 12319, 12397, 11719, 11342, 12279, 11887,
    11831};
    
    int OR_paper[] = {5199, 5796, 6389, 5948, 6773, 6982, 8653, 9854, 11034, 11851, 12670, 14531,
    14395, 16625, 15720, 16546, 18196, 17952, 18652, 19123, 18126, 19332, 19184, 17813, 18548,
    18582};
    
    int NS_paper[] = {1711, 1834, 1867, 1474, 1922, 2278, 2375, 2436, 2710, 2652, 2757, 2938, 3698,
    3989, 3380, 4303, 4134, 4520, 4870, 5015, 5232, 5484, 5771, 6501, 7360, 8119};
    
    int* all_observations[4] = { IH_paper, OH_paper, OR_paper, NS_paper };

    // Pup production estimates for the year before the survey:
    int y0_array[] = { 1332, 7594, 4741, 1325 };
    
    ////////////////////////////////// MAKING THE PRIORS OBJECTS ///////////////////////////////////
    parameterSpec adult_survival = { &scaledBetaLogDensity, 1.6, 1.2, 0.17, 0.8 };
    parameterSpec pup_survival = { &scaledBetaLogDensity, 2.87, 1.78, 1, 0 };
    parameterSpec fecundity = { &scaledBetaLogDensity, 2, 1.5, 0.4, 0.6 };
    parameterSpec psi = { &scaledGammaLogDensity, 2.1, 66.67, 1, 0 };
    parameterSpec chi_IH = { &scaledGammaLogDensity, 4, 1250, 1, 0 };
    parameterSpec chi_OH = { &scaledGammaLogDensity, 4, 3750, 1, 0 };
    parameterSpec chi_OR = { &scaledGammaLogDensity, 4, 10000, 1, 0 };
    parameterSpec chi_NS = { &scaledGammaLogDensity, 4, 5000, 1, 0 };
    parameterSpec rho = { &scaledGammaLogDensity, 4, 2.5, 1, 0 };
    parameterSpec omega = {&scaledGammaLogDensity, 28.08, 3.70e-3, 1, 1.6};
    parameterSpec* prior_specs_pointer = (parameterSpec*)malloc(sizeof(parameterSpec) * n_theta);
    prior_specs_pointer[0] = pup_survival;
    prior_specs_pointer[1] = adult_survival;
    prior_specs_pointer[2] = fecundity;
    prior_specs_pointer[3] = rho;
    prior_specs_pointer[4] = psi;
    prior_specs_pointer[5] = chi_IH;
    prior_specs_pointer[6] = chi_OH;
    prior_specs_pointer[7] = chi_OR;
    prior_specs_pointer[8] = chi_NS;
    prior_specs_pointer[9] = omega;

    // Shift parameters for total pop size distribution:
    float k0 = 59167.84161;

    ////////////////////// STARTING PARAMETER VALUES, SETTINGS, AND RESULTS FILE ///////////////////
    float theta[] = {0.465894, 0.955375, 0.862145, 5.27171, 154.050995, 3096.870117, 11746.099609,
        18226.699219, 7896.390137, 1.69039};

    // Hard-coded proposal VCM, found using an iterative process:
    #include "covarStructures/MainProposals.cpp"
    
    //////////////////////////////////////////// MCMC //////////////////////////////////////////////
    
    std::cout << "Starting a " << n_samples << " sample MCMC on " << G << " GPU(s) with " <<
    n_particles << " total particles and " << L << " filters per call" << std::endl;
    
    for (unsigned int i = 0; i < n_chains; i++){

        // Create the correct output filename:
        
        // Add P, L, and B info:
        char c[10];
        strcpy(mcmc_name, mcmc_basename);
        sprintf(c, "%d", P);
        strcat(mcmc_name, "_P");
        strcat(mcmc_name, c);
        sprintf(c, "%d", L);
        strcat(mcmc_name, "_L");
        strcat(mcmc_name, c);
        sprintf(c, "%d", B);
        strcat(mcmc_name, "_B");
        strcat(mcmc_name, c);
        
        // Add chain number:
        sprintf(c, "%d", i);
        strcat(mcmc_name, "_");
        strcat(mcmc_name, c);
        
        // Add file extension:
        strcat(mcmc_name, mcmc_endname);
        
        // Launch the GPU MCMC routine:
        sealsMultiRegionMCMC(
            n_samples,
            n_particles, 
            theta,
            r0_paper,

            mcmc_name,
            debug_filename,
            prior_specs_pointer,
            n_states,
            n_theta,
            n_theta_single_region,
            n_regions,
            tpb,
            T,
            all_observations,
            y0_array,
            create_csv_header,
            &sealsLegalityChecks,
            &sealsThetaMapGeneral,
            &k0,
            1, // # of indep estimates
            G,
            B,
            verbose,
            oneby1,
            useR,
            &EEI,
            M,
            L,
            1, // Data-cloning parameter, set to 1 for regular analysis;
            ieCV
        );
    }

    // cleanup:
    free(prior_specs_pointer);
    free(mcmc_name);
    resetAllDevices();
    return 0;
}
