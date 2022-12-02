/*
 * High-Perfomance Computing - Coursework Assignment
 *
 * Solves a 2D smoothed particle hydrodynamic (SPH)
 * formulation of the Navier-Stokes equations.
 *
 * Results of particle position and the system's
 * kinetic, potential and total energy are written
 * to files output.txt and energy.txt respectively.
 *
 * Written by Jonathan De Sousa
 * Date: 24/03/2021
 */

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace std;

#include "SPH.h"
#include <boost/program_options.hpp>
#include <mpi.h>

// Boost program options alias
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    // Specify the options available to user
    po::options_description opts("Available options: ");
    opts.add_options()
    ("ic-dam-break", "Dam-break initial condition")
    ("ic-block-drop", "Block-drop initial condition")
    ("ic-droplet", "Droplet initial condition [set as default I.C.]")
    ("ic-one-particle", "One particle validation case initial condition")
    ("ic-two-particles", "Two particles validation case initial condition")
    ("ic-three-particles", "Three particles validation case initial condition")
    ("ic-four-particles", "Four particles validation case initial condition")
    ("dt", po::value<double>()->default_value(1e-4), "Time step to use.")
    ("T", po::value<double>()->default_value(2), "Total integration time")
    ("h", po::value<double>()->default_value(0.01), "Radius of influence of each particle.")
    ("help", "Print help message.");

    // Generate map (vm) containing options and values
    // specified by the user.
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    // Print program usage if user enters "--help" option.
    if(vm.count("help")) {
        cout << "Simulates a 2D smoothed particle hydrodynamic (SPH) formulation "
             << "of the Navier-Stokes equations." << endl;
        cout << opts << endl;
        return 0;
    }

    // Extract the input values to relevant parameters
    // ic-droplet is set as default so its value is not extracted
    const bool ic_DmBk = vm.count("ic-dam-break");
    const bool ic_BkDp = vm.count("ic-block-drop");
    const bool ic_P1 = vm.count("ic-one-particle");
    const bool ic_P2 = vm.count("ic-two-particles");
    const bool ic_P3 = vm.count("ic-three-particles");
    const bool ic_P4 = vm.count("ic-four-particles");
    const double dt = vm["dt"].as<double>();
    const double T = vm["T"].as<double>();
    const double h = vm["h"].as<double>();

    // Declare pointers and variables
    int N = 0;           // number of particles
    int N_t = T / dt;    // number of time-steps
    double* x = nullptr; // x-coordinates
    double* y = nullptr; // y-coordinates

    int loc_N = 0; // process's local number of particles
    int rank = 0;  // process rank
    int size = 0;  // communicator size

    // Initialise MPI
    int retval = MPI_Init(&argc, &argv);

    // Check no error in initialisation
    if(retval != MPI_SUCCESS) {
        cout << "Error in initialising MPI." << endl;
        return -1;
    }

    // Find rank and size on each process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set particle coordinates based on initial condition.
    // Done on root only then broadcast so that all processes
    // have initial test case conditions with exactly the same
    // small amplitude noise.
    if(rank == 0) {
        int iRow = 0; // number of rows in initial particle grid
        int iCol = 0; // number of rows in initial particle grid

        // Seed random number generator
        srand(time(0));

        // Pre-computation of part of noise expression for test cases.
        // Noise added is no greater than 20% of h
        double noise = (1.0 / RAND_MAX) * h * 0.2;

        // Set particle coordinates based on initial conditions
        // Validation cases
        if(ic_P1 == 1) {
            N = 1;
            x = new double[N];
            y = new double[N];

            x[0] = 0.5;
            y[0] = 0.5;
        } else if(ic_P2 == 1) {
            N = 2;
            x = new double[N];
            y = new double[N];

            x[0] = 0.5;
            x[1] = 0.5;

            y[0] = 0.5;
            y[1] = h;
        } else if(ic_P3 == 1) {
            N = 3;
            x = new double[N];
            y = new double[N];

            x[0] = 0.5;
            x[1] = 0.495;
            x[2] = 0.505;

            y[0] = 0.5;
            y[1] = h;
            y[2] = h;

        } else if(ic_P4 == 1) {
            N = 4;
            x = new double[N];
            y = new double[N];

            x[0] = 0.505;
            x[1] = 0.515;
            x[2] = 0.51;
            x[3] = 0.5;

            y[0] = 0.5;
            y[1] = 0.5;
            y[2] = 0.45;
            y[3] = 0.45;
        }
        // Test cases
        else if(ic_DmBk == 1 || ic_BkDp == 1) {
            double x0, x1; // ends of rectangular domain
            double y0, y1; //
            if(ic_DmBk == 1) {
                x1 = 0.2;
                y1 = 0.2;
                x0 = h; // set to h instead of 0...
                y0 = h; // ...due to boundary condition
            } else {
                x1 = 0.3;
                x0 = 0.1;
                y1 = 0.6;
                y0 = 0.3;
            }
            iRow = int((y1 - y0) / h) + 1;
            iCol = int((x1 - x0) / h) + 1;
            N = iRow * iCol;

            x = new double[N];
            y = new double[N];

            for(int i = 0; i < iCol; i++) {
                for(int j = 0; j < iRow; j++) {
                    x[i * iRow + j] = x0 + i * h + (double)rand() * noise * pow(-1, rand());
                    y[i * iRow + j] = y0 + j * h + (double)rand() * noise * pow(-1, rand());
                }
            }
        } else { // Droplet is default case for un-specified initial condition
            // Create square grid encapsulating circle centre [0.5, 0.7], radius 0.1
            // Then only include points within circle to be set into x and y arrays.

            double x_temp; // temporary coordinates of square grid
            double y_temp; //

            iRow = int((0.8 - 0.6) / h) + 1;
            iCol = int((0.6 - 0.4) / h) + 1;
            N = (iRow * iCol); // N number of elements in square grid

            // Ratio (Area_circle : Area_square) = pi/4 = 0.785, so can roughly set N for droplet
            // N_circle <= N_square * pi/4 always (N_circle = N_square * pi/4 as N-->infty)
            N = int(N * 0.79);

            x = new double[N];
            y = new double[N];
            N = 0; // number of particles in circle

            for(int i = 0; i < iCol; i++) {
                for(int j = 0; j < iRow; j++) {
                    x_temp = 0.4 + i * h;
                    y_temp = 0.6 + j * h;
                    // Add all points inside circle to x & y arrays
                    if((pow((x_temp - 0.5), 2) + pow((y_temp - 0.7), 2)) <= pow(0.1, 2)) {
                        x[N] = x_temp + (double)rand() * noise * pow(-1, rand());
                        y[N] = y_temp + (double)rand() * noise * pow(-1, rand());
                        N++;
                    }
                }
            }
        }

        // Ensure number of processes is not larger than number of particles
        if(size > N) {
            cout << "Number of particles, N = " << N << endl;
            cout << "Number of processes cannot be greater than N." << endl;
            MPI_Finalize();
            return -1;
        }
    }

    // Broadcast N from root process
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0) {
        x = new double[N];
        y = new double[N];
    }

    // Broadcast x,y coordinates from root process
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local number of particles,
    // accounting for N not divisible by size
    if(N % size == 0) {
        loc_N = int(N / size);
    } else {
        loc_N = int(N / (size));
        if(rank == size - 1) {
            loc_N += N % size;
        }
    }

    // Instantiate SPH object
    SPH sys(x, y, N, loc_N, dt, T, rank, size);
    sys.set_h(h);

    // Solve SPH problem
    sys.solve_SPH();

    // Obtain results and write to files on root process only
    if(rank == 0) {
        x = sys.get_x();
        y = sys.get_y();

        double* EK = sys.get_EK();
        double* EP = sys.get_EP();
        double* ET = sys.get_ET();

        // Write final spacial coordinates to .txt file
        ofstream fOut1("output.txt");
        fOut1 << setw(10) << "x" << setw(10) << "y" << endl;

        fOut1.precision(6);
        for(int i = 0; i < N; i++) {
            fOut1 << setw(10) << x[i] 
                  << setw(10) << y[i] << endl;
        }
        fOut1.close(); // close file

        // Write energies at each time-step to .txt file
        ofstream fOut2("energy.txt");

        fOut2 << setw(10) << "Time" 
              << setw(20) << "Kinetic Energy" 
              << setw(20) << "Potential Energy" 
              << setw(15) << "Total Energy" << endl;

        fOut2.precision(6);
        for(int i = 0; i < N_t + 1; i++) {
            fOut2 << setw(10) << i * dt 
                  << setw(20) << EK[i] 
                  << setw(20) << EP[i] 
                  << setw(15) << ET[i] << endl;
        }
        fOut2.close(); // close file
    }

    // De-allocate dynamic memory
    delete[] x;
    delete[] y;

    // Terminate parallel environment
    MPI_Finalize();

    // Return success
    return 0;
}