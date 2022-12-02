/*
 * High-Perfomance Computing - Coursework Assignment
 *
 * SPH class functions' implementations to solve the
 * 2D smoothed particle hydrodynamic (SPH)
 * formulation of the Navier-Stokes equations.
 *
 * Written by Jonathan De Sousa
 * Date: 24/03/2021
 */

#include <cblas.h>
#include <cmath>
#include <cstring>
using namespace std;

#include "SPH.h"
#include <mpi.h>

// Assigns values and required dynamic memory to member variables & pointers
SPH::SPH(double* p_x,
    double* p_y,
    const int p_N,
    const int p_loc_N,
    const double p_dt,
    const double p_T,
    const double p_rank,
    const double p_size)
    : N(p_N)
    , loc_N(p_loc_N)
    , dt(p_dt)
    , T(p_T)
    , rank(p_rank)
    , size(p_size)
{
    N_t = int(p_T / p_dt);
    loc_N0 = int(p_N / size);

    x = p_x;
    y = p_y;

    loc_x = new double[p_loc_N];
    loc_y = new double[p_loc_N];

    vx = new double[p_N];
    vy = new double[p_N];
    loc_vx = new double[p_loc_N];
    loc_vy = new double[p_loc_N];

    // Initialise zero intial velocity
    memset(vx, 0, p_N * sizeof(double));
    memset(vy, 0, p_N * sizeof(double));
    memset(loc_vx, 0, p_loc_N * sizeof(double));
    memset(loc_vy, 0, p_loc_N * sizeof(double));

    loc_rx = new double[p_loc_N * p_N];
    loc_ry = new double[p_loc_N * p_N];
    loc_q = new double[p_loc_N * p_N];

    rho = new double[p_N];
    p = new double[p_N];
    loc_rho = new double[p_loc_N];
    loc_p = new double[p_loc_N];

    loc_Fpx = new double[p_loc_N];
    loc_Fpy = new double[p_loc_N];

    loc_Fvx = new double[p_loc_N];
    loc_Fvy = new double[p_loc_N];

    loc_Fg = new double[p_loc_N];

    loc_ax = new double[p_loc_N];
    loc_ay = new double[p_loc_N];

    EK = new double[N_t + 1]; // allocate memory for all time-steps
    EP = new double[N_t + 1]; // plus initial energy at "zeroth timestep"
    ET = new double[N_t + 1]; //
}

// De-allocates dynamic memory
SPH::~SPH()
{
    delete[] loc_x;
    delete[] loc_y;

    delete[] vx;
    delete[] vy;
    delete[] loc_vx;
    delete[] loc_vy;

    delete[] loc_rx;
    delete[] loc_ry;
    delete[] loc_q;

    delete[] rho;
    delete[] p;
    delete[] loc_rho;
    delete[] loc_p;

    delete[] loc_Fpx;
    delete[] loc_Fpy;

    delete[] loc_Fvx;
    delete[] loc_Fvy;

    delete[] loc_Fg;

    delete[] loc_ax;
    delete[] loc_ay;

    delete[] EK;
    delete[] EP;
    delete[] ET;
}

// Computes normalised distance between particles
void SPH::solve_q(double inv_h, int offset)
{
    double xi;
    double yi;

    for(int i = 0; i < loc_N; i++) {
        xi = x[i + offset]; // prevents continuous...
        yi = y[i + offset]; // ...calling in j loop
        for(int j = 0; j < N; j++) {
            loc_rx[i * N + j] = xi - x[j];
            loc_ry[i * N + j] = yi - y[j];
            loc_q[i * N + j] =
                sqrt(loc_rx[i * N + j] * loc_rx[i * N + j] + loc_ry[i * N + j] * loc_ry[i * N + j]) * inv_h;
        }
    }
}

// Computes fluid density associated with each particle
void SPH::solve_rho(double valrho)
{
    memset(loc_rho, 0, loc_N * sizeof(double));
    for(int i = 0; i < loc_N; i++) {
        for(int j = 0; j < N; j++) {
            if(loc_q[i * N + j] < 1) {
                loc_rho[i] += valrho * pow((1 - loc_q[i * N + j] * loc_q[i * N + j]), 3);
            }
        }
    }
}

// Computes the fluid pressure associated with each particle
void SPH::solve_p(double valp)
{
    for(int i = 0; i < loc_N; i++) {
        loc_p[i] = k * loc_rho[i] + valp;
    }
}

// Computes the pressure, viscous and gravitational forces, Fp*, Fv* and Fg
// respectively, and acceleration, a*.
// * = x or y, for the two directional components of quantities
void SPH::solve_F(double valFp, double valFv, int offset)
{
    memset(loc_Fpx, 0, loc_N * sizeof(double));
    memset(loc_Fpy, 0, loc_N * sizeof(double));

    memset(loc_Fvx, 0, loc_N * sizeof(double));
    memset(loc_Fvy, 0, loc_N * sizeof(double));

    double inv_rhoj;
    double vxi;
    double vyi;
    double pi;

    for(int i = 0; i < loc_N; i++) {
        vxi = vx[i + offset]; // prevents continuous...
        vyi = vy[i + offset]; // ...calling in j loop.
        pi = p[i + offset];   //
        for(int j = 0; j < N; j++) {
            if(i != j && loc_q[i * N + j] < 1) {
                inv_rhoj = 1.0 / rho[j];
                loc_Fpx[i] += valFp * ((pi + p[j]) * inv_rhoj) * loc_rx[i * N + j] * pow((1 - loc_q[i * N + j]), 2) /
                    loc_q[i * N + j];
                loc_Fpy[i] += valFp * ((pi + p[j]) * inv_rhoj) * loc_ry[i * N + j] * pow((1 - loc_q[i * N + j]), 2) /
                    loc_q[i * N + j];
                loc_Fvx[i] += valFv * (loc_q[i * N + j] - 1.0) * (vxi - vx[j]) * inv_rhoj;
                loc_Fvy[i] += valFv * (loc_q[i * N + j] - 1.0) * (vyi - vy[j]) * inv_rhoj;
            }
        }
        loc_Fg[i] = -loc_rho[i] * g;
        loc_ax[i] = (loc_Fpx[i] + loc_Fvx[i]) / loc_rho[i];
        loc_ay[i] = (loc_Fpy[i] + loc_Fvy[i] + loc_Fg[i]) / loc_rho[i];
    }
}

// Enforces the solid wall boundary conditions of the unit box
void SPH::checkBC()
{
    for(int i = 0; i < loc_N; i++) {
        // Right wall
        if(loc_x[i] <= h) {
            loc_x[i] = h;
            loc_vx[i] = -e * loc_vx[i];
        } else if(loc_x[i] >= 1.0 - h) {
            loc_x[i] = 1.0 - h;
            loc_vx[i] = -e * loc_vx[i];
        }
        if(loc_y[i] <= h) {
            loc_y[i] = h;
            loc_vy[i] = -e * loc_vy[i];
        } else if(loc_y[i] >= 1.0 - h) {
            loc_y[i] = 1.0 - h;
            loc_vy[i] = -e * loc_vy[i];
        }
    }
}

// Solves the dynamic SPH problem over the integration time T
void SPH::solve_SPH()
{
    // Compute arguments of MPI functions
    int* recvsendcnts = new int[size]; // number of elements recv'ed/sent per process
    int* displs = new int[size];       // displacement from recv/send buffer to send/take data

    MPI_Allgather(&loc_N, 1, MPI_INT, recvsendcnts, 1, MPI_INT, MPI_COMM_WORLD); // gather loc_N's

    for(int i = 0; i < size; i++) {
        displs[i] = i * loc_N0;
    }

    // Pre-computations of constants for the above member functions
    int offset = loc_N0 * rank;
    double inv_h = 1.0 / h;
    double valp = -k * rho_0;
    double valrho = m * 4.0 / (M_PI * h * h);

    solve_q(inv_h, offset);
    solve_rho(valrho);

    // Scale mass so that density is equal to reference density
    MPI_Allgatherv(loc_rho, loc_N, MPI_DOUBLE, rho, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    m = N * rho_0 / cblas_dasum(N, rho, 1);

    // Pre-computations of partial EK & EP expressions
    double valEK = 0.5 * m;
    double valEP = m * g;

    // Initial energies ("zeroth time-step")
    // EK[0] calculated for generality, i.e. non-zero initial velocity.
    EK[0] = valEK * (cblas_ddot(N, vx, 1, vx, 1) + cblas_ddot(N, vy, 1, vy, 1));
    EP[0] = valEP * cblas_dasum(N, y, 1);
    ET[0] = EP[0] + EK[0];

    if(N_t > 0) {

        // Pre-compute mass-dependent constants in above member functions
        valrho = m * 4.0 / (M_PI * h * h);
        double valFp = m * 30.0 / (2.0 * M_PI * h * h * h);
        double valFv = mu * m * 40.0 / (M_PI * pow(h, 4));

        solve_rho(valrho); // re-calculate density with new mass
        solve_p(valp);

        // Gather loc_rho, loc_p components into full system arrays
        MPI_Allgatherv(loc_rho, loc_N, MPI_DOUBLE, rho, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(loc_p, loc_N, MPI_DOUBLE, p, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        solve_F(valFp, valFv, offset);

        // Split up coordinates between processes
        MPI_Scatterv(x, recvsendcnts, displs, MPI_DOUBLE, loc_x, loc_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(y, recvsendcnts, displs, MPI_DOUBLE, loc_y, loc_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // First time-step
        cblas_daxpy(loc_N, dt / 2.0, loc_ax, 1, loc_vx, 1); // vx = vx + ax * dt/2.0
        cblas_daxpy(loc_N, dt / 2.0, loc_ay, 1, loc_vy, 1); // vy = vy + ay * dt/2.0

        cblas_daxpy(loc_N, dt, loc_vx, 1, loc_x, 1); // x = x + vx * dt
        cblas_daxpy(loc_N, dt, loc_vy, 1, loc_y, 1); // y = y + vy * dt

        checkBC(); // enforce boundary conditions

        // Re-combine loc_x, loc_y into full system x, y
        MPI_Allgatherv(loc_x, loc_N, MPI_DOUBLE, x, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(loc_y, loc_N, MPI_DOUBLE, y, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // Re-combine loc_vx, loc_vy into full system vx, vy
        MPI_Allgatherv(loc_vx, loc_N, MPI_DOUBLE, vx, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(loc_vy, loc_N, MPI_DOUBLE, vy, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // First time-step energies
        EK[1] = valEK * (cblas_ddot(N, vx, 1, vx, 1) + cblas_ddot(N, vy, 1, vy, 1));
        EP[1] = valEP * cblas_dasum(N, y, 1);
        ET[1] = EP[1] + EK[1];

        // Advance for remaining timesteps
        for(int i = 1; i < N_t; i++) {
            solve_q(inv_h, offset);
            solve_rho(valrho);
            solve_p(valp);

            // Gather loc_rho, loc_p components into full system arrays
            MPI_Allgatherv(loc_rho, loc_N, MPI_DOUBLE, rho, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgatherv(loc_p, loc_N, MPI_DOUBLE, p, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

            solve_F(valFp, valFv, offset);

            cblas_daxpy(loc_N, dt, loc_ax, 1, loc_vx, 1); // advance velocities
            cblas_daxpy(loc_N, dt, loc_ay, 1, loc_vy, 1); //

            cblas_daxpy(loc_N, dt, loc_vx, 1, loc_x, 1); // advance positions
            cblas_daxpy(loc_N, dt, loc_vy, 1, loc_y, 1); //

            checkBC(); // enforce boundary conditions

            // Re-combine loc_x, loc_y into full system x, y
            MPI_Allgatherv(loc_x, loc_N, MPI_DOUBLE, x, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgatherv(loc_y, loc_N, MPI_DOUBLE, y, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

            // Re-combine loc_vx, loc_vy into full system vx, vy
            MPI_Allgatherv(loc_vx, loc_N, MPI_DOUBLE, vx, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgatherv(loc_vy, loc_N, MPI_DOUBLE, vy, recvsendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

            EK[i + 1] = valEK * (cblas_ddot(N, vx, 1, vx, 1) + cblas_ddot(N, vy, 1, vy, 1));
            EP[i + 1] = valEP * cblas_dasum(N, y, 1);
            ET[i + 1] = EP[i + 1] + EK[i + 1];
        }
    }
    delete[] recvsendcnts;
    delete[] displs;
}
