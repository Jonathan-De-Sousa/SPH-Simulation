/*
 * High-Perfomance Computing - Coursework Assignment
 *
 * Class to solve a 2D smoothed particle hydrodynamic (SPH)
 * formulation of the Navier-Stokes equations.
 *
 * Written by Jonathan De Sousa
 * Date: 24/03/2021
 */

#ifndef SPH_H
#define SPH_H

class SPH
{
public:
    // Constructors
    SPH() = delete; // do not allow without input parameters
    SPH(double* p_x,
        double* p_y,
        const int p_N,
        const int p_loc_N,
        const double p_dt,
        const double p_T,
        const double p_rank, 
        const double p_size);    
    SPH(const SPH&) = delete; // no need for copy constructor
    SPH(SPH&&) = delete;      // no need for move constructor
    ~SPH();                   

    // Functions to compute relevant quantities
    void solve_q(double inv_h, int offset);  
    void solve_rho(double valrho);
    void solve_p(double valp);
    void solve_F(double valFp, double valFv, int offset);   
    void solve_SPH(); 
    void checkBC();   

    // Getter functions for particle & system properties
    double get_k(){return k;}
    double get_rho_0(){return rho_0;}
    double get_mu(){return mu;}
    double get_g(){return g;}
    double get_h(){return h;}
    double get_e(){return e;}
    double* get_x(){return x;}
    double* get_y(){return y;}
    double* get_EK(){return EK;}
    double* get_EP(){return EP;}
    double* get_ET(){return ET;}

    // Setter functions for particle properties
    // [All but set_h() not used in coursework
    //  but makes class flexible for simulations
    //  with other fluids.]
    void set_k(const double& p_k){k = p_k;}
    void set_rho_0(const double& prho_0){rho_0 = prho_0;}
    void set_mu(const double& p_mu){mu = p_mu;}
    void set_g(const double& p_g){g = p_g;}
    void set_h(const double& p_h){h = p_h;}
    void set_e(const double& p_e){e = p_e;}

    // Data held privately
private:
    double k = 2000.0;     // gas constant
    double rho_0 = 1000.0; // resting density
    double mu = 1.0;       // viscosity
    double g = 9.81;       // gravitational acceleration
    double h = 0.01;       // radius of influence
    double e = 0.5;        // coefficient of restitution
    double m = 1.0;        // (initial) mass of each particle

    // Pointers/variables prefixed with loc_ represent local 
    // segmented arrays/local quantities, otherwise they 
    // refer to full-particle system.
    int N = 0;       // number of particles
    int loc_N = 0;   //  
    int loc_N0 = 0;  // number of particles (= int(N/size))
    double dt = 0.0; // time-step
    double T = 0.0;  // total integration time
    int N_t = 0;     // number of time-steps
    int loc_N_t = 0; // 
    
    int rank = 0;   // processor rank         
    int size = 0;   // communicator size
 
    double* x = nullptr;      // x-coordinates
    double* loc_x = nullptr;  //
    double* y = nullptr;      // y-coordinates
    double* loc_y = nullptr;  //
    
    double* vx = nullptr;     // x-direction velocity
    double* loc_vx = nullptr; // 
    double* vy = nullptr;     // y-direction velocity
    double* loc_vy = nullptr; //

    double* loc_rx = nullptr; // x-distance separation
    double* loc_ry = nullptr; // y-distance separation
    double* loc_q = nullptr;  // normalised absolute separation

    double* rho = nullptr;     // density
    double* loc_rho = nullptr; //
    double* p = nullptr;       // pressure
    double* loc_p = nullptr;   // 

    double* loc_Fpx = nullptr; // x-direction pressure force
    double* loc_Fpy = nullptr; // y-direction pressure force

    double* loc_Fvx = nullptr; // x-direction viscous force
    double* loc_Fvy = nullptr; // y-direction viscous force

    double* loc_Fg = nullptr; // gravitational force

    double* loc_ax = nullptr; // x-direction acceleration
    double* loc_ay = nullptr; // y-direction acceleration
    
    double* EK = nullptr; // kinetic energy
    double* EP = nullptr; // potential energy
    double* ET = nullptr; // total energy 
};

#endif