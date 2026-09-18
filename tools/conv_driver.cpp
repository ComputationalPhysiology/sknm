// Integrates the original gotran C++ base_model_IM for 300 ms and prints V_m every 1 ms.
// Usage: conv_driver <dt> <scheme>   scheme: 0 = forward_rush_larsen, 1 = forward_explicit_euler
#include <cstdio>
#include <cstdlib>
#include "base_model_IM.h"

int main(int argc, char** argv) {
    double dt = atof(argv[1]);
    int euler = atoi(argv[2]);
    double T = 300.0;
    double s[25], p[83];
    init_state_values(s);
    init_parameters_values(p);
    p[parameter_index("stim_amplitude")] = 20.0;
    long N = (long)(T / dt + 0.5);
    long every = (long)(1.0 / dt + 0.5);
    int Vi = state_index("V_m");
    double t = 0;
    for (long n = 0; n < N; n++) {
        if (euler) forward_explicit_euler(s, t, dt, p);
        else forward_rush_larsen(s, t, dt, p);
        t += dt;
        if ((n + 1) % every == 0) printf("%.17g\n", s[Vi]);
    }
    return 0;
}
