// Reads test cases on stdin, prints rhs() from the original gotran C++.
#include <cstdio>
#include "PBM.h"

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 1;
    double s[5], p[25], v[5], t;
    for (int c = 0; c < n; c++) {
        for (int i = 0; i < 5; i++) {
            if (scanf("%lf", &s[i]) != 1) return 1;
        }
        for (int i = 0; i < 25; i++) {
            if (scanf("%lf", &p[i]) != 1) return 1;
        }
        if (scanf("%lf", &t) != 1) return 1;
        rhs(s, t, p, v);
        for (int i = 0; i < 5; i++) printf("%.17g%c", v[i], i == 4 ? '\n' : ' ');
    }
    return 0;
}
