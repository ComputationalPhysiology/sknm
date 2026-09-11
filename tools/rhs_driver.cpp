// Reads test cases on stdin, prints rhs() from the original gotran C++.
#include <cstdio>
#include "base_model_IM.h"

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 1;
    double s[25], p[83], v[25], t;
    for (int c = 0; c < n; c++) {
        for (int i = 0; i < 25; i++) scanf("%lf", &s[i]);
        for (int i = 0; i < 83; i++) scanf("%lf", &p[i]);
        scanf("%lf", &t);
        rhs(s, t, p, v);
        for (int i = 0; i < 25; i++) printf("%.17g%c", v[i], i == 24 ? '\n' : ' ');
    }
    return 0;
}
