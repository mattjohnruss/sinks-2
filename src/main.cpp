#include <solver.h>

#include <iostream>
#include <fstream>
#include <random>

int main()
{
    // params
    const unsigned N = 99;
    const double ep = 1.0/static_cast<double>(N);
    const double Pe = ep;
    const double Da = ep*ep;
    const unsigned n_output = 100*(N+1) + 1;

    // probabiltiy distribution
    std::random_device rd;
    std::normal_distribution<double> norm_dist(0,0.01);

    // sink locations
    std::vector<double> xi(N+2,0);

    // 0 and N+1 elements always the same
    xi[0] = 0.0;
    xi[N+1] = static_cast<double>(N+1);

    Solver<double> solver(Pe,Da,N,xi);

    unsigned runs = 1000;

    std::vector<std::vector<double> > cs(runs, std::vector<double>(n_output));

    // loop over the runs
    for(unsigned r = 0; r < runs; ++r)
    {
        // loop over the sinks (from 1 to N since 0, N+1 are already fixed)
        for(unsigned j = 1; j <= N; ++j)
        {
            xi[j] = j + norm_dist(rd);
        }

        solver.construct_system();
        solver.solve();
        solver.output(n_output, cs[r]);
    }

    //std::ofstream outfile("output.dat");
    //solver.output(n_output, outfile);
}
