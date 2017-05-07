#include <solver.h>

#include <iostream>
#include <fstream>
#include <random>

int main(int argc, char **argv)
{
    if(argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " N Pe Da points/cell runs\n";
        exit(1);
    }

    // params
    const unsigned N = std::atoi(argv[1]);
    const double Pe = std::atof(argv[2]);
    const double Da = std::atof(argv[3]);
    const unsigned n_output = std::atoi(argv[4])*(N+1) + 1;
    unsigned runs = std::atoi(argv[5]);

    // probabiltiy distribution
    std::random_device rd;
    std::normal_distribution<double> norm_dist(0,0.1);

    // store all runs in a contiguous vector: must be careful to offset indices
    std::vector<double> cs(runs*n_output);

    // loop over the runs in parallel
    #pragma omp parallel for
    for(unsigned r = 0; r < runs; ++r)
    {
        // sink locations
        std::vector<double> xi(N+2,0);

        // 0 and N+1 elements always the same
        xi[0] = 0.0;

        // loop over the actual sinks (from 1 to N)
        for(unsigned j = 1; j <= N; ++j)
        {
            if(runs == 1)
            {
                xi[j] = j;
            }
            else
            {
                xi[j] = j + norm_dist(rd);
            }
        }

        xi[N+1] = static_cast<double>(N+1);

        // solve
        Solver<double> solver(Pe,Da,N,xi);
        solver.construct_system();
        solver.solve();
        solver.output(n_output, cs, r);
    }

    Eigen::IOFormat plain_fmt(8, Eigen::DontAlignCols);

    // map our output to an Eigen matrix - allows use of matrix methods with
    // ~zero overhead!
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > cs_mat(&cs[0], runs, n_output);

    if(runs == 1)
    {
        // output the whole dataset
        std::ofstream cs_outfile("cs.dat");
        cs_outfile << cs_mat.format(plain_fmt) << std::endl;
    }
    else // runs > 1
    {
        // calculate the covariance using Eigen tricks and output
        Eigen::MatrixXd centred = cs_mat.rowwise() - cs_mat.colwise().mean();
        Eigen::MatrixXd cov = (centred.adjoint()*centred)/static_cast<double>(cs_mat.rows()-1);

        std::ofstream cov_outfile("cov.dat");
        cov_outfile << cov.format(plain_fmt) << std::endl;

        std::ofstream var_outfile("var.dat");
        var_outfile << cov.diagonal().format(plain_fmt) << std::endl;
    }
}
