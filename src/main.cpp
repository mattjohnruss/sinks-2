#include <solver.h>

#include <iostream>
#include <fstream>
#include <random>

int main(int argc, char **argv)
{
    if(argc < 6 || argc > 8)
    {
        std::cerr << "Usage: " << argv[0] << " N Pe Da points/cell runs [ sink_dist [ std_dev ] ]" << std::endl
                  << "sink_dist: n - normally perturbed; u - uniformly distributed"
                  << std::endl;
        exit(1);
    }

    // params
    const unsigned N = std::atoi(argv[1]);
    const double Pe = std::atof(argv[2]);
    const double Da = std::atof(argv[3]);
    const unsigned n_output = std::atoi(argv[4])*(N+1) + 1;
    const unsigned runs = std::atoi(argv[5]);
    double std_dev = 0.01;
    char sink_dist = 'n';

    // see if sinks are randomly distributed
    if(argc >= 7)
    {
        // take the first char out of the arg for the distribution
        sink_dist = argv[6][0];

        // only bother setting std_dev if sinks are random
        if(argc == 8)
        {
            std_dev = std::atof(argv[7]);
        }
    }

    // probabiltiy distributions
    std::random_device rd;
    std::normal_distribution<double> norm_dist(0,std_dev);
    std::uniform_real_distribution<double> uniform_dist(0,static_cast<double>(N+1));

    // store all runs in a contiguous vector: must be careful to offset indices
    std::vector<double> cs(runs*n_output);

    // only need one set of corrections regardless of runs
    std::vector<double> corrections(n_output);

    // loop over the runs in parallel
    #pragma omp parallel for
    for(unsigned r = 0; r < runs; ++r)
    {
        // sink locations
        std::vector<double> xi(N+2,0);

        // 0 and N+1 elements always the same
        xi[0] = 0.0;

        // loop over the actual sinks (from 1 to N)
        if(runs == 1)
        {
            for(unsigned j = 1; j <= N; ++j)
            {
                xi[j] = j;
            }
        }
        else // runs > 1
        {
            if(sink_dist == 'n')
            {
                for(unsigned j = 1; j <= N; ++j)
                {
                    xi[j] = j + norm_dist(rd);
                }
            }
            if(sink_dist == 'u')
            {
                for(unsigned j = 1; j <= N; ++j)
                {
                    xi[j] = uniform_dist(rd);
                }

                // must sort the sinks if uniformly distributed, excluding the first and last entries
                std::sort(xi.begin()+1,xi.end()-1);
            }
        }

        xi[N+1] = static_cast<double>(N+1);

        // solve
        Solver<double> solver(Pe,Da,N,xi);
        solver.construct_system();
        solver.solve();
        solver.output(n_output, cs, r);
    }

    // make a dummy solver here to just output the corrections
    std::vector<double> xi(N+2,0);
    Solver<double> solver(Pe,Da,N,xi);
    solver.output_corrections(n_output, corrections);

    // sensible output format for Eigen matrices
    const Eigen::IOFormat plain_fmt(8, Eigen::DontAlignCols);

    // map our output to Eigen matrices - allows use of matrix methods with
    // ~zero overhead!
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > cs_mat(&cs[0], runs, n_output);
    Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> > corrections_mat(&corrections[0], n_output);

    std::ofstream outfile;

    bool output_all = false;

    if(runs == 1 || output_all)
    {
        // output the whole dataset
        outfile.open("cs.dat");
        outfile << cs_mat.format(plain_fmt);
        outfile.close();

        // output the residual for each run
        outfile.open("cs_residual.dat");
        outfile << (cs_mat.rowwise() - corrections_mat).format(plain_fmt);
        outfile.close();
    }
    if(runs > 1)
    {
        // calculate the mean and covariance using Eigen tricks
        Eigen::MatrixXd mean = cs_mat.colwise().mean();
        Eigen::MatrixXd centred = cs_mat.rowwise() - cs_mat.colwise().mean();
        Eigen::MatrixXd cov = (centred.adjoint()*centred)/static_cast<double>(cs_mat.rows()-1);

        // output the mean
        outfile.open("mean.dat");
        outfile << mean.format(plain_fmt);
        outfile.close();

        // output the mean residual
        outfile.open("mean_residual.dat");
        outfile << (mean - corrections_mat).format(plain_fmt);
        outfile.close();

        // output the covariance
        outfile.open("cov.dat");
        outfile << cov.format(plain_fmt);
        outfile.close();

        // output the variance
        outfile.open("var.dat");
        outfile << cov.diagonal().transpose().format(plain_fmt);
        outfile.close();

        // output the transverse covariance
        outfile.open("tcov.dat");
        outfile << cov.rowwise().reverse().diagonal().transpose().format(plain_fmt);
        outfile.close();
    }
}
