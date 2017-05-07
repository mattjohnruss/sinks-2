#include <solver.h>

#include <cmath>
#include <iostream>

template <class FloatT>
Solver<FloatT>::Solver(const FloatT &Pe_,
               const FloatT &Da_,
               const unsigned &N_,
               std::vector<FloatT> &xi_)
    :
        Pe(Pe_),
        Da(Da_),
        N(N_),
        xi(xi_),
        A(2*(N+1),2*(N+1)),
        f(2*(N+1)),
        u(2*(N+1))
{
}

template <class FloatT>
Solver<FloatT>::~Solver()
{
}

template <class FloatT>
void Solver<FloatT>::set_xi(std::vector<FloatT> &xi_)
{
    xi = xi_;
}

template <class FloatT>
void Solver<FloatT>::construct_system()
{
    using std::exp;

    std::vector<T> triplet_list;

    // Set epsilon
    const FloatT ep = static_cast<FloatT>(1)/static_cast<FloatT>(N+1);

    // 1 for inlet BC, 6*N for bulk, 2 for outlet BC
    const unsigned n_entries = 1 + 6*N + 2;

    // Reserve elements in the triplet list
    triplet_list.reserve(n_entries);

    // Inlet BC
    triplet_list.push_back( T(0, N+1, static_cast<FloatT>(1)) );
    f[0] = ep/Pe;
    
    // Bulk
    for(unsigned j = 1; j <= N; ++j)
    {
        triplet_list.push_back( T(j, j,       static_cast<FloatT>(1) - Da/Pe) );
        triplet_list.push_back( T(j, j-1,     -exp(Pe*(xi[j] - xi[j-1]))) );
        triplet_list.push_back( T(j, N+1 + j, -Da/Pe) );
        f[j] = static_cast<FloatT>(0);

        triplet_list.push_back( T(N+1 + j-1, j,         Da/Pe) );
        triplet_list.push_back( T(N+1 + j-1, N+1 + j,   static_cast<FloatT>(1) + Da/Pe) );
        triplet_list.push_back( T(N+1 + j-1, N+1 + j-1, static_cast<FloatT>(-1)) );
        f[N+1 + j] = static_cast<FloatT>(0);
    }

    // Outlet BC
    triplet_list.push_back( T(N+1 + N, N,       exp(Pe*(static_cast<FloatT>(N+1) - xi[N]))) );
    triplet_list.push_back( T(N+1 + N, N+1 + N, static_cast<FloatT>(1)) );
    f[N+1 + N] = static_cast<FloatT>(0);

    // Construct the matrix out of the triplets
    A.setFromTriplets(triplet_list.begin(), triplet_list.end());
    A.makeCompressed();

    linear_solver.compute(A);

    if(linear_solver.info() != Eigen::Success)
    {
        std::cerr << "Compute failed!\n";
    }
}

template <class FloatT>
void Solver<FloatT>::solve()
{
    u = linear_solver.solve(f);

    if(linear_solver.info() != Eigen::Success)
    {
        std::cerr << "Solve failed!\n";
    }
}

template <class FloatT>
void Solver<FloatT>::output(const unsigned &n_output, std::ostream &outstream) const
{
    FloatT x = 0;
    FloatT dx = static_cast<FloatT>(N+1)/static_cast<FloatT>(n_output-1);

    // loop over output points and call our solution helper
    for(unsigned p = 0; p < n_output; ++p)
    {
        x = p*dx;
        outstream << x << ' ' << solution_helper(x) << std::endl;
    }
}

template <class FloatT>
void Solver<FloatT>::output(const unsigned &n_output, std::vector<FloatT> &outvec, const unsigned offset) const
{
    FloatT x = 0;
    FloatT dx = static_cast<FloatT>(N+1)/static_cast<FloatT>(n_output-1);

    // loop over output points and call our solution helper
    for(unsigned p = 0; p < n_output; ++p)
    {
        x = p*dx;
        outvec[offset*n_output + p] = solution_helper(x);
    }
}

template <class FloatT>
const FloatT Solver<FloatT>::solution_helper(const FloatT &x) const
{
    using std::exp;

    // find the cell that contains output point x
    unsigned cell = 0;

    for(unsigned j = 0; j < N+1; ++j)
    {
        if(x >= xi[j] && x <= xi[j+1])
        {
            cell = j;
            break;
        }
    }

    // return the solution at x using the cell we found
    return u[cell]*exp(Pe*(x - xi[cell])) + u[N+1 + cell];
}

template class Solver<double>;
//template class Solver<long double>;
