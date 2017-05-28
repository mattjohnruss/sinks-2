#ifndef SOLVER_H_
#define SOLVER_H_

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <Eigen/UmfPackSupport>
#include <Eigen/SuperLUSupport>

#include <vector>
#include <ostream>

template <class FloatT>
class Solver
{
    typedef Eigen::Triplet<FloatT> T;
    typedef Eigen::Matrix<FloatT, Eigen::Dynamic, 1> VectorXft;

public:
    Solver(const FloatT &Pe_,
           const FloatT &Da_,
           const unsigned &N_,
           std::vector<FloatT> &xi);

    virtual ~Solver();

    void set_xi(std::vector<FloatT> &xi_);

    void construct_system();
    void solve();

    void output(const unsigned &n_output, std::ostream &outstream) const;
    void output(const unsigned &n_output, std::vector<FloatT> &outvec, const unsigned &offset) const;

    void output_corrections(const unsigned &n_output, std::ostream &outstream) const;
    void output_corrections(const unsigned &n_output, std::vector<FloatT> &outvec) const;

    void output_corrections_classical(const unsigned &n_output, std::ostream &outstream) const;
    void output_corrections_classical(const unsigned &n_output, std::vector<FloatT> &outvec) const;

private:
    const FloatT Pe;
    const FloatT Da;
    const unsigned N;

    std::vector<FloatT> &xi;

    Eigen::SparseMatrix<FloatT> A;

    VectorXft f;
    VectorXft u;

    Eigen::SparseLU<Eigen::SparseMatrix<FloatT> > linear_solver;
    //Eigen::SuperLU<Eigen::SparseMatrix<FloatT> > linear_solver;
    //Eigen::UmfPackLU<Eigen::SparseMatrix<FloatT> > linear_solver;

    const FloatT solution_helper(const FloatT &x) const;

    const FloatT C_h(const FloatT &x) const;
    const FloatT C_h_diff(const FloatT &x) const;
    const FloatT C_a_min_fk(const FloatT &x) const;
    const FloatT C_aa(const FloatT &x) const;
};

#endif
