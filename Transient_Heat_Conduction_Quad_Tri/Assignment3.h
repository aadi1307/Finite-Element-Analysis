#ifndef ASSIGNMENT3_H
#define ASSIGNMENT3_H

#include <vector>
#include <Eigen/Dense>

struct Mesh {
    int nEl;
    int nNode;
    std::vector<std::vector<int>> connArr;
    Eigen::MatrixXd xGlo;
    };

// Function to create mesh
Mesh createmesh(int nElx, int nEly, double W, double H);

// Function declaration Dirichlet
void setDirichletConditions(std::vector<std::vector<double>>& dirich, int nElx, int nEly);

// Function declaration Initial condition
void setInitialCondition(Eigen::VectorXd& u0, const Mesh& mesh);

// Structure for local matrices
struct LocalMatrices {
    Eigen::MatrixXd stiffness;
    Eigen::MatrixXd mass;
    Eigen::VectorXd force;
};

// Function to compute local matrices
LocalMatrices computeLocalMatrices(const Eigen::MatrixXd& elementNodes, 
                                   const Eigen::MatrixXd& kappa, 
                                   double rho_c, 
                                   double distributedLoad, 
                                   int wq);

// Function to create global stiffness matrix
Eigen::MatrixXd createKGlobal(const Mesh& mesh, const Eigen::MatrixXd& kappa, int wq, double rho_c);

// Function to create global mass matrix
Eigen::MatrixXd createMGlobal(const Mesh& mesh, const Eigen::MatrixXd& kappa, double rho_c, int wq, double distributedLoad);

// Function to create global force matrix
Eigen::VectorXd createFGlobal(const Mesh& mesh, const Eigen::MatrixXd& kappa, 
                              double rho_c, int wq, double distributedLoad);

// Function to create reduced matrix 
Eigen::MatrixXd createReducedMatrix(const Eigen::MatrixXd& globalMatrix, const Eigen::VectorXi& nonDirichNodes);

// Function to create reduced force vector
Eigen::VectorXd createReducedForceVector(const Eigen::MatrixXd& kGlobal, 
                                         const Eigen::VectorXd& fGlobal, 
                                         const Eigen::VectorXi& nonDirichNodes, 
                                         const Eigen::MatrixXd& dirich);

/*// Function for time-stepping
Eigen::VectorXd createTimeStepping(const Eigen::MatrixXd& mDirichlet, 
                                   const Eigen::MatrixXd& kDirichlet, 
                                   const Eigen::VectorXd& fDirichlet, 
                                   const Eigen::VectorXd& dInitial, 
                                   double alpha, 
                                   int tSteps, 
                                   double tDelta);

void shapeFunc(double zhi, double eta, Eigen::VectorXd& N, Eigen::VectorXd& dNdz, Eigen::VectorXd& dNde);

Eigen::MatrixXd quadrature(int n); */

#endif
