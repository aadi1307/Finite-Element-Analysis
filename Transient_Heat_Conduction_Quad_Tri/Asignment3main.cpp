#include "assignment3.h"
#include <vector>
#include <iostream>

int main() {
    // Define material properties and mesh dimensions
    const double W = 1.0, H = 1.0;
    const double kappa_bar = 385.0;
    const double rho_c = 3.8151e6;
    const int nElx = 20; 
    const int nEly = 20;
    const double distributedLoad = 1000.0; // Example distributed loa
    const int wq = 4; // Number of quadrature points
    const double alpha = 0.5; // Damping factor - placeholder
    const int tSteps = 100;   // Number of time steps - placeholder
    const double tDelta = 0.01; // Time step size - placeholder
    double zhi = 0.5;
    double eta = -0.5;
    Eigen::VectorXd N, dNdz, dNde;
     int n = 3;


    // Creating a vector to store Dirichlet conditions
    std::vector<std::vector<double>> dirich;

    /*Eigen::MatrixXd kappa = Eigen::MatrixXd::Identity(2, 2) * kappa_bar;
    //Eigen::MatrixXd globalMatrix = Eigen::MatrixXd::Random(10, 10); // Replace with actual global matrix
    //Eigen::VectorXd fGlobal = Eigen::VectorXd::Random(10); // Replace with actual global force vector */
    Eigen::MatrixXd globalMatrix = Eigen::MatrixXd::Random(10, 10); // Placeholder for actual global matrix
    //Eigen::VectorXi nonDirichNodes = Eigen::VectorXi::LinSpaced(5, 1, 9); // Placeholder for actual non-Dirichlet nodes
    Eigen::VectorXi nonDirichNodes(5); // Example non-Dirichlet nodes
    nonDirichNodes << 0, 2, 4, 6, 8;
    //Eigen::MatrixXd dirich(2, 2); // Example Dirichlet conditions (node index, value)
    dirich[0] = {1, 1.0};  // Node index 1 with value 1.0
    dirich[1] = {3, -1.0}; // Node index 3 with value -1.0
    /*Eigen::MatrixXd mDirichlet = Eigen::MatrixXd::Random(10, 10); // Placeholder
    Eigen::MatrixXd kDirichlet = Eigen::MatrixXd::Random(10, 10); // Placeholder
    Eigen::VectorXd fDirichlet = Eigen::VectorXd::Random(10);     // Placeholder
    Eigen::VectorXd dInitial = Eigen::VectorXd::Random(10);       // Placeholder */

    
    // Create a Mesh object
    Mesh mesh;
    mesh.nNode = 4; // Example number of nodes
    mesh.xGlo = Eigen::MatrixXd(mesh.nNode, 2); // coordinates

    // Initialize coordinates for demonstration (normally set based on your problem)
    for (int i = 0; i < mesh.nNode; ++i) {
        mesh.xGlo(i, 0) = static_cast<double>(i) / (mesh.nNode - 1); // Example x-coordinates
        mesh.xGlo(i, 1) = 0; // Example y-coordinates
    }
    
    /*// Create an Eigen::VectorXd to store initial conditions
    Eigen::VectorXd u0;
    // Call the function to set initial conditions
    setInitialCondition(u0, mesh);
    // Optional: Print u0 for verification
    std::cout << "Initial Conditions:\n" << u0 << std::endl; */

    Eigen::MatrixXd elementNodes(4, 2); // Initialize with your element node coordinates
    Eigen::MatrixXd kappa = Eigen::MatrixXd::Identity(2, 2); // Material property matrix
    //double distributedLoad = 1000; // Example distributed load
    //int wq = 4; // Number of quadrature points

    // Compute local matrices
    LocalMatrices localMatrices = computeLocalMatrices(elementNodes, kappa, rho_c, distributedLoad, wq);

    // Example: Output the computed matrices for verification
    std::cout << "Local Stiffness Matrix:\n" << localMatrices.stiffness << "\n\n";
    std::cout << "Local Mass Matrix:\n" << localMatrices.mass << "\n\n";
    std::cout << "Local Force Vector:\n" << localMatrices.force << "\n"; 



    // Meshing
    //const int nElx = 20, nEly = 20;
    //Mesh mesh = createmesh(nElx, nEly, W, H);
    std::cout << "Number of Elements: " << mesh.nEl << std::endl;
    std::cout << "Number of Nodes: " << mesh.nNode << std::endl;

    // Dirichlet BC
    // Call the function to set Dirichlet conditions
    //std::vector<std::vector<double>> dirich((nEly + 1) * 2, std::vector<double>(2));
    setDirichletConditions(dirich, nElx, nEly);


    // Initial condition
    Eigen::VectorXd u0(mesh.nNode);
    setInitialCondition(u0, mesh);

    // Assemble global stiffness vector
    Eigen::MatrixXd kGlobal = createKGlobal(mesh, kappa, wq, rho_c);
    // Optional: Output the global stiffness matrix for verification
    std::cout << "Global Stiffness Matrix:\n" << kGlobal << "\n";

    // Assemble global Mass Global vector
    Eigen::MatrixXd dirichMatrix(dirich.size(), dirich[0].size());
    for (int i = 0; i < dirich.size(); ++i) {
        for (int j = 0; j < dirich[0].size(); ++j) {
            dirichMatrix(i, j) = dirich[i][j];
        }
    }

    Eigen::MatrixXd mGlobal = createMGlobal(mesh, kappa, wq, rho_c, distributedLoad);
    // Optional: Output the global mass matrix for verification
    std::cout << "Global Mass Matrix:\n" << mGlobal << "\n";

    // Assemble global force vector
    Eigen::VectorXd fGlobal = createFGlobal(mesh, kappa, wq, rho_c, distributedLoad);
    // Optional: Output the global force vector for verification
    std::cout << "Global Force Vector:\n" << fGlobal << "\n";


    // Reduced 
    Eigen::MatrixXd reducedMatrix = createReducedMatrix(globalMatrix, nonDirichNodes);
    
    //reducedForceVector
    Eigen::VectorXd reducedForceVector = createReducedForceVector(kGlobal, fGlobal, nonDirichNodes, dirichMatrix);
    //Eigen::VectorXd reducedForce = createReducedForceVector(kGlobal, fGlobal, nonDirichNodes, dirich);

    // Optional: Print the reduced force vector
    std::cout << "Reduced Force Vector:\n" << reducedForceVector << std::endl;
    
    
    /*// Call the function to create the reduced matrix
    Eigen::MatrixXd reducedMatrix = createReducedMatrix(globalMatrix, nonDirichNodes);
    // Optional: Print the reduced matrix
    std::cout << "Reduced Matrix:\n" << reducedMatrix << std::endl;


     // Call the function to perform time stepping
    Eigen::VectorXd dFinal = createTimeStepping(mDirichlet, kDirichlet, fDirichlet, dInitial, alpha, tSteps, tDelta);
    // Optional: Output the final displacement vector
    std::cout << "Final Displacement Vector:\n" << dFinal << std::endl;

    
    // Call the shape function
    shapeFunc(zhi, eta, N, dNdz, dNde);
    // Print the results
    std::cout << "N: " << N.transpose() << std::endl;
    std::cout << "dNdz: " << dNdz.transpose() << std::endl;
    std::cout << "dNde: " << dNde.transpose() << std::endl;

    // Call the quadrature function
    Eigen::MatrixXd A = quadrature(n);
    std::cout << "Quadrature Matrix for n = " << n << ":\n" << A << std::endl; */ 

    return 0;
}

