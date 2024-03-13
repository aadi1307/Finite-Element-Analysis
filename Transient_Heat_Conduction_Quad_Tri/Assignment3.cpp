#include "assignment3.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

Mesh createmesh(int nElx, int nEly, double W, double H) {
    Mesh mesh;
    // Calculate the number of nodes
    int nNodesX = nElx + 1;
    int nNodesY = nEly + 1;
    mesh.nNode = nNodesX * nNodesY;
    mesh.nEl = nElx * nEly;

    // Initialize global coordinates matrix
    mesh.xGlo = Eigen::MatrixXd(mesh.nNode, 2);

    // Spacing between nodes
    double dx = W / nElx;
    double dy = H / nEly;

    // Filling the global coordinates
    for (int i = 0; i < nNodesY; ++i) {
        for (int j = 0; j < nNodesX; ++j) {
            int nodeIndex = i * nNodesX + j;
            mesh.xGlo(nodeIndex, 0) = j * dx; // x coordinate
            mesh.xGlo(nodeIndex, 1) = i * dy; // y coordinate
        }
    }

    // Initialize the connectivity array
    mesh.connArr.resize(mesh.nEl);
    for (int elY = 0; elY < nEly; ++elY) {
        for (int elX = 0; elX < nElx; ++elX) {
            int elIndex = elY * nElx + elX;
            mesh.connArr[elIndex].resize(4);
            
            // Calculate node indices for the corners of the element
            int nodeIndex = elY * nNodesX + elX;
            mesh.connArr[elIndex][0] = nodeIndex;                // Bottom left
            mesh.connArr[elIndex][1] = nodeIndex + 1;            // Bottom right
            mesh.connArr[elIndex][2] = nodeIndex + nNodesX + 1;  // Top right
            mesh.connArr[elIndex][3] = nodeIndex + nNodesX;      // Top left
        }
    }

    return mesh;
}

/*The function setDirichletConditions sets the Dirichlet boundary conditions for nodes on the left and right edges of a rectangular mesh.
The dirich vector is resized to hold the node index and the corresponding Dirichlet condition value.
The loop for the left edge assigns a fixed value (e.g., 300) to each node along this edge. The node indices are calculated based on the mesh structure.
Similarly, the loop for the right edge assigns another fixed value (e.g., 310) to each node along the right edge*/

void setDirichletConditions(std::vector<std::vector<double>>& dirich, int nElx, int nEly) {
    // The Dirichlet conditions are defined along the left and right edges.
    // Resize dirich to hold the conditions for each node on these edges.
    int totalDirichNodes = (nEly + 1) * 2; // Nodes on the left and right edges
    dirich.resize(totalDirichNodes, std::vector<double>(2));

    // Set conditions on the left edge (constant value)
    for (int i = 0; i <= nEly; ++i) {
        dirich[i][0] = 1 + i * (nElx + 1); // Node index on the left edge
        dirich[i][1] = 300;                // Temperature or value at this node
    }

    // Set conditions on the right edge (different constant value)
    for (int i = 0; i <= nEly; ++i) {
        int index = nEly + 1 + i;
        dirich[index][0] = (i + 1) * (nElx + 1); // Node index on the right edge
        dirich[index][1] = 310;                  // Temperature or value at this node
    }
}

void setInitialCondition(Eigen::VectorXd& u0, const Mesh& mesh) {
    // Resize u0 to match the number of nodes in the mesh
    u0.resize(mesh.nNode);

    // Set the initial condition based on the x-coordinate of each node
    for (int i = 0; i < mesh.nNode; ++i) {
        double x = mesh.xGlo(i, 0); // x-coordinate of the ith node
        if (x < 0.5) {
            u0(i) = 300; // Initial condition for x < 0.5
        } else {
            u0(i) = 300 + 20 * (x - 0.5); // Initial condition for x >= 0.5
        }
    }
}

LocalMatrices computeLocalMatrices(const Eigen::MatrixXd& elementNodes, 
                                   const Eigen::MatrixXd& kappa, 
                                   double rho_c, 
                                   double distributedLoad, 
                                   int wq) {
    LocalMatrices local;
    local.stiffness = Eigen::MatrixXd::Zero(4, 4); // Assuming 4 nodes per element
    local.mass = Eigen::MatrixXd::Zero(4, 4);
    local.force = Eigen::VectorXd::Zero(4);

    // Define quadrature points and weights for Gauss quadrature
    Eigen::MatrixXd quadPoints(4, 2); // 4 quadrature points for 2D integration
    Eigen::VectorXd quadWeights(4);
    double a = std::sqrt(1.0 / 3.0);
    quadPoints << -a, -a,  a, -a,  a,  a, -a,  a;
    quadWeights << 1, 1, 1, 1;

    // Numerical integration over the element
    for (int q = 0; q < quadPoints.rows(); ++q) {
        double xi = quadPoints(q, 0); // Local coordinates
        double eta = quadPoints(q, 1);

        // Evaluate shape functions N and their derivatives dN/dxi, dN/deta
        Eigen::VectorXd N(4);
        Eigen::MatrixXd dNdxi(4, 2);
        N << (1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4, 
             (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4;
        dNdxi << -(1 - eta), -(1 - xi),
                  (1 - eta), -(1 + xi),
                  (1 + eta),  (1 + xi),
                 -(1 + eta),  (1 - xi);

        // Compute the Jacobian matrix for coordinate transformation
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 2);
        for (int i = 0; i < 4; ++i) {
            J(0, 0) += dNdxi(i, 0) * elementNodes(i, 0);
            J(0, 1) += dNdxi(i, 0) * elementNodes(i, 1);
            J(1, 0) += dNdxi(i, 1) * elementNodes(i, 0);
            J(1, 1) += dNdxi(i, 1) * elementNodes(i, 1);
        }

        double detJ = J.determinant();

        // Calculate the derivatives of shape functions with respect to x and y
        Eigen::MatrixXd dNdxy = dNdxi * J.inverse();

        // Integrate stiffness, mass, and force contributions over the element
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                local.stiffness(i, j) += (dNdxy.row(i) * kappa * dNdxy.row(j).transpose() * detJ * quadWeights(q))(0, 0);
                local.mass(i, j) += (N(i) * N(j) * rho_c * detJ * quadWeights(q));
            }
            local.force(i) += (N(i) * distributedLoad * detJ * quadWeights(q));
        }
    }

    return local;
}

Eigen::MatrixXd createKGlobal(const Mesh& mesh, const Eigen::MatrixXd& kappa, int wq, double rho_c) {
    Eigen::MatrixXd kGlobal = Eigen::MatrixXd::Zero(mesh.nNode, mesh.nNode);

    for (int el = 0; el < mesh.nEl; ++el) {
        // Extract nodes for this element
        Eigen::MatrixXd elementNodes(4, 2); // Assuming 4 nodes per element
        for (int i = 0; i < 4; ++i) {
            int nodeId = mesh.connArr[el][i];
            elementNodes.row(i) = mesh.xGlo.row(nodeId);
        }

        // Compute the local stiffness, mass matrices, and local force vector for this element
        LocalMatrices localMatrices = computeLocalMatrices(elementNodes, kappa, rho_c, 0, wq); // Assuming no distributed load for stiffness calculation

        // Assemble the local stiffness matrix into the global matrix
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int globalI = mesh.connArr[el][i];
                int globalJ = mesh.connArr[el][j];
                kGlobal(globalI, globalJ) += localMatrices.stiffness(i, j);
            }
        }
    }

    return kGlobal;
}

Eigen::MatrixXd createMGlobal(const Mesh& mesh, const Eigen::MatrixXd& kappa, double rho_c, int wq, double distributedLoad) {
    Eigen::MatrixXd mGlobal = Eigen::MatrixXd::Zero(mesh.nNode, mesh.nNode);

    for (int el = 0; el < mesh.nEl; ++el) {
        // Extract nodes for this element
        Eigen::MatrixXd elementNodes(4, 2); // Assuming 4 nodes per element
        for (int i = 0; i < 4; ++i) {
            int nodeId = mesh.connArr[el][i];
            elementNodes.row(i) = mesh.xGlo.row(nodeId);
        }

        // Compute the local stiffness, mass matrices, and local force vector for this element
        LocalMatrices localMatrices = computeLocalMatrices(elementNodes, kappa, rho_c, distributedLoad, wq);

        // Assemble the local mass matrix into the global matrix
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int globalI = mesh.connArr[el][i];
                int globalJ = mesh.connArr[el][j];
                mGlobal(globalI, globalJ) += localMatrices.mass(i, j);
            }
        }
    }

    return mGlobal;
}

Eigen::VectorXd createFGlobal(const Mesh& mesh, const Eigen::MatrixXd& kappa, double rho_c, int wq, double distributedLoad) {
    Eigen::VectorXd fGlobal = Eigen::VectorXd::Zero(mesh.nNode);

    for (int el = 0; el < mesh.nEl; ++el) {
        // Extract nodes for this element
        Eigen::MatrixXd elementNodes(4, 2); // Assuming 4 nodes per element
        for (int i = 0; i < 4; ++i) {
            int nodeId = mesh.connArr[el][i];
            elementNodes.row(i) = mesh.xGlo.row(nodeId);
        }

        // Compute the local stiffness, mass matrices, and local force vector for this element
        LocalMatrices localMatrices = computeLocalMatrices(elementNodes, kappa, rho_c, distributedLoad, wq);

        // Assemble the local force vector into the global vector
        for (int i = 0; i < 4; ++i) {
            int globalIndex = mesh.connArr[el][i];
            fGlobal(globalIndex) += localMatrices.force(i);
        }
    }

    return fGlobal;
}

// Assuming Eigen::MatrixXd for kGlobal, mGlobal, fGlobal
// and Eigen::VectorXi for dirich (first column: node indices, second column: prescribed values)

Eigen::MatrixXd createReducedMatrix(const Eigen::MatrixXd& globalMatrix, const Eigen::VectorXi& nonDirichNodes) {
    int size = nonDirichNodes.size();
    Eigen::MatrixXd reducedMatrix(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            reducedMatrix(i, j) = globalMatrix(nonDirichNodes(i), nonDirichNodes(j));
        }
    }
    return reducedMatrix;
}

Eigen::VectorXd createReducedForceVector(const Eigen::MatrixXd& kGlobal, 
                                         const Eigen::VectorXd& fGlobal, 
                                         const Eigen::VectorXi& nonDirichNodes, 
                                         const Eigen::MatrixXd& dirich) {
    Eigen::VectorXd fDash = Eigen::VectorXd::Zero(nonDirichNodes.size());
    for (int i = 0; i < nonDirichNodes.size(); ++i) {
    // Correct way to access elements of Eigen::VectorXi
    int nodeId = nonDirichNodes(i);
    for (int j = 0; j < dirich.rows(); ++j) {
            fDash(i) += (kGlobal(nonDirichNodes(i), dirich(j, 0)) * dirich(j, 1))(0);
        }
    }
    Eigen::VectorXd reducedForce = fGlobal(nonDirichNodes) - fDash;
    return reducedForce;
}

// Usage
// Eigen::MatrixXd kDirichlet = createReducedMatrix(kGlobal, nonDirichNodes);
// Eigen::MatrixXd mDirichlet = createReducedMatrix(mGlobal, nonDirichNodes);
// Eigen::VectorXd fDirichlet = createReducedForceVector(kGlobal, fGlobal, nonDirichNodes, dirich);

/*Eigen::VectorXd createTimeStepping(const Eigen::MatrixXd& mDirichlet, 
                                   const Eigen::MatrixXd& kDirichlet, 
                                   const Eigen::VectorXd& fDirichlet, 
                                   const Eigen::VectorXd& dInitial, 
                                   double alpha, 
                                   int tSteps, 
                                   double tDelta) {
    // Initialize displacement vector with initial conditions
    Eigen::VectorXd d = dInitial;
    Eigen::VectorXd dNext = Eigen::VectorXd::Zero(d.size());

    // Compute the effective stiffness matrix for time stepping
    Eigen::MatrixXd kEffective = kDirichlet + alpha * tDelta * mDirichlet;

    // Time-stepping loop
    for (int step = 0; step < tSteps; ++step) {
        // Compute the effective force vector for the current time step
        Eigen::VectorXd fEffective = fDirichlet + mDirichlet * d / tDelta;

        // Solve for the next displacement
        dNext = kEffective.ldlt().solve(fEffective); // Using LDLT decomposition for solving

        // Update displacement
        d = dNext;
    }

    return d;
}

void shapeFunc(double zhi, double eta, Eigen::VectorXd& N, Eigen::VectorXd& dNdz, Eigen::VectorXd& dNde) {
    N.resize(4);
    dNdz.resize(4);
    dNde.resize(4);

    N(0) = (1 - zhi) * (1 - eta) / 4;
    N(1) = (1 + zhi) * (1 - eta) / 4;
    N(2) = (1 + zhi) * (1 + eta) / 4;
    N(3) = (1 - zhi) * (1 + eta) / 4;

    dNdz(0) = -(1 - eta) / 4;
    dNdz(1) =  (1 - eta) / 4;
    dNdz(2) =  (1 + eta) / 4;
    dNdz(3) = -(1 + eta) / 4;

    dNde(0) = -(1 - zhi) / 4;
    dNde(1) = -(1 + zhi) / 4;
    dNde(2) =  (1 + zhi) / 4;
    dNde(3) =  (1 - zhi) / 4;
}

Eigen::MatrixXd quadrature(int n) {
    Eigen::MatrixXd A;

    if (n == 1) {
        A.resize(1, 2);
        A(0, 0) = 0;
        A(0, 1) = 2;
    }
    else if (n == 2) {
        A.resize(2, 2);
        A(0, 0) = -1.0 / std::sqrt(3.0);
        A(0, 1) = 1;
        A(1, 0) =  1.0 / std::sqrt(3.0);
        A(1, 1) = 1;
    }
    else if (n == 3) {
        A.resize(3, 2);
        A(0, 0) = -std::sqrt(3.0 / 5.0);
        A(0, 1) = 5.0 / 9.0;
        A(1, 0) = 0;
        A(1, 1) = 8.0 / 9.0;
        A(2, 0) = std::sqrt(3.0 / 5.0);
        A(2, 1) = 5.0 / 9.0;
    }

    return A;
} */


















/*Eigen::MatrixXd computeLocalStiffness(const Eigen::MatrixXd& elementNodes, const Eigen::MatrixXd& kappa, int wq) {
    Eigen::MatrixXd kLocal = Eigen::MatrixXd::Zero(4, 4); // Assuming 4 nodes per element

    // Define quadrature points and weights for Gauss quadrature
    Eigen::MatrixXd quadPoints(4, 2); // 4 quadrature points for 2D integration
    Eigen::VectorXd quadWeights(4);

    // Define quadrature points and weights (for 2x2 Gauss Quadrature)
    double a = std::sqrt(1.0 / 3.0);
    quadPoints << -a, -a,  a, -a,  a,  a, -a,  a;
    quadWeights << 1, 1, 1, 1; // Equal weights for 2x2 Gauss quadrature

    // Numerical integration over the element
    for (int i = 0; i < quadPoints.rows(); ++i) {
        double xi = quadPoints(i, 0);  // Local coordinates
        double eta = quadPoints(i, 1);

        // Evaluate shape functions N and their derivatives dN/dxi, dN/deta at quadrature point
        Eigen::VectorXd N(4);
        Eigen::MatrixXd dNdxi(4, 2);
        // Shape functions for a quadrilateral element
        N << (1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4, 
             (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4;
        // Derivatives of shape functions
        dNdxi << -(1 - eta) / 4, -(1 - xi) / 4,
                  (1 - eta) / 4, -(1 + xi) / 4,
                  (1 + eta) / 4,  (1 + xi) / 4,
                 -(1 + eta) / 4,  (1 - xi) / 4;

        // Compute the Jacobian matrix for coordinate transformation
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 2);
        for (int j = 0; j < 4; ++j) {
            J(0, 0) += elementNodes(j, 0) * dNdxi(j, 0);
            J(0, 1) += elementNodes(j, 1) * dNdxi(j, 0);
            J(1, 0) += elementNodes(j, 0) * dNdxi(j, 1);
            J(1, 1) += elementNodes(j, 1) * dNdxi(j, 1);
        }

        // Calculate the determinant of the Jacobian
        double detJ = J.determinant();

        // Calculate the derivative of shape functions with respect to x and y
        Eigen::MatrixXd dNdxy = dNdxi * J.inverse();

        // Integrate stiffness contribution over the element
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                kLocal(a, b) += dNdxy.row(a) * kappa * dNdxy.row(b).transpose() * detJ * quadWeights(i);
            }
        }
    }

    return kLocal;
}*/



/*Eigen::MatrixXd createKGlobal(const Mesh& mesh, const Eigen::MatrixXd& kappa, int wq) {
    Eigen::MatrixXd kGlobal = Eigen::MatrixXd::Zero(mesh.nNode, mesh.nNode);

    for (int el = 0; el < mesh.nEl; ++el) {
        // Extract nodes for this element
        Eigen::MatrixXd elementNodes(4, 2); // Assuming 4 nodes per element
        for (int i = 0; i < 4; ++i) {
            int nodeId = mesh.connArr[el][i];
            elementNodes.row(i) = mesh.xGlo.row(nodeId);
        }

        // Compute the local stiffness matrix for this element
        Eigen::MatrixXd kLocal = computeLocalStiffness(elementNodes, kappa, wq);

        // Assemble the local stiffness matrix into the global matrix
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int globalI = mesh.connArr[el][i];
                int globalJ = mesh.connArr[el][j];
                kGlobal(globalI, globalJ) += kLocal(i, j);
            }
        }
    }

    return kGlobal;
} */

/*Eigen::MatrixXd computeLocalMass(const Eigen::MatrixXd& elementNodes, double rho_c, int wq) {
    Eigen::MatrixXd mLocal = Eigen::MatrixXd::Zero(4, 4); // Assuming 4 nodes per element

    // Define quadrature points and weights for numerical integration
    // This example uses a simple 2x2 Gauss quadrature for demonstration purposes
    Eigen::MatrixXd quadPoints(4, 2); // 4 quadrature points for 2D integration
    Eigen::VectorXd quadWeights(4);

    // Define quadrature points and weights (for 2x2 Gauss Quadrature)
    double a = std::sqrt(1.0 / 3.0);
    quadPoints << -a, -a,  a, -a,  a,  a, -a,  a;
    quadWeights << 1, 1, 1, 1; // Equal weights for 2x2 Gauss quadrature

    // Numerical integration over the element
    for (int i = 0; i < quadPoints.rows(); ++i) {
        double xi = quadPoints(i, 0);  // Local coordinates
        double eta = quadPoints(i, 1);

        // Evaluate shape functions at quadrature point
        Eigen::VectorXd N(4);
        N << (1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4, 
             (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4;

        // Compute the Jacobian matrix for coordinate transformation
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 2);
        for (int j = 0; j < 4; ++j) {
            J(0, 0) += elementNodes(j, 0) * N(j);
            J(0, 1) += elementNodes(j, 1) * N(j);
            J(1, 0) += elementNodes(j, 0) * N(j);
            J(1, 1) += elementNodes(j, 1) * N(j);
        }

        // Calculate determinant of Jacobian for area element
        double detJ = J.determinant();

        // Integrate mass contribution over the element
        mLocal += N * N.transpose() * rho_c * detJ * quadWeights(i);
    }

    return mLocal;
} */




/*Eigen::MatrixXd createMGlobal(const Mesh& mesh, double rho_c, int wq) {
    Eigen::MatrixXd mGlobal = Eigen::MatrixXd::Zero(mesh.nNode, mesh.nNode);

    for (int el = 0; el < mesh.nEl; ++el) {
        // Extract nodes for this element
        Eigen::MatrixXd elementNodes(4, 2); // Assuming 4 nodes per element
        for (int i = 0; i < 4; ++i) {
            int nodeId = mesh.connArr[el][i];
            elementNodes.row(i) = mesh.xGlo.row(nodeId);
        }

        // Compute the local mass matrix for this element
        Eigen::MatrixXd mLocal = computeLocalMass(elementNodes, rho_c, wq);

        // Assemble the local mass matrix into the global matrix
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int globalI = mesh.connArr[el][i];
                int globalJ = mesh.connArr[el][j];
                mGlobal(globalI, globalJ) += mLocal(i, j);
            }
        }
    }

    return mGlobal;
} */

/*Eigen::VectorXd computeLocalForce(const Eigen::MatrixXd& elementNodes, int wq, double distributedLoad) {
    Eigen::VectorXd fLocal = Eigen::VectorXd::Zero(4); // Assuming 4 nodes per element

    // Define quadrature points and weights for numerical integration
    // This example uses a simple 2x2 Gauss quadrature for demonstration purposes
    Eigen::MatrixXd quadPoints(4, 2); // 4 quadrature points for 2D integration
    Eigen::VectorXd quadWeights(4);

    // Define quadrature points and weights (for 2x2 Gauss Quadrature)
    double a = std::sqrt(1.0 / 3.0);
    quadPoints << -a, -a,  a, -a,  a,  a, -a,  a;
    quadWeights << 1, 1, 1, 1; // Equal weights for 2x2 Gauss quadrature

    // Numerical integration over the element
    for (int i = 0; i < quadPoints.rows(); ++i) {
        double xi = quadPoints(i, 0);  // Local coordinates
        double eta = quadPoints(i, 1);

        // Evaluate shape functions at quadrature point
        Eigen::VectorXd N(4);
        N << (1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4, 
             (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4;

        // Compute the Jacobian matrix for coordinate transformation
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 2);
        for (int j = 0; j < 4; ++j) {
            J(0, 0) += elementNodes(j, 0) * N(j);
            J(0, 1) += elementNodes(j, 1) * N(j);
            J(1, 0) += elementNodes(j, 0) * N(j);
            J(1, 1) += elementNodes(j, 1) * N(j);
        }

        // Calculate determinant of Jacobian for area element
        double detJ = J.determinant();

        // Integrate the distributed load over the element
        fLocal += N * distributedLoad * detJ * quadWeights(i);
    }

    return fLocal;
} */
