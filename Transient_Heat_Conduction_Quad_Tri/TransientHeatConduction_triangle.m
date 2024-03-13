%% Defining Constants and Parameters (opted from reference code)
W = 1;  %%% width
H = 1;  %%% height
kappa_bar = 385;
rho = 6000;
rho_c = 3.8151e6;
kappa = kappa_bar*eye(2);    %%% Conductivity tensor

%% Parameters for Time Integration DEtAils for Incremental Time Advancement Scheme
alpha = 1; %0 for forward, 0.5 for central, 1 for backward
tSteps = 10;  % number of time steps

%%  Determination of Integration Points for Numerical Quadrature(options are 1, 2, or 3)
wq = 3;

%% Construction of the Finite Element Mesh
%%%% Define the Element Count Along X and Y Axes
nElx = 20;
nEly = 20;
% For triangular mesh, each element has 3 nodes
nNoEl = 3;
% Total number of elements is twice the number in the quadrilateral mesh
% because each quadrilateral is split into two triangles
[nEl, nNode, connArr, xGlo] = createmesh(nElx, nEly, W, H);
%%
% Initialize Dirichlet boundary conditions array
dirich = [];

% Apply Dirichlet boundary conditions along the left edge where x = 0
for i = 1 : nEly+1
    nodeIndex = 1 + (i-1) * (nElx + 1); % Nodes along the left edge
    dirich = [dirich; nodeIndex, 300];  % Temperature of 300 K
end

% Apply Dirichlet boundary conditions along the right edge where x = W
for i = 1 : nEly+1
    nodeIndex = nElx + 1 + (i-1) * (nElx + 1); % Nodes along the right edge
    dirich = [dirich; nodeIndex, 305];         % Temperature of 305 K
end

% Total number of Dirichlet boundary conditions
nDirich = size(dirich, 1);
% Initialize the initial condition vector
u0 = zeros(nNode, 1);
%%
% Set the initial condition for each node based on its x-coordinate
for i = 1:nNode
    if xGlo(i, 1) < 0.5
        u0(i) = 300; % Temperature is 300K for x < 0.5m
    else
        u0(i) = 300 + 10 * (xGlo(i, 1) - 0.5); % Linear increase from 300K to 310K for x >= 0.5m
    end
end
%% K, M and f Matrices global and Local
% Obtain Gauss quadrature points and weights for triangular elements
gaussMatrix = triQuadrature(wq);

% Initialize global matrices and vector
kGlobal = zeros(nNode, nNode);
mGlobal = zeros(nNode, nNode);
fGlobal = zeros(nNode, 1);

% Loop over all elements to assemble global matrices
for e = 1:nEl
    kLocal = zeros(nNoEl, nNoEl);
    mLocal = zeros(nNoEl, nNoEl);
    fLocal = zeros(nNoEl, 1);

    % Get the coordinates of the nodes of element e
    x = xGlo(connArr(e,:), 1);
    y = xGlo(connArr(e,:), 2);

    % Loop over all quadrature points for the current element
    for q = 1:size(gaussMatrix, 1)
        xi = gaussMatrix(q, 1);
        eta = gaussMatrix(q, 1);
        weight = gaussMatrix(q, 2);

        % Compute shape functions and their derivatives for triangular elements
        [N, dNdxi, dNdeta] = triShapeFunc(xi, eta);

        % Compute the Jacobian matrix for triangular elements
        J = [dNdxi' * x, dNdeta' * x; dNdxi' * y, dNdeta' * y];
        detJ = det(J);

        % % Check for a non-invertible Jacobian matrix
        % if abs(detJ) < 1e-10
        %     error('Element has near-zero area or is badly shaped.');
        % end

        % Compute derivatives of N with respect to x and y using the backslash operator
        dNdX = J \ [dNdxi'; dNdeta'];

        % Local stiffness matrix
        kLocal = kLocal + (dNdX' * kappa * dNdX) * detJ * weight;

        % Local mass matrix
        mLocal = mLocal + (N * N') * rho_c * detJ * weight;

        % Local force vector
        fLocal=fLocal+0;

        % Local force vector (if applicable)
        % fLocal = fLocal + ... (depends on the source term, if present)
    end

    % Assembly into global matrices
    kGlobal(connArr(e,:), connArr(e,:)) = kGlobal(connArr(e,:), connArr(e,:)) + kLocal;
    mGlobal(connArr(e,:), connArr(e,:)) = mGlobal(connArr(e,:), connArr(e,:)) + mLocal;
    fGlobal(connArr(e,:), 1) = fGlobal(connArr(e,:), 1) + fLocal;
end
% Initialize the Dirichlet matrices and vector
kDirichlet = zeros(nNode - nDirich, nNode - nDirich);
fDirichlet = zeros(nNode - nDirich, 1);
mDirichlet = zeros(nNode - nDirich, nNode - nDirich);

% Initialize the list of non-Dirichlet nodes
nonDirichnodes = [];

% Identify the non-Dirichlet nodes
for i = 1:nNode
    if ~ismember(i, dirich(:,1))
        nonDirichnodes = [nonDirichnodes; i];
    end
end

% Construct the reduced stiffness matrix and mass matrix
for i = 1:length(nonDirichnodes)
    for j = 1:length(nonDirichnodes)
        kDirichlet(i, j) = kGlobal(nonDirichnodes(i), nonDirichnodes(j));
        mDirichlet(i, j) = mGlobal(nonDirichnodes(i), nonDirichnodes(j));
    end
end

% Adjust the force vector
f_dash = zeros(length(nonDirichnodes), 1);
for i = 1:length(nonDirichnodes)
    for j = 1:nDirich
        % Accumulate the influence of the Dirichlet boundary on the force vector
        f_dash(i) = f_dash(i) + kGlobal(nonDirichnodes(i), dirich(j, 1)) * dirich(j, 2);
    end
end
% Subtract the Dirichlet influences from the global force vector
fDirichlet = fGlobal(nonDirichnodes) - f_dash;
%%
% Calculate the time step size
tDelta = tDelta_calculate(alpha, mDirichlet, kDirichlet);

% Initialize global displacement with initial condition
dGlobal = u0;

% Initialize displacement and velocity for non-Dirichlet nodes
d = zeros(length(nonDirichnodes), tSteps+1);
v = zeros(length(nonDirichnodes), tSteps+1);
d(:, 1) = u0(nonDirichnodes);
v(:, 1) = mDirichlet \ (fDirichlet - kDirichlet * d(:, 1));

% Time stepping loop
for i = 1:tSteps
    % Predict the displacement 'd_tilde' at the next time step
    d_tilde = d(:, i) + (1 - alpha) * tDelta * v(:, i);

    % Update the velocity at the next time step
    v(:, i+1) = (mDirichlet + alpha * tDelta * kDirichlet) \ (fDirichlet - kDirichlet * d_tilde);

    % Correct the displacement at the next time step
    d(:, i+1) = d_tilde + alpha * tDelta * v(:, i+1);
end

% Update the global displacement vector with the final time step results
dGlobal(nonDirichnodes) = d(:, end);
dGlobal(dirich(:, 1)) = dirich(:, 2); % Apply Dirichlet conditions

%% fem to vtu
fem_to_vtk ('HW4', xGlo, connArr, dGlobal);
%%
function [N, dNdxi, dNdeta] = triShapeFunc(xi, eta)
    % N1, N2, and N3 are the shape functions for a linear triangular element
    N = [1 - xi - eta;  % Shape function for Node 1
         xi;            % Shape function for Node 2
         eta];          % Shape function for Node 3

    % Derivatives of the shape functions with respect to local coordinates xi and eta
    dNdxi = [-1;  % dN1/dxi
              1;  % dN2/dxi
              0]; % dN3/dxi

    dNdeta = [-1;  % dN1/deta
               0;  % dN2/deta
               1]; % dN3/deta
end

function [A] = triQuadrature(n)

if(n==1)
    A(1,1:2)=[0 2];
end

if(n==2)
    A(1,1:2)=[-1/sqrt(3) 1];
    A(2,1:2)=[1/sqrt(3) 1];
end

if(n==3)
    A(1,1:2)=[-sqrt(3/5) 5/9];
    A(2,1:2)=[0 8/9];
    A(3,1:2)=[sqrt(3/5) 5/9];
end
end

%% Mesh creation function modified for triangular elements
function [nEl, nNode, connArr, xGlo] = createmesh(nElx, nEly, W, H)
    % Total number of triangular elements
    nEl = nElx * nEly * 2; 

    % Total number of nodes
    nNode = (nElx + 1) * (nEly + 1);

    % Node coordinates
    xGlo = zeros(nNode, 2);
    nodeIndex = 1;
    for i = 1:nEly+1
        for j = 1:nElx+1
            xGlo(nodeIndex, :) = [(j-1) * W/nElx, (i-1) * H/nEly];
            nodeIndex = nodeIndex + 1;
        end
    end

    % Connectivity array for triangular elements
    connArr = zeros(nEl, 3);
    elemIndex = 1;
    for i = 1:nEly
        for j = 1:nElx
            bl = (i-1) * (nElx + 1) + j;
            br = bl + 1;
            tl = bl + (nElx + 1);
            tr = tl + 1;

            connArr(elemIndex, :) = [bl, br, tl];
            elemIndex = elemIndex + 1;
            connArr(elemIndex, :) = [tr, tl, br];
            elemIndex = elemIndex + 1;
        end
    end
end
%%
function[tDelta]=tDelta_calculate(alpha, mDirichlet, kDirichlet)
if(alpha<0.5)
    e=eig(mDirichlet\kDirichlet);
    e_max=max(e);
    tDelta=2/(1-2*alpha)/e_max;
end

if(alpha>=0.5)
    tDelta=10;
end

end
