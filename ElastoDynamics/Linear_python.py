import numpy as np
import vtk
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def ShpF(q):
    # Unpack the local coordinates
    xi, eta, zeta = q

    # Shape functions for a linear 8-node hexahedral element
    N = 1/8 * np.array([(1 - xi) * (1 - eta) * (1 - zeta),
                        (1 + xi) * (1 - eta) * (1 - zeta),
                        (1 + xi) * (1 + eta) * (1 - zeta),
                        (1 - xi) * (1 + eta) * (1 - zeta),
                        (1 - xi) * (1 - eta) * (1 + zeta),
                        (1 + xi) * (1 - eta) * (1 + zeta),
                        (1 + xi) * (1 + eta) * (1 + zeta),
                        (1 - xi) * (1 + eta) * (1 + zeta)])

    # Derivatives of the shape functions with respect to xi, eta, and zeta
    dN_dxi = 1/8 * np.array([-(1 - eta) * (1 - zeta),  (1 - eta) * (1 - zeta),
                              (1 + eta) * (1 - zeta), -(1 + eta) * (1 - zeta),
                             -(1 - eta) * (1 + zeta),  (1 - eta) * (1 + zeta),
                              (1 + eta) * (1 + zeta), -(1 + eta) * (1 + zeta)])

    dN_deta = 1/8 * np.array([-(1 - xi) * (1 - zeta), -(1 + xi) * (1 - zeta),
                               (1 + xi) * (1 - zeta),  (1 - xi) * (1 - zeta),
                              -(1 - xi) * (1 + zeta), -(1 + xi) * (1 + zeta),
                               (1 + xi) * (1 + zeta),  (1 - xi) * (1 + zeta)])

    dN_dzeta = 1/8 * np.array([-(1 - xi) * (1 - eta), -(1 + xi) * (1 - eta),
                               -(1 + xi) * (1 + eta), -(1 - xi) * (1 + eta),
                                (1 - xi) * (1 - eta),  (1 + xi) * (1 - eta),
                                (1 + xi) * (1 + eta),  (1 - xi) * (1 + eta)])

    # Combine the derivatives into a single matrix
    dN = np.vstack([dN_dxi, dN_deta, dN_dzeta])

    return N, dN



def ElementStiffnessMatrix(LocalESF, q4, C, rho_c):
    # Initialize element stiffness and mass matrices for 8 nodes, 3 DOF each
    Ke = np.zeros((24, 24))
    Me = np.zeros((24, 24))
    
    # Gauss weights for 3D integration
    W = np.array([5/9, 8/9, 5/9])
    
    # Loop over quadrature points in 3 dimensions
    for z_idx, zi in enumerate(q4):
        for y_idx, yi in enumerate(q4):
            for x_idx, xi in enumerate(q4):
                q = [xi, yi, zi]
                N, dN = ShpF(q)  # Implement or provide this function
                
                J = dN @ LocalESF  # Jacobian matrix for 3D
                detJ = np.linalg.det(J)  # Determinant of Jacobian for 3D
                dN = np.linalg.solve(J, dN)  # Derivative of shape functions with respect to x, y, z
                
                #  matrix (strain-displacement matrix) for 3D
                SD = np.zeros((6, 24))
                for i in range(8):
                    SD[0, 3*i] = dN[0, i]
                    SD[1, 3*i+1] = dN[1, i]
                    SD[2, 3*i+2] = dN[2, i]
                    SD[3, 3*i+1] = dN[2, i]
                    SD[3, 3*i+2] = dN[1, i]
                    SD[4, 3*i] = dN[2, i]
                    SD[4, 3*i+2] = dN[0, i]
                    SD[5, 3*i] = dN[1, i]
                    SD[5, 3*i+1] = dN[0, i]
                
                # Expand N for mass matrix construction
                Ne = np.zeros((24, 1))
                for i in range(8):
                    Ne[3*i:3*i+3] = np.tile(N[i], (3, 1))


                
                # Integrate to construct the mass matrix Me
                for i in range(8):
                    for j in range(8):
                         Me[3*i:3*i+3, 3*j:3*j+3] += N[i] * N[j] * rho_c * detJ * W[z_idx] * W[y_idx] * W[x_idx] * np.eye(3)
                
                Ke += SD.T @ C @ SD * detJ * W[z_idx] * W[y_idx] * W[x_idx]
    
    return Ke, Me

# Time stepping parameters
beta = 1/4
gamma = 1/2

# Rayleigh damping constants
a = 1
b = 0.001
rho = 1  # density

tDelta = 0.01  # deltaT
tSteps = 20  # number of time steps

# Mesh properties for nodes in x, y, z direction
# Number of elements in each direction
Num_Elx  = 4
Num_Ely = 4
Num_Elz = 10 

# Length of the mesh in each direction
width = 0.1
length = 0.1
hieght = 1

 # Number of nodes in each direction
NumNx = Num_Elx + 1 
NumNy = Num_Ely + 1
NumNz = Num_Elz + 1 

num_nodes = NumNx * NumNy * NumNz  # Total number of nodes

# Nodes - 3D
nodes = np.array([[x, y, z] for z in np.linspace(0, hieght, NumNz)
                              for y in np.linspace(0, length, NumNy)
                              for x in np.linspace(0, width, NumNx)])

# Initialize the connectivity matrix for 3D
conn = []
for k in range(Num_Elz):
    for j in range(Num_Ely):
        for i in range(Num_Elx):
            n0 = i + j * NumNx + k * NumNx * NumNy
            conn.append([n0, n0 + 1, n0 + 1 + NumNx, n0 + NumNx,
                         n0 + NumNx * NumNy, n0 + 1 + NumNx * NumNy,
                         n0 + 1 + NumNx + NumNx * NumNy, n0 + NumNx + NumNx * NumNy])
conn = np.array(conn)

# Display the nodes and connectivity matrix
print("Nodes:")
print(nodes)
print("\nConnectivity Matrix:")
print(conn)

# Material properties
E = 1000  # Young's modulus in Pa
nu = 0.3  # Poisson's ratio

# Calculate Lamé constants
mu = E * nu / ((1 + nu) * (1 - 2 * nu))
lambda_ = E / (2 * (1 + nu))

# Constitutive matrix for isotropic linear elasticity in 3D
C = np.array([[lambda_ + 2*mu, lambda_, lambda_, 0, 0, 0],
              [lambda_, lambda_ + 2*mu, lambda_, 0, 0, 0],
              [lambda_, lambda_, lambda_ + 2*mu, 0, 0, 0],
              [0, 0, 0, 2*mu, 0, 0],
              [0, 0, 0, 0, 2*mu, 0],
              [0, 0, 0, 0, 0, 2*mu]])

# Initialize Global Stiffness and Mass Matrices for 3 DOF per node
K = np.zeros((3*num_nodes, 3*num_nodes))
M = np.zeros((3*num_nodes, 3*num_nodes))

# Quadrature points
q4 = [-np.sqrt(3/5), 0, np.sqrt(3/5)]

# Assembly of Global Stiffness Matrix (Vectorized where possible)
for e, node_indices in enumerate(conn):
    LocalESF = nodes[node_indices, :]  # Coordinates of the element's nodes
    Ke, Me = ElementStiffnessMatrix(LocalESF, q4, C, rho)  # Call your element stiffness matrix function
    
    for i, ni in enumerate(node_indices):
        for j, nj in enumerate(node_indices):
            global_i_indices = np.arange(3*ni, 3*ni+3)
            global_j_indices = np.arange(3*nj, 3*nj+3)
            
            K[np.ix_(global_i_indices, global_j_indices)] += Ke[3*i:3*i+3, 3*j:3*j+3]
            M[np.ix_(global_i_indices, global_j_indices)] += Me[3*i:3*i+3, 3*j:3*j+3]

Kglobal = K
Mglobal = M

# Initialize displacement and velocity vectors
u0 = np.zeros(3*num_nodes)  # Displacement field: Three DOF per node (u, v, w)
v0 = np.zeros(3*num_nodes)  # Velocity field: Three DOF per node (u, v, w)

# After the initial condition
UY = np.zeros(3*num_nodes)
Constrained_DOF = []

# Loop through each node to apply boundary conditions
for i in range(num_nodes):
    x, y, z = nodes[i]

    # Node index in the global displacement vector
    node_dof_start = 3*i
    node_dof_end = node_dof_start + 2

    # Fully constrained on z = 0m face
    if z == 0.0:
        Constrained_DOF.extend(range(node_dof_start, node_dof_end + 1))

    # Y-displacement on z = hieght face
    if z == hieght:
        Constrained_DOF.append(node_dof_start + 1)  # Constrain Y DOF for this node
        UY[node_dof_start + 1] = 0.05  # Set the prescribed Y displacement

# Now remove the constrained DOF from the global stiffness matrix and force vector
GlobalU = np.zeros(3*num_nodes)
GlobalU[Constrained_DOF] = UY[Constrained_DOF]
Fglobal = -Kglobal @ UY
Reducek, ReduceF, ReduceM = Kglobal.copy(), Fglobal.copy(), Mglobal.copy()

Reducek = np.delete(Reducek, Constrained_DOF, axis=0)
Reducek = np.delete(Reducek, Constrained_DOF, axis=1)
ReduceF = np.delete(ReduceF, Constrained_DOF)
ReduceM = np.delete(ReduceM, Constrained_DOF, axis=0)
ReduceM = np.delete(ReduceM, Constrained_DOF, axis=1)
Uunknown = np.linalg.solve(Reducek, ReduceF)

unconstrained_indices = [i for i in range(3*num_nodes) if i not in Constrained_DOF]

# Transient Analysis
kDirichlet = Reducek
fDirichlet = ReduceF
mDirichlet = ReduceM
cDirichlet = a * mDirichlet + b * kDirichlet  # Rayleigh damping

# Initializing
dGlobal = np.zeros((3*num_nodes, tSteps+1))
vGlobal = np.zeros((3*num_nodes, tSteps+1))
dGlobal[:, 0] = u0
vGlobal[:, 0] = v0

# Time stepping
d = np.zeros((len(unconstrained_indices), tSteps+1))
v = np.zeros((len(unconstrained_indices), tSteps+1))
acc = np.zeros((len(unconstrained_indices), tSteps+1))
d[:, 0] = dGlobal[unconstrained_indices, 0]
v[:, 0] = vGlobal[unconstrained_indices, 0]
acc[:, 0] = np.linalg.solve(mDirichlet, fDirichlet - kDirichlet @ d[:, 0] - cDirichlet @ v[:, 0])

for i in range(tSteps):
    d_tilde = d[:, i] + tDelta * v[:, i] + (1 - 2 * beta) / 2 * tDelta**2 * acc[:, i]
    v_tilde = v[:, i] + (1 - gamma) * tDelta * acc[:, i]

    acc[:, i+1] = np.linalg.solve(mDirichlet + gamma * tDelta * cDirichlet + beta * tDelta**2 * kDirichlet,
                                  fDirichlet - kDirichlet @ d_tilde - cDirichlet @ v_tilde)

    d[:, i+1] = d_tilde + beta * tDelta**2 * acc[:, i+1]
    v[:, i+1] = v_tilde + gamma * tDelta * acc[:, i+1]

# Extracting the coordinates and displacements from GlobalU
scale_factor = 1  # Define the scale factor
X, Y, Z = nodes.T  # Transpose for easier unpacking
G_x, G_y, G_z = dGlobal[0::3], dGlobal[1::3], dGlobal[2::3]

# Adjust the scale_factor as necessary
#X_def = X + scale_factor * G_x
#Y_def = Y + scale_factor * G_y
#Z_def = Z + scale_factor * G_z

# Select the final time step displacements
final_G_x = G_x[:, -1]  # Last column for x-displacement
final_G_y = G_y[:, -1]  # Last column for y-displacement
final_G_z = G_z[:, -1]  # Last column for z-displacement

# Apply the scale factor and calculate the deformed coordinates
X_def = X + scale_factor * final_G_x
Y_def = Y + scale_factor * final_G_y
Z_def = Z + scale_factor * final_G_z

for i in range(tSteps):
    # Your transient analysis update equations here
    pass

# Extract final displacements from the global displacement array at the last time step
final_displacements = dGlobal[:, -1]

# Update node positions with final displacements for the deformed configuration
deformed_nodes = nodes.copy()  # Copy to keep original nodes
deformed_nodes[:, 0] += final_displacements[0::3]
deformed_nodes[:, 1] += final_displacements[1::3]
deformed_nodes[:, 2] += final_displacements[2::3]

# Function to create a VTK grid from nodes and connectivity
def create_vtk_grid(nodes, conn):
    points = vtk.vtkPoints()
    for node in nodes:
        points.InsertNextPoint(node.tolist())

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)

    for element in conn:
        hexa = vtk.vtkHexahedron()
        for i, node_id in enumerate(element):
            hexa.GetPointIds().SetId(i, node_id)
        grid.InsertNextCell(hexa.GetCellType(), hexa.GetPointIds())

    return grid

# Create VTK grids for undeformed and deformed configurations
undeformed_grid = create_vtk_grid(nodes, conn)
deformed_grid = create_vtk_grid(deformed_nodes, conn)

# Calculate displacement magnitudes for coloring the deformed grid
displacement_magnitudes = np.linalg.norm(final_displacements.reshape(-1, 3), axis=1)
displacement_array = vtk.vtkDoubleArray()
displacement_array.SetName("Displacement Magnitude")
for magnitude in displacement_magnitudes:
    displacement_array.InsertNextValue(magnitude)
deformed_grid.GetPointData().AddArray(displacement_array)
deformed_grid.GetPointData().SetActiveScalars("Displacement Magnitude")

# Setup visualization for undeformed grid
undeformed_mapper = vtk.vtkDataSetMapper()
undeformed_mapper.SetInputData(undeformed_grid)
undeformed_actor = vtk.vtkActor()
undeformed_actor.SetMapper(undeformed_mapper)

# Setup visualization for deformed grid
deformed_mapper = vtk.vtkDataSetMapper()
deformed_mapper.SetInputData(deformed_grid)
deformed_mapper.SetScalarRange(displacement_magnitudes.min(), displacement_magnitudes.max())
deformed_actor = vtk.vtkActor()
deformed_actor.SetMapper(deformed_mapper)

# Renderer, render window, and interactor
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Add the undeformed and deformed actor to the renderer
#renderer.AddActor(undeformed_actor)  # For undeformed mesh
renderer.AddActor(deformed_actor)  # For deformed mesh

# Add a scalar bar to show displacement magnitudes
scalar_bar = vtk.vtkScalarBarActor()
scalar_bar.SetLookupTable(deformed_mapper.GetLookupTable())
scalar_bar.SetTitle("Displacement Magnitude")
renderer.AddActor2D(scalar_bar)

# Render and start interaction
renderer.SetBackground(0.1, 0.2, 0.4)  # Background color
renderWindow.Render()
renderWindowInteractor.Start()

# Setup for visualization (create mappers, actors, renderer, etc.)
# [This section includes creating undeformed_mapper, undeformed_actor, deformed_mapper, deformed_actor, renderer, renderWindow, renderWindowInteractor, scalar_bar as provided in your script]

def create_vtk_grid(nodes, conn):
    points = vtk.vtkPoints()
    for node in nodes:
        points.InsertNextPoint(node.tolist())

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)

    for element in conn:
        hexa = vtk.vtkHexahedron()
        for i, node_id in enumerate(element):
            hexa.GetPointIds().SetId(i, node_id)
        grid.InsertNextCell(hexa.GetCellType(), hexa.GetPointIds())

    return grid

# Function to save VTK grid as a .vtu file
def save_vtk_grid_as_vtu(grid, file_path):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_path)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(grid)
    else:
        writer.SetInputData(grid)
    writer.Write()

# Extract final displacements from dGlobal for the last time step
final_displacements = dGlobal[:, -1]

# Update node positions with final displacements to get deformed positions
deformed_nodes = nodes + np.column_stack((final_displacements[0::3], final_displacements[1::3], final_displacements[2::3]))

# Create grids for undeformed and deformed meshes
undeformed_grid = create_vtk_grid(nodes, conn)
deformed_grid = create_vtk_grid(deformed_nodes, conn)

# Specify output file paths
output_path_undeformed = "C:/Advance_FEA/HW01/python/HW01_1/undeformed_mesh.vtu"  # Update this path
output_path_deformed = "C:/Advance_FEA/HW01/python/HW01_1/deformed_mesh.vtu"      # Update this path

# Save undeformed and deformed grids as .vtu files
save_vtk_grid_as_vtu(undeformed_grid, output_path_undeformed)
save_vtk_grid_as_vtu(deformed_grid, output_path_deformed)

print("VTU files saved:")
print(output_path_undeformed)
print(output_path_deformed)
