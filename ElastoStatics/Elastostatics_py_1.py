import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vtk

#Make a function for ShapeFunction for 8 node HexaHedral Elements
def ShpF(q):
    xi, eta, zeta = q

    # ShapeFunction for 8 node HexaHedral Elements
    N = 1/8 * np.array([(1 - xi) * (1 - eta) * (1 - zeta),
                        (1 + xi) * (1 - eta) * (1 - zeta),
                        (1 + xi) * (1 + eta) * (1 - zeta),
                        (1 - xi) * (1 + eta) * (1 - zeta),
                        (1 - xi) * (1 - eta) * (1 + zeta),
                        (1 + xi) * (1 - eta) * (1 + zeta),
                        (1 + xi) * (1 + eta) * (1 + zeta),
                        (1 - xi) * (1 + eta) * (1 + zeta)])

    # Taking the derivatives of the shape functions (eg. dN/dxi)
    dN_dxi = 1/8 * np.array([-(1-eta)*(1-zeta),  (1-eta)*(1-zeta),
                             (1+eta)*(1-zeta), -(1+eta)*(1-zeta),
                            -(1-eta)*(1+zeta),  (1-eta)*(1+zeta),
                             (1+eta)*(1+zeta), -(1+eta)*(1+zeta)])

    dN_deta = 1/8 * np.array([-(1-xi)*(1-zeta), -(1+xi)*(1-zeta),
                               (1+xi)*(1-zeta),  (1-xi)*(1-zeta),
                              -(1-xi)*(1+zeta), -(1+xi)*(1+zeta),
                               (1+xi)*(1+zeta),  (1-xi)*(1+zeta)])

    dN_dzeta = 1/8 * np.array([-(1-xi)*(1-eta), -(1+xi)*(1-eta),
                               -(1+xi)*(1+eta), -(1-xi)*(1+eta),
                                (1-xi)*(1-eta),  (1+xi)*(1-eta),
                                (1+xi)*(1+eta),  (1-xi)*(1+eta)])

    dN = np.vstack([dN_dxi, dN_deta, dN_dzeta])
    return N, dN

#Make a function Element Stiffness Matrix and calling ShpF
def ElementStiffnessMatrix(LocalESF, q4, C):
    Ke = np.zeros((24, 24))

    # Gauss weights for 3D integration
    W = np.array([5/9, 8/9, 5/9])

    for zi in range(len(q4)):
        for yi in range(len(q4)):
            for xi in range(len(q4)):
                q = np.array([q4[xi], q4[yi], q4[zi]])
                N, dN = ShpF(q)

                J = np.dot(dN, LocalESF)
                detJ = np.linalg.det(J)
                dN = np.linalg.solve(J, dN)

                # SD matrix (strain-displacement matrix) for 3D
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

                # Integration (Gauss Quadrature) for 3D
                Ke += np.dot(SD.T, np.dot(C, SD)) * detJ * W[zi] * W[yi] * W[xi]
    return Ke


#Mesh Properties
#Number of elements in each direction
Num_Elx = 4
Num_Ely = 4
Num_Elz = 40

# Mesh Properties
# Length in each direction
Width = 0.1
Length = 0.1
Height = 1

# Mesh properties for nodes in x, y, z direction
NumNx = Num_Elx + 1   # No. of N in x
NumNy = Num_Ely + 1   # No. of N in y
NumNz = Num_Elz + 1   # No. of N in z

#Total No. of Nodes
Total_NumNd = NumNx * NumNy * NumNz 

# Nodes - 3D
nodes = np.array([[x, y, z] for z in np.linspace(0, Height, NumNz)
                  for y in np.linspace(0, Length, NumNy)
                  for x in np.linspace(0, Width, NumNx)])

# Initialize the connectivity matrix for 3D
conn = []

# Loop over the elements in the mesh - 3D
for k in range(Num_Elz):
    for j in range(Num_Ely):
        for i in range(Num_Elx):
            # Calculate the index of the bottom-left-front node of the current element
            n0 = i + j * NumNx + k * NumNx * NumNy

            # Append the indices of the eight nodes of the current element
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
mu = (E * nu) / ((1 + nu) * (1 - 2 * nu))
lambda_ = E / (2 * (1 + nu))  # 'lambda_' is used instead of 'lambda' because lambda is a reserved keyword in Python

# Constitutive matrix for isotropic linear elasticity in 3D
C = np.array([[lambda_ + 2*mu, lambda_, lambda_, 0, 0, 0],
              [lambda_, lambda_ + 2*mu, lambda_, 0, 0, 0],
              [lambda_, lambda_, lambda_ + 2*mu, 0, 0, 0],
              [0, 0, 0, 2*mu, 0, 0],
              [0, 0, 0, 0, 2*mu, 0],
              [0, 0, 0, 0, 0, 2*mu]])

# Display the constitutive matrix easier to debug
print("Constitutive Matrix (C):")
print(C)

# Initialize Global Stiffness Matrix for 3 DOF per node
Kglobal = np.zeros((3 * Total_NumNd, 3 * Total_NumNd))
q4 = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])

# Assembly of Global Stiffness Matrix (Vectorized where possible)
for e in range(conn.shape[0]):
   # Get the global node indices for the current element
   node_indices = conn[e, :]

   # Compute the local stiffness matrix for the current element
   LocalESF = nodes[node_indices, :]  # LocalESF contains the coordinates of the element's nodes
   Ke = ElementStiffnessMatrix(LocalESF, q4, C)  # Call your element stiffness matrix function

   # Map the local stiffness matrix Ke to the correct location in the global matrix K
   for i in range(len(node_indices)):
       for j in range(len(node_indices)):
           # Global DOF indices
           global_i_indices = 3 * node_indices[i] + np.arange(3)
           global_j_indices = 3 * node_indices[j] + np.arange(3)

           # Add the local stiffness matrix Ke to the global matrix K
           Kglobal[np.ix_(global_i_indices, global_j_indices)] += Ke[3*i:3*(i+1), 3*j:3*(j+1)]


# Initialize displacement vectors and constrained DOF list
IniDisp = np.zeros((3 * Total_NumNd, 1))
B_C = np.zeros((3 * Total_NumNd, 1))
Con_DOF = []

# Apply the boundary conditions
for i in range(Total_NumNd):
   x, y, z = nodes[i, :]
   start_idx_dof_node = 3 * i
   end_idx_dof_node = start_idx_dof_node + 2

   if z == 0.0:  # Fully constrained on z = 0m face
       Con_DOF.extend(range(start_idx_dof_node, end_idx_dof_node + 1))
   elif z == Height:  # Y-displacement on z = 1m face
       Con_DOF.append(start_idx_dof_node + 1)
       B_C[start_idx_dof_node + 1, 0] = 0.05

# Remove all the constrained Degree Of Freedom from global
GlobalU = np.zeros((3 * Total_NumNd, 1))
GlobalU[Con_DOF] = B_C[Con_DOF]
Fglobal = -Kglobal @ B_C
Reducek = np.delete(np.delete(Kglobal, Con_DOF, axis=0), Con_DOF, axis=1)
ReduceF = np.delete(Fglobal, Con_DOF, axis=0)

# Solving for the unknown displacements
# Check for the Mismatch in the number of unconstrained DOFs
G_unKnown = np.linalg.solve(Reducek, ReduceF)
unconstrained_indices = np.setdiff1d(np.arange(3 * Total_NumNd), Con_DOF)
if G_unKnown.shape[0] != unconstrained_indices.shape[0]:
    raise ValueError("Mismatch in the number of unconstrained DOFs") 

GlobalU[unconstrained_indices] = G_unKnown

# Extracting the coordinates and displacements
X = nodes[:, 0]
Y = nodes[:, 1]
Z = nodes[:, 2]
G_x = GlobalU[0::3]
G_y = GlobalU[1::3]
G_z = GlobalU[2::3]

# Visualization (adjust the scale_factor)
scale_factor = 1
X_def = X + scale_factor * G_x
Y_def = Y + scale_factor * G_y
Z_def = Z + scale_factor * G_z
DisplacementMagnitude = np.sqrt(G_x**2 + G_y**2 + G_z**2)

# Prepare for visualization (subplots)
nRows = 1
nCols = 2


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

# Create grids for undeformed and deformed meshes
undeformed_grid = create_vtk_grid(nodes, conn)
deformed_nodes = nodes + np.column_stack([G_x, G_y, G_z])
deformed_grid = create_vtk_grid(deformed_nodes, conn)

# Calculate displacement magnitudes
displacement_magnitudes = np.sqrt(G_x**2 + G_y**2 + G_z**2)
displacement_array = vtk.vtkDoubleArray()
displacement_array.SetName("Displacement Magnitude")
for magnitude in displacement_magnitudes:
    displacement_array.InsertNextValue(magnitude)
deformed_grid.GetPointData().AddArray(displacement_array)
deformed_grid.GetPointData().SetActiveScalars("Displacement Magnitude")

# Create a mapper and actor for undeformed mesh
undeformed_mapper = vtk.vtkDataSetMapper()
undeformed_mapper.SetInputData(undeformed_grid)
undeformed_actor = vtk.vtkActor()
undeformed_actor.SetMapper(undeformed_mapper)

# Create a mapper and actor for deformed mesh
deformed_mapper = vtk.vtkDataSetMapper()
deformed_mapper.SetInputData(deformed_grid)
deformed_mapper.SetScalarRange(displacement_magnitudes.min(), displacement_magnitudes.max())
deformed_actor = vtk.vtkActor()
deformed_actor.SetMapper(deformed_mapper)

# Create a renderer, render window, and interactor
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Add the actor to the scene (toggle these to switch between undeformed and deformed mesh visualization)
renderer.AddActor(undeformed_actor)  # For undeformed mesh
# renderer.AddActor(deformed_actor)  # For deformed mesh

# Add a scalar bar to show displacement magnitudes
scalar_bar = vtk.vtkScalarBarActor()
scalar_bar.SetLookupTable(deformed_mapper.GetLookupTable())
scalar_bar.SetTitle("Displacement Magnitude")
renderer.AddActor2D(scalar_bar)

# Set background color and initialize
renderer.SetBackground(0.1, 0.2, 0.4)  # Background color
renderWindow.Render()
renderWindowInteractor.Start()

# Function to save VTK grid as a .vtu file
def save_vtk_grid_as_vtu(grid, HW4):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(HW4)
    writer.SetInputData(grid)
    writer.Write()

# Full file paths for the output files
output_path_undeformed = "C:/FEA/HW04/python/itr2/undeformed_mesh.vtu"  # Update this path
output_path_deformed = "C:/FEA/HW04/python/itr2/deformed_mesh.vtu"      # Update this path

# Save undeformed and deformed grids as .vtu files
save_vtk_grid_as_vtu(undeformed_grid, output_path_undeformed)
save_vtk_grid_as_vtu(deformed_grid, output_path_deformed)




