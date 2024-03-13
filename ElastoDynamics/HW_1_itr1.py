import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vtk


def createmesh(nElx, nEly, nElz, L, W, H, order):
    if order == 2:
        nEl = nElx * nEly * nElz
        elemL = L / nElz
        elemH = H / nEly
        elemW = W / nElx

        # Global node coordinates (initialization)
        xGlo = np.zeros(((nElx + 1) * (nEly + 1) * (nElz + 1), 3))

        # Base layer nodes
        for i in range(nElz + 1):
            for j in range(nEly + 1):
                for k in range(nElx + 1):
                    index = k + (j - 1) * (nElx + 1) + (i - 1) * (nElx + 1) * (nEly + 1)
                    xGlo[index, 0] = (k - 1) * elemW
                    xGlo[index, 1] = (j - 1) * elemH
                    xGlo[index, 2] = (i - 1) * elemL

        typical = (nElx + 1) * (nEly + 1) * (nElz + 1)
        count = typical  # Initialize count

        # Additional nodes for quadratic elements
        for k in range(1, 2 * nElz + 2):
            if k % 2 == 1:
                for i in range(1, 2 * nEly + 2):
                    if i % 2 == 1:
                        for j in range(1, nElx + 1):
                            xGlo[count, 0] = (2 * j - 1) * elemW / 2
                            xGlo[count, 1] = (i - 1) * elemH / 2
                            xGlo[count, 2] = (k - 1) * elemL / 2
                            count += 1
                    else:  # i % 2 == 0
                        for j in range(1, nElx + 2):
                            xGlo[count, 0] = (j - 1) * elemW
                            xGlo[count, 1] = (i - 1) * elemH / 2
                            xGlo[count, 2] = (k - 1) * elemL / 2
                            count += 1
            else:  # k % 2 == 0:
                for i in range(1, nEly + 2):
                    for j in range(1, nElx + 2):
                        xGlo[count, 0] = (j - 1) * elemW
                        xGlo[count, 1] = (i - 1) * elemH
                        xGlo[count, 2] = (k - 1) * elemL / 2
                        count += 1

        nNodes = count  # Update the number of nodes

        # Connectivity array creation
    connArr = np.zeros((nEl, 20), dtype=int)  # Initialize with integers

    for i in range(nElz):
        for j in range(nEly):
            for k in range(nElx):
                l = k + (j - 1) * nElx + (i - 1) * nElx * nEly

            # Node indexing (adjustments for zero-based indexing)
            connArr[l, 0] = l + j - 1 + (i - 1) * (nElx + nEly + 1)
            connArr[l, 1] = l + j + (i - 1) * (nElx + nEly + 1)
            connArr[l, 2] = l + j + nElx + (i - 1) * (nElx + nEly + 1)  
            connArr[l, 3] = l + j + nElx - 1 + (i - 1) * (nElx + nEly + 1)
            connArr[l, 4] = l + j + i * (nElx + nEly + 1) + nElx * nEly
            connArr[l, 5] = l + j + 1 + i * (nElx + nEly + 1) + nElx * nEly
            connArr[l, 6] = l + j + nElx + 2 + i * (nElx + nEly + 1) + nElx * nEly
            connArr[l, 7] = l + j + nElx + 1 + i * (nElx + nEly + 1) + nElx * nEly
                    
            connArr[l, 8] = typical + k + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            connArr[l, 9] = typical + nElx + k + 1 + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            connArr[l, 10] = typical + 2 * nElx + 1 + k + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            connArr[l, 11] = typical + nElx + k + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    
            connArr[l, 12] = typical + k + j * (2 * nElx + 1) + (i + 1) * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            connArr[l, 13] = typical + nElx + k + 1 + j * (2 * nElx + 1) + (i + 1) * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            connArr[l, 14] = typical + 2 * nElx + 1 + k + j * (2 * nElx + 1) + (i + 1) * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            connArr[l, 15] = typical + nElx + k + j * (2 * nElx + 1) + (i + 1) * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    
            connArr[l, 16] = typical + (nElx + 1) * nEly + (nEly + 1) * nElx + k + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            connArr[l, 17] = typical + (nElx + 1) * nEly + (nEly + 1) * nElx + k + 1 + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            connArr[l, 18] = typical + (nElx + 1) * nEly + (nEly + 1) * nElx + k + nElx + 2 + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            connArr[l, 19] = typical + (nElx + 1) * nEly + (nEly + 1) * nElx + k + nElx + 1 + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
            
        return nEl, nNodes, connArr, xGlo
    
    elif order == 1:
        
    # Linear elements
    nNodes = (nElx + 1) * (nEly + 1) * (nElz + 1)
    nEl = nElx * nEly * nElz
    elemL, elemH, elemW = L / nElz, H / nEly, W / nElx

    # Initialize global node coordinates array for linear elements
    xGlo = np.zeros((nNodes, 3))

    # Define global node coordinates
    for i in range(nElz + 1):
        for j in range(nEly + 1):
            for k in range(nElx + 1):
                index = k + j * (nElx + 1) + i * (nElx + 1) * (nEly + 1)
                xGlo[index, :] = [(k - 1) * elemW, (j - 1) * elemH, (i - 1) * elemL]

    # Initialize connectivity array for linear elements
    connArr = np.zeros((nEl, 8), dtype=int)

    # Define connectivity array
    for i in range(nElz):
        for j in range(nEly):
            for k in range(nElx):
                l = k + j * nElx + i * nElx * nEly
                if l >= nEl: 
                    break 

                connArr[l, 0] = l + j + i * (nElx + 1) * (nEly + 1)
                connArr[l, 1] = l + j + 1 + i * (nElx + 1) * (nEly + 1) 
                connArr[l, 2] = l + j + nElx + 2 + i * (nElx + 1) * (nEly + 1) 
                connArr[l, 3] = l + j + nElx + 1 + i * (nElx + 1) * (nEly + 1) 
                connArr[l, 4] = l + j + i * (nElx + 1) * (nEly + 1) + nElx * nEly
                connArr[l, 5] = l + j + 1 + i * (nElx + 1) * (nEly + 1) + nElx * nEly
                connArr[l, 6] = l + j + nElx + 2 + i * (nElx + 1) * (nEly + 1) + nElx * nEly 
                connArr[l, 7] = l + j + nElx + 1 + i * (nElx + 1) * (nEly + 1) + nElx * nEly

    return nEl, nNodes, connArr, xGlo 

""""
def createmesh(nElx, nEly, nElz, L, W, H, order):
    if order == 2:
        nEl = nElx * nEly * nElz
        elemL = L / nElz
        elemH = H / nEly
        elemW = W / nElx

        

        total_nodes_estimate = ((nElx + 1) * (nEly + 1) * (nElz + 1)) + \
                               (nElx * nEly * nElz * 8)
        
        
        xGlo = np.zeros((total_nodes_estimate, 3))
        count = 0

        # Initialize the global node coordinates array
        #total_nodes_estimate = (nElx + 1) * (nEly + 1) * (nElz + 1) + 2 * nElx * (2 * nEly + 1) * (2 * nElz + 1)
        #xGlo = np.zeros((total_nodes_estimate, 3))
        #count = 0

        # Base layer nodes
        #for i in range(nElz + 1):
         #   for j in range(nEly + 1):
          #      for k in range(nElx + 1):
           #         if count >= total_nodes_estimate:
            #            break
             #       xGlo[count, :] = [(k - 1) * elemW, (j - 1) * elemH, (i - 1) * elemL]
              #      count += 1
        
        for i in range(nElz + 1):
            for j in range(nEly + 1):
                for k in range(nElx + 1):
                    #print("i:", i, "j:", j, "k:", k, "count:", count)  # Print before node assignment
                    
                    xGlo[count, :] = [k * elemW, j * elemH, i * elemL]
                    count += 1

        # Additional nodes for quadratic elements
        for k in range(1, 2 * nElz + 2):
            if k % 2 == 1:
                for i in range(1, 2 * nEly + 2):
                    if i % 2 == 1:
                        for j in range(1, nElx + 1):
                            if count >= total_nodes_estimate:
                                break
                            
                            xGlo[count, :] = [(2 * j - 1) * elemW / 2, (i - 1) * elemH / 2, (k - 1) * elemL / 2]
                            count += 1
                    else:
                        for j in range(1, nElx + 2):
                            if count >= total_nodes_estimate:
                                break
                            xGlo[count, :] = [(j - 1) * elemW, (i - 1) * elemH / 2, (k - 1) * elemL / 2]
                            count += 1
            else:
                for i in range(1, nEly + 2):
                    for j in range(1, nElx + 2):
                        if count >= total_nodes_estimate:
                            break
                        xGlo[count, :] = [(j - 1) * elemW, (i - 1) * elemH, (k - 1) * elemL / 2]
                        count += 1

        # Trim xGlo to actual size used
        xGlo = xGlo[:count, :]

        #nNodes = len(xGlo)

        #connArr = np.zeros((nEl, 20), dtype=int)
        nNodes = count  # Update the number of nodes
        connArr = np.zeros((nEl, 20), dtype=int) 

        for i in range(nElz):
            for j in range(nEly):
                for k in range(nElx):
                    l = k + j * nElx + i * nElx * nEly
                    
                    if l >= nEl:
                        break
                    print("Element index (l):", l)  # Print element index
                    print("connArr[l]:", connArr[l])  # Print the row before assignment

                    connArr[l, 0] = l + j + i * (nElx + nEly + 1)
                    connArr[l, 1] = l + j + 1 + i * (nElx + nEly + 1)
                    connArr[l, 2] = l + j + nElx + 2 + i * (nElx + nEly + 1)
                    connArr[l, 3] = l + j + nElx + 1 + i * (nElx + nEly + 1)
                    
                    connArr[l, 4] = l + j + i * (nElx + nEly + 1) + nElx * nEly
                    connArr[l, 5] = l + j + 1 + i * (nElx + nEly + 1) + nElx * nEly
                    connArr[l, 6] = l + j + nElx + 2 + i * (nElx + nEly + 1) + nElx * nEly
                    connArr[l, 7] = l + j + nElx + 1 + i * (nElx + nEly + 1) + nElx * nEly
                    
                    connArr[l, 8] = typical + k + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    connArr[l, 9] = typical + nElx + k + 1 + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    connArr[l, 10] = typical + 2 * nElx + 1 + k + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    connArr[l, 11] = typical + nElx + k + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    
                    connArr[l, 12] = typical + k + j * (2 * nElx + 1) + (i + 1) * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    connArr[l, 13] = typical + nElx + k + 1 + j * (2 * nElx + 1) + (i + 1) * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    connArr[l, 14] = typical + 2 * nElx + 1 + k + j * (2 * nElx + 1) + (i + 1) * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    connArr[l, 15] = typical + nElx + k + j * (2 * nElx + 1) + (i + 1) * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    
                    connArr[l, 16] = typical + (nElx + 1) * nEly + (nEly + 1) * nElx + k + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    connArr[l, 17] = typical + (nElx + 1) * nEly + (nEly + 1) * nElx + k + 1 + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    connArr[l, 18] = typical + (nElx + 1) * nEly + (nEly + 1) * nElx + k + nElx + 2 + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))
                    connArr[l, 19] = typical + (nElx + 1) * nEly + (nEly + 1) * nElx + k + nElx + 1 + j * (2 * nElx + 1) + i * ((nElx + 1) * (2 * nEly + 1) + nElx * (nEly + 1))

        return nEl, nNodes, connArr, xGlo
    
    elif order == 1:
        # Linear elements
        nNodes = (nElx + 1) * (nEly + 1) * (nElz + 1)
        nEl = nElx * nEly * nElz
        elemL, elemH, elemW = L / nElz, H / nEly, W / nElx

        # Initialize global node coordinates array for linear elements
        xGlo = np.zeros((nNodes, 3))

        # Define global node coordinates
        for i in range(nElz + 1):
            for j in range(nEly + 1):
                for k in range(nElx + 1):
                    index = k + j * (nElx + 1) + i * (nElx + 1) * (nEly + 1)
                    xGlo[index, :] = [(k - 1) * elemW, (j - 1) * elemH, (i - 1) * elemL]

        # Initialize connectivity array for linear elements
        connArr = np.zeros((nEl, 8), dtype=int)

        # Define connectivity array
        for i in range(nElz):
            for j in range(nEly):
                for k in range(nElx):
                    l = k + j * nElx + i * nElx * nEly
                    # Make sure connArr index doesn't exceed nEl
                    if l >= nEl:
                        break
                    connArr[l, 0] = l + j + i * (nElx + 1) * (nEly + 1)
                    connArr[l, 0] = l + j + i * (nElx + 1) * (nEly + 1)
                    connArr[l, 1] = l + j + (i * (nElx + 1) * (nEly + 1)) + 1
                    connArr[l, 2] = l + j + (nElx + 1) + (i * (nElx + 1) * (nEly + 1)) + 1
                    connArr[l, 3] = l + j + (nElx + 1) + i * (nElx + 1) * (nEly + 1)
                    connArr[l, 4] = l + j + (i + 1) * (nElx + 1) * (nEly + 1)
                    connArr[l, 5] = l + j + ((i + 1) * (nElx + 1) * (nEly + 1)) + 1
                    connArr[l, 6] = l + j + (nElx + 1) + ((i + 1) * (nElx + 1) * (nEly + 1)) + 1
                    connArr[l, 7] = l + j + (nElx + 1) + (i + 1) * (nElx + 1) * (nEly + 1)

    return nEl, nNodes, connArr, xGlo

"""
def shapeFunc(zhi, eta, gamma, order):
    if order == 2:
        xi = zhi
        zeta = gamma

        # Shape functions for quadratic elements
        N = np.array([
            -(1 - xi) * (1 - eta) * (1 - zeta) * (2 + xi + eta + zeta),
            -(1 + xi) * (1 - eta) * (1 - zeta) * (2 - xi + eta + zeta),
            -(1 + xi) * (1 + eta) * (1 - zeta) * (2 - xi - eta + zeta),
            -(1 - xi) * (1 + eta) * (1 - zeta) * (2 + xi - eta + zeta),
            -(1 - xi) * (1 - eta) * (1 + zeta) * (2 + xi + eta - zeta),
            -(1 + xi) * (1 - eta) * (1 + zeta) * (2 - xi + eta - zeta),
            -(1 + xi) * (1 + eta) * (1 + zeta) * (2 - xi - eta - zeta),
            -(1 - xi) * (1 + eta) * (1 + zeta) * (2 + xi - eta - zeta),
            (1 - xi**2) * (1 - eta) * (1 - zeta) * 2,
            (1 - eta**2) * (1 + xi) * (1 - zeta) * 2,
            (1 - xi**2) * (1 + eta) * (1 - zeta) * 2,
            2 * (1 - eta**2) * (1 - xi) * (1 - zeta),
            2 * (1 - xi**2) * (1 - eta) * (1 + zeta),
            2 * (1 - eta**2) * (1 + xi) * (1 + zeta),
            2 * (1 - xi**2) * (1 + eta) * (1 + zeta),
            2 * (1 - eta**2) * (1 - xi) * (1 + zeta),
            2 * (1 - zeta**2) * (1 - xi) * (1 - eta),
            2 * (1 - zeta**2) * (1 + xi) * (1 - eta),
            2 * (1 - zeta**2) * (1 + xi) * (1 + eta),
            2 * (1 - zeta**2) * (1 - xi) * (1 + eta)
        ]) / 8

        # Derivatives of N with respect to xi, eta, and zeta for quadratic elements
        dNdz = np.array([
            ((eta - 1) * (zeta - 1) * (eta + 2 * xi + zeta + 1)) / 8,
            -((eta - 1) * (zeta - 1) * (eta - 2 * xi + zeta + 1)) / 8,
            -((eta + 1) * (zeta - 1) * (eta + 2 * xi - zeta - 1)) / 8,
            -((eta + 1) * (zeta - 1) * (2 * xi - eta + zeta + 1)) / 8,
            -((eta - 1) * (zeta + 1) * (eta + 2 * xi - zeta + 1)) / 8,
            ((eta - 1) * (zeta + 1) * (eta - 2 * xi - zeta + 1)) / 8,
            ((eta + 1) * (zeta + 1) * (eta + 2 * xi + zeta - 1)) / 8,
            -((eta + 1) * (zeta + 1) * (eta - 2 * xi + zeta - 1)) / 8,
            -(xi * (eta - 1) * (zeta - 1)) / 2,
            ((eta**2 - 1) * (zeta - 1)) / 4,
            (xi * (eta + 1) * (zeta - 1)) / 2,
            -((2 * eta**2 - 2) * (zeta - 1)) / 8,
            (xi * (eta - 1) * (zeta + 1)) / 2,
            -((2 * eta**2 - 2) * (zeta + 1)) / 8,
            -(xi * (eta + 1) * (zeta + 1)) / 2,
            ((2 * eta**2 - 2) * (zeta + 1)) / 8,
            -((2 * zeta**2 - 2) * (eta - 1)) / 8,
            ((2 * zeta**2 - 2) * (eta - 1)) / 8,
            -((2 * zeta**2 - 2) * (eta + 1)) / 8,
            ((2 * zeta**2 - 2) * (eta + 1)) / 8
        ])

        dNde = np.array([
            ((xi - 1) * (zeta - 1) * (2 * eta + xi + zeta + 1)) / 8,
            -((xi + 1) * (zeta - 1) * (2 * eta - xi + zeta + 1)) / 8,
            -((xi + 1) * (zeta - 1) * (2 * eta + xi - zeta - 1)) / 8,
            -((xi - 1) * (zeta - 1) * (xi - 2 * eta + zeta + 1)) / 8,
            -((xi - 1) * (zeta + 1) * (2 * eta + xi - zeta + 1)) / 8,
            ((xi + 1) * (zeta + 1) * (2 * eta - xi - zeta + 1)) / 8,
            ((xi + 1) * (zeta + 1) * (2 * eta + xi + zeta - 1)) / 8,
            -((xi - 1) * (zeta + 1) * (2 * eta - xi + zeta - 1)) / 8,
            -((xi**2 - 1) * (zeta - 1)) / 4,
            (eta * (xi + 1) * (zeta - 1)) / 2,
            ((xi**2 - 1) * (zeta - 1)) / 4,
            -(eta * (xi - 1) * (zeta - 1)) / 2,
            ((2 * xi**2 - 2) * (zeta + 1)) / 8,
            -(eta * (xi + 1) * (zeta + 1)) / 2,
            -((2 * xi**2 - 2) * (zeta + 1)) / 8,
            (eta * (xi - 1) * (zeta + 1)) / 2,
            -((2 * zeta**2 - 2) * (xi - 1)) / 8,
            ((2 * zeta**2 - 2) * (xi + 1)) / 8,
            -((2 * zeta**2 - 2) * (xi + 1)) / 8,
            ((2 * zeta**2 - 2) * (xi - 1)) / 8
        ])

        dNdg = np.array([
            ((eta - 1) * (xi - 1) * (eta + xi + 2 * zeta + 1)) / 8,
            -((eta - 1) * (xi + 1) * (eta - xi + 2 * zeta + 1)) / 8,
            -((eta + 1) * (xi + 1) * (eta + xi - 2 * zeta - 1)) / 8,
            -((eta + 1) * (xi - 1) * (xi - eta + 2 * zeta + 1)) / 8,
            -((eta - 1) * (xi - 1) * (eta + xi - 2 * zeta + 1)) / 8,
            ((eta - 1) * (xi + 1) * (eta - xi - 2 * zeta + 1)) / 8,
            ((eta + 1) * (xi + 1) * (eta + xi + 2 * zeta - 1)) / 8,
            -((eta + 1) * (xi - 1) * (eta - xi + 2 * zeta - 1)) / 8,
            -((xi**2 - 1) * (eta - 1)) / 4,
            ((eta**2 - 1) * (xi + 1)) / 4,
            ((xi**2 - 1) * (eta + 1)) / 4,
            -((2 * eta**2 - 2) * (xi - 1)) / 8,
            ((2 * xi**2 - 2) * (eta - 1)) / 8,
            -((2 * eta**2 - 2) * (xi + 1)) / 8,
            -((2 * xi**2 - 2) * (eta + 1)) / 8,
            ((2 * eta**2 - 2) * (xi - 1)) / 8,
            -(zeta * (eta - 1) * (xi - 1)) / 2,
            (zeta * (eta - 1) * (xi + 1)) / 2,
            -(zeta * (eta + 1) * (xi + 1)) / 2,
            (zeta * (eta + 1) * (xi - 1)) / 2])
        
    elif order == 1:
    # Shape functions for linear elements
        N = np.array([
        (1 - zhi) * (1 - eta) * (1 - gamma) / 8,
        (1 + zhi) * (1 - eta) * (1 - gamma) / 8,
        (1 + zhi) * (1 + eta) * (1 - gamma) / 8,
        (1 - zhi) * (1 + eta) * (1 - gamma) / 8,
        (1 - zhi) * (1 - eta) * (1 + gamma) / 8,
        (1 + zhi) * (1 - eta) * (1 + gamma) / 8,
        (1 + zhi) * (1 + eta) * (1 + gamma) / 8,
        (1 - zhi) * (1 + eta) * (1 + gamma) / 8
    ])

    # Derivatives of N with respect to zhi, eta, and gamma for linear elements
    dNdz = np.array([
        -(1 - eta) * (1 - gamma) / 8,
        (1 - eta) * (1 - gamma) / 8,
        (1 + eta) * (1 - gamma) / 8,
        -(1 + eta) * (1 - gamma) / 8,
        -(1 - eta) * (1 + gamma) / 8,
        (1 - eta) * (1 + gamma) / 8,
        (1 + eta) * (1 + gamma) / 8,
        -(1 + eta) * (1 + gamma) / 8
    ])

    dNde = np.array([
        -(1 - zhi) * (1 - gamma) / 8,
        -(1 + zhi) * (1 - gamma) / 8,
        (1 + zhi) * (1 - gamma) / 8,
        (1 - zhi) * (1 - gamma) / 8,
        -(1 - zhi) * (1 + gamma) / 8,
        -(1 + zhi) * (1 + gamma) / 8,
        (1 + zhi) * (1 + gamma) / 8,
        (1 - zhi) * (1 + gamma) / 8
    ])

    dNdg = np.array([
        -(1 - zhi) * (1 - eta) / 8,
        -(1 + zhi) * (1 - eta) / 8,
        -(1 + zhi) * (1 + eta) / 8,
        -(1 - zhi) * (1 + eta) / 8,
        (1 - zhi) * (1 - eta) / 8,
        (1 + zhi) * (1 - eta) / 8,
        (1 + zhi) * (1 + eta) / 8,
        (1 - zhi) * (1 + eta) / 8
    ])

    return N, dNdz, dNde, dNdg

def quadrature(n):
    if n == 1:
        A = np.array([[0, 2]])
    elif n == 2:
        A = np.array([[-1/np.sqrt(3), 1], [1/np.sqrt(3), 1]])
    elif n == 3:
        A = np.array([[-np.sqrt(3/5), 5/9], [0, 8/9], [np.sqrt(3/5), 5/9]])
    return A

def nodalStiffness(dNdX, C, A, B):
    K = np.zeros((3, 3))
    dNdX_t = dNdX.T  # Transpose of dNdX
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    K[i, k] += dNdX[A, j] * C[i, j, k, l] * dNdX_t[l, B]
    return K

def nodalMass(N, rho, delta, A, B):
    M = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            M[i, k] = N[A] * rho * delta[i, k] * N[B]
    return M

# Variable values
W = 0.1  # z-direction
H = 0.1  # y-direction
L = 1.0  # x-direction

E = 1000  # Young's modulus
nu = 0.3  # Poisson's ratio
rho = 1  # Density

# Rayleigh damping constants
a = 1
b = 0.001

# Lame's parameters
lambda_ = E / (2 * (1 + nu))
mu = E * nu / ((1 + nu) * (1 - 2 * nu))

# Identity matrix for use in defining Cijkl Elastic moduli
delta = np.eye(3)

# Initialize the Elastic moduli tensor
C = np.zeros((3, 3, 3, 3))

# Cijkl Elastic moduli
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                C[i, j, k, l] = lambda_ * delta[i, j] * delta[k, l] + 2 * mu * (delta[i, k] * delta[j, l] + delta[i, l] * delta[j, k])

# Order of shape function (1-Linear or 2-Quadratic)
order = 1

# Dirichlet displacement at z=L
yDisp = 0.05

# Time stepping parameters
beta = 0.255
gamma = 2 * beta

# Transient analysis flag
transient = 1  # (transient = 1) for performing transient analysis, otherwise (transient = 0) for static

tDelta = 1000  # DeltaT
tSteps = 10  # Number of time steps

# Gaussian quadrature selection (1, 2, or 3 points)
wq = 3

# Meshing parameters
nElx = 2  # Number of elements along W
nEly = 1  # Number of elements along H
nElz = 1  # Number of elements along L

# Number of nodes per element
nNoEl = 20 if order == 2 else 8

# Number of degrees of freedom
nDof = 3

# Create the mesh
nEl, nNodes, connArr, xGlo = createmesh(nElx, nEly, nElz, L, W, H, order)

# Dirichlet Boundary Conditions
dirich = []
for i in range(nNodes):
    if xGlo[i, 2] == 0:  # Dirichlet at z=0
        dirich.extend([(3 * i, 0), (3 * i + 1, 0), (3 * i + 2, 0)])
    if xGlo[i, 2] == L:  # Dirichlet at z=L
        dirich.append((3 * i + 1, yDisp))

dirich = np.array(dirich)
nDirich = len(dirich)

# Initial conditions
u0 = np.zeros(nDof * nNodes)  # Initial displacement
v0 = np.zeros_like(u0)  # Initial velocity

# Gaussian quadrature
wq = 3  # Select from 1, 2, or 3

#  kLocal and kGlobal
gaussMatrix = quadrature(wq)  # Gaussian quadrature

# Initialize global matrices
kGlobal = np.zeros((nDof * nNodes, nDof * nNodes))
mGlobal = np.zeros((nDof * nNodes, nDof * nNodes))
fGlobal = np.zeros((nDof * nNodes, 1))

# Loop over elements to construct local matrices and assemble into global matrices
for e in range(nEl):
    kLocal = np.zeros((nDof * nNoEl, nDof * nNoEl))
    mLocal = np.zeros((nDof * nNoEl, nDof * nNoEl))
    if e >= connArr.shape[0]:
        break
    kLocal = np.zeros((nDof * nNoEl, nDof * nNoEl))
    mLocal = np.zeros((nDof * nNoEl, nDof * nNoEl))
    fLocal = np.zeros((nDof * nNoEl, 1))
    
    for wqX in range(wq):
        for wqY in range(wq):
            for wqZ in range(wq):
                N, dNdz, dNde, dNdg = shapeFunc(gaussMatrix[wqX, 0], gaussMatrix[wqY, 0], gaussMatrix[wqZ, 0], order)
                
                # Check if any of the node indices exceed the bounds of xGlo
                node_indices = connArr[e, :]
                if np.any(node_indices >= xGlo.shape[0]):
                    print("Error: Node index exceeds bounds of xGlo")
                    continue

                Jac = np.array([
                    [dNdz.dot(xGlo[node_indices, 0]), dNde.dot(xGlo[node_indices, 0]), dNdg.dot(xGlo[node_indices, 0])],
                    [dNdz.dot(xGlo[node_indices, 1]), dNde.dot(xGlo[node_indices, 1]), dNdg.dot(xGlo[node_indices, 1])],
                    [dNdz.dot(xGlo[node_indices, 2]), dNde.dot(xGlo[node_indices, 2]), dNdg.dot(xGlo[node_indices, 2])]
                ])
                
                dNdX = np.dot(np.array([dNdz, dNde, dNdg]).T, np.linalg.inv(Jac))
                
                for A in range(nNoEl):
                    for B in range(nNoEl):
                        K = nodalStiffness(dNdX, C, A, B)
                        M = nodalMass(N, rho, delta, A, B)
                        kL = np.zeros((3, 3))
                        mL = np.zeros((3, 3))
                        kL[0:3, 0:3] = K
                        mL[0:3, 0:3] = M
                        kLocal[3 * A:3 * (A + 1), 3 * B:3 * (B + 1)] += kL * np.linalg.det(Jac) * gaussMatrix[wqX, 1] * gaussMatrix[wqY, 1] * gaussMatrix[wqZ, 1]
                        mLocal[3 * A:3 * (A + 1), 3 * B:3 * (B + 1)] += mL * np.linalg.det(Jac) * gaussMatrix[wqX, 1] * gaussMatrix[wqY, 1] * gaussMatrix[wqZ, 1]
                        fLocal[3 * A:3 * (A + 1)] += 0  # No forcing term

    # Assembly
for i in range(nNoEl):
    for j in range(nNoEl):
        I = slice(3 * (connArr[e, i] - 1), 3 * connArr[e, i])
        J = slice(3 * (connArr[e, j] - 1), 3 * connArr[e, j])
        print("I:", I)
        print("J:", J)
        k_slice = kLocal[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
        m_slice = mLocal[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
    if l == 2:  # Focus on the problematic element
        #print("Element index:", l)  # Keep this
        #print("k_slice:", k_slice)
        #print("m_slice:", m_slice)
        #print("I:", I)
        #print("J:", J)
        #print("connArr[l]:", connArr[l])
        print("k_slice:", k_slice)
        print("m_slice:", m_slice)
        print("I:", I)
        print("J:", J)
        print("k_slice shape:", k_slice.shape)
        print("m_slice shape:", m_slice.shape)
        print("Dimensions of kglobal:", kGlobal.shape)
        print("I,J",I, J)
        print(connArr[e, i])
        print("nElx:", nElx, "nEly:", nEly, "nElz:", nElz, "order:", order)  # Print input parameters
        print("Element index (l):", l)  # Print element index
        print("connArr[l]:", connArr[l])  # Print the row before assignment
        #print("Estimated total nodes:", total_nodes_estimate)  # Check if this is correct
        #print("i:", i, "j:", j, "k:", k, "count:", count)  # Print before node assignment

        kGlobal[I, J] += k_slice
        mGlobal[I, J] += m_slice

# kDirichlet and fDirichlet
kDirichlet = np.zeros((nDof * nNodes - nDirich, nDof * nNodes - nDirich))
mDirichlet = np.zeros((nDof * nNodes - nDirich, nDof * nNodes - nDirich))
fDirichlet = np.zeros((nDof * nNodes - nDirich, 1))

nonDirichnodes = np.zeros((nDof * nNodes - nDirich, 1))
j = 0

for i in range(nDof * nNodes):
    flag = False
    for k in range(nDirich):
        if i == dirich[k, 0]:
            flag = True
            break
    if not flag:
        nonDirichnodes[j] = i
        j += 1

# kDirichlet and mDirichlet
for i in range(len(nonDirichnodes)):
    for j in range(len(nonDirichnodes)):
        kDirichlet[i, j] = kGlobal[int(nonDirichnodes[i]), int(nonDirichnodes[j])]
        mDirichlet[i, j] = mGlobal[int(nonDirichnodes[i]), int(nonDirichnodes[j])]

f_dash = np.zeros((len(nonDirichnodes), 1))

# fDirichlet
for i in range(len(nonDirichnodes)):
    for j in range(nDirich):
        f_dash[i] = kGlobal[int(nonDirichnodes[i]), int(dirich[j, 0])] * dirich[j, 1] + f_dash[i]

fDirichlet = fGlobal[nonDirichnodes.astype(int)] - f_dash

# Static Analysis
if not transient:
    kDirichlet = sparse.csr_matrix(kDirichlet)
    d = spsolve(kDirichlet, fDirichlet)
    
    dGlobal = np.zeros((nDof * nNodes, 1))
    dGlobal[dirich[:, 0]] = dirich[:, 1]
    dGlobal[nonDirichnodes] = d
    
    if order == 1:
        x, y, z = np.zeros(nNodes), np.zeros(nNodes), np.zeros(nNodes)
        for i in range(nNodes):
            x[i] = dGlobal[i * nDof]
            y[i] = dGlobal[i * nDof + 1]
            z[i] = dGlobal[i * nDof + 2]
        u = np.column_stack((x, y, z))
        # Code for converting fem to vtk_Vector here
    
    if order == 2:
        typical = (nElx + 1) * (nEly + 1) * (nElz + 1)
        dPlot = dGlobal[:nDof * typical]
        x, y, z = np.zeros(typical), np.zeros(typical), np.zeros(typical)
        for i in range(typical):
            x[i] = dPlot[i * nDof]
            y[i] = dPlot[i * nDof + 1]
            z[i] = dPlot[i * nDof + 2]
        u = np.column_stack((x, y, z))
        # Code for converting fem to vtk_Vector here

# Transient Analysis
if transient:
    cDirichlet = a * mDirichlet + b * kDirichlet  # Rayleigh damping
    
    # Initializing
    dGlobal = np.zeros((nDof * nNodes, 1))
    vGlobal = np.zeros_like(dGlobal)
    
    # Time stepping
    d = dGlobal[nonDirichnodes, 0]
    v = vGlobal[nonDirichnodes, 0]
    acc = spsolve(mDirichlet + gamma * tDelta * cDirichlet + beta * tDelta ** 2 * kDirichlet, fDirichlet - kDirichlet @ d - cDirichlet @ v)
    
    for i in range(tSteps):
        d_tilde = d + tDelta * v + (1 - 2 * beta) / 2 * tDelta ** 2 * acc
        v_tilde = v + (1 - gamma) * tDelta * acc
        
        acc = spsolve(mDirichlet + gamma * tDelta * cDirichlet + beta * tDelta ** 2 * kDirichlet, fDirichlet - kDirichlet @ d_tilde - cDirichlet @ v_tilde)
        
        d = d_tilde + beta * tDelta ** 2 * acc
        v = v_tilde + gamma * tDelta * acc
    
    # Plotting
    tStepPlot = 11  # timestep to plot (should be less than or equal to tSteps+1)
    dGlobal[dirich[:, 0]] = dirich[:, 1]
    dGlobal[nonDirichnodes] = d[:, tStepPlot]
    
    x, y, z = np.zeros(nNodes), np.zeros(nNodes), np.zeros(nNodes)
    for i in range(nNodes):
        x[i] = dGlobal[i * nDof]
        y[i] = dGlobal[i * nDof + 1]
        z[i] = dGlobal[i * nDof + 2]
    u = np.column_stack((x, y, z))
    # Code for converting fem to vtk_Vector here
    # Extracting the coordinates and displacements
X = xGlo[:, 0]
Y = xGlo[:, 1]
Z = xGlo[:, 2]
G_x = dGlobal[0::3]
G_y = dGlobal[1::3]
G_z = dGlobal[2::3]

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
undeformed_grid = create_vtk_grid(xGlo, connArr)
deformed_nodes = xGlo + np.column_stack([G_x, G_y, G_z])
deformed_grid = create_vtk_grid(deformed_nodes, connArr)

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
def save_vtk_grid_as_vtu(grid, HW5):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(HW5)
    writer.SetInputData(grid)
    writer.Write()

# Full file paths for the output files
output_path_undeformed = "C:/Advance_FEA/HW01/python/itr2/undeformed_mesh.vtu"  # Update this path
output_path_deformed = "C:/Advance_FEA/HW01/python/itr2/deformed_mesh.vtu"      # Update this path

# Save undeformed and deformed grids as .vtu files
save_vtk_grid_as_vtu(undeformed_grid, output_path_undeformed)
save_vtk_grid_as_vtu(deformed_grid, output_path_deformed)
