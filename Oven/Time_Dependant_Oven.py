#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:14:22 2021

@author: Ugo Pelissier

[1] https://bthierry.pages.math.cnrs.fr/course-fem/download/FEM.pdf
[2] http://hplgit.github.io/INF5620/doc/pub/sphinx-fem/._main_fem019.html
"""
#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from Delaunay_Triangulation import *
from matplotlib import cm
from scipy.interpolate import griddata

from tqdm import tqdm

#--- INIT FUNCTION ------------------------------------------------------------+
def read_triangulation(filename):
    """
    Convertir une triangulation .csv en une liste de triangle
    """
    triangulation = []
    df = pd.read_csv(filename)
    for i in range(len(df)):
        p1 = df.values[i,0:2].tolist()
        p2 = df.values[i,2:4].tolist()
        p3 = df.values[i,4:6].tolist()
        p4 = df.values[i,6:8].tolist()
        triangle = [p1, p2, p3, p4]
        triangulation.append(triangle)
    return triangulation

#--- MATHS OPERATIONS ---------------------------------------------------------+
def xij(triangle, i, j):
    return (triangle[i][0]-triangle[j][0])

def yij(triangle, i, j):
    return (triangle[i][1]-triangle[j][1])

def aire(triangle):
    return (xij(triangle,1,0)*yij(triangle,2,0)-yij(triangle,1,0)*xij(triangle,2,0))/2

#--- LIST OPERATIONS ----------------------------------------------------------+
def point_list(triangulation):
    L=[]
    for triangle in triangulation:
        for i in range(3):
            L.append(triangle[i])
    x,y = zip(*L)
    df = pd.DataFrame({'x':x, 'y':y})
    df = df.drop_duplicates()
    return (df.values.tolist())

def triangle_index_of(point, triangulation):
    index = []
    for i in range(len(triangulation)):
        triangle = triangulation[i]
        if is_inside(point, triangle):
            index.append(i)
    return index

def mean_value_triangle(K, triangulation, Points, triangle_index):
    index_global = [local2global(triangulation, Points, triangle_index, i) for i in range(2)]
    k = 0
    for i in range(2):
        k += K[index_global[i]]
    return k/3

#--- TRIANGULATION OPERATIONS --------------------------------------------------+
def phi(triangle, sommet, x, y):
    if (sommet==0):
        return (yij(triangle,1,2)*(x-triangle[2][0])-xij(triangle,1,2)*(y-triangle[2][1]))/(2*aire(triangle))
    if (sommet==1):
        return (yij(triangle,2,0)*(x-triangle[0][0])-xij(triangle,2,0)*(y-triangle[0][1]))/(2*aire(triangle))
    if (sommet==2):
        return (yij(triangle,0,1)*(x-triangle[1][0])-xij(triangle,0,1)*(y-triangle[1][1]))/(2*aire(triangle))
  
def grad_phi_ref(sommet):
    if (sommet==0):
        return np.array([[-1],[-1]])
    if (sommet==1):
        return np.array([[1],[0]])
    if (sommet==2):
        return np.array([[0],[1]])
    
def local2global(triangulation, points, triangle_index, sommet_index):
    point = triangulation[triangle_index][sommet_index]
    return point_in_list_index(point, points)

def index_point_Dirichlet(triangulation, domain_Dirichlet):
    index = []
    Points = point_list(triangulation)
    for i in range(len(Points)):
        for segment in domain_Dirichlet:
            if is_between(segment[0], Points[i], segment[1]):
                index.append(i)
    return list(set(index))

def u_h(Points,z,X,Y):
    x,y=zip(*Points)
    Z = griddata((x,y),z,(X,Y),method='linear')
    return Z

#--- ELEMENTARY MATRIXES ------------------------------------------------------+
def elementary_mass_matrix(triangle):
    return (((abs(2*aire(triangle))/24)*(np.ones([3,3])+np.eye(3))))

def B(triangle):
    p1_x = triangle[0][0]
    p1_y = triangle[0][1]
    p2_x = triangle[1][0]
    p2_y = triangle[1][1]
    p3_x = triangle[2][0]
    p3_y = triangle[2][1]
    return np.array([[p3_y-p1_y, p1_y-p2_y],[p1_x-p3_x, p2_x-p1_x]])/(2*aire(triangle))

def elementary_diffusion_matrix(triangle, i, j):
    coeff = abs(aire(triangle))*np.matmul(np.matmul(grad_phi_ref(j).transpose(),np.matmul(B(triangle).transpose(),B(triangle))),grad_phi_ref(i))
    return round(coeff[0,0],15)

#--- ASSEMBLY FUNCTIONS -------------------------------------------------------+
def mass_matrix(triangulation, domain_Dirichlet):
    I_Points, D_Points = points_list_Dirichlet(triangulation, domain_Dirichlet)
    Points = I_Points + D_Points
    M = np.zeros([len(Points),len(Points)])
    for k in range(len(triangulation)):
        triangle = triangulation[k]
        M_elem = elementary_mass_matrix(triangle)
        for i in range(3):
            I = local2global(triangulation, Points, k, i)
            for j in range(3):
                J = local2global(triangulation, Points, k, j)
                M[I,J] += M_elem[i,j]
    I = len(I_Points)
    M_II = M[0:I,0:I]
    M_ID = M[0:I,I:len(M)]
    return M, M_II, M_ID

def equation_matrix(triangulation, domain_Dirichlet, K):
    I_Points, D_Points = points_list_Dirichlet(triangulation, domain_Dirichlet)
    Points = I_Points + D_Points
    A = np.zeros([len(Points),len(Points)])
    for k in range(len(triangulation)):
        triangle = triangulation[k]
        a = mean_value_triangle(K, triangulation, Points, k)
        for i in range(3):
            I = local2global(triangulation, Points, k, i)
            for j in range(3):
                J = local2global(triangulation, Points, k, j)
                A[I,J] += a*elementary_diffusion_matrix(triangle, i, j)
    I = len(I_Points)
    A_II = A[0:I,0:I]
    A_ID = A[0:I,I:len(A)]
    return A_II, A_ID

def source_term_matrix(triangulation, domain_Dirichlet, geometries, t):
    I_Points, D_Points = points_list_Dirichlet(triangulation, domain_Dirichlet)
    Points = I_Points + D_Points
    F = np.zeros(len(Points))
    M, M_II, M_ID = mass_matrix(triangulation, domain_Dirichlet)
    for i in range(len(F)):
        for j in range(len(F)):
            F[i] += f(Points[j], geometries, t)*M[i,j]
    F_I = F[0:len(I_Points)]
    return F_I

#--- DIRICHLET CONDITION FUNCTIONS --------------------------------------------+
def points_list_Dirichlet(triangulation, domain_Dirichlet):
    Points = point_list(triangulation)
    L = index_point_Dirichlet(triangulation, domain_Dirichlet)
    I_Points = []
    D_Points = []
    for i in range(len(Points)):
        if i not in L:
            I_Points.append(Points[i])
        else:
            D_Points.append(Points[i])
    return I_Points, D_Points

def dirichlet_vector(triangulation, domain_Dirichlet):
    I_Points, D_Points = points_list_Dirichlet(triangulation, domain_Dirichlet)
    g_h = np.zeros(len(D_Points))
    for i in range(len(D_Points)):
        g_h[i] = g(D_Points[i])
    return g_h

#--- TIME SCHEME FUNCTIONS ----------------------------------------------------+
def Euler_Forward_Dirichlet(triangulation, domain_Dirichlet, U_I, t, dt, K, geometries):
    F_I = source_term_matrix(triangulation, domain_Dirichlet, geometries, t)
    A_II, A_ID = equation_matrix(triangulation, domain_Dirichlet, K)
    M, M_II, M_ID = mass_matrix(triangulation, domain_Dirichlet)
    g_D = dirichlet_vector(triangulation, domain_Dirichlet)
    A = M_II
    b = np.matmul((M_II-dt*A_II),U_I) - dt*np.matmul(A_ID,g_D) + dt*F_I
    return np.linalg.solve(A,b)

def Euler_Backward_Dirichlet(triangulation, domain_Dirichlet, U_I, t, dt, K, geometries):
    f_I = source_term_matrix(triangulation, domain_Dirichlet, geometries, t+dt)
    A_II, A_ID = equation_matrix(triangulation, domain_Dirichlet, K)
    M, M_II, M_ID = mass_matrix(triangulation, domain_Dirichlet)
    g_D = dirichlet_vector(triangulation, domain_Dirichlet)
    A = M_II+dt*A_II
    b = np.matmul(M_II,U_I) - dt*np.matmul(A_ID,g_D) + dt*f_I
    return np.linalg.solve(A,b)

def Crank_Nicholson_Dirichlet(triangulation, domain_Dirichlet, U_I, t, dt, K, geometries):
    F_I = source_term_matrix(triangulation, domain_Dirichlet, geometries, t)
    f_I = source_term_matrix(triangulation, domain_Dirichlet, geometries, t+dt)
    A_II, A_ID = equation_matrix(triangulation, domain_Dirichlet, K)
    M, M_II, M_ID = mass_matrix(triangulation, domain_Dirichlet)
    g_D = dirichlet_vector(triangulation, domain_Dirichlet)
    A = M_II+(dt/2)*A_II
    b = np.matmul((M_II-(dt/2)*A_II),U_I) - dt*np.matmul(A_ID,g_D) + (dt/2)*(F_I+f_I)
    return np.linalg.solve(A,b)

#--- PLOT FUNCTIONS -----------------------------------------------------------+
def plot_2d(X,Y,Z, triangulation, geometries):
    plt.contourf(X, Y, Z, 40, cmap=cm.jet)
    plt.colorbar()
    for triangle in triangulation:
        x,y = zip(*triangle)
        plt.plot(x, y, 'k', linewidth=0.1)
        plt.plot(x, y, 'ok',markersize=0.1)
    for geometry in geometries:
        xGeom,yGeom = zip(*geometry)
        plt.plot(xGeom, yGeom, 'k', linewidth=0.1)
        plt.plot(xGeom,yGeom, 'ok',markersize=0.1)
    plt.gca().axis('equal')
    plt.savefig('Solution_2D.eps', format = 'eps', dpi=1200)
    plt.show()
    
def plot_solution(delta, points, U, triangulation, geometry):
    X = np.arange(-1, 1, delta)
    Y = np.arange(-1, 1, delta)
    X,Y = np.meshgrid(X,Y)
    U_h = u_h(points, U, X, Y)
    plot_2d(X,Y,U_h, triangulation, geometry)
    
def write_vtu(Triangulation, Points, U, n):
    path = '/Users/fp/Desktop/Ugo/Projets/Elements_Finis/Poisson/2d/'
    filename = 'bulles_00' + '{:03d}'.format(n) + '.vtu'
    f = open(path+filename,"w+")
    f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <UnstructuredGrid>\n')
    f.write('    <Piece NumberOfPoints="%d"' %len(Points))
    f.write(' NumberOfCells="%d">\n' %len(Triangulation))
    f.write('      <Points>\n')
    f.write('        <DataArray type = "Float32" NumberOfComponents="3" format="ascii">\n')
    for point in Points:
        f.write('{} {} {}\n'.format(point[0],point[1],0.0))
    f.write('        </DataArray>\n')
    f.write('      </Points>\n') 
    f.write('      <Cells>\n') 
    f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
    for i in range(len(Triangulation)):
        I = local2global(Triangulation, Points, i, 0)
        J = local2global(Triangulation, Points, i, 1)
        K = local2global(Triangulation, Points, i, 2)
        f.write('{} {} {}\n'.format(I, J, K))
    f.write('        </DataArray>\n')
    f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
    for i in range(1, len(Triangulation)+1):
        f.write('%d\n' %(3*i))
    f.write('        </DataArray>\n')
    f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
    for i in range(len(Triangulation)):
           f.write('%d\n' %5)
    f.write('        </DataArray>\n')
    f.write('      </Cells>\n')
    f.write('<PointData>\n')
    f.write('  <DataArray type = "Float32" Name="Erreur" format="ascii">\n')
    for i in range(len(Triangulation)):
           f.write('%d\n' %0)
    f.write('  </DataArray>\n')
    f.write('  <DataArray type = "Float32" Name="Temperature" format="ascii">\n')
    for u in U:
        f.write('%d\n' %u)
    f.write('  </DataArray>\n')
    f.write('</PointData>\n')
    f.write('<CellData>\n')
    f.write('</CellData>\n')
    f.write('<UserData>\n')
    f.write('  <DataArray type = "Float32" Name="CompteurTemps" format="ascii">\n')
    f.write('%d\n' %0)
    f.write('  </DataArray>\n')
    f.write('  <DataArray type = "Float32" Name="Temps" format="ascii">\n')
    f.write('%d\n' %0)
    f.write('  </DataArray>\n')
    f.write('</UserData>\n')
    f.write('    </Piece>\n')
    f.write('  </UnstructuredGrid>\n')
    f.write('</VTKFile>')
    
def residu(U_old, U_new, R):
    R = np.delete(R, 0)
    res = abs(U_old[0] - U_new[0])
    for i in range(len(U_old)):
        if (abs(U_old[i] - U_new[i])>res):
            res = abs(U_old[i] - U_new[i])
    R = np.append(R,res)
    r = np.mean(R)
    print('\r', 'Résidu : ', str(np.round(r, 5)), end="", flush = True)
    return r, R

#--- PARAMETERS ---------------------------------------------------------------+
start_time = time.time()

path = '/Users/fp/Desktop/Ugo/Projets/Elements_Finis/Maillage/csv/'
filename = 'Oven_20_refined.csv'

a = 1

delta = 0.001

bord_Dirichlet = [[[-1,-1],[1,-1]],
                  [[-1,1],[1,1]]]

epsilon = 1e-3
r = 100

def i(point):
    """
    Space init function
    """
    y = point[1]
    if y==1:
        return 50
    return 10

def f(point, geometries, t):
    """
    Source function
    """
    sources = [geometries[i]for i in range(2,len(geometries))]
    x = np.round(point[0],8)
    y= np.round(point[1],8)
    for source in sources:
        if inside_geometry([x,y], source):
            return 100000
    return 0
    
def g(point):
    """
    Dirichlet conditions function
    """
    y = point[1]
    if y==-1:
        return 10
    elif y==1:
        return 50
    
def k(point, geometry):
    """
    Diffusion coefficient function
    """
    k_oven = 10
    k_piece = 1
    x = np.round(point[0],8)
    y = np.round(point[1],8)
    if inside_geometry([x,y], geometry):
        return k_piece
    return k_oven 

#--- TRIANGULATION ------------------------------------------------------------+
Delaunay_triangulation = read_triangulation(path+filename)
Points = point_list(Delaunay_triangulation)
I_Points, D_Points = points_list_Dirichlet(Delaunay_triangulation, bord_Dirichlet)
Points = I_Points + D_Points

#--- MATRICES -----------------------------------------------------------------+
K = [k(Points[i], piece) for i in range(len(Points))]
M, M_II, M_ID = mass_matrix(Delaunay_triangulation, bord_Dirichlet)
U_D = dirichlet_vector(Delaunay_triangulation, bord_Dirichlet)

#--- INITIALIZATION -----------------------------------------------------------+
I = [i(point) for point in Points]
U_I = [I[i] for i in range(len(I_Points))]
U = I

#--- TIME DISCRETIZATION ------------------------------------------------------+
t_0 = 0
dt = 0.001
T_save = 10
R = np.array([r for i in range(T_save)])

#--- COMPUTE ------------------------------------------------------------------+
t = t_0
compteur = 0
while (r>epsilon):
    t += dt
    compteur += 1
    U_I = Crank_Nicholson_Dirichlet(Delaunay_triangulation, bord_Dirichlet, U_I, t, dt, K, geom)
    U_new = U_I.tolist() + U_D.tolist()
    r, R = residu(U, U_new, R)
    if ((compteur-1)%T_save==0):
        U = U_new
        write_vtu(Delaunay_triangulation, Points, U, compteur-1)
    
print('\n')
print("--- %s seconds ---" % (time.time() - start_time))