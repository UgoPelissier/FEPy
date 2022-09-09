#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:57:10 2022

@author: Ugo PELISSIER
"""

#--- IMPORT DEPENDENCIES -----------------------------------------------------+

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
from Build_Geometry import *
from time import sleep

#--- MATHS OPERATIONS FUNCTIONS ----------------------------------------------+
def progress(text, percent=0, width=40):
    left = width * percent // 100
    right = width - left
    
    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    
    print("\r", text, "[", tags, spaces, "]", percents, sep="", end="", flush=True)
    
def progress_bad_triangles(i):
    print("\r", "Number of bad triangles : ", i, sep="", end="", flush=True)
    
def progress_encorached_segments(i):
    print("\r", "Number of encroached segments : ", i, sep="", end="", flush=True)
    
def progress_mesh_size(i):
    print("\r", "Number of too long edges : ", i, sep="", end="", flush=True)


def equal(a,b):
    if abs(a-b)<1e-10:
        return True
    return False

def midpoint(segment):
    return [(segment[1][0]+segment[0][0])/2,(segment[1][1]+segment[0][1])/2]

def distance(p1,p2):
    """
    Calcule la distance entre deux points
    """
    return sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def point_in_circle(point, circle):
    """
    Determine si un point se situe dans un cercle
    """
    center,r = circle
    d = distance(point,center)
    if (d-r)<1e-10:
        return True
    else:
        return False
    
def collinear(x1, y1, x2, y2, x3, y3):
    a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    if equal(a,0):
        return True
    return False
    
def point_in_list(p, lst):
    """
    Determine si un point appartient à une liste de points
    """
    for i in range(len(lst)):
        point = lst[i]
        if (equal(p[0],point[0]) and equal(p[1],point[1])):
            return True
    return False

def point_in_list_index(p, lst):
    """
    Determine l'index d'un point s'il appartient à une liste de point
    """
    index = -1
    for i in range(len(lst)):
        point = lst[i]
        if (p[0]==point[0] and p[1]==point[1]):
            index = i
    return index

def ascending_trigonometric_angle_sort(pts):
    """
    Trie une liste de points dans le sens trigonometrique croissant
    """
    geometric_center = np.mean(pts,axis=0)
    translated_pts   = pts - geometric_center
    angles           = np.arctan2(translated_pts[:,1], translated_pts[:,0])
    x                = angles.argsort()
    pts2             = np.array(pts)
    return pts2[x,:].tolist()

def identical_points(p1,p2):
    """
    Indique si deux points sont identiques
    """
    if ((equal(p1[0],p2[0])) and (equal(p1[1],p2[1]))):
        return True
    return False

def identical_edges(first_edge,second_edge):
    """
    Indique si deux segments sont identiques
    """
    p1 = first_edge[0]
    q1 = first_edge[1]
    p2 = second_edge[0]
    q2 = second_edge[1]
    if ((identical_points(p1,p2) and identical_points(q1,q2)) or (identical_points(p1,q2) and identical_points(q1,p2))):
        return True
    return False

def is_between(p1,p3,p2):
    return equal(distance(p1,p3) + distance(p3,p2),distance(p1,p2))

def segment_in_list(segment,lst):
    for edge in lst:
        if identical_edges(edge, segment):
            return True
    return False

#--- POLYGON FUNCTIONS -------------------------------------------------------+
def line_intersection(a,b,c,d):
    if equal(a, c):
        return -1
    else:
        x = (b-d)/(c-a)
        y = a*x+b
    return [x,y]

def line_edge_intersection(a,b,edge):
    p1 = edge[0]
    p2 = edge[1]
    if equal(p2[0],p1[0]):
        if is_between(p1,[p1[0],a*p1[0]+b],p2):
            return [p1[0],a*p1[0]+b]
    c = (p2[1]-p1[1])/(p2[0]-p1[0])
    d = p1[1] - c*p1[0]
    if line_intersection(a,b,c,d)==-1:
        if equal(a*p1[0]+b,p1[1]):
            return [a*p1[0]+b,p1[1]]
    else:
        p = line_intersection(a,b,c,d)
        if is_between(p1,p,p2):
            return p

def line_intersect_edge(a,b,edge):
    p1 = edge[0]
    p2 = edge[1]
    if equal(p2[0],p1[0]):
        if is_between(p1,[p1[0],a*p1[0]+b],p2):
            return True
        return False
    c = (p2[1]-p1[1])/(p2[0]-p1[0])
    d = p1[1] - c*p1[0]
    if line_intersection(a,b,c,d)==-1:
        if equal(a*p1[0]+b,p1[1]):
            return True
        else:
            return False
    else:
        p = line_intersection(a,b,c,d)
        if is_between(p1,p,p2):
            return True
    return False

def geometry_to_polygon(geometry):
    polygon = [[geometry[i],geometry[i+1]] for i in range(len(geometry)-1)]
    return polygon

def polygon_to_geometry(polygon):
    geometry = [polygon[i][0] for i in range(len(polygon))]
    geometry.append(geometry[0])
    return geometry

def ray_casting_number(point, geometry):
    left = 0
    right = 0
    polygon = geometry_to_polygon(geometry)
    x = point[0]
    y = point[1]
    for edge in polygon:
        if line_intersect_edge(0,y,edge):
            p = line_edge_intersection(0,y,edge)
            if p[0]<x:
                left = left+1
            else:
                right = right+1
    return max(left,right)

def point_belongs_to_polygon(point, polygon):
    for edge in polygon:
        if is_between(edge[0],point,edge[1]):
            return True
    return False

def inside_geometry(point, geometry):
    polygon = geometry_to_polygon(geometry)
    if point_belongs_to_polygon(point, polygon):
        return True
    n = ray_casting_number(point, geometry)
    if equal(n%2,0):
        return False
    return True

def outside_geometries(point, geometries):
    for geometry in geometries:
        if inside_geometry(point, geometry):
            return False
    return True

#--- TRIANGLE OPERATIONS FUNCTIONS -------------------------------------------+
def circumcircle(triangle):
    """
    Calcule le cercle circonscrit à un triangle
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = triangle
    A = np.array([[x3-x1,y3-y1],[x3-x2,y3-y2]])
    Y = np.array([(x3**2 + y3**2 - x1**2 - y1**2),(x3**2+y3**2 - x2**2-y2**2)])
    if (abs(np.linalg.det(A)) < 1e-15):
        return False
    Ainv = np.linalg.inv(A)
    X = 0.5*np.dot(Ainv,Y)
    x,y = X[0],X[1]
    r = sqrt((x-x1)**2+(y-y1)**2)
    return (x,y),r

def circumcircle_edge(edge):
    mid = midpoint(edge)
    r = distance(edge[0],edge[1])/2
    return (mid[0],mid[1]),r

def shortest_edge(triangle):
    d = distance(triangle[0], triangle[1])
    for i in range(len(triangle)-1):
        if distance(triangle[i], triangle[i+1])<d:
            d = distance(triangle[i], triangle[i+1])
    return d

def shortest_edges(triangulation):
    d = shortest_edge(triangulation[0])
    for triangle in triangulation:
        if shortest_edge(triangle)<d:
            d = shortest_edge(triangle)
    return d
        
def longest_edge(triangle):
    d = distance(triangle[0], triangle[1])
    edge = [triangle[0], triangle[1]]
    for i in range(len(triangle)-1):
        if distance(triangle[i], triangle[i+1])>d:
            d = distance(triangle[i], triangle[i+1])
            edge = [triangle[i], triangle[i+1]]
    return d, edge

def longest_edges(triangulation):
    d, edge = longest_edge(triangulation[0])
    for triangle in triangulation:
        if longest_edge(triangle)[0]>d:
            d, edge = longest_edge(triangle)
    return d, edge

def too_long_edges(triangulation, criterion):
    L = []
    for i in range(len(triangulation)):
        triangle = triangulation[i]
        for j in range(len(triangle)-1):
            edge = [triangle[j], triangle[j+1]]
            d = distance(triangle[j], triangle[j+1])
            if d>criterion:
                if not segment_in_list(edge, L):
                    L.append(edge)
    return L

def too_long_segments(geometry, criterion):
    L = []
    for i in range(len(geometry)-1):
        edge = [geometry[i], geometry[i+1]]
        d = distance(geometry[i], geometry[i+1])
        if d>criterion:
            if not segment_in_list(edge, L):
                L.append(edge)
    return L

def edge_triangle_belongs_geometry(geometry, triangle):
    geometry = geometry_to_polygon(geometry)
    triangle = geometry_to_polygon(triangle)
    for edge in triangle:
        if segment_in_list(edge, geometry):
            return True
    return False

def edge_triangle_belongs_geometries(geometries, triangle):
    for geometry in geometries:
        if edge_triangle_belongs_geometry(geometry, triangle):
            return True
    return False

#--- TRIANGULATION OPERATIONS FUNCTIONS --------------------------------------+
def cavity_index(triangulation, vertex):
    """
    Retourne la liste des indexs des triangles formant la cavité
    """
    index = []
    for i in range(len(triangulation)):
        triangle = triangulation[i]
        if (circumcircle(triangle) != False):
            circle = circumcircle(triangle)
            if (point_in_circle(vertex, circle)):
                index.append(i)
    return index

def point_belong_to_triangle(p, triangle):
    """
    Determine si un point est un sommet d'un triangle
    """
    for i in range(len(triangle)):
        point = triangle[i]
        if identical_points(p,point):
            return True
    return False

def edge_in_triangles(p1, p2, triangulation):
    """
    Determine si un segment appartient à un triangle
    """
    triangle_index=[]
    for i in range(len(triangulation)):
        triangle = triangulation[i]
        if ((point_belong_to_triangle(p1, triangle)) & (point_belong_to_triangle(p2, triangle))):
            triangle_index.append(i)
    return triangle_index

def find_next_edge(edge, edge_list):
    for i in range(len(edge_list)):
        if not identical_edges(edge, edge_list[i]):
            if point_in_list(edge[1], edge_list[i]):
                if identical_points(edge[1], edge_list[i][0]):
                    return i, edge_list
                else :
                    edge_list[i][0], edge_list[i][1] = edge_list[i][1], edge_list[i][0]
                    return i, edge_list
    return -1

def convex_cavity(triangulation, index):
    """
    Determine les segments de la triangulation qui forment la cavité
    """
    cavity=[]
    for i in index:
        triangle = triangulation[i]
        for j in range(3):
            p1= triangle[j]
            p2 = triangle[j+1]
            edge = [p1, p2]
            triangle_index = edge_in_triangles(p1, p2, triangulation)
            if (len(triangle_index)==1):
                cavity.append(edge)
            else:
                for l in triangle_index:
                    if (l!=i):
                        index_tri_op = l
                if (index_tri_op not in index):
                    cavity.append(edge)
    if (len(cavity)==0):
        return cavity
    i = -1
    j = -1
    index = []
    while (i<0):
        j+=1
        i, cavity = find_next_edge(cavity[j], cavity)
    index.append(j)
    k, cavity = find_next_edge(cavity[j], cavity)
    index.append(k)
    while(k!=j):
        k, cavity = find_next_edge(cavity[k], cavity)
        index.append(k)
    index.pop()
    ordered_cavity = []
    for l in index:
        ordered_cavity.append(cavity[l])
    cavity_points = [ordered_cavity[i][0] for i in range(len(ordered_cavity))]
    cavity_points.append(cavity_points[0])
    return cavity_points
    
def add_vertex(triangulation, vertex, geometries, animation):
    """
    Retourne la nouvelle triangulation suite à l'ajout d'un point
    """
    new_triangulation = []
    index = cavity_index(triangulation, vertex)
    # if animation:
    #     plot_triangulaion(triangulation, geometries)
    #     plt.plot(vertex[0], vertex[1], 'og',markersize=6)
    #     for i in index:
    #         fill_triangle(triangulation[i])
    #     plt.pause(0.1)
    cavity = convex_cavity(triangulation, index)
    for i in range(len(triangulation)):
        if i not in index:
            new_triangulation.append(triangulation[i])
    # if animation:
    #     plot_triangulaion(new_triangulation, geometries)
    #     plot_cavity(cavity,vertex)
    #     plt.plot(vertex[0], vertex[1], 'og',markersize=6)
    #     plt.pause(0.1)
    for j in range(len(cavity)-1):
        if not collinear(cavity[j][0], cavity[j][1], vertex[0], vertex[1], cavity[j+1][0], cavity[j+1][1]):
            new_triangle = [cavity[j], vertex, cavity[j+1], cavity[j]]
            new_triangulation.append(new_triangle)
    if animation:
        plot_triangulaion(new_triangulation, geometries)
        plt.pause(0.1)
    return new_triangulation


#--- GEOMETRY OPERATIONS FUNCTIONS -------------------------------------------+
def read_geometry(filename):
    """
    Renvoie la liste des points d'une géométrie
    renseignée sous la forme d'un fichier csv
    """
    df = pd.read_csv(filename, names = ['x','y','z'])
    df = df.drop('z', axis=1)
    df = df.iloc[::4]
    return (df.values.tolist())

def prepare_geometry(geometry, h):
    L = too_long_segments(geometry, h)
    old_geometry = geometry_to_polygon(geometry)
    while(len(L)!=0):
        new_geometry = []
        for edge in old_geometry:
            if not segment_in_list(edge, L):
                new_geometry.append(edge)
            else:
                vertex = midpoint(edge)
                new_geometry.append([edge[0],vertex])
                new_geometry.append([vertex,edge[1]])
        L = too_long_segments(polygon_to_geometry(new_geometry),h)
        old_geometry = new_geometry
    old_geometry = polygon_to_geometry(old_geometry)
    return old_geometry    

def prepare_geometries(geometries, h):
    corrected_geometries = []
    for geometry in geometries:
        corrected_geometries.append(prepare_geometry(geometry,h))
    return corrected_geometries

def super_triangle(geometry):
    """
    Calcule le super triangle englobant une géométrie
    """
    eps=0.1
    x,y=zip(*geometry)
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    A = [xmin-2*eps,ymin-eps]
    B = [xmin+2*(xmax-xmin)+3*eps,ymin-eps]
    C = [xmin-2*eps,ymin+2*(ymax-ymin)+3*eps]
    return ([A,B,C,A])

def point_belongs_to_geometry(point, geometry):
    for vertex in geometry:
        if identical_points(point, vertex):
            return True
    return False

def first_delaunay_triangulation(geometry, geometries, animation):
    geometry_vertex = [geometry[i] for i in range(len(geometry)-1)]
    delaunay_triangulation = [super_triangle(geometry)]
    for vertex in geometry_vertex:
        delaunay_triangulation = add_vertex(delaunay_triangulation, vertex, geometries, animation)
    return delaunay_triangulation

def remove_super_triangle(delaunay_triangulation, geometry):
    new_delaunay_triangulation = []
    for triangle in delaunay_triangulation:
        if point_belongs_to_geometry(triangle[0], geometry):
            if point_belongs_to_geometry(triangle[1], geometry):
                if point_belongs_to_geometry(triangle[2], geometry):
                    new_delaunay_triangulation.append(triangle)
    return new_delaunay_triangulation

def remove_out_triangles(delaunay_triangulation, geometry):
    new_delaunay_triangulation = []
    for j in range(len(delaunay_triangulation)):
        triangle = delaunay_triangulation[j]
        n = 0
        for i in range(len(triangle)-1):
            edge = [triangle[i],triangle[i+1]]
            mid = midpoint(edge)
            mid[0] = mid[0]+1e-10
            mid[1] = mid[1]+1e-10
            if not inside_geometry(mid, geometry):
                n=n+1
        if equal(n,0):
            new_delaunay_triangulation.append(triangle)
    return new_delaunay_triangulation

#--- CONSTRAINED BOUNDARIES FUNCTIONS ----------------------------------------+
def edge_is_subsegment(edge, segment, triangulation):
    if not identical_edges(segment, edge):
        if (is_between(segment[0], edge[0], segment[1])):
            if (is_between(segment[0], edge[1], segment[1])):
                return True
    return False

def edge_is_subsegment_geometry(edge, geometry, triangulation):
    for boundary in geometry:
        if edge_is_subsegment(edge, boundary, triangulation):
            return True
    return False
            

def subsegments(segment, triangulation):
    points = []
    subsegments = []
    missing_subsegments = []
    for triangle in triangulation:
        for i in range(len(triangle)-1):
            edge = [triangle[i], triangle[i+1]]
            point = triangle[i]
            if edge_is_subsegment(edge, segment, triangulation):
                if not segment_in_list(edge, subsegments):
                    subsegments.append(edge)
                if not point_in_list(edge[0], points):
                    points.append(edge[0])
                if not point_in_list(edge[1], points):
                    points.append(edge[1])
            if is_between(segment[0], point, segment[1]):
                if not point_in_list(point, points):
                    points.append(point)
    if len(points)==0:                  
        missing_subsegments.append(segment)
        return missing_subsegments
    d = [distance(segment[0], points[i]) for i in range(len(points))]
    points = np.array(points)
    points = points[np.argsort(d),:].tolist()
    for i in range(len(points)-1):
        edge = [points[i], points[i+1]]
        if not segment_in_list(edge, subsegments):
            if not segment_in_list(edge, missing_subsegments):
                missing_subsegments.append(edge)
    return missing_subsegments
                    
def segment_in_triangulation(segment, triangulation):
    for triangle in triangulation:
        for i in range(len(triangle)-1):
            edge = [triangle[i], triangle[i+1]]
            if identical_edges(segment, edge):
                return True
    missings = subsegments(segment, triangulation)
    if len(missings)==0:
        return True
    return False

def missing_boundaries(triangulation, geometry):
    geom = geometry_to_polygon(geometry)
    missing = []
    for boundary in geom:
        if not segment_in_triangulation(boundary, triangulation):
            missing.append(boundary)
    return missing

def constrained_triangulation(geometry, geometries, triangulation, animation):
    missings = missing_boundaries(triangulation, geometry)
    while len(missings)!=0:
        for missing in missings:
            missing_subsegments = subsegments(missing, triangulation)
            while(len(missing_subsegments)!=0):
                for missing_subsegment in missing_subsegments:
                    mid = midpoint(missing_subsegment)
                    triangulation = add_vertex(triangulation, mid, geometries, animation)
                missing_subsegments = subsegments(missing, triangulation)
        missings = missing_boundaries(triangulation, geometry)
    return triangulation

#--- ENCROACHED SEGMENTS FUNCTIONS -------------------------------------------+
def encroached(triangulation, edge):
    circle = circumcircle_edge(edge)
    for triangle in triangulation:
        for vertex in triangle:
            if not identical_points(vertex, edge[0]):
                if not identical_points(vertex, edge[1]):
                    if (point_in_circle(vertex, circle)):
                        return True
    return False

def point_encroached_upon(triangulation, vertex, t):
    t = geometry_to_polygon(t)
    for triangle in triangulation:
        for i in range(len(triangle)-1):
            edge = [triangle[i],triangle[i+1]]
            circle = circumcircle_edge(edge)
            if (point_in_circle(vertex, circle)) and (not segment_in_list(edge, t)):
                return edge
    return False

def encroached_segments(triangulation, geometry):
    geometry = geometry_to_polygon(geometry)
    segments = []
    for triangle in triangulation:
        for i in range(len(triangle)-1):
            edge = [triangle[i],triangle[i+1]]
            if encroached(triangulation, edge) and (segment_in_list(edge, geometry) or edge_is_subsegment_geometry(edge, geometry, triangulation)):
                if not segment_in_list(edge,segments):
                    segments.append(edge)
    return segments

def correct_encroached_segments(delaunay_triangulation, geometry, geometries, animation):
    already_good_triangulation = False
    segments = encroached_segments(delaunay_triangulation, geometry)
    if len(segments)==0:
        already_good_triangulation = True
        return delaunay_triangulation, already_good_triangulation
    while(len(segments)!=0):
        segment = segments[0]
        mid = midpoint(segment)
        delaunay_triangulation = add_vertex(delaunay_triangulation, mid, geometries, animation)
        segments = encroached_segments(delaunay_triangulation, geometry)
        progress_encorached_segments(len(segments))
    return delaunay_triangulation, already_good_triangulation

#--- TRIANGULATION QUALITY FUNCTIONS -----------------------------------------+
def minimum_angle(triangle):
    center,r = circumcircle(triangle)
    d = shortest_edge(triangle)
    theta = np.arcsin(d/(2*r))
    return theta   

def worst_triangle(index, triangulation):
    triangle = triangulation[index[0]]
    worst = triangle
    theta = minimum_angle(triangle)
    for i in index:
        triangle = triangulation[i]
        if minimum_angle(triangle)<theta:
            theta = minimum_angle(triangle)
            worst = triangle
    return worst
            

def good_quality_measure(triangle, min_angle):
    theta = np.rad2deg(minimum_angle(triangle))
    if theta<min_angle:
        return False
    return True

def bad_triangles(triangulation, min_angle):
    index =[]
    for i in range(len(triangulation)):
        triangle = triangulation[i]
        if not good_quality_measure(triangle, min_angle):
            index.append(i)
    return index

def correct_bad_triangles(delaunay_triangulation, geometry, geometries, min_angle, animation):
    already_good_triangulation = False
    index = bad_triangles(delaunay_triangulation, min_angle)
    if len(index)==0:
        already_good_triangulation = True
        return delaunay_triangulation, already_good_triangulation
    while (len(index) != 0):
        triangle = worst_triangle(index, delaunay_triangulation)
        center,r = circumcircle(triangle)
        if (point_encroached_upon(delaunay_triangulation, center, triangle)!=False):
            edge = point_encroached_upon(delaunay_triangulation, center, triangle)
            mid = midpoint(edge)
            delaunay_triangulation = add_vertex(delaunay_triangulation, mid, geometries, animation)
        else:
            if outside_geometries(center, geometries):
                for i in range(len(triangle)-1):
                    polygons = [geometry_to_polygon(geometries[j]) for j in range(len(geometries))]
                    for polygon in polygons:
                        if point_belongs_to_polygon(triangle[i], polygon) and point_belongs_to_polygon(triangle[i+1], polygon):
                            edge = [triangle[i],triangle[i+1]]
                            mid = midpoint(edge)
                            delaunay_triangulation = add_vertex(delaunay_triangulation, mid, geometries, animation)
            else:
                delaunay_triangulation = add_vertex(delaunay_triangulation, center, geometries, animation)
        index = bad_triangles(delaunay_triangulation, min_angle)
        progress_bad_triangles(len(index))
    return delaunay_triangulation, already_good_triangulation

def correct_mesh_size(triangulation, geometries, animation, h):
    already_good_triangulation = False
    d, edge = longest_edges(triangulation)
    L = too_long_edges(triangulation, h)
    if len(L)==0:
        already_good_triangulation = True
        return triangulation, already_good_triangulation
    while (len(L)!=0):
        vertex = midpoint(edge)
        triangulation = add_vertex(triangulation, vertex, geometries, animation)
        d, edge = longest_edges(triangulation)
        L = too_long_edges(triangulation, h)
        progress_mesh_size(len(L))
    return triangulation, already_good_triangulation

#--- CREATE DELAUNAY TRIANGULATION -------------------------------------------+
def make_delaunay_triangulation(geometry, geometries, min_angle, animation):
    dt = first_delaunay_triangulation(geometry, geometries, animation)
    dt = remove_super_triangle(dt, geometry)
    dt = remove_out_triangles(dt, geometry)
    plot_triangulaion(dt, geometries)
    plt.pause(0.1)
    dt = constrained_triangulation(geometry, geometries, dt, animation)
    return dt

def multiple_geometries_delaunay_triangulation(geometries, min_angle, centers, animation, h):
    geometry = geometries[0]
    dt = make_delaunay_triangulation(geometry, geometries, min_angle, animation)
    for i in range(len(geometries)-1):
        i=i+1
        geometry = geometries[i]
        for j in range(len(geometry)-1):
            vertex = geometry[j]
            dt = add_vertex(dt, vertex, geometries, animation)
            t = "Geometry "+str(i)+" : "
            percent = int(((j+1)/(len(geometry)-1))*100)
            progress(t, percent)
        plot_triangulaion(dt, geometries)
        plt.pause(0.1)
    # dt, state_encroached_segments = correct_encroached_segments(dt, geometry, geometries, animation)
    dt, state_mesh_size = correct_mesh_size(dt, geometries, animation, h)
    dt, state_bad_triangles = correct_bad_triangles(dt, geometry, geometries, min_angle, animation)
    plot_triangulaion(dt, geometries)
    plt.pause(0.1)
    state = state_bad_triangles and state_mesh_size # and state_encroached_segments
    while (state!=True):
        for k in range(len(geometries)):
            dt = constrained_triangulation(geometries[k], geometries, dt, animation)
            t = "Constrained Triangulation  : "
            percent = int(((k+1)/(len(geometries)))*100)
            progress(t, percent)
        plot_triangulaion(dt, geometries)
        plt.pause(0.1)
        # dt, state_encroached_segments = correct_encroached_segments(dt, geometry, geometries, animation)
        dt, state_mesh_size = correct_mesh_size(dt, geometries, animation, h)
        dt, state_bad_triangles = correct_bad_triangles(dt, geometry, geometries, min_angle, animation)
        state = state_bad_triangles and state_mesh_size # and state_encroached_segments
        plot_triangulaion(dt, geometries)
        plt.pause(0.1)
    save_triangulation(dt, "Oven_20_refined.csv")
    plot_triangulaion(dt, geometries)
    return dt

def save_triangulation(triangulation, name):
    df = pd.DataFrame(columns=['p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y'])
    for triangle in triangulation:
        p1 = triangle[0]
        p1_x = round(p1[0],15)
        p1_y = round(p1[1],15)
        p2 = triangle[1]
        p2_x = round(p2[0],15)
        p2_y = round(p2[1],15)
        p3 = triangle[2]
        p3_x = round(p3[0],15)
        p3_y = round(p3[1],15)
        p4 = triangle[3]
        p4_x = round(p4[0],15)
        p4_y = round(p4[1],15)
        new_row = {'p1_x':p1_x, 'p1_y':p1_y, 'p2_x':p2_x, 'p2_y':p2_y, 'p3_x':p3_x, 'p3_y':p3_y, 'p4_x':p4_x, 'p4_y':p4_y}
        df = df.append(new_row, ignore_index=True)
    df.to_csv(name, index = False)

#--- PLOT FUNCTIONS ----------------------------------------------------------+
def fill_triangle(triangle):
    x = [triangle[i][0] for i in range(len(triangle))]
    y = [triangle[i][1] for i in range(len(triangle))]
    plt.fill(x, y)

def plot_triangulaion(triangulation, geometries):
    """
    Trace la triangulation finale
    """
    for triangle in triangulation:
        x,y = zip(*triangle)
        plt.plot(x, y, 'b', linewidth=0.1)
        # plt.plot(x, y, 'ok',markersize=0.1)
    for geometry in geometries:
        xGeom,yGeom = zip(*geometry)
        plt.plot(xGeom, yGeom, 'k', linewidth=0.2)
        plt.plot(xGeom,yGeom, 'ok',markersize=0.2)
    # plt.xlim(0.65,0.81)
    # plt.ylim(0.65,0.81)
    # plt.gca().axis('equal')
    plt.savefig('Oven_20_refined.eps', format = 'eps', dpi=1200)
    
def plot_cavity(cavity, vertex):
    """
    Trace la cavité associé au placement d'un nouveau point
    """
    cavity = [[cavity[i],cavity[i+1]] for i in range(len(cavity)-1)]
    for edge in cavity:
        x,y = zip(*edge)
        plt.plot(x, y, 'r', linewidth=1)
        plt.plot(x, y, 'ok',markersize=1)
    plt.xlim(vertex[0]-0.1,vertex[0]+0.1)
    plt.ylim(vertex[1]-0.1,vertex[1]+0.1)

#--- MAIN: ONE GEOETRY -------------------------------------------------------+
# geom = [[0,0],
#         [0,1],
#         [0.5,0.25],
#         [0.5,1],
#         [1,0.25],
#         [1,1],
#         [1.75,0],
#         [0,0]]

# path = '/Users/up/Desktop/Mines/Eléments Finis/Implementation/'
# filename = 'Input/NACA64.csv'
# geom = read_geometry(path+filename)

# anim = False
# minimum_required_angle = 0

# out = make_delaunay_triangulation(geom, minimum_required_angle, anim)
# plot_triangulaion(out, geom)

#--- MAIN: MULTIPLE GEOETRIES ------------------------------------------------+
cavity = box([-1,-1],2,2)

geom = [cavity] #, piece, c[0], c[1], c[2], c[3]]

anim = False
mesh_size = 0.25
minimum_required_angle = 20

# geom = prepare_geometries(geom, mesh_size)
# out = multiple_geometries_delaunay_triangulation(geom, minimum_required_angle, P, anim, mesh_size)

#--- TEST --------------------------------------------------------------------+