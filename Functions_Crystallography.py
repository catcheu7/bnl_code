import numpy as np
import xraylib as xb
# from xraylib import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from crystals import Crystal
import tifffile as tif
import imutils as imu
from time import time, localtime, strftime

from multiprocessing import Pool, cpu_count
from functools import partial

from Functions_General_Geometry import *
from Functions_File_Operation import *


def CalLat_vectors(a=5.431, b=5.431, c=5.431, alpha=90, beta=90, gamma=90, display=False): 
    ''' >>> Instruction <<< 
        This function calculates the real and reciprocal vectors of a lattice. 
        
        The input a, b, and c are in Angstrom. The alpha, beta, and gamma are in degree. 
            alpha is the angle between b and c
            beta  is the angle between c and a
            gamma is the angle between a and b
        
        The output is {'Direct': [a1, a2, a3], 'Reciprocal': [b1, b2, b3]}
        
        Both direct lattice vectors [a1, a2, a3] and reciprocal lattice vectors [b1, b2, b3]
        are described in a right-handed Cartesian coordinates [X, Y, Z]. [a1, a2, a3] are 
        aligned as following: 
                a1 is along the X axis, 
                    i.e. [a*X, 0*Y, 0*Z]
                a2 is in the X-Y plane with an angle gamma related to a1, 
                    i.e. [b*cos*(gamma)*X, b*sin(gamma)*Y, 0*Z]
                a3 is defined by a1, a2, alpha, and beta, 
                    i.e. [c*cos(beta)*X, 
                          c*[cos(alpha)-cos(beta)*cos(gamma)], 
                          c*sqrt( sin(beta)**2 - [cos(alpha) - cos(beta)*cos(gamma)]**2 )]
    '''
    # Initialization
    a1 = np.zeros(3)
    a2 = np.zeros(3)
    a3 = np.zeros(3)
    b1 = np.zeros(3)
    b2 = np.zeros(3)
    b3 = np.zeros(3)
    
    alpha = np.deg2rad(alpha)
    beta  = np.deg2rad(beta )
    gamma = np.deg2rad(gamma)
    
    # Define direct lattice vectors
    a1[0] = a
    a2[0] = b * np.cos(gamma)
    a2[1] = b * np.sin(gamma)
    a3[0] = c * np.cos(beta)
    a3[1] = c * ( np.cos(alpha) - np.cos(beta) * np.cos(gamma) )
    a3[2] = c * np.sqrt(np.sin(beta)**2 - (np.cos(alpha) - np.cos(beta) * np.cos(gamma))**2)
    
    # Calculate reciprocal lattice vectors
    b1 = 2*np.pi * np.cross(a2, a3) / (np.dot( a1, np.cross(a2, a3) ))
    b2 = 2*np.pi * np.cross(a3, a1) / (np.dot( a2, np.cross(a3, a1) ))
    b3 = 2*np.pi * np.cross(a1, a2) / (np.dot( a3, np.cross(a1, a2) ))
    
    # Plot the result
    if display: 
        # Initialize figure and plot the origin point
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(0, 0, 0, color='k')
        ax.text(0, 0, 0, 'O')
        # Plot and label the direct vectors
        ax.scatter(a1[0], a1[1], a1[2], color='r')
        ax.text(a1[0], a1[1], a1[2], 'a1')
        ax.plot([0, a1[0]], [0, a1[1]], [0, a1[2]], color='r')
        ax.scatter(a2[0], a2[1], a2[2], color='r')
        ax.text(a2[0], a2[1], a2[2], 'a2')
        ax.plot([0, a2[0]], [0, a2[1]], [0, a2[2]], color='r')
        ax.scatter(a3[0], a3[1], a3[2], color='r')
        ax.text(a3[0], a3[1], a3[2], 'a3')
        ax.plot([0, a3[0]], [0, a3[1]], [0, a3[2]], color='r')
        # Plot and label the reciprocal vectors
        ax.scatter(b1[0], b1[1], b1[2], color='b')
        ax.text(b1[0], b1[1], b1[2], 'b1')
        ax.plot([0, b1[0]], [0, b1[1]], [0, b1[2]], color='b')
        ax.scatter(b2[0], b2[1], b2[2], color='b')
        ax.text(b2[0], b2[1], b2[2], 'b2')
        ax.plot([0, b2[0]], [0, b2[1]], [0, b2[2]], color='b')
        ax.scatter(b3[0], b3[1], b3[2], color='b')
        ax.text(b3[0], b3[1], b3[2], 'b3')
        ax.plot([0, b3[0]], [0, b3[1]], [0, b3[2]], color='b')
        # Show the image and adjust the aspect ratio
        plt.show()
        # ax.set_aspect('equal')
        
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5*max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    return {'Direct': [a1, a2, a3], 'Reciprocal': [b1, b2, b3]}


def CalLat_vectors_rot(vectors, matrix=None): 
    ''' >>> Instruction <<< 
        This function takes the crystal rotation into account and generates the direct and 
        reciprocal lattice vectors after rotating the lattice
        
        Inputs: 
            vectors      Should use the output of function CalLat_vectors().
            matrix       (3,3) matrix. From function Rot_Matrix().
        
        The output is {'Direct': [a1, a2, a3], 'Reciprocal': [b1, b2, b3]}
    '''
    # Initialization
    if matrix is None: 
        return vectors
    
    a1 = vectors['Direct'][0]
    a2 = vectors['Direct'][1]
    a3 = vectors['Direct'][2]
    b1 = vectors['Reciprocal'][0]
    b2 = vectors['Reciprocal'][1]
    b3 = vectors['Reciprocal'][2]
    
    new = Rotate_vectors([a1,a2,a3,b1,b2,b3], matrix)
    return {'Direct': new[:3], 'Reciprocal': new[3:]}


def CalLat_plane_and_gv(HKL=[1, 0, 0], direct=[[1,0,0], [0,1,0], [0,0,1]], 
                        reciprocal=[[1,0,0], [0,1,0], [0,0,1]], display=False):
    ''' >>> Instruction <<< 
        Input:
            'HKL'           Miller indices
            'direct'        Direct lattice unit vectors
            'reciprocal'    Reciprocal lattice unit vectors
        
        Output: {'Plane_Normal': Plane[:3], 'G_vector': Peak}
            'Plane_Normal'  Normal vector of the lattice plane
            'G_Vector'      The corresponding G vector
    '''
    
    h = HKL[0]
    k = HKL[1]
    l = HKL[2]
    
    a1 = np.asarray(direct[0])
    a2 = np.asarray(direct[1])
    a3 = np.asarray(direct[2])
    
    b1 = np.asarray(reciprocal[0])
    b2 = np.asarray(reciprocal[1])
    b3 = np.asarray(reciprocal[2])
    
    point1 = np.asarray([0, 0, 0])
    point2 = np.asarray([0, 0, 0])
    point3 = np.asarray([0, 0, 0])
    
    # Calculate the intersection points and plane normal in the direct space
    if h == 0 and k == 0 and l == 0 : 
        print('>>>>>> Wrong Miller indices!!! <<<<<<')
        return
    elif k == 0 and l == 0:
        Normal = a1
        point1 = a1/h
        Plane = Plane_func('Normal_and_Point', Normal, point1)
    elif l == 0 and h == 0:
        Normal = a2
        point2 = a2/k
        Plane = Plane_func('Normal_and_Point', Normal, point2)
    elif h == 0 and k == 0:
        Normal = a3
        point3 = a3/l
        Plane = Plane_func('Normal_and_Point', Normal, point3)  
    elif h == 0: 
        point2 = a2/k
        point3 = a3/l
        vector = point2 - point3
        Plane = Plane_func('Vectors', vector, a1, point2)
        Normal = Plane[:3]
    elif k == 0:
        point1 = a1/h
        point3 = a3/l
        vector = point3 - point1
        Plane = Plane_func('Vectors', vector, a2, point3)
        Normal = Plane[:3]
    elif l == 0:
        point1 = a1/h
        point2 = a2/k
        vector = point1 - point2
        Plane = Plane_func('Vectors', vector, a3, point1)
        Normal = Plane[:3]
    else: 
        point1 = a1/h
        point2 = a2/k
        point3 = a3/l
        Plane = Plane_func('Points', point1, point2, point3)
        Normal = Plane[:3]

    # Peak in the reciprocal space
    Peak = h*b1 + k*b2 + l*b3
    
    # Plot the lattice plane and Bragg peak
    if display: 
        # Initialize the figure and plot the origin point
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(0, 0, 0, color='k')
        ax.text(0, 0, 0, 'O')
        # plot the axes of the direct space
        ax.scatter(a1[0], a1[1], a1[2], color='r')
        ax.text(a1[0], a1[1], a1[2], 'a1')
        ax.plot([0, a1[0]], [0, a1[1]], [0, a1[2]], color='r')
        ax.scatter(a2[0], a2[1], a2[2], color='r')
        ax.text(a2[0], a2[1], a2[2], 'a2')
        ax.plot([0, a2[0]], [0, a2[1]], [0, a2[2]], color='r')
        ax.scatter(a3[0], a3[1], a3[2], color='r')
        ax.text(a3[0], a3[1], a3[2], 'a3')
        ax.plot([0, a3[0]], [0, a3[1]], [0, a3[2]], color='r')
        # Show the image and adjust the aspect ratio
        plt.show()
        # ax.set_aspect('equal')
        
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5*max([x_range, y_range, z_range])
        
        xx_min = x_middle - plot_radius
        xx_max = x_middle + plot_radius
        yy_min = y_middle - plot_radius
        yy_max = y_middle + plot_radius
        zz_min = x_middle - plot_radius
        zz_max = x_middle + plot_radius
        
        # Plot and label the reciprocal vectors
        ax.scatter(b1[0], b1[1], b1[2], color='b')
        ax.text(b1[0], b1[1], b1[2], 'b1')
        ax.plot([0, b1[0]], [0, b1[1]], [0, b1[2]], color='b')
        ax.scatter(b2[0], b2[1], b2[2], color='b')
        ax.text(b2[0], b2[1], b2[2], 'b2')
        ax.plot([0, b2[0]], [0, b2[1]], [0, b2[2]], color='b')
        ax.scatter(b3[0], b3[1], b3[2], color='b')
        ax.text(b3[0], b3[1], b3[2], 'b3')
        ax.plot([0, b3[0]], [0, b3[1]], [0, b3[2]], color='b')
        
        # plot the G vector
        Intersection = Inter_plane_and_line(Plane, np.asarray([0, 0, 0]), Peak)
        ax.scatter(Intersection[0] , Intersection[1] , Intersection[2],  color='gray')
        ax.plot([0, Intersection[0]], [0, Intersection[1]], [0, Intersection[2]], dashes=[6, 2], color='gray')
        
        ax.scatter(Peak[0] , Peak[1] , Peak[2],  color='green')
        ax.plot([0, Peak[0]], [0, Peak[1]], [0, Peak[2]], color='green')
        
        # Show the image and adjust the aspect ratio
        plt.show()
        # ax.set_aspect('equal')
        
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5*max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
        # plot the intersections of the lattice plane and axes
        ax.scatter(point1[0] , point1[1] , point1[2],  color='gray')
        ax.scatter(point2[0] , point2[1] , point2[2],  color='gray')
        ax.scatter(point3[0] , point3[1] , point3[2],  color='gray')
        
        # plot the lattice plane
        Distance = - Plane[3] / np.sqrt(Plane[0]**2 + Plane[1]**2 + Plane[2]**2)
        xx, yy, zz = Generate_plane(xx_max-xx_min, yy_max-yy_min, 10, 10, Normal, Distance)
            
        fig = plt.gca(projection='3d')
        fig.plot_surface(xx, yy, zz, alpha=0.2)
        
    return {'Plane_Normal': Plane[:3]/np.linalg.norm(Plane[:3]), 'G_Vector': Peak}


def CalLat_HKLs(Type='P', H_max=3, K_max=3, L_max=3, Progress=False): 
    ''' >>> Instruction <<< 
        Generate the Miller indices
        
        Input:
            Type       Unit cell type, one of the following: 
               'P'        Primitive
               'BC'       Body-centered
               'FC'       Face-centered
               'Diamond'  FCC Diamond
               'HCP'      Hexagonal
            H/K/L_max  Maximum number for each index
            
        Output: 
            HKLs       (n, 3) array, Miller indices of the diffraction peaks. 
    '''
    H_max = int(H_max)
    K_max = int(K_max)
    L_max = int(L_max)
    
    HKLs = np.zeros((1, 3))
    counter = 0
    
    if Progress: 
        if Type == 'BC':
            print('Body-centered: H + K + L is even.')
        if Type == 'FC':
            print('Face-centered: H, K, L all odd or all even.')
        if Type == 'Diamond':
            print('FCC Diamond: H, K, L all odd, or, all even and H + K + L = 4n.')
        if Type == 'HCP':
            print('Hexagonal: L is even, or, H + 2K != 3n.')
    
    if H_max == 0: 
        H_list = [0]
    else: 
        H_list = np.linspace(-H_max, H_max, 2*H_max+1).astype(int)
    if K_max == 0: 
        K_list = [0]
    else: 
        K_list = np.linspace(-K_max, K_max, 2*K_max+1).astype(int)
    if L_max == 0: 
        L_list = [0]
    else: 
        L_list = np.linspace(-L_max, L_max, 2*L_max+1).astype(int)
    
    for h in H_list: 
        for k in K_list:
            for l in L_list:
                # Selection rules
                if h == 0 and k == 0 and l == 0: 
                    continue
                if Type == 'BC': 
                    if np.mod(h+k+l, 2) == 0: 
                        if Progress: 
                            print('[%d %d %d]' % (h, k, l) )
                    else:
                        continue
                if Type == 'FC':
                    if np.mod(h, 2) == np.mod(k, 2) and np.mod(h, 2) == np.mod(l, 2) :
                        if Progress: 
                            print('[%d %d %d]' % (h, k, l) )
                    else: 
                        continue
                if Type == 'Diamond': 
                    if np.mod(h, 2) == 1 and np.mod(k, 2) == 1 and np.mod(l, 2) == 1:
                        if Progress: 
                            print('[%d %d %d]' % (h, k, l) )
                    elif np.mod(h, 2) == 0 and np.mod(k, 2) == 0 and np.mod(l, 2) == 0:
                        if np.mod(h+k+l, 4) == 0:
                            if Progress: 
                                print('[%d %d %d]' % (h, k, l) )
                        else: 
                            continue
                    else: 
                        continue
                if Type == 'HCP': 
                    if np.mod(l, 2) == 1: 
                        if np.mod(h + 2*k, 3) == 0: 
                            continue
                    if Progress: 
                        print('[%d %d %d]' % (h, k, l) )
                    
                # Save current HKL to the result
                counter = counter + 1
                HKL = np.asarray([[h, k, l]])
                if counter == 1: 
                    HKLs = HKL
                else: 
                    HKLs = np.append(HKLs, HKL, 0)
    
    return HKLs


def CalLat_reciprocal_peaks(HKLs=[[0, 0, 1],[1, 1, 1]], direct=[[1,0,0], [0,1,0], [0,0,1]], 
                            reciprocal=[[1,0,0], [0,1,0], [0,0,1]], display=False): 
    ''' >>> Instruction <<< 
        Input:
            HKLs            (n, 3) array, Miller indices of the diffraction peaks. 
            direct          Direct lattice unit vectors
            reciprocal      Reciprocal lattice unit vectors
        
        Output: 
            G_vectors       (n, 3) array, coordinates of the reciprocal peaks. 
    '''
    HKLs = np.asarray(HKLs)
    
    b1 = np.asarray(reciprocal[0])
    b2 = np.asarray(reciprocal[1])
    b3 = np.asarray(reciprocal[2])
    
    if np.size(np.shape(HKLs)) > 2: 
        print(">>>>>> Input HKL's dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        return
    if np.shape(HKLs)[0] == 3:
        n = int( np.size(HKLs) / 3 )
        Input = HKLs.transpose()
    elif np.shape(HKLs)[1] == 3: 
        n = int( np.shape(HKLs)[0] )
        Input = HKLs
    else: 
        print(">>>>>> Input HKL's size is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        return
    
    G_vectors = np.zeros((n, 3))
    
    for i in range(n): 
        temp = CalLat_plane_and_gv(HKL=HKLs[i], direct=direct, reciprocal=reciprocal, display=False)
        G_vectors[i] = temp['G_Vector']
    
    # Plot the reciprocal peaks
    if display: 
        # Initialize the figure and plot the origin point
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(0, 0, 0, color='k')
        ax.text(0, 0, 0, 'O')
        
        # Plot and label the reciprocal vectors
        ax.scatter(b1[0], b1[1], b1[2], color='b')
        ax.text(b1[0], b1[1], b1[2], 'b1')
        ax.plot([0, b1[0]], [0, b1[1]], [0, b1[2]], color='b')
        ax.scatter(b2[0], b2[1], b2[2], color='b')
        ax.text(b2[0], b2[1], b2[2], 'b2')
        ax.plot([0, b2[0]], [0, b2[1]], [0, b2[2]], color='b')
        ax.scatter(b3[0], b3[1], b3[2], color='b')
        ax.text(b3[0], b3[1], b3[2], 'b3')
        ax.plot([0, b3[0]], [0, b3[1]], [0, b3[2]], color='b')
        
        # plot the reciprocal peaks
        for i in range(n): 
            ax.scatter(G_vectors[i][0], G_vectors[i][1], G_vectors[i][2],  color='green')
            ax.text(G_vectors[i][0], G_vectors[i][1], G_vectors[i][2], 
                    '[%d %d %d]' %(HKLs[i][0], HKLs[i][1], HKLs[i][2]) )
        
        # Show the image and adjust the aspect ratio
        plt.show()
        # ax.set_aspect('equal')
        
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5*max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
    return G_vectors


def CalLat_lit_peaks(axis=[0,1,0], theta=0, PhotonE=12, G_vectors=[[0,0,1]], thres=0.1, IsRadian=True): 
    ''' >>> Instruction <<< 
        This function determines whether a peak is 'lit', given a rocking
        axis and a rocking angle. 
        
        Input:
            axis            rocking axis, format np.asarray([x, y, z])
            theta           rocking angle
            PhotonE         photon energy [keV]
            G_vectors       (n, 3) array, coordinates of the reciprocal peaks when theta = 0 
            thres           a peak is 'lit' if the distance between its G and Ewald sphere is 
                            less than this threshold
        Output: 
            lit             (n,), a boolean array as the status of the peaks. 
    '''
    G_vectors = np.asarray(G_vectors)
    N = np.shape(G_vectors)[0]
    dist = np.zeros(N)
    lit = np.zeros(N)
    
    matrix = Rot_Matrix(axis, theta, IsRadian=IsRadian)
    current_G = Rotate_vectors(G_vectors, matrix)
    
    WaveLen = 12.3980 / PhotonE   # unit is [A]
    WaveK   = 2 * np.pi / WaveLen
    Ewald_sph = np.asarray([0, 0, -WaveK, WaveK])
    
    for i in range(N): 
        dist[i] = Dist_point_to_sphere(current_G[i], Ewald_sph)
    print(dist)
    lit = (np.abs(dist) <= thres)
    return lit


def CalLat_Ewald_and_peaks(rot=[0, 0, 0], PhotonE=12, HKLs=[[0,0,1]], G_vectors=[[0,0,1]], 
                           reciprocal=[[1,0,0], [0,1,0], [0,0,1]], 
                           regular=False, display=False):
    ''' >>> Instruction <<< 
        Plot the reciprocal peaks and the Ewald sphere
        
        Input:
            rot = [alpha, beta, gamma]
            Rotations of the crystal:
               alpha is the rotation angle around x axis, in degree.
               beta  is the rotation angle around y axis, in degree.
               gamma is the rotation angle around z axis, in degree.
            
            PhotonE         Photon energy
            HKLs            (n, 3) array, Miller indices of the diffraction peaks. 
            G_vectors       (n, 3) array, coordinates of the reciprocal peaks. 
            reciprocal      Reciprocal lattice unit vectors
        
        Output: 
            G_vectors       (n, 3) array, coordinates of the reciprocal peaks after rotation. 
    '''    
    # Input Regularization
    if regular:
        if np.size(np.shape(HKLs)) > 2: 
            print(">>>>>> Input HKLs' dimension is wrong! <<<<<<")
            print(">>>>>> It should be a n by 3 array. <<<<<<")
            return
        if np.size(np.shape(G_vectors)) > 2: 
            print(">>>>>> Input G_vectors' dimension is wrong! <<<<<<")
            print(">>>>>> It should be a n by 3 array. <<<<<<")
            return
        if np.shape(HKLs)[0] == 3:
            n = int( np.size(HKLs) / 3 )
            HKLs = HKLs.transpose()
        elif np.shape(HKLs)[1] == 3: 
            n = int( np.shape(HKLs)[0] )
        else: 
            print(">>>>>> Input HKLs' size is wrong! <<<<<<")
            print(">>>>>> It should be a n by 3 array. <<<<<<")
            return
        if np.shape(G_vectors)[0] == 3:
            m = int( np.size(G_vectors) / 3 )
            G_vectors = G_vectors.transpose()
        elif np.shape(G_vectors)[1] == 3: 
            m = int( np.shape(G_vectors)[0] )
        else: 
            print(">>>>>> Input G_vectors' size is wrong! <<<<<<")
            print(">>>>>> It should be a n by 3 array. <<<<<<")
            return
        if not m == n: 
            print(">>>>>> HKLs' size doesn't match G_vectors' size! <<<<<<")
            return
    else: 
        n = int( np.shape(HKLs)[0] )
    
    # Calculate the wave length and vector
    WaveLen = 12.3980 / PhotonE   # unit is [A]
    WaveK   = 2 * np.pi / WaveLen
    Ewald_center = np.asarray([0, 0, -WaveK])
    
    # Rotation matrices
    Rx = Rot_Matrix([1, 0, 0], np.deg2rad(rot[0]))
    Ry = Rot_Matrix([0, 1, 0], np.deg2rad(rot[1]))
    Rz = Rot_Matrix([0, 0, 1], np.deg2rad(rot[2]))
    
    # Reciprocal lattice unit vectors
    b1 = np.asarray(reciprocal[0])
    b2 = np.asarray(reciprocal[1])
    b3 = np.asarray(reciprocal[2])
    
    # Rotate the unit vectors
    b1 = np.asarray(np.matmul(Rz, np.matmul(Ry, np.matmul(Rx, b1).transpose()))).reshape(3,)
    b2 = np.asarray(np.matmul(Rz, np.matmul(Ry, np.matmul(Rx, b2).transpose()))).reshape(3,)
    b3 = np.asarray(np.matmul(Rz, np.matmul(Ry, np.matmul(Rx, b3).transpose()))).reshape(3,)
    
    # Rotate the reciprocal peaks
    G_vectors = np.matmul(Rx, G_vectors.transpose())
    G_vectors = np.matmul(Ry, G_vectors)
    G_vectors = np.matmul(Rz, G_vectors)
    G_vectors = G_vectors.transpose()
    
    # Plot the result
    if display: 
        # Initialize the figure and plot the origin point
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(0, 0, 0, color='k')
        ax.text(0, 0, 0, 'O')
        
        # Plot and label the reciprocal vectors
        ax.scatter(b1[0], b1[1], b1[2], color='b')
        ax.text(b1[0], b1[1], b1[2], 'b1')
        ax.plot([0, b1[0]], [0, b1[1]], [0, b1[2]], color='b')
        ax.scatter(b2[0], b2[1], b2[2], color='b')
        ax.text(b2[0], b2[1], b2[2], 'b2')
        ax.plot([0, b2[0]], [0, b2[1]], [0, b2[2]], color='b')
        ax.scatter(b3[0], b3[1], b3[2], color='b')
        ax.text(b3[0], b3[1], b3[2], 'b3')
        ax.plot([0, b3[0]], [0, b3[1]], [0, b3[2]], color='b')
        
        # plot the reciprocal peaks
        for i in range(n): 
            dist = np.linalg.norm(G_vectors[i] - Ewald_center)
            if np.absolute(dist-WaveK) > 0.005 * WaveK: 
                ax.scatter(G_vectors[i, 0], G_vectors[i, 1], G_vectors[i, 2],  color='green')
                # ax.plot([0, G_vectors[i, 0]], [0, G_vectors[i, 1]], [0, G_vectors[i, 2]],  color='blue')
            else:
                ax.scatter(G_vectors[i, 0], G_vectors[i, 1], G_vectors[i, 2],  color='red')
                Ki = np.asarray( np.asarray([0, 0, 0]) - Ewald_center)
                Kf = np.asarray( G_vectors[i] - Ewald_center).reshape(3, 1)
                Twotheta = np.arccos( np.dot(Ki, Kf) / (np.linalg.norm(Ki) * np.linalg.norm(Kf)) )
                print('[%d %d %d] \t 2-Theta angle: %.3f' % (HKLs[i][0], HKLs[i][1], HKLs[i][2], np.rad2deg(Twotheta)) )
            
            ax.text(G_vectors[i, 0], G_vectors[i, 1], G_vectors[i, 2], 
                    '[%d %d %d]' %(HKLs[i][0], HKLs[i][1], HKLs[i][2]) )
        
        # Show the image and adjust the aspect ratio
        plt.show()
        # ax.set_aspect('equal')
        
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5*max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
        # plot the Ewald sphere
        xx, yy, zz = Generate_sphere(R=WaveK, C=Ewald_center, Nu=30, Nv=15)
        
        idx = ( (xx <= x_middle - 2*plot_radius) | (xx >= x_middle + 2*plot_radius) | 
                (yy <= y_middle - 2*plot_radius) | (yy >= y_middle + 2*plot_radius) |
                (zz <= z_middle - 2*plot_radius) | (zz >= z_middle + 2*plot_radius) )
        zz[idx] = np.nan
        
        fig = plt.gca(projection='3d')
        fig.plot_surface(xx, yy, zz, alpha=0.2)
        
    return G_vectors


def CalLat_grain_mapping(rot=[0, 0, 0], PhotonE=12, E_bandwidth=0.05, 
                         HKLs=[[0,0,1]], G_vectors=[[0,0,1]], 
                         reciprocal=[[1,0,0], [0,1,0], [0,0,1]], 
                         regular=False, display=False, label_peak=False, 
                         lit_peak_only=False):
    ''' >>> Instruction <<< 
        This function is modified from 'CalLat_Ewald_and_peaks'. It identifies Bragg peaks that
        can be lit by an incident X-ray beam with certain photon energy bandwidth.  
        
        Input:
            rot = [alpha, beta, gamma]
            Rotations of the crystal:
               alpha is the rotation angle around x axis, in degree.
               beta  is the rotation angle around y axis, in degree.
               gamma is the rotation angle around z axis, in degree.
            
            PhotonE         Photon energy
            E_bandwidth      Energy bandwidth is [PhotonE - E_bandwidth/2, PhotonE + E_bandwidth/2]
            HKLs            (n, 3) array, Miller indices of the diffraction peaks. 
            G_vectors       (n, 3) array, coordinates of the reciprocal peaks. 
            reciprocal      Reciprocal lattice unit vectors
        
        Output: 
            G_vectors       (n, 3) array, coordinates of the reciprocal peaks after rotation. 
            
        Examples: 
            1) Mapping peaks by rotating the crystal 
                # Graing-mapping parameters 
                N = 1800
                Angles = np.linspace(0,180,N)
                RotAx  = [[0, 1, 0]]
                CIF_file = 'Al.cif'
                PhotonE = 15    # [keV]
                Type = 'FC'
                H_max = 7
                K_max = 7
                L_max = 7
                E_bandwidth = 0.05

                # Calculate constants 
                Rotation = np.matmul(Angles.reshape((N,1)), np.asarray(RotAx))
                a, b, c, alpha, beta, gamma = CalLat_read_cif(CIF_name=CIF_file, 
                                    CIF_path=r'D:\GY\Dropbox\BNL-HXN\Detector_Fitting\CIFs')['UnitCellParams']
                Spaces = CalLat_vectors(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, display=False)
                HKLs = CalLat_HKLs(H_max=H_max, K_max=K_max, L_max=L_max, Type=Type, Progress=False)
                TwoThetas = CalLat_2Theta(HKL=HKLs, uc=[a, b, c, alpha, beta, gamma], PhotonE=PhotonE, display=False)
                G_vectors = CalLat_reciprocal_peaks(HKLs=HKLs, direct=Spaces['Direct'], reciprocal=Spaces['Reciprocal'], display=False)

                # Mapping - angle 
                result_mapping = np.zeros((N,1))
                for i in range(N): 
                    print('Calculating %d / %d ...  ' %(i, N), end='\r')
                    rot = Rotation[i]
                    temp_result = CalLat_grain_mapping(rot=rot, PhotonE=PhotonE, E_bandwidth=E_bandwidth, 
                                           HKLs=HKLs, G_vectors=G_vectors, 
                                           reciprocal=Spaces['Reciprocal'], regular=False, display=False)
                    G_vectors_rot = temp_result['G_vectors']
                    Flag_lit = temp_result['Flag_lit']
                    result_mapping[i] = np.sum(Flag_lit)

                print('Calculating %d / %d ... Done ' %(N, N))
                plt.figure()
                # plt.plot(Angles, result_mapping)
                plt.plot(result_mapping)
            
            2) Mapping peaks by changing the photon energy 
                # Graing-mapping parameters 
                N = 500
                Es = np.linspace(5,10,N)
                rot  = [0, 0, 0]
                CIF_file = 'Al.cif'
                Type = 'FC'
                H_max = 6
                K_max = 6
                L_max = 6
                E_bandwidth = 0.05

                # Calculate constants 
                a, b, c, alpha, beta, gamma = CalLat_read_cif(CIF_name=CIF_file, 
                                        CIF_path=r'D:\GY\Dropbox\BNL-HXN\Detector_Fitting\CIFs')['UnitCellParams']
                Spaces = CalLat_vectors(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, display=False)
                HKLs = CalLat_HKLs(H_max=H_max, K_max=K_max, L_max=L_max, Type=Type, Progress=False)

                # Mapping - energy 
                result_mapping = np.zeros((N,1))
                for i in range(N): 
                    print('Calculating %d / %d ...  ' %(i, N), end='\r')
                    PhotonE = Es[i]
                    TwoThetas = CalLat_2Theta(HKL=HKLs, uc=[a, b, c, alpha, beta, gamma], PhotonE=PhotonE, display=False)
                    G_vectors = CalLat_reciprocal_peaks(HKLs=HKLs, direct=Spaces['Direct'], reciprocal=Spaces['Reciprocal'], display=False)
                    temp_result = CalLat_grain_mapping(rot=rot, PhotonE=PhotonE, E_bandwidth=E_bandwidth, 
                                           HKLs=HKLs, G_vectors=G_vectors, 
                                           reciprocal=Spaces['Reciprocal'], regular=False, display=False)
                    G_vectors_rot = temp_result['G_vectors']
                    Flag_lit = temp_result['Flag_lit']
                    result_mapping[i] = np.sum(Flag_lit)

                print('Calculating %d / %d ... Done ' %(N, N))
                plt.figure()
                # plt.plot(Angles, result_mapping)
                plt.plot(result_mapping)
            
            3) Verify the result 
                rot = Rotation[***] // PhotonE = Es[***]
                Result = CalLat_grain_mapping(rot=rot, PhotonE=PhotonE, E_bandwidth=E_bandwidth, HKLs=HKLs, G_vectors=G_vectors, 
                                    reciprocal=Spaces['Reciprocal'], regular=False, display=True, label_peak=False)
                G_vectors_rot = Result['G_vectors']
                Flag_lit = Result['Flag_lit']
    '''    
    # Input Regularization
    if regular: 
        if np.size(np.shape(HKLs)) > 2: 
            print(">>>>>> Input HKLs' dimension is wrong! <<<<<<")
            print(">>>>>> It should be a n by 3 array. <<<<<<")
            return
        if np.size(np.shape(G_vectors)) > 2: 
            print(">>>>>> Input G_vectors' dimension is wrong! <<<<<<")
            print(">>>>>> It should be a n by 3 array. <<<<<<")
            return
        if np.shape(HKLs)[0] == 3:
            n = int( np.size(HKLs) / 3 )
            HKLs = HKLs.transpose()
        elif np.shape(HKLs)[1] == 3: 
            n = int( np.shape(HKLs)[0] )
        else: 
            print(">>>>>> Input HKLs' size is wrong! <<<<<<")
            print(">>>>>> It should be a n by 3 array. <<<<<<")
            return
        if np.shape(G_vectors)[0] == 3:
            m = int( np.size(G_vectors) / 3 )
            G_vectors = G_vectors.transpose()
        elif np.shape(G_vectors)[1] == 3: 
            m = int( np.shape(G_vectors)[0] )
        else: 
            print(">>>>>> Input G_vectors' size is wrong! <<<<<<")
            print(">>>>>> It should be a n by 3 array. <<<<<<")
            return
        if not m == n: 
            print(">>>>>> HKLs' size doesn't match G_vectors' size! <<<<<<")
            return
    else: 
        n = int( np.shape(HKLs)[0] )
    
    # Calculate the wave length and vector
    PhotonE_hi = PhotonE * (1 + E_bandwidth/2)
    PhotonE_lo = PhotonE * (1 - E_bandwidth/2)
    WaveLen_hi = 12.3980 / PhotonE_hi   # unit is [A]
    WaveLen_lo = 12.3980 / PhotonE_lo   # unit is [A]
    WaveK_hi   = 2 * np.pi / WaveLen_hi
    WaveK_lo   = 2 * np.pi / WaveLen_lo
    Ewald_center_hi = np.asarray([0, 0, -WaveK_hi])
    Ewald_center_lo = np.asarray([0, 0, -WaveK_lo])
    
    WaveLen = 12.3980 / PhotonE   # unit is [A]
    WaveK   = 2 * np.pi / WaveLen
    Ewald_center = np.asarray([0, 0, -WaveK])
    
    # Rotation matrices
    Rx = Rot_Matrix([1, 0, 0], np.deg2rad(rot[0]))
    Ry = Rot_Matrix([0, 1, 0], np.deg2rad(rot[1]))
    Rz = Rot_Matrix([0, 0, 1], np.deg2rad(rot[2]))
    
    # Reciprocal lattice unit vectors
    b1 = np.asarray(reciprocal[0])
    b2 = np.asarray(reciprocal[1])
    b3 = np.asarray(reciprocal[2])
    
    # Rotate the unit vectors
    b1 = np.asarray(np.matmul(Rz, np.matmul(Ry, np.matmul(Rx, b1).transpose()))).reshape(3,)
    b2 = np.asarray(np.matmul(Rz, np.matmul(Ry, np.matmul(Rx, b2).transpose()))).reshape(3,)
    b3 = np.asarray(np.matmul(Rz, np.matmul(Ry, np.matmul(Rx, b3).transpose()))).reshape(3,)
    
    # Rotate the reciprocal peaks
    G_vectors = np.matmul(Rx, G_vectors.transpose())
    G_vectors = np.matmul(Ry, G_vectors)
    G_vectors = np.matmul(Rz, G_vectors)
    G_vectors = G_vectors.transpose()
    
    # Identify G_vectors that are between two Ewald spheres
    Flag_lit = np.zeros((n,1))
    dist_hi = np.linalg.norm(G_vectors - np.tile(Ewald_center_hi, (n,1)), axis=1)
    dist_lo = np.linalg.norm(G_vectors - np.tile(Ewald_center_lo, (n,1)), axis=1)
    
    idx = (dist_hi <= WaveK_hi) & (dist_lo >= WaveK_lo)
    Flag_lit[idx] = 1
    
    # Plot the result
    if display: 
        # Initialize the figure and plot the origin point
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(0, 0, 0, color='k')
        ax.text(0, 0, 0, 'O')
        
        # Plot and label the reciprocal vectors
        ax.scatter(b1[0], b1[1], b1[2], color='b')
        ax.text(b1[0], b1[1], b1[2], 'b1')
        ax.plot([0, b1[0]], [0, b1[1]], [0, b1[2]], color='b')
        ax.scatter(b2[0], b2[1], b2[2], color='b')
        ax.text(b2[0], b2[1], b2[2], 'b2')
        ax.plot([0, b2[0]], [0, b2[1]], [0, b2[2]], color='b')
        ax.scatter(b3[0], b3[1], b3[2], color='b')
        ax.text(b3[0], b3[1], b3[2], 'b3')
        ax.plot([0, b3[0]], [0, b3[1]], [0, b3[2]], color='b')
        
        # plot the reciprocal peaks
        for i in range(n): 
            dist_hi = np.linalg.norm(G_vectors[i] - Ewald_center_hi)
            dist_lo = np.linalg.norm(G_vectors[i] - Ewald_center_lo)
            if (dist_hi >= WaveK_hi) | (dist_lo <= WaveK_lo): 
                if not lit_peak_only: 
                    ax.scatter(G_vectors[i, 0], G_vectors[i, 1], G_vectors[i, 2], color='green', s=3)
                    # ax.plot([0, G_vectors[i, 0]], [0, G_vectors[i, 1]], [0, G_vectors[i, 2]],  color='blue')
            else:
                ax.scatter(G_vectors[i, 0], G_vectors[i, 1], G_vectors[i, 2],  color='red')
                Ki = np.asarray( np.asarray([0, 0, 0]) - Ewald_center)
                Kf = np.asarray( G_vectors[i] - Ewald_center).reshape(3, 1)
                Twotheta = np.arccos( np.dot(Ki, Kf) / (np.linalg.norm(Ki) * np.linalg.norm(Kf)) )
                print('[%d %d %d] \t 2-Theta angle: %.3f' % (HKLs[i][0], HKLs[i][1], HKLs[i][2], np.rad2deg(Twotheta)) )
            
            if label_peak is True: 
                ax.text(G_vectors[i, 0], G_vectors[i, 1], G_vectors[i, 2], 
                        '[%d %d %d]' %(HKLs[i][0], HKLs[i][1], HKLs[i][2]) )
        
        # Show the image and adjust the aspect ratio
        plt.show()
        # ax.set_aspect('equal')
        
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5*max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
        # plot the Ewald sphere
        xx, yy, zz = Generate_sphere(R=WaveK_hi, C=Ewald_center_hi, Nu=30, Nv=15)
        
        idx = ( (xx <= x_middle - 2*plot_radius) | (xx >= x_middle + 2*plot_radius) | 
                (yy <= y_middle - 2*plot_radius) | (yy >= y_middle + 2*plot_radius) |
                (zz <= z_middle - 2*plot_radius) | (zz >= z_middle + 2*plot_radius) )
        zz[idx] = np.nan
        
        fig = plt.gca(projection='3d')
        fig.plot_surface(xx, yy, zz, alpha=0.2)
        
        xx, yy, zz = Generate_sphere(R=WaveK_lo, C=Ewald_center_lo, Nu=30, Nv=15)
        
        idx = ( (xx <= x_middle - 2*plot_radius) | (xx >= x_middle + 2*plot_radius) | 
                (yy <= y_middle - 2*plot_radius) | (yy >= y_middle + 2*plot_radius) |
                (zz <= z_middle - 2*plot_radius) | (zz >= z_middle + 2*plot_radius) )
        zz[idx] = np.nan
        
        fig = plt.gca(projection='3d')
        fig.plot_surface(xx, yy, zz, alpha=0.2)
        
    return {'G_vectors':G_vectors, 'Flag_lit':Flag_lit.reshape(-1)}


def CalLat_lattice_params(HKL=[[1, 1, 1]], uc=[5,5,5,90,90,90], CalAng=False): 
    ''' >>> Instruction <<< 
        The input uc = [a, b, c, alpha, beta, gamma] 
            The lattice constants a, b, and c are in Angstrom. 
            The alpha, beta, and gamma are in degree. 
        The HKL should be a (n, 3) array, n >= 1. 
        
        The output is a dictionary contains 'd_space', 'volumn', and 'Angle_P2P': 
        'd' is a (n, 1) array of the corresponding d spacing.
        'volumn' is the volumn of the unit cell.
        'Angle_P2P' is a (n, n) array, while each element showing the angle between two lattice planes. 
              For example, Angle_P2P[2, 3] stands for the angle between the lattice planes HKL[2, :] and HKL [3, :].
              Angle_P2P[i, i] should be zero, and Angle_P2P[i, j] should be same as Angle_P2P[j, i].
        
        The function use the formula for Triclinic lattice to calculate the spacing d_hkl: 
              1/d**2 = (1/V**2) * ( S_11 * h**2 
                                  + S_22 * k**2 
                                  + S_33 * l**2 
                                  + 2 * S_12 * h * k 
                                  + 2 * S_23 * k * l
                                  + 2 * S_31 * l * h )
        where 
              S_11 = b**2 * c**2 * sin(alpha)**2
              S_22 = c**2 * a**2 * sin(beta )**2
              S_33 = a**2 * b**2 * sin(gamma)**2
              S_12 = a * b * c**2 * [ cos(alpha) * cos(beta ) - cos(gamma) ]
              S_23 = a**2 * b * c * [ cos(beta ) * cos(gamma) - cos(alpha) ]
              S_31 = a * b**2 * c * [ cos(gamma) * cos(alpha) - cos(beta ) ]
        
        The unit cell volumn is calculated as: 
              V = a * b * c * sqrt[ 1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 
                                  + 2 * cos(alpha) * cos(beta) * cos(gamma) ]
        
        Angle phi between planes (h_1, k_1, l_1) and (h_2, k_2, l_2): 
              cos(phi) = ( d_1 * d_2 / V**2 ) * [ S_11 * h_1 * h_2 
                                                + S_22 * k_1 * k_2 
                                                + S_33 * l_1 * l_2 
                                                + S_23 * (k_1 * l_2 + l_1 * k_2) 
                                                + S_31 * (l_1 * h_2 + h_1 * l_2) 
                                                + S_12 * (h_1 * k_2 + k_1 * h_2) ]
    '''
    # Input Regularization
    HKL = np.asarray(HKL)
    a = uc[0]
    b = uc[1]
    c = uc[2]
    alpha = uc[3]
    beta  = uc[4]
    gamma = uc[5]
    
    if np.size(np.shape(HKL)) > 2: 
        print(">>>>>> Input HKL's dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        return
    if np.shape(HKL)[1] == 3: 
        n = np.shape(HKL)[0]
        Input = HKL
    elif np.shape(HKL)[0] == 3:
        n = int( np.size(HKL) / 3 )
        Input = HKL.transpose()
    else: 
        print(">>>>>> Input HKL's size is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        return
    alpha = np.deg2rad(alpha)
    beta  = np.deg2rad(beta )
    gamma = np.deg2rad(gamma)
    
    # Initialize the output arrays
    d = np.zeros((n, 1))
    Angle_P2P = np.zeros((n, n))
    
    # Calculate the contants
    V = a * b * c * np.sqrt(  1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 
                            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) )
    S_11 = b**2 * c**2 * np.sin(alpha)**2
    S_22 = c**2 * a**2 * np.sin(beta )**2
    S_33 = a**2 * b**2 * np.sin(gamma)**2
    S_12 = a    * b    * c**2 * ( np.cos(alpha) * np.cos(beta ) - np.cos(gamma) )
    S_23 = a**2 * b    * c    * ( np.cos(beta ) * np.cos(gamma) - np.cos(alpha) )
    S_31 = a    * b**2 * c    * ( np.cos(gamma) * np.cos(alpha) - np.cos(beta ) )
    
    # Calculate the d_space and Angle_P2P
    for i in range(n):
        h_1 = Input[i, 0]
        k_1 = Input[i, 1]
        l_1 = Input[i, 2]
        if h_1 == 0 and k_1 == 0 and l_1 ==0: 
            d[i] = 0
        else: 
            d[i] = np.sqrt( V**2 * 1/(  S_11 * h_1**2 
                                      + S_22 * k_1**2 
                                      + S_33 * l_1**2 
                                      + 2 * S_12 * h_1 * k_1 
                                      + 2 * S_23 * k_1 * l_1 
                                      + 2 * S_31 * l_1 * h_1 ) )
        if CalAng: 
            for j in range(i): 
                h_2 = Input[j, 0]
                k_2 = Input[j, 1]
                l_2 = Input[j, 2]
                costheta = ( (d[i] * d[j] / V**2) * 
                             (S_11 * h_1 * h_2 + 
                              S_22 * k_1 * k_2 + 
                              S_33 * l_1 * l_2 + 
                              S_23*(k_1 * l_2 + l_1 * k_2) + 
                              S_31*(l_1 * h_2 + h_1 * l_2) + 
                              S_12*(h_1 * k_2 + k_1 * h_2) ) )
                if costheta > 1: 
                    costheta = 1
                elif costheta < -1:
                    costheta = -1
                # print('cos value is ' + np.str(costheta) + ', arccos value is %0.3f' % (np.rad2deg(np.arccos(costheta))) )
                Angle_P2P[i, j] = np.arccos(costheta)
    
    if np.shape(d)[0] != np.shape(HKL)[0]: 
        d = np.transpose(d)
    if CalAng:
        Angle_P2P = Angle_P2P + np.transpose(Angle_P2P)
        Angle_P2P = np.rad2deg(Angle_P2P)
    
    return {'d': d, 'volumn': V, 'Angle_P2P': Angle_P2P}


def CalLat_2Theta(HKL=[[1, 1, 1]], uc=[5,5,5,90,90,90], PhotonE=12, display=False): 
    ''' >>> Instruction <<< 
        The input uc = [a, b, c, alpha, beta, gamma]
            The lattice constants a, b, and c are in A. 
            The alpha, beta, and gamma are in degree. 
        The PhotonE is the photon energy in keV. 
        The HKL should be a (n, 3) array, n >= 1. 
        
        The output is a (n, 1) array of TWO Theta angles.
        
        lambda = 2 * d * sin(Theta)
    '''
    # Input Regularization & Initialize the output arrays
    HKL = np.asarray(HKL)
    
    if np.size(np.shape(HKL)) > 2: 
        print(">>>>>> Input HKL's dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        return
    if np.shape(HKL)[1] == 3: 
        n = np.shape(HKL)[0]
        TwoTheta = np.zeros((n, 1))
    elif np.shape(HKL)[0] == 3:
        n = int( np.size(HKL) / 3 )
        TwoTheta = np.zeros((1, n))
    else: 
        print(">>>>>> Input HKL's size is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        return
    
    # Calculate thetas
    WaveLen = 12.3980 / PhotonE   # unit is [A]
    Result = CalLat_lattice_params(HKL=HKL, uc=uc, CalAng=False)
    d = Result['d'].copy()
    
    d[np.where(d == 0)] = np.inf    
    TwoTheta = 2 * np.rad2deg(np.arcsin(WaveLen/2/d))
    
    if display:
        for i in range(n): 
            print('[%d %d %d] \t 2-Theta angle: %.3f' 
                  % (HKL[i][0], HKL[i][1], HKL[i][2], TwoTheta[i]) )
    
    return TwoTheta.reshape(-1)


def CalLat_RotY_of_Q(k=0, Q=[0,0,0], OutputRadian=False): 
    ''' >>> Instruction <<<
        This function figures out the required rotation of Q around Y axis 
        for the Bragg condition. 
        
        Inputs: 
            k      [A^-1], Wave vector
            Q      (3,) array, the initial Q vector
            
        Math: 
            The incident and diffracted wave vectors are: 
                ki = [0, 0, k] 
                kf = RotY(Delta) * RotX(-Gamma) * ki
                   = [k*cos(Gamma)*sin(Delta), 
                      k*sin(Gamma), 
                      k*cos(Gamma)*cos(Delta)]
            
            The initial vector Q = [qx, qy, qz]
            After being rotated Theta around the Y axis,
            The final vector Q' = RotY(Theta) * Q
                                = [ qx * cos(Theta) + qz * sin(Theta), 
                                    qy, 
                                   -qx * sin(Theta) + qz * cos(Theta) ]
            
            Also, there is: kf - ki = Q', i.e.
                    k*cos(Gamma)*sin(Delta) =  qx * cos(Theta) + qz * sin(Theta)
                               k*sin(Gamma) =  qy
                k*cos(Gamma)*cos(Delta) - k = -qx * sin(Theta) + qz * cos(Theta)
            
            Therefore, 
                    Gamma = arcsin( qy/k )
                    Delta = arccos( (1-q**2/k**2/2) / cos(Gamma) )
                    Theta = arccos( k*(qx*cos(Gamma)*sin(Delta) + qz*(cos(Gamma)*cos(Delta)-1))
                                    / (qx**2 + qz**2) )
                        = arcsin( k*(qz*cos(Gamma)*sin(Delta) - qx*(cos(Gamma)*cos(Delta)-1))      
                                    / (qx**2 + qz**2) )
    '''
    qx, qy, qz = Q
    q = np.linalg.norm(Q)
    if q == 0 or k == 0:
        Rotat = 0
        Gamma = 0
        Delta = 0
        return {'Rotation':Rotat, 'Gamma':Gamma, 'Delta':Delta}
    
    Gamma = np.arcsin( qy/k )
    Delta = np.arccos( (1-q**2/k**2/2) / np.cos(Gamma) )
    cos_Rotat = ( k*(qx*np.cos(Gamma)*np.sin(Delta) + qz*(np.cos(Gamma)*np.cos(Delta)-1))
                       / (qx**2 + qz**2) )
    sin_Rotat = ( k*(qz*np.cos(Gamma)*np.sin(Delta) - qx*(np.cos(Gamma)*np.cos(Delta)-1))
                       / (qx**2 + qz**2) )
    Rotat = np.arctan2(sin_Rotat, cos_Rotat)
    
    if not OutputRadian: 
        Gamma = np.rad2deg(Gamma)
        Delta = np.rad2deg(Delta)
        Rotat = np.rad2deg(Rotat)
        
    return {'Rotation':Rotat, 'Gamma':Gamma, 'Delta':Delta}


def CalLat_RotX_of_Q(k=0, Q=[0,0,0], OutputRadian=False): 
    ''' >>> Instruction <<< 
        This function figures out the required rotation of Q around -X axis 
        for the Bragg condition. 
        
        Inputs: 
            k      [A^-1], Wave vector
            Q      (3,) array, the initial Q vector
            
        Math: 
            The incident and diffracted wave vectors are: 
                ki = [0, 0, k] 
                kf = RotY(Delta) * RotX(-Gamma) * ki
                   = [k*cos(Gamma)*sin(Delta), 
                      k*sin(Gamma), 
                      k*cos(Gamma)*cos(Delta)]
            
            The initial vector Q = [qx, qy, qz]
            After being rotated Theta around the X axis,
            The final vector Q' = RotX(Theta) * Q
                                = [ qx, 
                                    qy * cos(Theta) - qz * sin(Theta), 
                                    qy * sin(Theta) + qz * cos(Theta) ]
            
            Also, there is: kf - ki = Q', i.e.
                    k*cos(Gamma)*sin(Delta) = qx
                               k*sin(Gamma) = qy * cos(Theta) - qz * sin(Theta)
                k*cos(Gamma)*cos(Delta) - k = qy * sin(Theta) + qz * cos(Theta)
            
            Therefore, 
                    Delta = arctan( qx / k / (1-q**2/k**2/2) )
                    Gamma = arccos( qx / k / sin(Delta) )
                    Theta = arccos( k * (qy*sin(Gamma) + qz*(cos(Gamma)*cos(Delta)-1))
                                    / (qy**2 + qz**2) )
                        = arcsin( k * (qy*(cos(Gamma)*cos(Delta)-1) - qz*sin(Gamma)) 
                                    / (qy**2 + qz**2) )
    '''
    qx, qy, qz = Q
    q = np.linalg.norm(Q)
    if q == 0 or k == 0:
        Rotat = 0
        Gamma = 0
        Delta = 0
        return {'Rotation':Rotat, 'Gamma':Gamma, 'Delta':Delta}
    
    Delta = np.arctan2( qx/k, 1-q**2/k**2/2 )
    Gamma = np.arccos( qx / k / np.sin(Delta) )
    cos_Rotat = ( k*(qy*np.sin(Gamma) + qz*(np.cos(Gamma)*np.cos(Delta)-1))
                      / (qy**2 + qz**2) )
    sin_Rotat = ( k*( qy*(np.cos(Gamma)*np.cos(Delta)-1) - qz*np.sin(Gamma) ) 
                      / (qy**2 + qz**2) )
    Rotat = np.arctan2(sin_Rotat, cos_Rotat)
    
    if not OutputRadian: 
        Gamma = np.rad2deg(Gamma)
        Delta = np.rad2deg(Delta)
        Rotat = np.rad2deg(Rotat)
    
    return {'Rotation':Rotat, 'Gamma':Gamma, 'Delta':Delta}


def CalLat_read_cif(CIF_name='test.cif', CIF_path='', LatticeInfo=False): 
    # Get the current working directory
    retval = os.getcwd()
    
    # Generate the full file path and check if the file exists
    if len(CIF_path) == 0: 
        CIF_path = retval
    '''
    if CIF_path[-1] == '/' or CIF_path[-1] == '\\':
        File = CIF_path + CIF_name
    elif '/' in CIF_path: 
        File = CIF_path + '/' + CIF_name
    elif '\\' in CIF_path: 
        File = CIF_path + '\\' + CIF_name
    else: 
        print('>>>>>> Error! Unknown path separator in CIF file name. <<<<<<')
        input('>>>>>> Press any key to quit...')
        return
    '''
    File = os.path.join(CIF_path, CIF_name)
    
    if not os.path.isfile(File): 
        print('>>>>>> Error! CIF File does NOT exist. <<<<<<')
        input('>>>>>> Press any key to quit...')
        return
    
    # Read cif file
    cif_data = Crystal.from_cif(File)
    if LatticeInfo: 
        print(cif_data)
    
    Atoms = np.asarray(cif_data)
    N,_ = np.shape(Atoms)   # N is the number of atoms in the unit cell
    Z = np.zeros((N, 1))
    Atoms_coords = np.zeros((N, 3))
    Z[:, 0] = Atoms[:, 0]
    Atoms_coords[:, :] = Atoms[:, 1:]
    
    return {'UnitCellParams':cif_data.lattice_parameters, 'Atoms_Z':Z, 'Atoms_coords':Atoms_coords, 
            'LattVectors':cif_data.lattice_vectors, 'ReciVectors':cif_data.reciprocal_vectors}


def CalLat_plot_atoms(CIF_name='test.cif', CIF_path='', SuperCell=[1,1,1],
                      Elements=[['O',8,'r'],['Li',3,'g'],['Nb',41,'b']],
                      Cartesian=False, PrintCoords=False): 
    ''' >>> Instruction <<< 
        This function read the cif file and plot the atoms in 3D coordinates
        
        SuperCell:    Creat super cells using the unit cell
        Cartesian:    Plot the atoms in a Cartesian coords or a fraction coords
        PrintCoords:  Plot the lattice vectors or not
    '''
    
    # Get the current working directory
    retval = os.getcwd()
    
    # Generate the full file path and check if the file exists
    if len(CIF_path) == 0: 
        CIF_path = retval
    if CIF_path[-1] == '/' or CIF_path[-1] == '\\':
        File = CIF_path + CIF_name
    elif '/' in CIF_path: 
        File = CIF_path + '/' + CIF_name
    elif '\\' in CIF_path: 
        File = CIF_path + '\\' + CIF_name
    else: 
        print('>>>>>> Error! Unknown path separator. <<<<<<')
        input('>>>>>> Press any key to quit...')
        return
    if not os.path.isfile(File): 
        print('>>>>>> Error! File does NOT exist. <<<<<<')
        input('>>>>>> Press any key to quit...')
        return
    
    # Read cif file
    cif_data = Crystal.from_cif(File)
    a, b, c, alpha, beta, gamma = cif_data.lattice_parameters
    Basis = cif_data.lattice_vectors
    Atoms = np.asarray(cif_data.supercell(SuperCell[0],SuperCell[1],SuperCell[2]))
    N,_ = np.shape(Atoms)   # N is the number of atoms in the unit cell
    
    if Cartesian: 
        Atoms_coords = np.matmul(Atoms[:, 1:], Basis)
    else: 
        Atoms_coords = Atoms[:, 1:]
    
    if PrintCoords:
        print(Atoms_coords)
    
    # Initialize the figure
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(0, 0, 0, color='k')
    # Plot the vectors
    if Cartesian: 
        # Plot the lattice vectors
        a1, a2, a3 = Basis
        # ax.scatter(a1[0], a1[1], a1[2], color='c', marker='+', alpha=0.3)
        # ax.scatter(a2[0], a2[1], a2[2], color='c', marker='+', alpha=0.3)
        # ax.scatter(a3[0], a3[1], a3[2], color='c', marker='+', alpha=0.3)
        ax.plot([0, a1[0]], [0, a1[1]], [0, a1[2]], color='c', alpha=0.3)
        ax.plot([0, a2[0]], [0, a2[1]], [0, a2[2]], color='c', alpha=0.3)
        ax.plot([0, a3[0]], [0, a3[1]], [0, a3[2]], color='c', alpha=0.3)
        # Plot the unit cell
        if True: 
            P1 = a1 + a2
            P2 = a2 + a3
            P3 = a3 + a1
            P4 = a1 + a2 + a3
            ax.plot([P1[0], a1[0]], [P1[1], a1[1]], [P1[2], a1[2]], color='k', alpha=0.3)
            ax.plot([P1[0], a2[0]], [P1[1], a2[1]], [P1[2], a2[2]], color='k', alpha=0.3)
            ax.plot([P2[0], a2[0]], [P2[1], a2[1]], [P2[2], a2[2]], color='k', alpha=0.3)
            ax.plot([P2[0], a3[0]], [P2[1], a3[1]], [P2[2], a3[2]], color='k', alpha=0.3)
            ax.plot([P3[0], a3[0]], [P3[1], a3[1]], [P3[2], a3[2]], color='k', alpha=0.3)
            ax.plot([P3[0], a1[0]], [P3[1], a1[1]], [P3[2], a1[2]], color='k', alpha=0.3)
            ax.plot([P1[0], P4[0]], [P1[1], P4[1]], [P1[2], P4[2]], color='k', alpha=0.3)
            ax.plot([P2[0], P4[0]], [P2[1], P4[1]], [P2[2], P4[2]], color='k', alpha=0.3)
            ax.plot([P3[0], P4[0]], [P3[1], P4[1]], [P3[2], P4[2]], color='k', alpha=0.3)
        # Label the lattice vectors
        a1, a2, a3 = np.asarray(Basis) * 1.05
        ax.text(a1[0], a1[1], a1[2], 'a1')
        ax.text(a2[0], a2[1], a2[2], 'a2')
        ax.text(a3[0], a3[1], a3[2], 'a3')
    # Plot the atoms
    M = np.shape(Elements)[0]
    for i in range(N): 
        for m in range(M): 
            if Atoms[i, 0] == Elements[m][1]:
                ar = Atoms_coords[i]
                ax.scatter(ar[0], ar[1], ar[2], color=Elements[m][2])
                ax.text(ar[0], ar[1], ar[2], Elements[m][0])
        
    # Show the image and adjust the aspect ratio
    plt.show()
    # ax.set_aspect('equal')
    return


def CalLat_creat_crystal(L1=10, L2=10, L3=10): 
    ''' >>> Introduction <<< 
        This function creats a crystal in the CRYSTAL FRAME. 
        
        Inputs: 
            L1, L2, L3      Number of unit cells along lattice vectors
        
        Output:
            Crys_coords     (n,3) array, each (n,1) is a raveled axis
            Data_shape      (3,) array, how to reshape the raveled data
        
        The result can be reshaped and transposed using command like: 
                Crys_coords = Crys_coords.reshape(Data_shape).transpose(2, 1, 0)
            where transpose(2,1,0) is aimed to reorganize the strides to [X,Y,Z]
    '''
    X = np.asarray(range(L1)) - np.floor(L1/2)
    Y = np.asarray(range(L2)) - np.floor(L2/2)
    Z = np.asarray(range(L3)) - np.floor(L3/2)
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X)
    
    Crys_coords = np.stack((X3.ravel(), Y3.ravel()), axis=0)
    Crys_coords = np.concatenate((Crys_coords, [Z3.ravel()]), axis=0)
    
    Data_shape = (np.size(Z), np.size(Y), np.size(X))
    
    return {'Coords':Crys_coords.transpose(), 'Shape':Data_shape}


def CalLat_crop_array_plane(Crys_coords, Crys_center=[0,0,0], Plan_normal=[0,0,0], dist=0, 
                            Flag_InputReg=False): 
    ''' >>> Instruction <<< 
        This function crop a 3D array using an arbitrary plane. 
        
        Inputs: 
            Crys_coords     (n,3) array, n>>3, coordinates of the voxels of the crystal.
                            Generated using function "CalLat_creat_crystal()".
            Crys_center     (3, ), coordinate of the center of the crystal.
                            Center-side of the plane is kept, the other side is cropped.
            Plan_normal     (3, ), normal of the plane. This vector is considered as 
                            pointing from Crys_center to the plane. 
            dist            float, distance from Crys_center to the plane
        
        Output: 
            Rho_coords      (n,1) array, representing the "density" of the crystal. 
                            Each element is either 1 or 0, where 0 means no unit cell at the 
                            corresponding voxel. 
        
        For functions that generate dislocations, i.e. "CalLat_creat_disloc_***", input the original Crys_coords; 
        For function "CalLat_crystal_structure_factor", input Crys_coords[np.where(Rho_coords!=0)]
    '''
    # Input Regularization
    Crys_coords = np.asarray(Crys_coords)
    Crys_center = np.asarray(Crys_center)
    Plan_normal = np.asarray(Plan_normal/np.linalg.norm(Plan_normal))
    
    if Flag_InputReg: 
        
        if np.size(np.shape(Crys_coords)) != 2: 
            print(">>>>>> Crys_coords should be a (n, 3) array. <<<<<<")
            input('>>>>>> Press any key to quit...')
            return
        
        N = np.max(np.shape(Crys_coords))
        if np.shape(Crys_coords)[0] != N: 
            if np.shape(Crys_coords)[1] != N:
                print(">>>>>> Crys_coords should be a (n,3) array ! <<<<<<")
                input('>>>>>> Press any key to quit...')
                return
            else: 
                Crys_coords = np.transpose(Crys_coords)
    
    N = np.max(np.shape(Crys_coords))
    Rho_coords = np.ones(N)
    
    # Plane function
    Cent_proj = Plan_normal * dist + Crys_center
    Plane  = Plane_func('Normal_and_Point', Plan_normal, Cent_proj)
    
    # Segment
    Cent_side = (  Plane[0] * Crys_center[0] 
                 + Plane[1] * Crys_center[1] 
                 + Plane[2] * Crys_center[2] 
                 + Plane[3] )
    if Cent_side == 0: 
        Cent_side = 1
    Cent_side = Cent_side / np.abs(Cent_side) # get the sign of Cent_side
    
    Voxel_side = (  Plane[0] * Crys_coords[:, 0] 
                  + Plane[1] * Crys_coords[:, 1] 
                  + Plane[2] * Crys_coords[:, 2] 
                  + Plane[3] )
    Rho_coords[np.where(Voxel_side * Cent_side < 0)] = 0
    
    return Rho_coords


def CalLat_creat_disloc_screw(Crys_coords, ScrewLoc=None, ScrewEdge=None, ScrewDis=None, 
                              Progress=False, SaveResult=False, FileName='Disloc.npy'): 
    ''' >>> Introduction <<< 
        This function mordifies the Crys_coords (fractional coordiantes of unit cells) to
        creat a screw dislocation. 
        
        Inputs: 
            Crys_coords     (n,3) array, each (n,1) is a raveled axis
            ScrewLoc        (3,) int array, the location of the screw dislocation
                                [0,?,?] or [?,0,?] or [?,?,0]
            ScrewEdge       (3,) int array, edge of the screw dislocation. 
                                [0,?,0] or [0,0,?] or [?,0,0], ? is the edge length
            ScrewDis        (3,) int array, magnitude of the dislocation. 
                                [?,0,0] or [0,?,0] or [0,0,?], matching the ScrewLoc
        Output: 
            Disloc          (n,3) array, each (n,1) is a raveled axis
    '''
    if Progress: 
        print('>>>>>> Start calculating ... ' + strftime('%H:%M:%S', localtime()))
    
    ScrewLoc = np.asarray(ScrewLoc)
    ScrewEdge = np.asarray(ScrewEdge)
    ScrewDis = np.asarray(ScrewDis)
    
    Disloc = np.zeros_like(Crys_coords)
    EdgeLen = np.linalg.norm(ScrewEdge)
    EdgeOri = ScrewEdge / EdgeLen
    Ratio = ScrewDis / EdgeLen
    
    # No dislocation if displacement or edge is zero
    if np.linalg.norm(ScrewDis) == 0 or np.linalg.norm(ScrewEdge) == 0: 
        if Progress: 
            print('>>>>>> Done !                ' + strftime('%H:%M:%S', localtime()))
        return Disloc
    
    # Calculate the dislocation
    N,_ = np.shape(Crys_coords)
    ''' The input Crys_coords is organized as [X, Y, Z]
        Therfore, depending on the ScrewDis, the Crys_coords needs to be reshape to
            [Y, Z, X], if the screw dislocation is along X => neworder is [1,2,0]
            [Z, X, Y], if the screw dislocation is along Y => neworder is [2,0,1]
            [X, Y, Z], if the screw dislocation is along Z => neworder is [0,1,2]
        before performing the Cartesian2Cylindrical transform
    '''
    Temp = Crys_coords.copy().transpose()
    DisAx = np.nonzero(ScrewDis)[0][0]
    EdgAx = np.nonzero(ScrewEdge)[0][0]
    if DisAx == 0: 
        NewOrder = np.asarray([1, 2, 0])
        # Shift the screw dislocation to the center
        Temp[1, :] = Temp[1, :] - ScrewLoc[1]
        Temp[2, :] = Temp[2, :] - ScrewLoc[2]
        # Determine the location of edge
        CylAngOri = [0,1,0]
    elif DisAx == 1: 
        NewOrder = np.asarray([2, 0, 1])
        # Shift the screw dislocation to the center
        Temp[2, :] = Temp[2, :] - ScrewLoc[2]
        Temp[0, :] = Temp[0, :] - ScrewLoc[0]
        # Determine the location of edge
        CylAngOri = [0,0,1]
    elif DisAx == 2: 
        NewOrder = np.asarray([0, 1, 2])
        # Shift the screw dislocation to the center
        Temp[0, :] = Temp[0, :] - ScrewLoc[0]
        Temp[1, :] = Temp[1, :] - ScrewLoc[1]
        # Determine the location of edge
        CylAngOri = [1,0,0]
    AngOff = Included_Angle(CylAngOri, EdgeOri, RotAxis=ScrewDis)['theta']
        
    Temp = Cartesian2Cylindrical(Temp[NewOrder, :], OutputRadian=True)
    
    Disloc[:, DisAx] = Ratio[DisAx] * Temp[0] * ((Temp[1]+AngOff)%(2*np.pi)-np.pi)/2/np.pi
    Idx = np.where(Temp[0]>EdgeLen)
    Disloc[Idx, DisAx] = ScrewDis[DisAx] * ((Temp[1, Idx]+AngOff)%(2*np.pi)-np.pi)/2/np.pi
    
    if Progress: 
        print('>>>>>> Done !                ' + strftime('%H:%M:%S', localtime()))
    if SaveResult: 
        np.save(FileName, Disloc)
    return Disloc


def CalLat_creat_disloc_gauss(Crys_coords, GaussLoc=None, GaussWid=None, GaussDis=None, 
                               Progress=False, SaveResult=False, FileName='Disloc.npy'): 
    ''' >>> Introduction <<< 
        This function mordifies the Crys_coords (fractional coordiantes of unit cells) to
        creat a 3D Gaussian dislocation. 
        
        Inputs: 
            Crys_coords     (n,3) array, each (n,1) is a raveled axis
            GaussLoc        (3,) int array, the 3D location of the Gaussian dislocation
            GaussWid        (3,) int array, FWHM of the Gaussian dislocation in 3D. 
            GaussDis        (3,) int array, magnitude of the 3D dislocation. 
        Output: 
            Disloc          (n,3) array, each (n,1) is a raveled axis
            
        Math: 
            The function of a 3D Gaussian is 
                    A * exp(-( (x-x0)**2/2/a**2 + 
                               (y-y0)**2/2/b**2 + 
                               (z-z0)**2/2/c**2 ))
                where [x0, y0, z0] is the center location,
                      [a, b, c] are the sigmas in three dimensions.
    '''
    if Progress: 
        print('>>>>>> Start calculating ... ' + strftime('%H:%M:%S', localtime()))
    
    GaussLoc = np.asarray(GaussLoc)
    GaussWid = np.asarray(GaussWid)
    GaussDis = np.asarray(GaussDis)
    
    Disloc = np.zeros_like(Crys_coords)
    SigX, SigY, SigZ = GaussWid /2/np.sqrt(2*np.log(2))
    DisX, DisY, DisZ = GaussDis
    
    # No dislocation if displacement or edge is zero
    if np.linalg.norm(GaussWid) == 0 or np.linalg.norm(GaussDis) == 0: 
        if Progress: 
            print('       GaussWid and/or GaussDis are zero. ')
            print('       No dislocation added. ')
            print('>>>>>> Done !                ' + strftime('%H:%M:%S', localtime()))
        return Disloc
    
    # Calculate the dislocation
    N,_ = np.shape(Crys_coords)
    Temp = Crys_coords.copy().transpose()
    # Shift the elliposid dislocation to the center
    Temp[0, :] = Temp[0, :] - GaussLoc[0]
    Temp[1, :] = Temp[1, :] - GaussLoc[1]
    Temp[2, :] = Temp[2, :] - GaussLoc[2]
    # The dislocations are
    Disloc[:, 0] = DisX * np.exp(-( Temp[0, :]**2/2/SigX**2 + Temp[1, :]**2/2/SigY**2 + 
                                    Temp[2, :]**2/2/SigZ**2 ))
    Disloc[:, 1] = DisY * np.exp(-( Temp[0, :]**2/2/SigX**2 + Temp[1, :]**2/2/SigY**2 + 
                                    Temp[2, :]**2/2/SigZ**2 ))
    Disloc[:, 2] = DisZ * np.exp(-( Temp[0, :]**2/2/SigX**2 + Temp[1, :]**2/2/SigY**2 + 
                                    Temp[2, :]**2/2/SigZ**2 ))
    
    if Progress: 
        print('>>>>>> Done !                ' + strftime('%H:%M:%S', localtime()))
    if SaveResult: 
        np.save(FileName, Disloc)
    return Disloc


def CalLat_creat_disloc_lorentz(Crys_coords, LorentzLoc=None, LorentzWid=None, LorentzDis=None, 
                                Progress=False, SaveResult=False, FileName='Disloc.npy'): 
    ''' >>> Introduction <<< 
        This function mordifies the Crys_coords (fractional coordiantes of unit cells) to
        creat a 3D Lorentzian dislocation. 
        
        Inputs: 
            Crys_coords     (n,3) array, each (n,1) is a raveled axis
            LorentzLoc      (3,) int array, the 3D location of the Lorentzian dislocation
            LorentzWid      (3,) int array, FWHM of the Lorentzian dislocation in 3D. 
            LorentzDis      (3,) int array, magnitude of the 3D dislocation. 
        Output: 
            Disloc          (n,3) array, each (n,1) is a raveled axis
            
        Math: 
            The function of a 3D Lorentzian is 
                    A * a**2 / ((x-x0)**2 + a**2) 
                      * b**2 / ((y-y0)**2 + b**2)
                      * c**2 / ((z-z0)**2 + c**2) 
                where [x0, y0, z0] is the center location,
                      [a, b, c] are the sigmas in three dimensions.
    '''
    if Progress: 
        print('>>>>>> Start calculating ... ' + strftime('%H:%M:%S', localtime()))
    
    LorentzLoc = np.asarray(LorentzLoc)
    LorentzWid = np.asarray(LorentzWid)
    LorentzDis = np.asarray(LorentzDis)
    
    Disloc = np.zeros_like(Crys_coords)
    SigX, SigY, SigZ = LorentzWid / 2
    DisX, DisY, DisZ = LorentzDis
    
    # No dislocation if displacement or edge is zero
    if np.linalg.norm(LorentzWid) == 0 or np.linalg.norm(LorentzDis) == 0: 
        if Progress: 
            print('       LorentzWid and/or LorentzDis are zero. ')
            print('       No dislocation added. ')
            print('>>>>>> Done !                ' + strftime('%H:%M:%S', localtime()))
        return Disloc
    
    # Calculate the dislocation
    N,_ = np.shape(Crys_coords)
    Temp = Crys_coords.copy().transpose()
    # Shift the elliposid dislocation to the center
    Temp[0, :] = Temp[0, :] - LorentzLoc[0]
    Temp[1, :] = Temp[1, :] - LorentzLoc[1]
    Temp[2, :] = Temp[2, :] - LorentzLoc[2]
    # The dislocations are
    Disloc[:, 0] = ( DisX * SigX**2 / (Temp[0, :]**2 + SigX**2) 
                          * SigY**2 / (Temp[1, :]**2 + SigY**2) 
                          * SigZ**2 / (Temp[2, :]**2 + SigZ**2) )
    Disloc[:, 1] = ( DisY * SigX**2 / (Temp[0, :]**2 + SigX**2) 
                          * SigY**2 / (Temp[1, :]**2 + SigY**2) 
                          * SigZ**2 / (Temp[2, :]**2 + SigZ**2) )
    Disloc[:, 2] = ( DisZ * SigX**2 / (Temp[0, :]**2 + SigX**2) 
                          * SigY**2 / (Temp[1, :]**2 + SigY**2) 
                          * SigZ**2 / (Temp[2, :]**2 + SigZ**2) )
    if Progress: 
        print('>>>>>> Done !                ' + strftime('%H:%M:%S', localtime()))
    if SaveResult: 
        np.save(FileName, Disloc)
    return Disloc


def CalLat_atomic_form_factor(Z=20, q=0, PhotonE=12, IonCharge=0, IsIon=False): 
    ''' >>> Instruction <<< 
        ==============================================================================
        Working on adding Ionic form factor rather than Atomic form factor. 
        ==============================================================================
        Calculate the atomic/ionic form factor. 
            Z        Element Z numebr
            q        [A^-1]
            PhotonE     [keV] Photon energy, for fi and fii
            IonCharge   Charge of the ion, could be 0, +1, -2, etc.
        
            IsIon      If true, will use the most common charge state for the element
            
        Calculate q from twotheta: 
            q = np.sin(np.deg2rad(TwoTheta/2)) / WaveLen
    '''
    # WaveLen = 12.3980 / PhotonE   # unit is [A]
    # Q = 4 * np.pi * np.sin(np.deg2rad(TwoTheta/2)) / WaveLen
    # q = np.sin(np.deg2rad(TwoTheta/2)) / WaveLen
    
    Z = int(Z)
    q = float(q)
    
    # f' and f"
    fi  = xb.Fi(Z, PhotonE)
    fii = xb.Fii(Z, PhotonE)
    # f0
    if IonCharge == 0:   # for atom, simply call the Xraylib
        f0 = xb.FF_Rayl(Z, q)
    else: 
        f0 = Z
    
    return {'f0': f0, 'fi': fi, 'fii': fii, 'complex': f0 + fi + 1j * fii}


def CalLat_lattice_structure_factor(q=0, PhotonE=12, LattVects=None, Zs=None, Atoms_coords=None, 
                                    Flag_InputReg=False): 
    ''' >>> Introduction <<< 
        This function calculates the lattice structure factor in the CRYSTAL FRAME !
        
        Inputs: 
            q               (3,) array, momentum transfer vector in the crystal frame
            PhotonE         [keV] Photon energy, for atomic form factor
            LattVects       (3, 3) array, lattice vectors, from {cif_data.lattice_vectors}
            Zs              (n, 1) array, Z numbers of the atoms in the unit cell
            Atoms_coords    (n, 3) array, fractional coordinates of atoms
            Flag_InputReg   Boolean, wheather check the format of inputs
    '''
    # Input Regularization
    if Flag_InputReg: 
        q = np.asarray(q)
        LattVects = np.asarray(LattVects)
        Zs = np.asarray(Zs)
        Atoms_coords = np.asarray(Atoms_coords)
        
        if np.size(q) != 3: 
            print(">>>>>> q should be a (3,) array. <<<<<<")
            input('>>>>>> Press any key to quit...')
            return
        if np.shape(LattVects) != (3, 3): 
            print(">>>>>> LattVects should be a (3, 3) array. <<<<<<")
            input('>>>>>> Press any key to quit...')
            return
        if np.size(np.shape(Atoms_coords)) != 2: 
            print(">>>>>> Atoms_coords should be a (n, 3) array. <<<<<<")
            input('>>>>>> Press any key to quit...')
            return
        
        N = np.size(Zs)
        if np.shape(Atoms_coords)[0] != N: 
            if np.shape(Atoms_coords)[1] != N:
                print(">>>>>> Atoms_coords does not match the number of atoms ! <<<<<<")
                input('>>>>>> Press any key to quit...')
                return
            else: 
                Atoms_coords = np.transpose(Atoms_coords)
        
        q = np.reshape(q, (3,))
        Zs = np.reshape(Zs, (N,))
    
    # Calculate the atomic form factors
    N = np.size(Zs)
    Z_unique, Z_indices = np.unique(Zs, return_inverse=True)
    Uniq_forfac = [CalLat_atomic_form_factor(Z=i, q=np.linalg.norm(q), PhotonE=PhotonE)['complex'] 
                   for i in Z_unique]
    Atom_forfac = np.asarray(Uniq_forfac)[Z_indices]
    
    # Calculate the lattice vectors for each atom
    R = np.matmul(Atoms_coords, LattVects)
    
    # Calculate {exp(2j*pi*R*Q)}
    Matrix_exp = np.exp(1j*np.matmul(R, q))   # 2j*np.pi
    
    # Calculate F_q
    Latt_strfac = np.matmul(Atom_forfac, Matrix_exp)[0,0]
    
    return Latt_strfac


def CalLat_crystal_structure_factor(q=0, Latt_forfac=0, LattVects=None, Crys_coords=None, 
                                    Flag_InputReg=False): 
    ''' >>> Introduction <<< 
        This function calculates the structure factor of a crystal in the CRYSTAL FRAME !
        
        Inputs: 
            q               (3,) array, momentum transfer vector in the crystal frame
            Latt_forfac     complex number or (n,) complex array, lattice form factor
            LattVects       (3, 3) array, lattice vectors, from {cif_data.lattice_vectors}
            Crys_coords     (n, 3) array, n>>3, define the shape of the crystal 
            Flag_InputReg   Boolean, wheather check the format of inputs
            
        Note: 
            n stands for the total number of unitcells in the crystal. 
            For Crys_coords, 
                (n,3) is a reshaped meshgrid representing the fractional 
                coordinates of the unitcells
            For Latt_forfac, 
                use one complex number if all unitcells are same,
                use (n,) complex array to include deformations of unitcells
    '''
    # Input Regularization
    if Flag_InputReg: 
        q = np.asarray(q)
        LattVects = np.asarray(LattVects)
        Crys_coords = np.asarray(Crys_coords)
        
        if np.size(q) != 3: 
            print(">>>>>> q should be a (3,) array. <<<<<<")
            input('>>>>>> Press any key to quit...')
            return
        if np.shape(LattVects) != (3, 3): 
            print(">>>>>> LattVects should be a (3, 3) array. <<<<<<")
            input('>>>>>> Press any key to quit...')
            return
        if np.size(np.shape(Crys_coords)) != 2: 
            print(">>>>>> Crys_coords should be a (n, 3) array. <<<<<<")
            input('>>>>>> Press any key to quit...')
            return
        
        N = np.max(np.shape(Crys_coords))
        if np.shape(Crys_coords)[0] != N: 
            if np.shape(Crys_coords)[1] != N:
                print(">>>>>> Crys_coords should be a (n,3) array ! <<<<<<")
                input('>>>>>> Press any key to quit...')
                return
            else: 
                Crys_coords = np.transpose(Crys_coords)
        
        q = np.reshape(q, (3,))
        if np.size(Latt_forfac) != 1:
            if np.size(Latt_forfac) != N: 
                print(">>>>>> Latt_forfac's size is wrong ! <<<<<<")
                input('>>>>>> Press any key to quit...')
                return
            else: 
                Latt_forfac = np.reshape(Latt_forfac, (N,))
    N = np.max(np.shape(Crys_coords))
    
    # Calculate the lattice vectors for each unit cell
    R = np.matmul(Crys_coords, LattVects)
    
    # Calculate {exp(2j*pi*R*Q)}
    Matrix_exp = np.exp(1j*np.matmul(R, q))   # 2j*np.pi
    
    # Calculate F_q
    if np.size(Latt_forfac) == 1: 
        Crys_strfac = Latt_forfac * np.sum(Matrix_exp)
    else: 
        Crys_strfac = np.matmul(Latt_forfac, Matrix_exp)
    
    return Crys_strfac


def CalLat_CSF(q, PhotonE=12,LattVects=None,Zs=None,Atoms_coords=None, 
               Crys_coords=None,Disloc=None): 
    q = np.matrix([q]).transpose()
    Lattforfac = CalLat_lattice_structure_factor(q=q, PhotonE=PhotonE, LattVects=LattVects, 
                                  Zs=Zs, Atoms_coords=Atoms_coords, 
                                  Flag_InputReg=False)
    StruFactor = CalLat_crystal_structure_factor(q=q, Latt_forfac=Lattforfac, LattVects=LattVects,
                                  Crys_coords=Crys_coords+Disloc, 
                                  Flag_InputReg=False)
    return np.abs(StruFactor)**2


def CalLat_DifPatSimu_MultiProcess(CrySize=[3,3,3], CryShape='cubic', CryInFile=False, CryFile=r'Crystal.npy', 
                        DislocShape='gauss', DislocInFile=False, DislocFile=r'Disloc.npy', 
                        DisCent=[0,0,0], DisSize=[2,0,0], DisMagn =[0,0,0], 
                        CIF_path=r'/CIFs', CIF_file='Ag.cif', 
                        Crys_orie=[0,0,0], PhotonE=12,  
                        Delt_rang=1, Delt_step=0.05, Delt_cent=21.3,
                        Gamm_rang=1, Gamm_step=0.05, Gamm_cent=45.9, 
                        Rock_rang=1.2, Rock_step=0.06, Rock_cent=0, 
                        Rock_axis='X', IsRadian=False, RunSimu=False, 
                        MultiProcess=False, Processes_count=1, 
                        SaveResult=False, SaveName='SimuDifPat'): 
    if CryInFile: 
        Crys_coords = np.load(CryFile)
    else: 
        Crystal = CalLat_creat_crystal(L1=CrySize[0], L2=CrySize[1], L3=CrySize[2])
        Crys_coords = Crystal['Coords']
        Crys_datsha = Crystal['Shape']
        print('>>>>>> Crystal data shape is ', Crys_datsha)
    if DislocInFile: 
        Disloc = np.load(DislocFile)
        print('>>>>>> Crystal array shape is ', np.shape(Cyrs_coords))
        print('       Disloc  array shape is ', np.shape(Disloc))
    else: 
        if DislocShape == 'gauss': 
            Disloc = CalLat_creat_disloc_gauss(Crys_coords, GaussLoc=DisCent, GaussWid=DisSize, 
                               GaussDis=DisMagn, Progress=False)
        elif DislocShape == 'screw': 
            Disloc = CalLat_creat_disloc_screw(Crys_coords, ScrewLoc=DisCent, ScrewEdge=DisSize, 
                               ScrewDis=DisMagn, Progress=False)
    result = CalLat_read_cif(CIF_name=CIF_file, CIF_path=CIF_path)
    Diffraction = CalLat_diffraction_vector(Crys_orie=Crys_orie, PhotonE=PhotonE,  
                               Delt_rang=Delt_rang, Delt_step=Delt_step, Delt_cent=Delt_cent,
                               Gamm_rang=Gamm_rang, Gamm_step=Gamm_step, Gamm_cent=Gamm_cent, 
                               Rock_rang=Rock_rang, Rock_step=Rock_step, Rock_cent=Rock_cent, 
                               Rock_axis=Rock_axis, IsRadian=IsRadian)
    DiffVects = Diffraction['DiffVect']
    DiffShape = Diffraction['DataShape']
    dq = Diffraction['dq']
    Ax_gamma = Diffraction['GammaAxis']
    Ax_delta = Diffraction['DeltaAxis']
    Ax_rock = Diffraction['RockAxis']
    print('>>>>>> Array shape [rock, gamma, delta] is ', DiffShape)
    print('       dq in [ rock  ] is ', dq[0])
    print('       dq in [ gamma ] is ', dq[1])
    print('       dq in [ delta ] is ', dq[2])
    
    _,N = np.shape(DiffVects)
    LattVects = result['LattVectors']
    Zs = result['Atoms_Z']
    Atoms_coords = result['Atoms_coords']
    
    if not RunSimu: 
        return
    
    DifPat = np.zeros(N)
    if MultiProcess: 
        CalLat_CSF_partial = partial(CalLat_CSF, PhotonE=PhotonE, LattVects=LattVects, Zs=Zs, 
                                     Atoms_coords=Atoms_coords, Crys_coords=Crys_coords, Disloc=Disloc)
        pool = Pool(processes=Processes_count)
        par = [(DiffVects[0,0],DiffVects[1,0],DiffVects[2,0])]
        for i in range(N-1): 
            par.append((DiffVects[0,i+1],DiffVects[1,i+1],DiffVects[2,i+1]))
        print('>>>>>> Start %d-core multiprocessing ... ' %Processes_count 
              + strftime('%H:%M:%S', localtime()))
        DifPat = pool.map(CalLat_CSF_partial, par)
        print('       Processing finished! ' + strftime('%H:%M:%S', localtime()))
        pool.terminate()
    else: 
        print('>>>>>> Start calculating ... ' + strftime('%H:%M:%S', localtime()))
        for i in range(N): 
            q = tuple(np.squeeze(np.asarray(DiffVects[:,i])))
            DifPat[i] = CalLat_CSF(q, PhotonE=PhotonE, LattVects=LattVects, Zs=Zs, 
                            Atoms_coords=Atoms_coords, Crys_coords=Crys_coords, Disloc=Disloc)
            if np.remainder(i, 100) == 0: 
                print('       Processing %0.2f %% ... ' %((i+1)/N*100) + 
                      strftime('%H:%M:%S', localtime()), end='\r')
        print('       Processing 100.00 % ... Done! ' + strftime('%H:%M:%S', localtime()))
    DifPat = np.reshape(DifPat, DiffShape)
    DifPat = DifPat/np.max(DifPat)
    
    if SaveResult: 
        np.save(SaveName+'.npy', DifPat)
        Save_tif((DifPat*1e7).astype('int32'), SaveName+'.tif', OverWrite=True)
    return


def CalLat_CSF_inFile(Q_File, PhotonE=None, LattVects=None, Zs=None, ProgNum=100,
               Atoms_coords=None, Crys_coords_file=None, Disloc_file=None): 
    # Read q vectors from file
    DiffVects = np.matrix(np.load(Q_File))
    os.remove(Q_File)
    Crys_coords = np.load(Crys_coords_file)
    Disloc = np.load(Disloc_file)
    _,N = np.shape(DiffVects)
    DifPat = np.zeros(N)
    # Creat a progress file
    Progress_file = Q_File[:-4]+'_progress.dat'
    f = open(Progress_file, 'w+')
    f.write('Start processing %d voxels ... ' %N + strftime('%H:%M:%S', localtime()) + '\n' )
    f.write('\n')
    f.close()
    # Calculate diffraction intensity
    for i in range(N): 
        Lattforfac = CalLat_lattice_structure_factor(q=DiffVects[:,i], PhotonE=PhotonE, Zs=Zs, 
                                      LattVects=LattVects, Atoms_coords=Atoms_coords, 
                                      Flag_InputReg=False)
        StruFactor = CalLat_crystal_structure_factor(q=DiffVects[:,i], Crys_coords=Crys_coords+Disloc, 
                                      Latt_forfac=Lattforfac, LattVects=LattVects, 
                                      Flag_InputReg=False)
        DifPat[i] = np.abs(StruFactor)**2
        if np.remainder(i, ProgNum) == 0: 
            f = open(Progress_file, 'a')
            f.write('Processing %8s\t%8s %% ... ' %('%d'%(i+1), '%0.3f'%((i+1)/N*100))
                   + strftime('%H:%M:%S', localtime()) + '\n' )
            f.close()
    # Save result to a seperated file
    np.save(Q_File[:-4]+'_Result'+Q_File[-4:], DifPat)
    os.remove(Progress_file)


def CalLat_DifPatSimu_MultiNodes(CrySize=[3,3,3], CryShape='cubic', CryInFile=False, CryFile=r'Crystal.npy', 
                                 DislocShape='gauss', DislocInFile=False, DislocFile=r'Disloc.npy', 
                                 DisCent=[0,0,0], DisSize=[2,0,0], DisMagn =[0,0,0], 
                                 CIF_path=r'/CIFs', CIF_file='Ag.cif', 
                                 Crys_orie=[0,0,0], PhotonE=12,  
                                 Delt_rang=1, Delt_step=0.05, Delt_cent=21.3,
                                 Gamm_rang=1, Gamm_step=0.05, Gamm_cent=45.9, 
                                 Rock_rang=1.2, Rock_step=0.06, Rock_cent=0, 
                                 Rock_axis='X', IsRadian=False, RunSimu=False, 
                                 MultiProcess=False, Processes_count=1, ProgNum=100, 
                                 SaveResult=False, SaveName='SimuDifPat'): 
    if CryInFile: 
        Crys_coords = np.load(CryFile)
    else: 
        Crystal = CalLat_creat_crystal(L1=CrySize[0], L2=CrySize[1], L3=CrySize[2])
        Crys_coords = Crystal['Coords']
        Crys_datsha = Crystal['Shape']
        print('>>>>>> Crystal data shape is ', Crys_datsha)
    if DislocInFile: 
        Disloc = np.load(DislocFile)
        print('>>>>>> Crystal array shape is ', np.shape(Crys_coords))
        print('       Disloc  array shape is ', np.shape(Disloc))
    else: 
        if DislocShape == 'gauss': 
            Disloc = CalLat_creat_disloc_gauss(Crys_coords, GaussLoc=DisCent, GaussWid=DisSize, 
                               GaussDis=DisMagn, Progress=False)
        elif DislocShape == 'screw': 
            Disloc = CalLat_creat_disloc_screw(Crys_coords, ScrewLoc=DisCent, ScrewEdge=DisSize, 
                               ScrewDis=DisMagn, Progress=False)
    result = CalLat_read_cif(CIF_name=CIF_file, CIF_path=CIF_path)
    Diffraction = CalLat_diffraction_vector(Crys_orie=Crys_orie, PhotonE=PhotonE,  
                               Delt_rang=Delt_rang, Delt_step=Delt_step, Delt_cent=Delt_cent,
                               Gamm_rang=Gamm_rang, Gamm_step=Gamm_step, Gamm_cent=Gamm_cent, 
                               Rock_rang=Rock_rang, Rock_step=Rock_step, Rock_cent=Rock_cent, 
                               Rock_axis=Rock_axis, IsRadian=IsRadian)
    DiffVects = Diffraction['DiffVect']
    DiffShape = Diffraction['DataShape']
    dq = Diffraction['dq']
    Ax_gamma = Diffraction['GammaAxis']
    Ax_delta = Diffraction['DeltaAxis']
    Ax_rock = Diffraction['RockAxis']
    print('>>>>>> Array shape [rock, gamma, delta] is ', DiffShape)
    print('       dq in [ rock  ] is ', dq[0])
    print('       dq in [ gamma ] is ', dq[1])
    print('       dq in [ delta ] is ', dq[2])
    
    _,N = np.shape(DiffVects)
    LattVects = result['LattVectors']
    Zs = result['Atoms_Z']
    Atoms_coords = result['Atoms_coords']
    
    if not RunSimu: 
        print('>>>>>> Switch RunSimu to True to start simulation')
        return
    
    DifPat = np.zeros(N)
    if MultiProcess: 
        # creat a temp file name
        TempFileName = 'temp_A'
        i = 0
        while (os.path.isfile(TempFileName+'_%03d.npy' %0) or 
               os.path.isfile(TempFileName+'_%03d_progress.dat' %0)):  # check if file exists
            i = i + 1
            TempFileName = 'temp_' + chr(ord('A') + i)
        # for multi-nodes, save DiffVects to several .npy files
        print('>>>>>> Save q-vectors to .npy files ... ' + strftime('%H:%M:%S', localtime()))
        N_seg = int(np.ceil(N/Processes_count))
        for i in range(Processes_count): 
            CurrentFile = TempFileName+'_%03d.npy' %i
            if i == 0: 
                par = [CurrentFile]
            else: 
                par.append(CurrentFile)
            if (i+1) * N_seg <= N: 
                np.save(CurrentFile, DiffVects[:,i*N_seg:(i+1)*N_seg])
            else: 
                np.save(CurrentFile, DiffVects[:,i*N_seg:N])
        # for multi-nodes, save Crys_coords and Disloc to .npy files
        print('       Save Crys_coords/Disloc to files ... ' + strftime('%H:%M:%S', localtime()))
        Crys_coords_file = TempFileName + '_Crys_coords.npy'
        Disloc_file = TempFileName + '_Disloc.npy'
        np.save(Crys_coords_file, Crys_coords)
        np.save(Disloc_file, Disloc)
        # Multiprocessing
        CalLat_CSF_partial = partial(CalLat_CSF_inFile, PhotonE=PhotonE, ProgNum=ProgNum, 
                                     LattVects=LattVects, Zs=Zs, Atoms_coords=Atoms_coords, 
                                     Crys_coords_file=Crys_coords_file, Disloc_file=Disloc_file)
        pool = Pool(processes=Processes_count)
        print('>>>>>> Start %d-core multiprocessing ... ' %Processes_count 
              + strftime('%H:%M:%S', localtime()))
        DifPat = pool.map(CalLat_CSF_partial, par)
        print('       Processing finished! ' + strftime('%H:%M:%S', localtime()))
        pool.terminate()
        # Read and merge result files
        print('>>>>>> Read results from .npy files ... ' + strftime('%H:%M:%S', localtime()))
        for i in range(Processes_count): 
            CurrentFile = TempFileName + '_%03d_Result.npy' %i
            if i == 0:
                DifPat = np.load(CurrentFile)
            else: 
                DifPat = np.append(DifPat, np.load(CurrentFile))
            os.remove(CurrentFile)
        os.remove(Crys_coords_file)
        os.remove(Disloc_file)
    else: 
        print('>>>>>> Start calculating ... ' + strftime('%H:%M:%S', localtime()))
        for i in range(N): 
            q = tuple(np.squeeze(np.asarray(DiffVects[:,i])))
            DifPat[i] = CalLat_CSF(q, PhotonE=PhotonE, LattVects=LattVects, Zs=Zs, 
                            Atoms_coords=Atoms_coords, Crys_coords=Crys_coords, Disloc=Disloc)
            if np.remainder(i, 100) == 0: 
                print('       Processing %0.2f %% ... ' %((i+1)/N*100) + 
                      strftime('%H:%M:%S', localtime()), end='\r')
        print('       Processing 100.00 % ... Done! ' + strftime('%H:%M:%S', localtime()))
    DifPat = np.reshape(DifPat, DiffShape)
    DifPat = DifPat/np.max(DifPat)
    
    if SaveResult: 
        np.save(SaveName+'.npy', DifPat)
        Save_tif((DifPat*1e7).astype('int32'), SaveName+'.tif', OverWrite=True)
    return


# Need a version of "CalLat_diffraction_vector" that uses pixels instead of angular steps
def CalLat_diffraction_vector(Crys_orie=[0, 0, 0], PhotonE=12,  
                              Gamm_rang=5, Gamm_step=0.01, Gamm_cent=0, 
                              Delt_rang=5, Delt_step=0.01, Delt_cent=0, 
                              Rock_rang=1, Rock_step=0.01, Rock_cent=0, 
                              Rock_axis='Y', IsRadian=False): 
    ''' >>> Introduction <<< 
        This fuction calcualte the diffraction vectors in Crystal Frame, during a rocking
        curve scan. 
        
        Inputs: 
            Crys_orie             Crystal orientation in LAB FRAME
                                      (3,) the extrinsic rotation around X, Y, Z axes
            PhotonE               [keV] Photon energy
            Gamm_step/cent          Step-size/center of Gamma, unit is [deg]
            Delt_step/cent          Step-size/center of Delta, unit is [deg]
            Rock_step             Step-size of rocking angel, center is 0, unit is [deg] 
            Gamm/Delt/Rock_rang       Number of steps in half range. 
            Rock_axis             Rocking axis, either 'X' or 'Y'
            IsRadian              Angles are in [deg] or [rad]
        
        Note: 
            Gamma and Delta are defined as 34-ID-C
            Lab frame is defined as Z - downstream, Y - vertical, X - outboard
            
            !!! The output, Qs_crys is (3,n) array, where each (1,n) is raveled from a 3D array with 
            strides organized as [Z, Y, X]. 
                The result can be reshaped and transposed using command like: 
                    DifPat = DifPat.reshape(DataShape).transpose(2, 1, 0)
                where DataShape is the other output of this function, 
                    transpose(2, 1, 0) is aimed to reorganize the strides to [X, Y, Z]
    '''    
    WaveLen = 12.3980 / PhotonE   # [A]
    WaveVec = 2 * np.pi / WaveLen   # [A-1]
    
    # Calculate all the q in Lab Frame
    N_gamm = int(Gamm_rang * 2 + 1)   # N_gamm = int(np.round(Gamm_rang / Gamm_step) * 2 + 1)
    N_delt = int(Delt_rang * 2 + 1)   # N_delt = int(np.round(Delt_rang / Delt_step) * 2 + 1)
    N_rock = int(Rock_rang * 2 + 1)   # N_rock = int(np.round(Rock_rang / Rock_step) * 2 + 1)
    N = N_gamm*N_delt*N_rock
    
    X = np.asarray(range(N_delt)) - np.floor(N_delt/2)
    Y = np.asarray(range(N_gamm)) - np.floor(N_gamm/2)
    Z = np.asarray(range(N_rock)) - np.floor(N_rock/2)
    
    X2, Y2 = np.meshgrid(X, Y)   # strides organized as [Y, X]
    Y3, Z3, X3 = np.meshgrid(Y, Z, X)   # strides organized as [Z, Y, X]
    
    if not IsRadian: 
        Crys_orie = np.deg2rad(Crys_orie)
        
        Gamm_step = np.deg2rad(Gamm_step)
        Gamm_cent = np.deg2rad(Gamm_cent)
        
        Delt_step = np.deg2rad(Delt_step)
        Delt_cent = np.deg2rad(Delt_cent)
        
        Rock_step = np.deg2rad(Rock_step)
        Rock_cent = np.deg2rad(Rock_cent)
        
    DeltAxis = X2.ravel() * Delt_step + Delt_cent
    GammAxis = Y2.ravel() * Gamm_step + Gamm_cent
    
    Spherical = np.empty((3, int(N_gamm*N_delt)))
    Spherical[0, :] = np.abs(WaveVec)   # [A-1]
    Spherical[1, :] = np.pi/2 - GammAxis
    Spherical[2, :] = DeltAxis
    
    Cartesian = Spherical2Cartesian(Spherical, IsRadian=True)
    Cartesian[0] = Cartesian[0] - WaveVec
    if Rock_axis == 'Y' or Rock_axis == 'y': 
        ''' Crystal is rotated around Y axis. 
            Therfore, it is easier to have Angle2 defined as rotation around Y axis.
            The current Cartesian is already ordered as [Z, X, Y]. No change required.
        '''
        NewOrder = np.asarray([0, 1, 2])
    elif Rock_axis == 'X' or Rock_axis == 'x':
        ''' Crystal is rotated around X axis. 
            Therfore, it is easier to have Angle2 defined as rotation around X axis, 
            i.e. it is better to order Cartesian as [Y, Z, X]. 
            The current Cartesian is ordered as [Z, X, Y]. Reordering required.
        '''
        NewOrder = np.asarray([2, 0, 1])
    else: 
        print('>>>>>> Unknown rocking axis. ')
        input('>>>>>> Press any key to quit...')
        return
    Spherical = Cartesian2Spherical(Cartesian[NewOrder, :], OutputRadian=True)
    
    Rho    = np.repeat(Spherical[0,:].reshape((N_gamm, N_delt))[np.newaxis, :, :], 
                       N_rock, axis=0)
    Angle1 = np.repeat(Spherical[1,:].reshape((N_gamm, N_delt))[np.newaxis, :, :], 
                       N_rock, axis=0)
    Angle2 = np.repeat(Spherical[2,:].reshape((N_gamm, N_delt))[np.newaxis, :, :], 
                       N_rock, axis=0)
    Angle2 = Angle2 - Z3 * Rock_step + Rock_cent   # strides organized as [Z, Y, X]
    
    del Spherical
    Spherical = np.stack((Rho.ravel(), Angle1.ravel()), axis=0)
    Spherical = np.concatenate((Spherical, [Angle2.ravel()]), axis=0)
    
    del Cartesian
    Cartesian = Spherical2Cartesian(Spherical, IsRadian=True)   # [Z, X, Y]
    if Rock_axis == 'Y' or Rock_axis == 'y': 
        '''Cartesian is ordered as [Z, X, Y]'''
        NewOrder = np.asarray([1, 2, 0])
    elif Rock_axis == 'X' or Rock_axis == 'x': 
        '''Cartesian is ordered as [Y, Z, X]'''
        NewOrder = np.asarray([2, 0, 1])
    
    Qs_lab = Cartesian[NewOrder, :] # (3,n) array, each (3,1) is a q(x,y,z) in lab frame
    del Cartesian, Spherical, DeltAxis, GammAxis
    
    # Convert Qs from Lab Frame to Crystal Frame
    RotX = Rot_Matrix([1,0,0], -Crys_orie[0], IsRadian=True)
    RotY = Rot_Matrix([0,1,0], -Crys_orie[1], IsRadian=True)
    RotZ = Rot_Matrix([0,0,1], -Crys_orie[2], IsRadian=True)
    
    Qs_crys = np.matmul(RotX, np.matmul(RotY, np.matmul(RotZ, Qs_lab)))
    
    # Some parameters for output
    DeltAxis = X * Delt_step + Delt_cent
    GammAxis = Y * Gamm_step + Gamm_cent
    RockAxis = Z * Rock_step + Rock_cent
    
    ki = np.asarray([0, 0, WaveVec])
    kf = np.asarray([WaveVec*np.cos(Gamm_cent)*np.sin(Delt_cent), 
                   WaveVec*np.sin(Gamm_cent), 
                   WaveVec*np.cos(Gamm_cent)*np.cos(Delt_cent)])
    Q = kf - ki
    
    # dq_Delta = np.sin(Delt_step/2) * WaveVec * 2
    # dq_Gamma = np.sin(Gamm_step/2) * WaveVec * 2
    dq_Delta = WaveVec*np.asarray([np.cos(Gamm_cent)*np.sin(Delt_cent+Delt_step), 
                        np.sin(Gamm_cent), 
                        np.cos(Gamm_cent)*np.cos(Delt_cent+Delt_step)-1]) - Q
    dq_Gamma = WaveVec*np.asarray([np.cos(Gamm_cent+Gamm_step)*np.sin(Delt_cent), 
                        np.sin(Gamm_cent+Gamm_step), 
                        np.cos(Gamm_cent+Gamm_step)*np.cos(Delt_cent)-1]) - Q
    if Rock_axis == 'Y' or Rock_axis == 'y': 
        ''' Crystal is rotated around Y axis. '''
        # Qy = np.dot(Q, [0, 1, 0])
        # Q_proj = Q - np.asarray([0, Qy, 0])
        T_rock = Rot_Matrix([0, 1, 0], Rock_step, IsRadian=True)
    elif Rock_axis == 'X' or Rock_axis == 'x':
        ''' Crystal is rotated around X axis. '''
        # Qx = np.dot(Q, [1, 0, 0])
        # Q_proj = Q - np.asarray([Qx, 0, 0])
        T_rock = Rot_Matrix([1, 0, 0], Rock_step, IsRadian=True)
    else: 
        print('>>>>>> Unknown rocking axis. ')
        input('>>>>>> Press any key to quit...')
        return
    # dq_Rock  = np.sin(Rock_step/2) * np.linalg.norm(Q_proj) * 2
    dq_Rock = np.squeeze(np.asarray(np.matmul(T_rock, Q) - Q))
    
    return {'DiffVect':Qs_crys, 'DataShape':(N_rock, N_gamm, N_delt), 
            'DeltaAxis':DeltAxis, 'GammaAxis':GammAxis, 'RockAxis':RockAxis,
            'dq':(np.linalg.norm(dq_Rock), np.linalg.norm(dq_Gamma), np.linalg.norm(dq_Delta))}


def CalLat_gridsize_estimate(photonE=None, delta=None, gamma=None, drock=None, rockaxis='Y', nrock=None, 
                            nx=None, ny=None, detdist=None, px=None, py=None, disp_info=True): 
    ''' >>> Instruction <<< 
        This function calculate grid size in real and reciprocal spaces based on the diffraction geometry. 
        
        Note this is just a ROUGH estimation: 
            Ewald sphere is approximated to a flat plane. 
            Detector surface grid is approximated to a regular grid in reciprocal space. 
        
        Inputs: 
            photonE:     Photon energy in [keV]
            delta:       Detector angle around Lab Y axis in [deg]
            gamma:       Detector angle around Lab -X axis in [deg]
            drock:       Step of rocking curve scan in [deg]
            nrock:       number of rocking steps 
            rockaxis:    Rocking axis. Note drock may be negative for rocking around X. 
            detdist:     Sample-detector distance in [m]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [um]
            nx, ny:      Number of pixels in X(horizontal) and Y(vertical)
            disp_info:   Whether print out all parameters
        
        Output: 
            No output yet. Just printing out results. 
    '''
    wavelen = 1.2398 / photonE   # [nm]
    wavevec = 2 * np.pi / wavelen   # [nm-1]
    if disp_info: 
        print('photonE: %.3f [keV]' %photonE)
        print('wavelen: %.3f [nm]' %wavelen)
        print('wavevec: %.3f [nm-1]' %wavevec)
        print('')
        print('delta: %.3f [deg] (around +Y axis)' %delta)
        print('gamma: %.3f [deg] (around -X axis)' %gamma)
        print('drock: %.3f [deg] (around +Y or -X)' %drock)
        print('')
        print('detdist: %.3f [m]' %detdist)
        print('px: %.1f [um] (along -X when delta,gamma = 0)' %px)
        print('py: %.1f [um] (along +Y when delta,gamma = 0)' %py)
        print('')
    
    # Convert units 
    delta = np.deg2rad(delta)
    gamma = np.deg2rad(gamma)
    drock = np.deg2rad(drock)
    detdist *= 1e3   # [mm]
    px *= 1e-3   # [mm]
    py *= 1e-3   # [mm]
    
    # Rotation matrices of kf and detector 
    R_delta = Rot_Matrix([0,1,0], delta, IsRadian=True)
    R_gamma = Rot_Matrix([-1,0,0], gamma, IsRadian=True)
    
    # Calculate Q 
    ki = wavevec * np.asarray([0, 0, 1]) 
    kf = Rotate_vectors( Rotate_vectors( ki, R_gamma ), R_delta )
    Q = kf - ki   # [nm-1]
    
    # Grid in reciprocal space 
    ang_x_step = np.arctan2(px, detdist)   # [rad]
    ang_y_step = np.arctan2(py, detdist)   # [rad]
    ang_r_step = drock   # [rad]
    ang_x_edge = np.arctan2(px * np.floor(nx/2), detdist)   # [rad]
    ang_y_edge = np.arctan2(py * np.floor(ny/2), detdist)   # [rad]
    ang_r_edge = drock * np.floor(nrock/2)   # [rad]
    
    # dq_x/dq_y 
    if True: 
        # Rotation matrices of pixels
        R_detx_step = Rot_Matrix(Rotate_vectors(Rotate_vectors([0,-1,0], 
                                                               R_gamma), 
                                                R_delta), 
                                 ang_x_step)
        R_dety_step = Rot_Matrix(Rotate_vectors(Rotate_vectors([-1,0,0], 
                                                               R_gamma), 
                                                R_delta), 
                                 ang_y_step)
        R_detx_edge = Rot_Matrix(Rotate_vectors(Rotate_vectors([0,-1,0], 
                                                               R_gamma), 
                                                R_delta), 
                                 ang_x_edge)
        R_dety_edge = Rot_Matrix(Rotate_vectors(Rotate_vectors([-1,0,0], 
                                                               R_gamma), 
                                                R_delta), 
                                 ang_y_edge)

        # dq_x/dq_y
        dq_x_step = Rotate_vectors( kf, R_detx_step ) - ki - Q
        dq_y_step = Rotate_vectors( kf, R_dety_step ) - ki - Q
        dq_x_edge = Rotate_vectors( kf, R_detx_edge ) - ki - Q
        dq_y_edge = Rotate_vectors( kf, R_dety_edge ) - ki - Q
    else: 
        ''' >>> An alternative way to calculate dq <<< '''
        # kx/ky of each pixel at delta = 0 and gamma = 0
        k0_x = Rotate_vectors( ki, Rot_Matrix([0,-1,0], ang_x_step, IsRadian=True) )
        k0_y = Rotate_vectors( ki, Rot_Matrix([-1,0,0], ang_y_step, IsRadian=True) )
        k0_x_m = Rotate_vectors( ki, Rot_Matrix([0,-1,0], ang_x_edge, IsRadian=True) )
        k0_y_m = Rotate_vectors( ki, Rot_Matrix([-1,0,0], ang_y_edge, IsRadian=True) )

        # Rotate kx/ky to delta and gamma
        k_x = Rotate_vectors( Rotate_vectors( k0_x, R_gamma ), R_delta )
        k_y = Rotate_vectors( Rotate_vectors( k0_y, R_gamma ), R_delta )
        k_x_m = Rotate_vectors( Rotate_vectors( k0_x_m, R_gamma ), R_delta )
        k_y_m = Rotate_vectors( Rotate_vectors( k0_y_m, R_gamma ), R_delta )

        # dq_x/dq_y
        dq_x_step = k_x - ki - Q
        dq_y_step = k_y - ki - Q
        dq_x_edge = k_x_m - ki - Q
        dq_y_edge = k_y_m - ki - Q
    
    # dq_r 
    if rockaxis == 'Y' or rockaxis == 'y': 
        '''Crystal is rotated around Y axis. '''
        T_rock = Rot_Matrix([0, -1, 0], ang_r_step, IsRadian=True)
    elif rockaxis == 'X' or rockaxis == 'x': 
        '''Crystal is rotated around X axis. '''
        T_rock = Rot_Matrix([-1, 0, 0], ang_r_step, IsRadian=True)
    else: 
        print('>>>>>> Unknown RockAxis. ')
        input('       Press any key to quit...')
        return
    dq_r_step = np.squeeze(np.asarray(np.matmul(T_rock, Q) - Q))
    
    if rockaxis == 'Y' or rockaxis == 'y': 
        '''Crystal is rotated around Y axis. '''
        T_rock = Rot_Matrix([0, -1, 0], ang_r_edge, IsRadian=True)
    elif rockaxis == 'X' or rockaxis == 'x': 
        '''Crystal is rotated around X axis. '''
        T_rock = Rot_Matrix([-1, 0, 0], ang_r_edge, IsRadian=True)
    else: 
        print('>>>>>> Unknown RockAxis. ')
        input('       Press any key to quit...')
        return
    dq_r_edge = np.squeeze(np.asarray(np.matmul(T_rock, Q) - Q))
    
    # Grid in real space 
    ''' dr of entire grid '''
    denorm = np.dot(dq_x_step, np.cross(dq_y_step, dq_r_step))
    dr_x_edge = 2 * np.pi * np.cross(dq_y_step, dq_r_step) / denorm
    dr_y_edge = 2 * np.pi * np.cross(dq_r_step, dq_x_step) / denorm
    dr_r_edge = 2 * np.pi * np.cross(dq_x_step, dq_y_step) / denorm
    ''' dr of each voxel '''
    denorm = np.dot(dq_x_edge, np.cross(dq_y_edge, dq_r_edge))
    dr_x_step = 2 * np.pi * np.cross(dq_y_edge, dq_r_edge) / denorm
    dr_y_step = 2 * np.pi * np.cross(dq_r_edge, dq_x_edge) / denorm
    dr_r_step = 2 * np.pi * np.cross(dq_x_edge, dq_y_edge) / denorm
    
    if disp_info: 
        print('ki: [%.3f, %.3f, %.3f] [nm-1]' %(ki[0], ki[1], ki[2]))
        print('kf: [%.3f, %.3f, %.3f] [nm-1]' %(kf[0], kf[1], kf[2]))
        print('Q:  [%.3f, %.3f, %.3f] [nm-1]' %(Q[0], Q[1], Q[2]))
        print('')
        print('dq_x:     [%.3f, %.3f, %.3f] [um-1]' %(dq_x_step[0]*1e3, dq_x_step[1]*1e3, dq_x_step[2]*1e3))
        print('dq_y:     [%.3f, %.3f, %.3f] [um-1]' %(dq_y_step[0]*1e3, dq_y_step[1]*1e3, dq_y_step[2]*1e3))
        print('dq_rock:  [%.3f, %.3f, %.3f] [um-1]' %(dq_r_step[0]*1e3, dq_r_step[1]*1e3, dq_r_step[2]*1e3))
        print('========== Just a ROUGH estimation ==========')
        print('Reciprocal space: ')
        print('  > Resolution: ')
        print('        dq_x:     %.3f [um-1]' %(np.linalg.norm(dq_x_step)*1e3))
        print('        dq_y:     %.3f [um-1]' %(np.linalg.norm(dq_y_step)*1e3))
        print('        dq_rock:  %.3f [um-1]' %(np.linalg.norm(dq_r_step)*1e3))
        print('  > Field of view: ')
        print('        q_x:      %.3f [um-1] (+/-)' %(np.linalg.norm(dq_x_edge)*1e3))
        print('        q_y:      %.3f [um-1] (+/-)' %(np.linalg.norm(dq_y_edge)*1e3))
        print('        q_rock:   %.3f [um-1] (+/-)' %(np.linalg.norm(dq_r_edge)*1e3))
        print('-----------------------------------')
        print('Real space: ')
        print('  > Field of view: ')
        print('        r_x:     %.3f [nm]' %(np.linalg.norm(dr_x_edge)))
        print('        r_y:     %.3f [nm]' %(np.linalg.norm(dr_y_edge)))
        print('        r_rock:  %.3f [nm]' %(np.linalg.norm(dr_r_edge)))
        print('  > Resolution: ')
        print('        dr_x:     %.3f [nm], %.3f [nm] (half period)' %(np.linalg.norm(dr_x_step), np.linalg.norm(dr_x_step)/2))
        print('        dr_y:     %.3f [nm], %.3f [nm] (half period)' %(np.linalg.norm(dr_y_step), np.linalg.norm(dr_y_step)/2))
        print('        dr_rock:  %.3f [nm], %.3f [nm] (half period)' %(np.linalg.norm(dr_r_step), np.linalg.norm(dr_r_step)/2))
        print('  > Vectors: ')
        print('        dr_x:     [%.3f, %.3f, %.3f] [nm]' %(dr_x_step[0], dr_x_step[1], dr_x_step[2]))
        print('        dr_y:     [%.3f, %.3f, %.3f] [nm]' %(dr_y_step[0], dr_y_step[1], dr_y_step[2]))
        print('        dr_rock:  [%.3f, %.3f, %.3f] [nm]' %(dr_r_step[0], dr_r_step[1], dr_r_step[2]))
        print('      Half period, i.e. FOV/N')
        print('        dr_x:     [%.3f, %.3f, %.3f] [nm]' %(dr_x_edge[0]/nx, dr_x_edge[1]/nx, dr_x_edge[2]/nx))
        print('        dr_y:     [%.3f, %.3f, %.3f] [nm]' %(dr_y_edge[0]/ny, dr_y_edge[1]/ny, dr_y_edge[2]/ny))
        print('        dr_rock:  [%.3f, %.3f, %.3f] [nm]' %(dr_r_edge[0]/nrock, dr_r_edge[1]/nrock, dr_r_edge[2]/nrock))
        
    return


def CalLat_beamspec_estimate(z=0, fwhm=1e3, wave_len=0.1): 
    ''' >>> Instruction <<< 
        This function calculates common beam properties of a Gaussian beam
        like Rayleigh range and the radius of curvature at a specific z position. 
        
        Input:
            z               [nm], z position, where z = 0 is the beam waist. 
            fwhm            [nm], fwhm at the beam waist. 
            wave_len        [nm], wavelength of the incident wave.  
        
        Output: 
            Print out specifications. 
    '''
    w_0 = fwhm / 2   # [nm], beam radius at the waist 
    s_0 = np.sqrt(np.pi) * w_0   # [nm], scale parameter
    z_bar = s_0**2 / wave_len   # [nm], Rayleigh range
    w  = w_0 * np.sqrt(1 + (z /z_bar)**2)   # [nm], beam radius at z
    s  = np.sqrt(np.pi) * w    # [nm], scale parameter for wf
    R  = z  * ( 1 + (z_bar / z )**2 )   # [nm], radius of curvature at z
    
    unit = '[nm]'; temp = fwhm
    if   np.abs(temp) > 1e9: 
        unit = '[m]' ; temp /= 1e9
    elif np.abs(temp) > 1e6: 
        unit = '[mm]'; temp /= 1e6
    elif np.abs(temp) > 1e3: 
        unit = '[um]'; temp /= 1e3
    print('fwhm_0 = %.3f ' %temp + unit + '   # fwhm at waist')
    unit = '[nm]'; temp = z_bar
    if   np.abs(temp) > 1e9: 
        unit = '[m]' ; temp /= 1e9
    elif np.abs(temp) > 1e6: 
        unit = '[mm]'; temp /= 1e6
    elif np.abs(temp) > 1e3: 
        unit = '[um]'; temp /= 1e3
    print('z_bar  = %.3f ' %temp + unit + '   # Rayleigh range')
    unit = '[nm]'; temp = w * 2
    if   np.abs(temp) > 1e9: 
        unit = '[m]' ; temp /= 1e9
    elif np.abs(temp) > 1e6: 
        unit = '[mm]'; temp /= 1e6
    elif np.abs(temp) > 1e3: 
        unit = '[um]'; temp /= 1e3
    print('fwhm_z = %.3f ' %temp + unit + '   # fwhm at z')
    unit = '[nm]'; temp = R
    if   np.abs(temp) > 1e9: 
        unit = '[m]' ; temp /= 1e9
    elif np.abs(temp) > 1e6: 
        unit = '[mm]'; temp /= 1e6
    elif np.abs(temp) > 1e3: 
        unit = '[um]'; temp /= 1e3
    print('R_o_C  = %.3f ' %temp + unit + '   # radius of curvature')
    return


def CalLat_dq(Lambda=None, PhotonE=None, Delta=None, Gamma=None, dRock=None, RockAxis='Y', 
              dDelta=None, dGamma=None, detdist=None, px=None, py=None, IsRadian=False, 
              disp_info=False):
    ''' >>> Instruction <<< 
        This function calculate dQ based on the diffraction geometry.
        This function is written for forward simulation under kinematic approximation.
        
        * Definitions of inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Delta:       Detector angle around Lab Y axis
            Gamma:       Detector angle around Lab X axis (May need to input negative value)
            dRock:       Step of rocking curve scan
            RockAxis:    Rocking axis
            dGamma:      Step in Gamma. Will calculate from py if this is None. 
            dDelta:      Step in Delta. Will calculate from px if this is None. 
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, all the angles are in [rad]; otherwise, [deg]
            disp_info:   Whether print out all parameters
        
        * Output: 
            
    '''
    # Inputs regularization
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE   # [A]
    WaveVec = 2 * np.pi / Lambda   # [A-1]
    
    if Delta is None or Gamma is None: 
        print('>>>>>> Delta and/or Gamma is missing. ')
        input('       Press any key to quit...')
        return
    if dRock is None: 
        print('>>>>>> dRock is missing. ')
        input('       Press any key to quit...')
        return
    if dDelta is None: 
        dDelta = 0
    if dGamma is None: 
        dGamma = 0
    if px is None: 
        px = 0
    if py is None: 
        py = 0
    if detdist is None: 
        detdist = 0
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dRock = np.deg2rad(dRock)
    
    if dDelta == 0 or dGamma == 0: 
        # print('>>>>>> Calculate dDelta and dGamma from px, py, and detdist. ')
        dDelta = np.arctan2(px, detdist)
        dGamma = np.arctan2(py, detdist)
        # print('       dDelta is %f rad' %dDelta)
        # print('       dGamma is %f rad' %dGamma)
    else: 
        if not IsRadian: 
            dDelta = np.deg2rad(dDelta)
            dGamma = np.deg2rad(dGamma)
        if detdist != 0: 
            px = np.tan(dDelta) * detdist
            py = np.tan(dGamma) * detdist
    
    # Calculate Q
    ki = np.asarray([0, 0, WaveVec]) 
    kf = WaveVec * np.asarray([ np.cos(Gamma) * np.sin(Delta), 
                             -np.sin(Gamma), 
                              np.cos(Gamma) * np.cos(Delta)])
    Q = kf - ki   # [A-1]
    
    # Calculate dQ
    # dq_Delta = np.sin(dDelta/2) * WaveVec * 2
    # dq_Gamma = np.sin(dGamma/2) * WaveVec * 2
    dq_Delta = WaveVec * np.asarray([ np.cos(Gamma) * np.sin(Delta + dDelta), 
                                   -np.sin(Gamma), 
                                    np.cos(Gamma) * np.cos(Delta + dDelta) - 1]) - Q
    dq_Gamma = WaveVec * np.asarray([ np.cos(Gamma + dGamma) * np.sin(Delta), 
                                   -np.sin(Gamma + dGamma), 
                                    np.cos(Gamma + dGamma) * np.cos(Delta) - 1]) - Q
    if RockAxis == 'Y' or RockAxis == 'y': 
        '''Crystal is rotated around Y axis. '''
        # Qy = np.dot(Q, [0, 1, 0])
        # Q_proj = Q - np.asarray([0, Qy, 0])
        T_rock = Rot_Matrix([0, 1, 0], dRock, IsRadian=True)
    elif RockAxis == 'X' or RockAxis == 'x': 
        '''Crystal is rotated around X axis. '''
        # Qx = np.dot(Q, [1, 0, 0])
        # Q_proj = Q - np.asarray([Qx, 0, 0])
        T_rock = Rot_Matrix([1, 0, 0], dRock, IsRadian=True)
    else: 
        print('>>>>>> Unknown RockAxis. ')
        input('       Press any key to quit...')
        return
    # dq_Rock  = np.sin(dRock/2) * np.linalg.norm(Q_proj) * 2
    dq_Rock = np.squeeze(np.asarray(np.matmul(T_rock, Q) - Q))
    
    q_delta = np.linalg.norm(dq_Delta)
    q_gamma = np.linalg.norm(dq_Gamma)
    q_rock  = np.linalg.norm(dq_Rock)
    
    denorm = np.dot(dq_Delta, np.cross(dq_Gamma, dq_Rock))
    dr_Delta = 2 * np.pi * np.cross(dq_Gamma, dq_Rock ) / denorm
    dr_Gamma = 2 * np.pi * np.cross(dq_Rock,  dq_Delta) / denorm
    dr_Rock  = 2 * np.pi * np.cross(dq_Delta, dq_Gamma) / denorm
    
    r_delta = np.linalg.norm(dr_Delta)
    r_gamma = np.linalg.norm(dr_Gamma)
    r_rock  = np.linalg.norm(dr_Rock)
    
    if disp_info: 
        print('Lambda: %.3f [A]' %Lambda)
        print('WaveVec: %.3f [A-1]' %WaveVec)
        print('detdist: %.3f [m]' %(detdist/1e3))
        print('px, py: %.1f, %.1f [um]' %(px*1e3, py*1e3))
        print('ki: [%.3f, %.3f, %.3f] [um-1]' %(ki[0]*1e4, ki[1]*1e4, ki[2]*1e4))
        print('kf: [%.3f, %.3f, %.3f] [um-1]' %(kf[0]*1e4, kf[1]*1e4, kf[2]*1e4))
        print('Q:  [%.3f, %.3f, %.3f] [um-1]' %(Q[0]*1e4, Q[1]*1e4, Q[2]*1e4))
        print('')
        print('dq_Delta:  [%.3f, %.3f, %.3f] [um-1]' %(dq_Delta[0]*1e4, dq_Delta[1]*1e4, dq_Delta[2]*1e4))
        print('dq_Gamma:  [%.3f, %.3f, %.3f] [um-1]' %(dq_Gamma[0]*1e4, dq_Gamma[1]*1e4, dq_Gamma[2]*1e4))
        print('dq_Rock:   [%.3f, %.3f, %.3f] [um-1]' %(dq_Rock[0]*1e4, dq_Rock[1]*1e4, dq_Rock[2]*1e4))
        print('q_delta: %.3f [um-1]' %(q_delta*1e4))
        print('q_gamma: %.3f [um-1]' %(q_gamma*1e4))
        print('q_rock:  %.3f [um-1]' %(q_rock*1e4))
        print('')
        print('dr_Delta:  [%.3f, %.3f, %.3f] [um]' %(dr_Delta[0]/1e4, dr_Delta[1]/1e4, dr_Delta[2]/1e4))
        print('dr_Gamma:  [%.3f, %.3f, %.3f] [um]' %(dr_Gamma[0]/1e4, dr_Gamma[1]/1e4, dr_Gamma[2]/1e4))
        print('dr_Rock:   [%.3f, %.3f, %.3f] [um]' %(dr_Rock[0]/1e4, dr_Rock[1]/1e4, dr_Rock[2]/1e4))
        print('r_delta: %.3f [um]' %(r_delta/1e4))
        print('r_gamma: %.3f [um]' %(r_gamma/1e4))
        print('r_rock:  %.3f [um]' %(r_rock/1e4))
    
    return {'dq_Rock':dq_Rock, 'dq_Gamma':dq_Gamma, 'dq_Delta':dq_Delta, 'Q': Q, 
            'dr_Rock':dr_Rock, 'dr_Gamma':dr_Gamma, 'dr_Delta':dr_Delta, 
            'q_rock':q_rock, 'q_gamma':q_gamma, 'q_delta':q_delta, 
            'r_rock':r_rock, 'r_gamma':r_gamma, 'r_delta':r_delta, 
            'dDelta':dDelta, 'dGamma':dGamma, 'detdist':detdist, 'px':px, 'py':py}


def CalLat_angle2q(photonE=None, wav_len=None, angle=None, q=None, d=None, IsRadian=False, PrintInfo=False): 
    ''' >>> Instruction <<< 
        This function does convertion between scattering angle and spatial frequency q, 
        based on photon energy or wave length. 
        
        Units:  
            photonE:      [keV]
            wav_len:      [nm]
            angle:        [deg] or [rad], depending on "IsRadian"
            q:            [nm-1]
            d:            [nm]
    '''
    if photonE is None: 
        if wav_len is None: 
            print(">>>>>> No photon energy or wavelength is defined ! <<<<<<")
            return None
        else: 
            wav_vec = 2 * np.pi / wav_len   # [nm-1]
            photonE = 1.2398 / wav_len   # [keV]
    else: 
        if wav_len is None: 
            wav_len = 1.2398 / photonE   # [nm]
            wav_vec = 2 * np.pi / wav_len   # [nm-1]
        else: 
            print(">>>>>> Both photon energy and wavelength are defined ! <<<<<<")
            return None
    
    count = 0
    if angle is None: 
        count += 1
    if q is None: 
        count += 1
    if d is None: 
        count += 1
    if count != 2: 
        print(">>>>>> Should define one and only one of (angle, q, d) ! <<<<<<")
        return None
    
    if angle is not None: 
        if IsRadian: 
            q = np.sin(angle) * wav_vec   # [nm-1]
        else: 
            q = np.sin(np.deg2rad(angle)) * wav_vec   # [nm-1]
        d = 2 * np.pi / q   # [nm]
    else: 
        if q is None: 
            q = 2 * np.pi / d   # [nm-1]
            angle = np.arcsin( q / wav_vec)   # [rad]
        else: 
            d = 2 * np.pi / q   # [nm]
            angle = np.arcsin( q / wav_vec)   # [rad]
        if not IsRadian: 
            angle = np.rad2deg(angle)   # [deg]
    
    if PrintInfo: 
        print('Photon energy:    %.3f [keV]' %photonE)
        print('Wavelength:       %.3f [nm]' %wav_len)
        print('Wave vector:      %.3f [nm-1]' %wav_vec)
        if IsRadian: 
            print('Scattering angle: %.3g [rad], %.3g [deg]' %(angle, np.rad2deg(angle)))
        else: 
            print('Scattering angle: %.3g [deg], %.3g [rad]' %(angle,np.deg2rad(angle)))
        print('Spatial freq.:    %.3g [nm-1]' %q)
        print('Real space dimension: %.3g [nm]' %d)
    return {'angle': angle, 'q': q, 'd': d}


def CalLat_Bragg_intensity(CIF_name='test.cif', CIF_path='', HKL=[[0, 0, 1],[1, 1, 1]], 
                           PhotonE=12, UC_params=[0,0,0,0,0,0], 
                           Normalize=True, LatticeInfo=True, Progress=False): 
    ''' >>> Instruction <<< 
        This function calculates the Structure Factor of a lattice. 
        ==============================================================================
        Working on adding Ionic form factor rather than Atomic form factor. 
        ==============================================================================
    '''
    # Get the current working directory
    retval = os.getcwd()
    
    # Generate the full file path and check if the file exists
    if len(CIF_path) == 0: 
        CIF_path = retval
    if CIF_path[-1] == '/' or CIF_path[-1] == '\\':
        File = CIF_path + CIF_name
    elif '/' in CIF_path: 
        File = CIF_path + '/' + CIF_name
    elif '\\' in CIF_path: 
        File = CIF_path + '\\' + CIF_name
    else: 
        print('>>>>>> Error! Unknown path separator. <<<<<<')
        input('>>>>>> Press any key to quit...')
        return
    if not os.path.isfile(File): 
        print('>>>>>> Error! File does NOT exist. <<<<<<')
        input('>>>>>> Press any key to quit...')
        return
    
    # Input Regularization & Initialize the output arrays
    HKL = np.asarray(HKL)
    
    if np.size(np.shape(HKL)) > 2: 
        print(">>>>>> Input HKL's dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        input('>>>>>> Press any key to quit...')
        return
    if np.shape(HKL)[0] == 3:
        M = int( np.size(HKL) / 3 )
        HKL = HKL.transpose()
    elif np.shape(HKL)[1] == 3: 
        M = np.shape(HKL)[0]
    else: 
        print(">>>>>> Input HKL's size is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        input('>>>>>> Press any key to quit...')
        return
    
    TwoTheta = np.zeros((M, 1))
    Intensity = np.zeros((M, 1))
    
    # Read cif file
    cif_data = Crystal.from_cif(File)
    if LatticeInfo: 
        print(cif_data)
    a, b, c, alpha, beta, gamma = UC_params
    if a*b*c*alpha*beta*gamma == 0:    # read lattice shape from cif if it is not defined
        a, b, c, alpha, beta, gamma = cif_data.lattice_parameters
        
    Atoms = np.asarray(cif_data)
    N,_ = np.shape(Atoms)   # N is the number of atoms in the unit cell
    Z = np.zeros((N, 1))
    Atoms_coords = np.zeros((N, 3))
    Atomic_form_factors = np.zeros((N, 1), dtype=np.complex)
    
    Z[:, 0] = Atoms[:, 0]
    Atoms_coords[:, :] = Atoms[:, 1:]
    
    # Calculate 2-theta of the corresponding HKL
    WaveLen = 12.3980 / PhotonE   # unit is [A]
    if Progress:
        print('>>>>>> Start calculating 2-theta ... ' + strftime("%H:%M:%S", localtime()) )
    TwoTheta = CalLat_2Theta(HKL=HKL, uc=[a, b, c, alpha, beta, gamma],
                             PhotonE=PhotonE, display=False).reshape((M,1))
    
    # Calculate the structure factor for each HKL
    if Progress:
        print('>>>>>> Start calculating intensity ... ' + strftime("%H:%M:%S", localtime()) )
    for i in range(M):
        # Display the progress
        if Progress: 
            print('>>>>>> Processing %04d/%04d ... ' %(i, M) 
                  + strftime("%H:%M:%S", localtime()), end='\r')
        # Calculate the momentum transfer
        Q = 4 * np.pi * np.sin(np.deg2rad(TwoTheta[i]/2)) / WaveLen
        q = np.sin(np.deg2rad(TwoTheta[i]/2)) / WaveLen
        # Convert hkl to a (3, 1) array
        hkl = np.reshape(HKL[i],(3,1))
        # Calculate Atomic Form Factors
        '''
        The complete atomic form factor f(q,E) = f1 + if2 = f0(q,E) + f'(E) + if"(E)
        Here, 
            f0 is from the Xraylib function FF_Rayl(Z, q)
            f' is from the Xraylib function Fi(Z, E)
            f" is from the Xraylib function Fii(Z, E)
        Note: f0 should be 'Ionic' rather than 'Atomic'. Will update later.
         
        http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
        '''
        for m in range(N): 
            f0  = xb.FF_Rayl(int(Z[m]), float(q))
            fi  = Fi(int(Z[m]), PhotonE)
            fii = Fii(int(Z[m]), PhotonE)
            Atomic_form_factors[m] = f0 + fi - 1j * fii
        # Calculate exp(hx+ky+lz) for each atom
        Matrix_exp = np.exp(2j*np.pi*np.matmul(Atoms_coords, hkl))
        # Calculate F_hkl
        F_hkl = np.matmul(Atomic_form_factors.transpose(), Matrix_exp)
        # Calculate the Intensity
        Intensity[i] = np.absolute(F_hkl.item())   # **2
    if Progress: 
        print('>>>>>> Processing %04d/%04d ... ' %(M, M) + strftime("%H:%M:%S", localtime()))
        print('>>>>>> Done! <<<<<<')
    Intensity = Intensity.reshape((M,))
    if Normalize: 
        Intensity = Intensity / np.max(Intensity) * 100
        # Intensity = Intensity.astype(int)
    
    return Intensity


def CalLat_ListPeaks(HKLs, TwoThetas, G_vectors, Intensities=0, 
                     HideZeroInt=True, HideNegHKL=True, HideSamePeak=True,
                     SortTwoTheta=True, SortIntensity=False, 
                     SortH=False, SortK=False, SortL=False):
    ''' >>> Instruction <<< 
        Input:
            HKLs            (n, 3) array, Miller indices of the diffraction peaks. 
            TwoThetas       (n,  ) array, the corresponding 2-theta angles of HKLs.
            G_vectors       (n, 3) array, coordinates of the reciprocal peaks. 
            Intensities     (n, 1) array, the corresponding intensities of Bragg peaks.
    '''
    # Regularize the inputs and outputs
    HKLs = np.asarray(HKLs)
    G_vectors = np.asarray(G_vectors)
    TwoThetas = np.asarray(TwoThetas)
    IntenFlag = False   # A flag to tell if there is an intensity input
    if np.size(np.shape(Intensities)) != 0:
        IntenFlag = True
        Intensities = np.asarray(Intensities)
    
    if np.size(np.shape(G_vectors)) != 2: 
        print('>>>>>> Input G_vectors has a wrong dimensions. <<<<<<')
        return
    if np.size(np.shape(HKLs)) != 2: 
        print('>>>>>> Input HKLs has a wrong dimensions. <<<<<<')
        return
    if IntenFlag:
        if np.size(np.shape(Intensities)) != 1: 
            print('>>>>>> Input Intensities has a wrong dimensions. <<<<<<')
            return
    if np.size(np.shape(TwoThetas)) != 1: 
        print('>>>>>> Input TwoThetas has a wrong dimensions. <<<<<<')
        return
    
    if np.shape(G_vectors)[1] != 3:
        if np.shape(G_vectors)[0] == 3: 
            G_vectors = np.transpose(G_vectors)
        else: 
            print('>>>>>> Input G_vectors has a wrong size. <<<<<<')
            return
    if np.shape(HKLs)[1] != 3:
        if np.shape(HKLs)[0] == 3: 
            HKLs = np.transpose(HKLs)
        else: 
            print('>>>>>> Input HKLs has a wrong size. <<<<<<')
            return
    
    m_HKLs = np.shape(HKLs)[0]
    m_peaks = np.shape(G_vectors)[0]
    m_tts = np.size(TwoThetas)   
    TwoThetas = np.reshape(TwoThetas, (m_tts,))
    if IntenFlag:
        m_ints = np.size(Intensities)
        Intensities = np.reshape(Intensities, (m_ints,))
    else: 
        m_ints = m_tts
    
    if ((m_HKLs != m_peaks) or 
        (m_HKLs != m_tts) or 
        (m_peaks != m_tts) or 
        (m_ints != m_tts) ): 
        print('>>>>>> The sizes of Inputs are different. <<<<<<')
        return
    
    N = m_HKLs
    if IntenFlag:
        Temp = np.empty((N, 8))
        Temp = np.append(np.append(np.append(HKLs, 
                                             TwoThetas.reshape(N, 1), 1), 
                                   Intensities.reshape(N, 1), 1), 
                         G_vectors, 1)
    else: 
        Temp = np.empty((N, 7))
        Temp = np.append(np.append(HKLs, TwoThetas.reshape(N, 1), 1), G_vectors, 1)

    if SortTwoTheta: 
        Temp = np.asarray(sorted(Temp, key=lambda x: x[3]))
    elif SortIntensity and IntenFlag:
        Temp = np.asarray(sorted(Temp, key=lambda x: x[4], reverse=True))
    elif SortH:
        Temp = np.asarray(sorted(Temp, key=lambda x: x[0]))
    elif SortK:
        Temp = np.asarray(sorted(Temp, key=lambda x: x[1]))
    elif SortL:
        Temp = np.asarray(sorted(Temp, key=lambda x: x[2]))
    
    if HideNegHKL: 
        Temp = Temp[np.intersect1d(np.intersect1d(np.where(Temp[:,0]>=0), 
                                   np.where(Temp[:,1]>=0)), 
                    np.where(Temp[:,2]>=0))]
    
    if HideSamePeak and not (SortTwoTheta or SortIntensity): 
        print( '>>>>>> "HideSamePeak" works only when ' +  
               'either "SortTwoTheta" or "SortIntensity" is True. <<<<<<')
    
    if IntenFlag:
        print(' HKL', '\t\t', 'Two Theta', '\t', 'Intensity', '\t', 'G_vector')
        for i in range(np.shape(Temp)[0]): 
            if HideZeroInt and Temp[i, 4] < 0.1:
                continue
            if HideSamePeak and i != 0: 
                if SortTwoTheta: 
                    if np.abs(Temp[i, 3] - Temp[i-1, 3]) < 0.001: 
                        continue
                if SortIntensity: 
                    if np.abs(Temp[i, 4] - Temp[i-1, 4]) < 0.001: 
                        continue
            print('%d  %d  %d \t %.3f  \t %.3f  \t %.3f   %.3f   %.3f' 
                  %(Temp[i,0], Temp[i,1], Temp[i,2], 
                    Temp[i,3], Temp[i,4], Temp[i,5], 
                    Temp[i,6], Temp[i,7]) )
    else: 
        print(' HKL', '\t\t', 'Two Theta', '\t', 'G_vector')
        for i in range(np.shape(Temp)[0]): 
            if HideSamePeak and i != 0: 
                if SortTwoTheta: 
                    if np.abs(Temp[i, 3] - Temp[i-1, 3]) < 0.001: 
                        continue
            print('%d  %d  %d \t %.3f  \t %.3f   %.3f   %.3f' 
                  %(Temp[i,0], Temp[i,1], Temp[i,2], 
                    Temp[i,3], Temp[i,4], Temp[i,5], 
                    Temp[i,6]) )
    return


def CalLat_SF_with_dislocations(CIF_name='test.cif', CIF_path='', HKL=[[0,0,1],[1,1,1]], 
                                PhotonE=12, UC_params=[0,0,0,0,0,0], DisLoc=[[0,0,0],[0,0,0]], 
                                LatticeInfo=False, Progress=False):
    # Get the current working directory
    retval = os.getcwd()
    
    # Input Regularization & Initialize the output arrays
    HKL = np.asarray(HKL)
    if np.size(np.shape(HKL)) > 2: 
        print(">>>>>> Input HKL's dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        input('>>>>>> Press any key to quit...')
        return
    if np.shape(HKL)[1] == 3: 
        M = np.shape(HKL)[0]
    elif np.shape(HKL)[0] == 3:
        M = int( np.size(HKL) / 3 )
        HKL = HKL.transpose()
    else: 
        print(">>>>>> Input HKL's size is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        input('>>>>>> Press any key to quit...')
        return
    
    DisLoc = np.asarray(DisLoc)
    if np.size(np.shape(DisLoc)) > 2: 
        print(">>>>>> Input DisLoc's dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        input('>>>>>> Press any key to quit...')
        return
    if np.shape(DisLoc)[1] == 3: 
        L = np.shape(DisLoc)[0]
    elif np.shape(DisLoc)[0] == 3:
        L = int( np.size(DisLoc) / 3 )
        DisLoc = DisLoc.transpose()
    else: 
        print(">>>>>> Input DisLoc's size is wrong! <<<<<<")
        print(">>>>>> The Input should be a n by 3 array. <<<<<<")
        input('>>>>>> Press any key to quit...')
        return
    
    TwoTheta = np.zeros((M, 1))
    StrucFactor = np.zeros((M, 1), dtype=np.complex)
    
    # Read cif file
    cif_data = CalLat_read_cif(CIF_name=CIF_name, CIF_path=CIP_path, LatticeInfo=LatticeInfo)
    a, b, c, alpha, beta, gamma = UC_params
    if a*b*c*alpha*beta*gamma == 0:    # read lattice shape from cif if it is not defined
        a, b, c, alpha, beta, gamma = cif_data['UnitCellParams']
    Z = cif_data['Atoms_Z']
    Atoms_coords = cif_data['Atoms_coords']
    
    N,_ = np.shape(Z)   # N is the number of atoms in the unit cell
    Atomic_form_factors = np.zeros((N, 1), dtype=np.complex)
    
    # Apply dislocations
    if L == N: 
        Atoms_coords = Atoms_coords + DisLoc
    else: 
        print(">>>>>> Input DisLoc's size does not match the atoms in cif file ! <<<<<<")
        print(">>>>>> No Dislocation applied ! <<<<<<")
    
    # Calculate 2-theta of the corresponding HKL
    WaveLen = 12.3980 / PhotonE   # unit is [A]
    if Progress:
        print('>>>>>> Calculating 2-theta ... ' + strftime("%H:%M:%S", localtime()) )
    TwoTheta = CalLat_2Theta(HKL=HKL, uc=[a, b, c, alpha, beta, gamma],
                             PhotonE=PhotonE, display=False).reshape((M,1))
    '''
    # For Debug
    print(a,b,c,alpha,beta,gamma)
    print(HKL)
    print(TwoTheta)
    print(Atoms_coords)
    print('==================================')
    # '''
    # Calculate the structure factor for each HKL
    if Progress:
        print('>>>>>> Calculating structure factor ... ' + strftime("%H:%M:%S", localtime()) )
    for i in range(M):
        # Display the progress
        if Progress: 
            print('>>>>>> Processing %04d/%04d ... ' %(i, M) 
                  + strftime("%H:%M:%S", localtime()), end='\r')
        # Calculate the momentum transfer
        Q = 4 * np.pi * np.sin(np.deg2rad(TwoTheta[i]/2)) / WaveLen
        q = np.sin(np.deg2rad(TwoTheta[i]/2)) / WaveLen
        # Convert hkl to a (3, 1) array
        hkl = np.reshape(HKL[i],(3,1))
        # Calculate Atomic Form Factors
        '''
        The complete atomic form factor f(q,E) = f1 + if2 = f0(q,E) + f'(E) + if"(E)
        Here, 
            f0 is from the Xraylib function FF_Rayl(Z, q)
            f' is from the Xraylib function Fi(Z, E)
            f" is from the Xraylib function Fii(Z, E)
        Note: f0 should be 'Ionic' rather than 'Atomic'. Will update later.
         
        http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
        '''
        for n in range(N): 
            f0  = xb.FF_Rayl(int(Z[n]), float(q))
            fi  = Fi(int(Z[n]), PhotonE)
            fii = Fii(int(Z[n]), PhotonE)
            Atomic_form_factors[n] = f0 + fi - 1j * fii
            
        # Calculate exp(hx+ky+lz) for each atom
        Matrix_exp = np.exp(2j*np.pi*np.matmul(Atoms_coords, hkl))
        
        # Calculate F_hkl
        F_hkl = np.matmul(Atomic_form_factors.transpose(), Matrix_exp)
        StrucFactor[i] = F_hkl
        
        '''
        # For Debug
        print('hkl = ', hkl.reshape(-1))
        print('q = ', q)
        print('Theta = ', TwoTheta[i]/2)
        print('f0 = ', f0)
        print('fi = ', fi)
        print('fii = ', fii)
        print('--- hx + ky + lz -----------------')
        print(np.matmul(Atoms_coords, hkl))
        # print('--- exps -------------------------')
        # print(Matrix_exp)
        print('--- F_hkl ------------------------')
        print(F_hkl)
        print('==================================')
        # '''
        
    if Progress: 
        print('>>>>>> Processing %04d/%04d ... ' %(M, M) + strftime("%H:%M:%S", localtime()))
        print('>>>>>> Done! <<<<<<')
    
    StrucFactor = StrucFactor.reshape((M,))
    
    return StrucFactor


def CalLat_Angles_between_Gvecs(Sample_Gvecs, Debug=False): 
    ''' >>> Instruction <<< 
        This function takes the result from function 'Retrieve_Gvecs' and calculate the included 
        angles between each Gvec pair.
        
        Input: 
            Sample_Gvecs     (4, n) array, each (4, 1) is a Bragg peak's [GvecX, GvecY, GvecZ, 2Theta]
                             where GvecX/Y/Z is the orientation vector in the Lab coordinates.
        
        Output: 
            Angles           (n, n) array
    '''
    Sample_Gvecs = np.asarray(Sample_Gvecs)
    # Regularize the input and output
    if np.size(np.shape(Sample_Gvecs)) != 2: 
        print('>>>>>> Input Sample_Gvecs has wrong dimensions. <<<<<<')
        return
    
    if np.shape(Sample_Gvecs)[0] != 4:
        if np.shape(Sample_Gvecs)[1] == 4: 
            Sample_Gvecs = np.transpose(Sample_Gvecs)
            print('>>>>>> Warning!!! Input Bragg_peaks is transposed. <<<<<<')
        else: 
            print('>>>>>> Input Bragg_peaks has a wrong size. <<<<<<')
            return
    
    n_peaks = np.shape(Sample_Gvecs)[1]
    Output = np.zeros((n_peaks, n_peaks))
    
    # Calculate the included angles
    for i in range(n_peaks): 
        for j in range(n_peaks): 
            Output[i, j] = Included_Angle(Sample_Gvecs[0:3, i], Sample_Gvecs[0:3, j], 
                                          OutputRadian=False)['theta']
            if Debug: 
                print('Vector %d is ' %i, Sample_Gvecs[0:3, i])
                print('Vector %d is ' %j, Sample_Gvecs[0:3, j])
                print('The Angle is ', Output[i,j])
                print('------------------------------------')
    
    return Output


def wavefront_propagator_angspc(wf_in, dist_z=5, grid_size=[5,5], wave_len=0.1, 
                                pad_size=[501,501], freq_offset=0, lowthres=1.0, 
                                flag_cropout=True, flag_norm=False, flag_lowthres=False): 
    ''' >>> Instruction <<< 
        This function propagates 2D wavefront using angular spectrum. 
        See Goodman's Fourier Optics and DOI:/10.1155/2017/7293905 for more details.
        
        !!! Note that wf_in dims should be even !!!
        
        Input:
            wf_in           (N, M) array representing a complex wavefront
            dist_z          [nm], propagation distance
            grid_size       [nm, nm], pixel size of the incident 2D wavefront
            wave_len        [nm], wavelength of the incident light
            pad_size        (>=N, >=M), zero padding for fourier transform
            freq_offset     remove phase ramp
            lowthres        absolute value of the threshold
            flag_cropout    True: crop output to same size as input
            flag_norm       True: normalize wf_out intensity to wf_in
            flag_lowthres   True: eliminate voxels with values less than a threshold
        
        Output: 
            wf_out          (N, M) array, complex wavefront after propagation 
    '''
    # Regularize input and get dimension
    wf_in = np.asarray(wf_in)
    input_x, input_y = np.shape(wf_in)
    # Convert input wavefront to angular specturm via FFT
    grid_x, grid_y = grid_size   # [nm]
    pad_x, pad_y = pad_size
    wf_in_as = sp.fft.fftshift(sp.fft.fft2((Array_zeropad_crop_2D(wf_in, AdjTo=[pad_x,pad_y])))) 
    # Calculate the additional phase
    idx_x = np.arange(pad_x) - pad_x/2 + freq_offset
    idx_y = np.arange(pad_y) - pad_y/2 + freq_offset
    fq_x = 1 / pad_x / grid_x * idx_x   # [nm-1]
    fq_y = 1 / pad_y / grid_y * idx_y   # [nm-1]
    fq_yy, fq_xx = np.meshgrid(fq_y, fq_x)
    # Angular spectrum phase term
    ph_term = np.sqrt((1/wave_len**2 - fq_xx**2 - fq_yy**2).astype('complex128'))
    ph_term *= 2j * np.pi * dist_z 
    # Eliminate anything outside 
    circ = (np.sqrt( (fq_xx*wave_len)**2 + (fq_yy*wave_len)**2 ) < 1).astype('float')
    wf_in_as *= circ
    ph_term *= circ
    # Propagation
    wf_out_as = wf_in_as * np.exp(ph_term)
    wf_out = sp.fft.ifft2(sp.fft.fftshift(wf_out_as))
    # post processing
    if flag_cropout: 
        wf_out = Array_crop_2D(wf_out, CropTo=(input_x, input_y))
    if flag_norm or flag_lowthres: 
        wf_out_int = np.abs( wf_out * np.conjugate(wf_out) )
    if flag_norm: 
        wf_out *= np.sqrt(np.sum( np.abs( wf_in  * np.conjugate(wf_in)  )) / 
                          np.sum( wf_out_int ) )
    if flag_lowthres: 
        wf_out[np.where(np.sqrt( wf_out_int ) < lowthres)] = 0.0 + 0.0j
    return wf_out


def wavefront_propagator_angspc_1D(wf_in, dist_z=5, grid_size=5, wave_len=0.1, 
                                   pad_size=501, freq_offset=0, flag_cropout=True, 
                                   flag_norm=False): 
    ''' >>> Instruction <<< 
        This function propagates 1D wavefront using angular spectrum. 
        See Goodman's Fourier Optics and DOI:/10.1155/2017/7293905 for more details.
        
        !!! Note that wf_in dims should be even !!!
        
        Input:
            wf_in           (N,) array representing a complex wavefront
            dist_z          [nm], propagation distance
            grid_size       [nm], pixel size of the incident 1D wavefront
            wave_len        [nm], wavelength of the incident light
            pad_size        (>=N, ), zero padding for fourier transform
            freq_offset     remove phase ramp
            flag_cropout    True: crop output to same size as input
            flag_norm       True: normalize wf_out intensity to wf_in
        
        Output: 
            wf_out          (N, ) array, complex wavefront after propagation 
    '''
    # Regularize input and get dimension
    wf_in = np.asarray(wf_in)
    input_x = len(wf_in)
    # Convert input wavefront to angular specturm via FFT
    grid_x = grid_size   # [nm]
    pad_x = pad_size
    wf_in_as = sp.fft.fftshift(sp.fft.fft(( Array_size_adjust(wf_in, adj_to=pad_x) ))) 
    # Calculate the additional phase
    idx_x = np.arange(pad_x) - pad_x/2 + freq_offset
    fq_x = 1 / pad_x / grid_x * idx_x   # [nm-1]
    # Angular spectrum phase term
    ph_term = np.sqrt((1/wave_len**2 - fq_x**2).astype('complex128'))
    ph_term *= 2j * np.pi * dist_z 
    # Eliminate anything outside 
    circ = ((fq_x*wave_len) < 1).astype('float')
    wf_in_as *= circ
    ph_term *= circ
    # Propagation
    wf_out_as = wf_in_as * np.exp(ph_term)
    wf_out = sp.fft.ifft(sp.fft.fftshift(wf_out_as))
    if flag_cropout: 
        wf_out = Array_size_adjust(wf_out, adj_to=input_x)
    # Normalization
    if flag_norm: 
        wf_out *= np.sqrt(np.sum( np.abs( wf_in  * np.conjugate(wf_in)  )) / 
                          np.sum( np.abs( wf_out * np.conjugate(wf_out) )) )
    return wf_out


def wavefront_propagator_fresnel(wf_in, dist_z=5, grid_size=[5,5], wave_len=0.1, 
                                 pad_size=[501,501], freq_offset=0, flag_cropout=True, 
                                 flag_norm=False): 
    ''' >>> Instruction <<< 
        This function propagates 2D wavefront under Fresenl approximation. 
        See Goodman's Fourier Optics and DOI:/10.1155/2017/7293905 for more details.
        
        !!! Note that wf_in dims should be even !!!
        
        Input:
            wf_in           (N, M) array representing a complex wavefront
            dist_z          [nm], propagation distance
            grid_size       [nm, nm], pixel size of the incident 2D wavefront
            wave_len        [nm], wavelength of the incident light
            pad_size        (>=N, >=M), zero padding for fourier transform
            freq_offset     remove phase ramp
            flag_cropout    True: crop output to same size as input
            flag_norm       True: normalize wf_out intensity to wf_in
        
        Output: 
            wf_out          (N, M) array, complex wavefront after propagation 
    '''
    # Regularize input and get dimension
    wf_in = np.asarray(wf_in)
    input_x, input_y = np.shape(wf_in)
    # Convert input wavefront to angular specturm via FFT
    grid_x, grid_y = grid_size   # [nm]
    pad_x, pad_y = pad_size
    wf_in_as = sp.fft.fftshift(sp.fft.fft2((Array_zeropad_crop_2D(wf_in, AdjTo=[pad_x,pad_y])))) 
    # Calculate the additional phase
    idx_x = np.arange(pad_x) - pad_x/2 + freq_offset
    idx_y = np.arange(pad_y) - pad_y/2 + freq_offset
    fq_x = 1 / pad_x / grid_x * idx_x   # [nm-1]
    fq_y = 1 / pad_y / grid_y * idx_y   # [nm-1]
    fq_yy, fq_xx = np.meshgrid(fq_y, fq_x)
    # Fresenl approximation phase term
    ph_term = 2j * np.pi * dist_z / wave_len - 1j * np.pi * wave_len * dist_z * (fq_xx**2 + fq_yy**2)
    # Propagation
    wf_out_as = wf_in_as * np.exp(ph_term.astype('complex128'))
    wf_out = sp.fft.ifft2(sp.fft.fftshift(wf_out_as))
    if flag_cropout: 
        wf_out = Array_crop_2D(wf_out, CropTo=(input_x, input_y))
    # Normalization
    if flag_norm: 
        wf_out *= np.sqrt(np.sum( np.abs( wf_in  * np.conjugate(wf_in)  )) / 
                          np.sum( np.abs( wf_out * np.conjugate(wf_out) )) )
    return wf_out


def wavefront_propagator_frft(wf_in, z_in=-50, z_out=100, w_0=1e3, wave_len=0.1, 
                              grid_in=[1,1], pad_size=[501,501], flag_cropout=True, flag_info=False): 
    ''' >>> Instruction <<< 
        This function propagates 2D wavefront using fractional fourier transform (FRFT). 
        See https://github.com/siddharth-maddali/frft for more details. 
        
        Input:
            wf_in           (N, M) array representing the complex incident wavefront. 
            z_in            [nm], z position of wf_in, where z = 0 is the beam waist. 
            z_out           [nm], z position of the propagated wavefront. 
            w_0             [nm], beam radius at the waist. 
            wave_len        [nm], wavelength of the incident wave. 
            grid_in         [nm], 2D grid size of wf_in, defaut is [1, 1]. 
            pad_size        (>=N, >=M), zero padding for fourier transform. 
            flag_cropout    True: crop output to same size as input. 
            flag_info       True: print information of parameters. 
        
        Output: 
            wf_out          (N, M) or larger array, complex wavefront after propagation. 
    '''
    # Regularize input and get dimension
    wf_in = np.asarray(wf_in)
    grid_in = np.asarray(grid_in)
    input_x, input_y = np.shape(wf_in)
    pad_x, pad_y = pad_size
    if pad_x < input_x: 
        print('>>>>>> Force the original size in the 1st axis ... ')
        pad_x = input_x
    if pad_y < input_y: 
        print('>>>>>> Force the original size in the 2nd axis ... ')
        pad_y = input_y
    # Calculate necessary parameters
    s_0 = np.sqrt(np.pi) * w_0   # [nm], scale parameter
    z_bar = s_0**2 / wave_len   # [nm], Rayleigh range
    w_in  = w_0 * np.sqrt(1 + (z_in /z_bar)**2)   # [nm], beam radius at wf_in
    w_out = w_0 * np.sqrt(1 + (z_out/z_bar)**2)   # [nm], beam radius at wf_out
    s_in  = np.sqrt(np.pi) * w_in    # [nm], scale parameter for wf_in
    s_out = np.sqrt(np.pi) * w_out   # [nm], scale parameter for wf_out
    R_in  = z_in  * ( 1 + (z_bar / z_in )**2 )   # [nm], radius of curvature of wf_in
    R_out = z_out * ( 1 + (z_bar / z_out)**2 )   # [nm], radius of curvature of wf_out
    xi_in  = np.arctan2(z_in,  z_bar)   # Gouy phase shift of wf_in
    xi_out = np.arctan2(z_out, z_bar)   # Gouy phase shift of wf_out
    a_in  = xi_in  / np.pi * 2   # order of FRFT from beam waist to wf_in 
    a_out = xi_out / np.pi * 2   # order of FRFT from beam waist to wf_out
    a_tot = a_out - a_in   # order of FRFT from wf_in to wf_out
    # Calculate the 2D grid size of wf_out
    grid_out = grid_in / s_in * s_out
    # FRFT propagation
    wf_in_pad = Array_size_adjust(wf_in, adj_to=[pad_x, pad_y])
    wf_out = sp.fft.fftshift(frft.frft(sp.fft.fftshift(wf_in_pad), a_tot))
    norm = ( np.sum(np.sum(np.abs( wf_in_pad * np.conjugate(wf_in_pad) ))) / 
             np.sum(np.sum(np.abs( wf_out    * np.conjugate(wf_out)    ))) )
    wf_out = wf_out * np.sqrt(norm)
    if flag_cropout: 
        wf_out = Array_size_adjust(wf_out, adj_to=[input_x, input_y])
    if flag_info: 
        print('s_0   = %.6f [nm]' %s_0)
        print('s_in  = %.6f [nm]' %s_in)
        print('s_out = %.6f [nm]' %s_out)
        print('z_bar = %.6f [um]' %(z_bar/1e3))
        print('w_0   = %.6f [um]' %(w_0/1e3))
        print('w_in  = %.6f [um]' %(w_in/1e3))
        print('w_out = %.6f [um]' %(w_out/1e3))
        print('R_in  = %.6f [mm]' %(R_in/1e6))
        print('R_out = %.6f [mm]' %(R_out/1e6))
        print('xi_in  = %.6f ' %xi_in)
        print('xi_out = %.6f ' %xi_out)
        print('a_in  = %.6f ' %a_in)
        print('a_out = %.6f ' %a_out)
        print('a_tot = %.6f ' %a_tot)
        print('grid_out = [%.3f, %.3f] [nm, nm] ' %(grid_out[0], grid_out[1]) )
    return {'wf': wf_out, 'grid': grid_out}


def wavefront_propagator_penetration_old(wf_in, samp_slice, thickness=5, wave_len=0.1, delta=1e-5, beta=1e-7): 
    ''' >>> Instruction <<< 
        This function propagates 2D wavefront through a slice of sample material. 
                
        Input:
            wf_in           (N, M) array representing a complex wavefront
            samp_slice      (N, M) binary array representing the sample slice
            thickness       [nm], penetration thickness
            wave_len        [nm], wavelength of the incident light
            delta, beta     complex refractive index, 1 - delta + i*beta
        
        Output: 
            wf_out          (N, M) array, complex wavefront after penegration 
    '''
    # Regularize input and get dimension
    wf_in = np.asarray(wf_in)
    input_x, input_y = np.shape(wf_in)
    samp_slice = np.asarray(samp_slice)
    temp_x, temp_y = np.shape(samp_slice)
    if (input_x != temp_x) or (input_y != temp_y): 
        raise TypeError('Dimensions of wf_in and sample_slice do NOT match. ')
    wave_vec = 2*np.pi/wave_len   # [nm-1]
    wf_out = wf_in
    # attenuation 
    sample_abs = np.exp(    - wave_vec *  beta * thickness * samp_slice)
    wf_out *= sample_abs
    # phase delay 
    sample_phd = np.exp(-1j * wave_vec * delta * thickness * samp_slice)
    wf_out *= sample_phd
    return wf_out


def wavefront_propagator_penetration_1D(wf_in, samp_slice, thickness=5, wave_len=0.1, delta=1e-5, beta=1e-7): 
    ''' >>> Instruction <<< 
        This function propagates 1D wavefront through a slice of sample material. 
                
        Input:
            wf_in           (N,) array representing a complex wavefront
            samp_slice      (N,) binary array representing the sample slice
            thickness       [nm], penetration thickness
            wave_len        [nm], wavelength of the incident light
            delta, beta     complex refractive index, 1 - delta + i*beta
        
        Output: 
            wf_out          (N,) array, complex wavefront after penegration 
    '''
    # Regularize input and get dimension
    wf_in = np.asarray(wf_in)
    input_x = len(wf_in)
    samp_slice = np.asarray(samp_slice)
    temp_x = len(samp_slice)
    if (input_x != temp_x): 
        raise TypeError('Dimensions of wf_in and sample_slice do NOT match. ')
    wave_vec = 2*np.pi/wave_len   # [nm-1]
    wf_out = wf_in
    # attenuation 
    sample_abs = np.exp(    - wave_vec *  beta * thickness * samp_slice)
    wf_out *= sample_abs
    # phase delay 
    sample_phd = np.exp(-1j * wave_vec * delta * thickness * samp_slice)
    wf_out *= sample_phd
    return wf_out


def CalLat_DynCrysProp(dth, params=None, varis=None, detpix=None, SavetoFile=False): 
    ''' >>> Instruction <<< 
        This function propagates a wavefront through an arbitrary shaped crystal. 
        
        For multi-thread processing. 
        
        Inputs: 
            dth         [rad], Deviation from Bragg angle. 
            params      dictionary, parameters for setting up simulation
            varis       dictionary, temporary variables required for simulation
            detpix      detector orientation vectors for pix-to-pix farfield propagation. 
        
        Output: 
            exit_wavefronts      dictionary, exit wavefront and metadata
    '''
    with open('temp_progress.txt', 'a+') as progress_file: 
        progress_file.writelines(f'dth{dth:>{10}.3f}    ... ' + strftime('%H:%M:%S', localtime()) + ' \n')
    
    save_dir = params['save_dir']
    num_pos = params['num_pos']
    phi = params['phi']
    tolerance = params['tolerance']
    rotax_crys = params['rotax_crys']
    d_spacing = params['d_spacing']
    chi_0 = params['chi_0']
    chi_h = params['chi_h']
    grid_cat_x = params['grid_cat_xyz'][0]
    grid_cat_y = params['grid_cat_xyz'][1]
    grid_cat_z = params['grid_cat_xyz'][2]
    grid_dif = params['grid_dif']
    grid_out = params['grid_out']
    unit_s0_0 = params['unit_s0_0']
    rot_mat_crys2lab = params['rot_mat_crys2lab']
    
    wave_number = varis['wave_number']
    h = varis['h']
    ori_coords = varis['ori_coords']
    Crys_pad = varis['Crys_pad']
    Crys_disloc_pad = varis['Crys_disloc_pad']
    incident_wavefront_0_abs = varis['incident_wavefront_0_abs']
    incident_wavefront_0_ang = varis['incident_wavefront_0_ang']
    det_x_0 = varis['det_x_0']
    det_y_0 = varis['det_y_0']
    det_z_0 = varis['det_z_0']
    axis_1 = varis['axis_1']
    axis_2 = varis['axis_2']
    axis_3 = varis['axis_3']
    flag_eqt = varis['flag_eqt']
    rgi_method = varis['rgi_method']
    
    # Output dictionary 
    exit_wavefronts = {}
    
    ''' ================================================================================
        Transform crystal to diffraction base, using interpolation 
        ================================================================================ '''
    
    # Beam position for beam scanning mode
    beam_center = [ 0.025*np.sqrt(num_pos) * np.cos(num_pos * phi), 
                    0.025*np.sqrt(num_pos) * np.sin(num_pos * phi) ]
    
    # Vectors for current rocking angle
    R = Rot_Matrix(rotax_crys, dth)
    k0 = wave_number * np.squeeze(np.asarray(np.matmul(R, unit_s0_0)))
    kh = k0 + h
    unit_s0 = k0 / np.linalg.norm(k0) 
    unit_sh = kh / np.linalg.norm(kh)
    unit_sy = np.cross(unit_s0, unit_sh)
    unit_sy /= np.linalg.norm(unit_sy)
    
    unit_sh_lab = Rotate_vectors( np.matmul(Rot_Matrix(rotax_crys, -dth), 
                                            unit_sh), 
                                  rot_mat_crys2lab )
    unit_sh_lab /= np.linalg.norm(unit_sh_lab)
    
    ''' ====== For pix-to-pix propagation - start ====== '''
    det_x = np.squeeze(np.asarray(np.matmul(R, det_x_0))) 
    det_y = np.squeeze(np.asarray(np.matmul(R, det_y_0))) 
    det_z = np.squeeze(np.asarray(np.matmul(R, det_z_0))) 
    det_pix = np.asarray(np.matmul(R, detpix.transpose())).transpose()
    ''' ======= For pix-to-pix propagation - end ======= '''
    
    # Coordinates in diffraction base
    Dif_ax_1 = unit_s0   # Incident beam (Z) 
    Dif_ax_2 = unit_sy   # Out of plane (Y) 
    Dif_ax_3 = unit_sh   # Diffracted beam (X) 
    New_coords = Base_transform_3D(ori_coords, ob_to_tb=False, 
                                   regularize_inputs=True, normalize_bases=False, 
                                   ob_1=[0, 0, grid_cat_z], 
                                   ob_2=[0, grid_cat_y, 0], 
                                   ob_3=[grid_cat_x, 0, 0], 
                                   tb_1=Dif_ax_3 * grid_dif, 
                                   tb_2=Dif_ax_2 * grid_out, 
                                   tb_3=Dif_ax_1 * grid_dif)['new_coords']
    ''' Here New_coords is in original base, i.e. a non-regular grid 
        corresponding to a regular grid in target base. '''
    # Interpolation - crystal shape
    if flag_eqt: 
        new_crys = Array_interpolation_3D_spline(Crys_pad, New_coords, 
                                                 ori_ax_1=axis_3, ori_ax_2=axis_2, ori_ax_3=axis_1, 
                                                 PadSize=None, PadIncr=[5,5,5], Progress=False)
    else: 
        new_crys = Array_interpolation_3D_rgi(Crys_pad, New_coords, method=rgi_method,
                                             ori_ax_1=axis_3, ori_ax_2=axis_2, ori_ax_3=axis_1, 
                                             PadSize=None, PadIncr=[5,5,5], Progress=False)
    new_crys = np.around(new_crys, decimals=0)
    
    # Interpolation - crystal displacement induced phase
    Crys_phase_pad = np.matmul(Crys_disloc_pad, h) * d_spacing
    if flag_eqt: 
        new_phase = Array_interpolation_3D_spline(Crys_phase_pad, New_coords, 
                                                  ori_ax_1=axis_3, ori_ax_2=axis_2, ori_ax_3=axis_1, 
                                                  PadSize=None, PadIncr=[5,5,5], Progress=False)
    else: 
        new_phase = Array_interpolation_3D_rgi(Crys_phase_pad, New_coords, method=rgi_method, 
                                               ori_ax_1=axis_3, ori_ax_2=axis_2, ori_ax_3=axis_1, 
                                               PadSize=None, PadIncr=[5,5,5], Progress=False)
    
    # Incident wavefront in diffraction base
    new_incid_wf = np.exp(1j * incident_wavefront_0_ang)
    new_incid_wf *= incident_wavefront_0_abs
    
    ''' ============================================================================
        Find boundaries of the crystal 
        ============================================================================ '''
    sh_us = Array_occurrence(new_crys, non_occ_val=0,axis=0,last_occ=False,invalid_val=-1)
    sh_ds = Array_occurrence(new_crys, non_occ_val=0,axis=0,last_occ=True, invalid_val=-1)
    sy_us = Array_occurrence(new_crys, non_occ_val=0,axis=1,last_occ=False,invalid_val=-1)
    sy_ds = Array_occurrence(new_crys, non_occ_val=0,axis=1,last_occ=True, invalid_val=-1)
    s0_us = Array_occurrence(new_crys, non_occ_val=0,axis=2,last_occ=False,invalid_val=-1)
    s0_ds = Array_occurrence(new_crys, non_occ_val=0,axis=2,last_occ=True, invalid_val=-1)
    
    bound_sh_us = np.min(sh_us[np.where(sh_us>=0)])
    bound_sh_ds = np.max(sh_ds) + 1
    bound_sy_us = np.min(sy_us[np.where(sy_us>=0)])
    bound_sy_ds = np.max(sy_ds) + 1
    bound_s0_us = np.min(s0_us[np.where(s0_us>=0)])
    bound_s0_ds = np.max(s0_ds) + 1
    crys_box = new_crys[ bound_sh_us-1 : bound_sh_ds+1, 
                         bound_sy_us-1 : bound_sy_ds+1, 
                         bound_s0_us-1 : bound_s0_ds+1 ]   # This is used as a mask
    grid_sh, grid_sy, grid_s0 = np.shape(crys_box)
    
    sh_us_box = Array_occurrence(crys_box, non_occ_val=0,axis=0,last_occ=False,invalid_val=-1)
    sh_ds_box = Array_occurrence(crys_box, non_occ_val=0,axis=0,last_occ=True, invalid_val=-1)
    sy_us_box = Array_occurrence(crys_box, non_occ_val=0,axis=1,last_occ=False,invalid_val=-1)
    sy_ds_box = Array_occurrence(crys_box, non_occ_val=0,axis=1,last_occ=True, invalid_val=-1)
    s0_us_box = Array_occurrence(crys_box, non_occ_val=0,axis=2,last_occ=False,invalid_val=-1)
    s0_ds_box = Array_occurrence(crys_box, non_occ_val=0,axis=2,last_occ=True, invalid_val=-1)
    
    ''' ============================================================================
        Initialize wavefields 
        ============================================================================ '''
    incid_wf_box = new_incid_wf[ bound_sh_us-1 : bound_sh_ds+1, 
                                 bound_sy_us-1 : bound_sy_ds+1 ]
    d0 = np.repeat(incid_wf_box[:, :, np.newaxis], grid_s0, axis=2)
    d0 *= crys_box
    dh = np.zeros_like(crys_box) * complex(0,0)
    phase = new_phase[ bound_sh_us-1 : bound_sh_ds+1, 
                       bound_sy_us-1 : bound_sy_ds+1, 
                       bound_s0_us-1 : bound_s0_ds+1 ]
    
    ''' ============================================================================
        Iterative process to find the dynamical solution.
        ============================================================================ '''
    # Indices arrays for wavefield calculation
    temp_arr = np.ones_like(crys_box)
    ind_arr_s0 = np.zeros_like(crys_box)
    ind_arr_sh = np.zeros_like(crys_box)
    for i in range(grid_s0-1): 
        ind_arr_s0 [:, :, i+1] = np.sum(temp_arr[:, :, :(i+1)], axis=2)-1
    for i in range(grid_sh-1): 
        ind_arr_sh [i+1, :, :] = np.sum(temp_arr[:(i+1), :, :], axis=0)-1
    ind_arr_s0 *= crys_box
    ind_arr_sh *= crys_box
    
    temp_arr = crys_box.copy()
    sh_mesh_s0, sh_mesh_sy = np.meshgrid(np.arange(0,grid_s0,1),np.arange(0,grid_sy,1))
    temp_arr[sh_us_box, sh_mesh_sy, sh_mesh_s0] = 0
    crys_idx_sh = np.where(temp_arr)   # Indices of voxels without sh up-stream boundary
    ind_arr_sh[sh_us_box, sh_mesh_sy, sh_mesh_s0] = 0
    
    temp_arr = crys_box.copy()
    s0_mesh_sy, s0_mesh_sh = np.meshgrid(np.arange(0,grid_sy,1),np.arange(0,grid_sh,1))
    temp_arr[s0_mesh_sh, s0_mesh_sy, s0_us_box] = 0
    crys_idx_s0 = np.where(temp_arr)   # Indices of voxels without s0 up-stream boundary
    ind_arr_s0[s0_mesh_sh, s0_mesh_sy, s0_us_box] = 0
    
    dis_arr_s0 = ind_arr_s0 - np.repeat(s0_us_box[:,:,np.newaxis], grid_s0, axis=2)
    dis_arr_s0 *= crys_box   # Distance from s0 up-stream boundary to each voxel
    del temp_arr
    
    ''' ------ Flip the optical path ------ '''
    if True: # True: Flipped optical path
        ind_arr_s0 *= -1
        ind_arr_sh *= -1
        dis_arr_s0 *= -1
    ''' ----------------------------------- '''
    
    d_inc_pos = np.matmul(s0_mesh_sh.ravel()[:,np.newaxis]  * grid_dif, [unit_sh]) +\
                np.matmul(s0_mesh_sy.ravel()[:,np.newaxis] * grid_out, [unit_sy]) +\
                np.matmul(s0_us_box.ravel()[:,np.newaxis] * grid_dif, [unit_s0])
    d_inc_pos = d_inc_pos.reshape(grid_sh, grid_sy, 3)
    
    # Variables for wavefield calculation
    phase_shift = np.pad(phase, ((1,0), (0,0), (0,0)), 
                         mode='constant', constant_values=0)[:-1, :, :]
    phase_shift *= crys_box
    
    ch = 1j * wave_number/2.0 * chi_h
    c0 = 1j * wave_number/2.0 * chi_0
    ''' ------ With flipped optical path ------ '''
    if True: # True: Flipped optical path
        w0 = chi_0 - (1 - (np.linalg.norm(kh)/wave_number)**2.0) 
    else: 
        w0 = chi_0 + (1 - (np.linalg.norm(kh)/wave_number)**2.0) 
    ''' --------------------------------------- '''
    cw = 1j * wave_number/2.0 * w0 
    
    cp = np.zeros_like(phase) * complex(0,0)
    gp = np.zeros_like(phase) * complex(0,0)
    cp[crys_idx_sh] = 1j * phase_shift[crys_idx_sh]
    gp[crys_idx_sh] = 1j * (phase[crys_idx_sh] - phase_shift[crys_idx_sh]) / grid_dif
    
    dh_tmp = np.zeros_like(dh[sh_ds_box, sh_mesh_sy, sh_mesh_s0])
    integral_s0 = np.zeros_like(d0)
    integral_sh = np.zeros_like(dh)
    
    res = 1.1 
    iter_counter = 0
    while res > tolerance: 
        iter_counter += 1
        res = 0 
        avg = 0
        # updating d0
        dh_shift = np.pad(dh, ((0,0), (0,0), (1,0)), 
                          mode='constant', constant_values=0)[:, :, :-1]
        dh_shift *= crys_box
        temp_s0 = np.zeros_like(dh)
        temp_s0[crys_idx_s0] = (dh[crys_idx_s0] + dh_shift[crys_idx_s0]) * ch / 2 \
                             * np.exp(-c0 * ind_arr_s0[crys_idx_s0] * grid_dif) \
                             * ( 1 - np.exp(-c0 * grid_dif) ) / c0
        for i in range(grid_s0): 
            integral_s0[:, :, i] = np.sum(temp_s0[:, :, :(i+1)], axis=2)
        integral_s0 *= crys_box
        d0 = np.repeat(d0[s0_mesh_sh, s0_mesh_sy, s0_us_box][:,:,np.newaxis], 
                       grid_s0, axis=2)
        d0 *= crys_box
        d0[crys_idx_s0] *= np.exp(c0 * (dis_arr_s0[crys_idx_s0] + 1) * grid_dif)
        d0[crys_idx_s0] += np.exp(c0 * (ind_arr_s0[crys_idx_s0] + 1) * grid_dif) \
                         * integral_s0[crys_idx_s0]
        
        # updating dh
        d0_shift = np.pad(d0, ((1,0), (0,0), (0,0)), 
                          mode='constant', constant_values=0)[:-1, :, :]
        d0_shift *= crys_box
        temp_sh = np.zeros_like(d0)
        temp_sh[crys_idx_sh] = (d0[crys_idx_sh] + d0_shift[crys_idx_sh]) / 2 \
                             * np.exp(-cp[crys_idx_sh]) \
                             * (1 - np.exp( -(cw + gp[crys_idx_sh]) * grid_dif) ) \
                             * np.exp( -cw * ind_arr_sh[crys_idx_sh] * grid_dif ) \
                             / (cw + gp[crys_idx_sh])
        for i in range(grid_sh): 
            integral_sh[i, :, :] = np.sum(temp_sh[:(i+1), :, :], axis=0)
        integral_sh *= crys_box
        dh[crys_idx_sh] = ch * integral_sh[crys_idx_sh] \
                        * np.exp(  1j * phase[crys_idx_sh] 
                                 + cw * (ind_arr_sh[crys_idx_sh]+1) * grid_dif)
        
        res = np.sum(np.abs( (dh_tmp - dh[sh_ds_box, sh_mesh_sy, sh_mesh_s0])**2 ))
        avg = np.sum(np.abs( dh[sh_ds_box, sh_mesh_sy, sh_mesh_s0]**2 ))
        dh_tmp = dh[sh_ds_box, sh_mesh_sy, sh_mesh_s0]
        
        if avg == 0: 
            res = 0
        else: 
            res = np.sqrt(res/avg)
            
    ''' ============================================================================
        Calculate poynting vectors.
        ============================================================================ '''
    poynting = kh.reshape((3,1)) * (np.abs(dh).ravel())**2 \
             + k0.reshape((3,1)) * (np.abs(d0).ravel())**2
    poynting = poynting.reshape((3, grid_sh, grid_sy, grid_s0))
    
    ''' ============================================================================
        Exiting wavefront in cartesian base. 
        ============================================================================ '''
    
    d_ext_val = dh[sh_ds_box.ravel(), 
                   sh_mesh_sy.ravel(), 
                   sh_mesh_s0.ravel()].reshape(grid_sy, grid_s0)
    d_ext_pos = np.matmul(sh_ds_box.ravel()[:,np.newaxis]  * grid_dif, [unit_sh]) +\
                np.matmul(sh_mesh_sy.ravel()[:,np.newaxis] * grid_out, [unit_sy]) +\
                np.matmul(sh_mesh_s0.ravel()[:,np.newaxis] * grid_dif, [unit_s0])
    
    # Calculate "factor"
    d_ext_pos = d_ext_pos.reshape(grid_sy, grid_s0, 3)
    d_ext_val = d_ext_val[1:-1, 1:-1] # Remove the zero voxels on edges
    d_ext_pos = d_ext_pos[1:-1, 1:-1, :]  # Remove the corresponding coordinates
    d_ext_pos_dif = np.diff(d_ext_pos, axis=1)
    d_ext_pos_avg = sp.ndimage.convolve1d(d_ext_pos, np.ones(2), axis=1, 
                                          mode='constant', cval=0)[:,:-1,:] / 2
    d_ext_val_avg = sp.ndimage.convolve1d(d_ext_val, np.ones(2), axis=1, 
                                          mode='constant', cval=0)[:,:-1] / 2
    # True wavefield at exit surface - so far dh is an envelope function
    factor = np.exp(-1j*np.dot(np.zeros_like(d_ext_pos_avg), h))\
             * d_ext_val_avg   # phase term exp{-ih.u}
    ''' np.zeros_like(d_ext_pos_avg) should use displacement (u of ih.u) '''
    factor *= np.exp( 1j*np.dot(d_ext_pos_avg, kh) ) # phase term exp{ikh.r}
    factor *= np.sqrt( np.linalg.norm(d_ext_pos_dif, axis=2)**2.0 
                     -(np.dot(d_ext_pos_dif, kh)/np.linalg.norm(kh))**2.0 )\
              * grid_out   # add geometric factor of crystal surface shape
    
    # Phase for propagating factor to a plane
    d_ext_addpath = np.dot(d_ext_pos_avg, unit_sh)
    d_ext_addpath -= np.max(d_ext_addpath)
    d_ext_addph = d_ext_addpath * wave_number
    
    ''' ============================================================================
        Save necessary variables for farfield propagation. 
        ============================================================================ '''
    idx = np.argmin(np.abs(varis['dth_range'] - dth))
    
    exit_wavefronts['esw_%03d' %idx] = factor
    exit_wavefronts['addph_%03d' %idx] = d_ext_addph
    exit_wavefronts['dims_%03d' %idx] = [grid_sh, grid_sy, grid_s0]
    exit_wavefronts['unit_sh_lab_%03d' %idx] = unit_sh_lab
    if np.abs(dth) < 1e-6: # Save these at Bragg for 3D visualization
        exit_wavefronts['d_inc_pos'] = d_inc_pos
        exit_wavefronts['d_ext_pos'] = d_ext_pos
            
    ''' ====== For pix-to-pix propagation - start ====== '''
    exit_wavefronts['eswpos_%03d' %idx] = d_ext_pos_avg
    exit_wavefronts['detpix_%03d' %idx] = det_pix
    ''' ======= For pix-to-pix propagation - end ======= '''
    
    if SavetoFile: 
        np.save(os.path.join(save_dir, 'exit_wavefronts_%03d.npy' %idx), exit_wavefronts)
    
    return exit_wavefronts


def CalLat_MesoCrysProp(rot_ang, params=None, crystal=None, wf_in=None, flag_progress=False): 
    ''' >>> Instruction <<< 
        This function propagates a wavefront through a meso crystal. 
        
        For multi-thread processing. 
        
        Inputs: 
            rot_ang     [deg], rocking angle of the meso-crystal 
            params      dictionary, parameters for setting up simulation
            crystal     3D array, meso crystal structure
            wf_in       2D complex array, incident wavefront 
        
        Output: 
            wf_out      2D complex array, exit wavefront (near field)
    '''
    with open('temp_progress.txt', 'a+') as progress_file: 
        progress_file.writelines(f'rot_ang{rot_ang:>{10}.3f}    ... ' + strftime('%H:%M:%S', localtime()) + ' \n')
    
    grid_size = params['grid_size']
    photonE = params['photonE']
    element_Z = params['element_Z'] 
    crystal_size_rot = params['crystal_size_rot']
    pad_size = params['pad_size'] 
    rot_ax = params['rot_ax'] 
    
    # Propagation 
    wave_len = 1.2398 / photonE   # [nm]
    wave_vec = 2*np.pi/wave_len   # [nm-1]
    density = xb.ElementDensity(element_Z)   # [g/cm3]
    # Index of Refraction = (1-delta) - i(beta)
    delta = 1 - xb.Refractive_Index_Re(xb.AtomicNumberToSymbol(element_Z), photonE, density)
    beta  = xb.Refractive_Index_Im(xb.AtomicNumberToSymbol(element_Z), photonE, density)

    crystal = Array_zeropad_3D(crystal, AddTo=crystal_size_rot)
    
    # rock crystal 
    crystal_rot = np.zeros_like(crystal)
    for ii in range(np.dot(rot_ax, crystal_size_rot)): 
        if   rot_ax[0] == 1: 
            crystal_rot[ii, :, :] = imu.rotate(crystal[ii, :, :], rot_ang)
        elif rot_ax[1] == 1: 
            crystal_rot[:, ii, :] = imu.rotate(crystal[:, ii, :], rot_ang)
        elif rot_ax[2] == 1: 
            crystal_rot[:, :, ii] = imu.rotate(crystal[:, :, ii], rot_ang)
    # propagate through crystal 
    if not np.array_equal(np.shape(wf_in), pad_size):  
        wf_out = Array_zeropad_2D(wf_in, AddTo=pad_size)
    else: 
        wf_out = np.copy(wf_in)
    for ii in range(np.shape(crystal_rot)[2]): 
        if flag_progress: 
            print('ii = %d / %d' %(ii+1, L), end='\r')
        sample = Array_zeropad_2D(crystal_rot[:,:,ii]*grid_size[2], AddTo=pad_size)
        # attenuation 
        sample_abs = np.exp(- wave_vec * beta * sample)**2
        wf_out *= sample_abs
        # phase delay 
        sample_phd = np.exp(-1j * wave_vec * delta * sample)
        wf_out *= sample_phd
        # propagation 
        wf_out = wavefront_propagator_angspc(wf_out, dist_z=grid_size[2], grid_size=grid_size[:2], 
                                              wave_len=wave_len, pad_size=pad_size, flag_cropout=False)
    return wf_out


def CalLat_MesoCrysProp_cryori(rot_ang, params=None, crystal=None, wf_in=None, crys_ori_params=None, 
                               flag_diff_attenu=True, flag_progress=False, flag_update_temp=False): 
    ''' >>> Instruction <<< 
        This function propagates a wavefront through a meso crystal, 
        taking account of single-particle diffractions. 
        
        For multi-thread processing. 
        
        Inputs: 
            rot_ang     [deg], rocking angle of the meso-crystal 
            params      dictionary, parameters for setting up simulation
            crystal     3D array, meso crystal structure
            wf_in       2D complex array, incident wavefront 
        
        Output: 
            wf_out      2D complex array, exit wavefront (near field)
    '''
    if flag_update_temp: 
        with open('temp_progress.txt', 'a+') as progress_file: 
            progress_file.writelines(f'rot_ang{rot_ang:>{10}.3f}    ... ' + strftime('%H:%M:%S', localtime()) + ' \n')
    
    grid_size = params['grid_size']
    photonE   = params['photonE']
    element_Z = params['element_Z'] 
    crystal_size_rot = params['crystal_size_rot']
    pad_size  = params['pad_size'] 
    rot_ax    = params['rot_ax'] 
    
    vec_particles    = crys_ori_params['vec_particles']
    num_particles    = crys_ori_params['num_particles']
    Bragg_angle      = crys_ori_params['Bragg_angle']
    peak_width       = crys_ori_params['peak_width']
    idx_particle     = crys_ori_params['idx_particle']
    chb_particle_eve = crys_ori_params['chb_particle_eve']
    chb_particle_odd = crys_ori_params['chb_particle_odd']
    
    # Propagation 
    wave_len = 1.2398 / photonE   # [nm]
    wave_vec = 2*np.pi/wave_len   # [nm-1]
    density = xb.ElementDensity(element_Z)   # [g/cm3]
    # Index of Refraction = (1-delta) - i(beta)
    delta = 1 - xb.Refractive_Index_Re(xb.AtomicNumberToSymbol(element_Z), photonE, density)
    beta  = xb.Refractive_Index_Im(xb.AtomicNumberToSymbol(element_Z), photonE, density)

    crystal          = Array_zeropad_3D(crystal,          AddTo=crystal_size_rot)
    idx_particle     = Array_zeropad_3D(idx_particle,     AddTo=crystal_size_rot)
    chb_particle_eve = Array_zeropad_3D(chb_particle_eve, AddTo=crystal_size_rot)
    chb_particle_odd = Array_zeropad_3D(chb_particle_odd, AddTo=crystal_size_rot)
    
    if flag_update_temp: 
        with open('temp_progress.txt', 'a+') as progress_file: 
            progress_file.writelines(f'rot_ang{rot_ang:>{10}.3f}    ... variables obtained ... ' + strftime('%H:%M:%S', localtime()) + ' \n')
    
    # rock crystal 
    crystal_rot = np.zeros_like(crystal)
    for ii in range(np.dot(rot_ax, crystal_size_rot)): 
        if   rot_ax[0] == 1: 
            crystal_rot[ii, :, :] = imu.rotate(crystal[ii, :, :], rot_ang)
        elif rot_ax[1] == 1: 
            crystal_rot[:, ii, :] = imu.rotate(crystal[:, ii, :], rot_ang)
        elif rot_ax[2] == 1: 
            crystal_rot[:, :, ii] = imu.rotate(crystal[:, :, ii], rot_ang)
    
    if flag_update_temp: 
        with open('temp_progress.txt', 'a+') as progress_file: 
            progress_file.writelines(f'rot_ang{rot_ang:>{10}.3f}    ... crystal rotated ... ' + strftime('%H:%M:%S', localtime()) + ' \n')
    
    # get arrays for diff-induced attenuations 
    if flag_diff_attenu: 
        # diff-induced attenuations
        vec_particles_rot = Rotate_vectors(vec_particles, Rot_Matrix(rot_ax, rot_ang, IsRadian=False))
        diff_attenu = np.zeros(num_particles)
        ang_off = np.zeros(num_particles)
        for ii in range(num_particles): 
            ang_off[ii] = Included_Angle(vec_particles_rot[ii,:], [0,0,1], OutputRadian=False)['theta'] - (90.0 - Bragg_angle)
            diff_attenu[ii] = max_diff_att * np.exp( -ang_off[ii]**2 / 2 / (peak_width/2.355)**2 )
        # rotate cry-ori arrays 
        for ii in range(np.dot(rot_ax, np.shape(idx_particle))): 
            # print('ii = %d / %d' %(ii+1, np.dot(rot_ax, np.shape(idx_particle))), end='\r')
            if   rot_ax[0] == 1: 
                idx_particle_rot[ii, :, :]     = imu.rotate(idx_particle[ii, :, :],     rot_ang)
                chb_particle_eve_rot[ii, :, :] = imu.rotate(chb_particle_eve[ii, :, :], rot_ang)
                chb_particle_odd_rot[ii, :, :] = imu.rotate(chb_particle_odd[ii, :, :], rot_ang)
            elif rot_ax[1] == 1: 
                idx_particle_rot[:, ii, :]     = imu.rotate(idx_particle[:, ii, :],     rot_ang)
                chb_particle_eve_rot[:, ii, :] = imu.rotate(chb_particle_eve[:, ii, :], rot_ang)
                chb_particle_odd_rot[:, ii, :] = imu.rotate(chb_particle_odd[:, ii, :], rot_ang)
            elif rot_ax[2] == 1: 
                idx_particle_rot[:, :, ii]     = imu.rotate(idx_particle[:, :, ii],     rot_ang)
                chb_particle_eve_rot[:, :, ii] = imu.rotate(chb_particle_eve[:, :, ii], rot_ang)
                chb_particle_odd_rot[:, :, ii] = imu.rotate(chb_particle_odd[:, :, ii], rot_ang)
        # get indices of all particles 
        idx_particle_rot_clean = ( np.multiply(idx_particle_rot, np.floor(chb_particle_eve_rot)) +
                                   np.multiply(idx_particle_rot, np.floor(chb_particle_odd_rot)) ).astype('int')
        # assign diff-induced attenuations to 3D array
        att_particle_rot_clean = np.zeros(np.shape(idx_particle_rot_clean))
        for ii in range(np.shape(att_particle_rot_clean)[0]): 
            for jj in range(np.shape(att_particle_rot_clean)[1]): 
                for kk in range(np.shape(att_particle_rot_clean)[2]): 
                    if idx_particle_rot_clean[ii, jj, kk] != 0: 
                        att_particle_rot_clean[ii, jj, kk] = diff_attenu[idx_particle_rot_clean[ii, jj, kk]-1]
    
    if flag_update_temp: 
        with open('temp_progress.txt', 'a+') as progress_file: 
            progress_file.writelines(f'rot_ang{rot_ang:>{10}.3f}    ... attenuation calculated ... ' + strftime('%H:%M:%S', localtime()) + ' \n')
    
    # propagate through crystal 
    if not np.array_equal(np.shape(wf_in), pad_size):  
        wf_out = Array_zeropad_2D(wf_in, AddTo=pad_size)
    else: 
        wf_out = np.copy(wf_in)
    
    if flag_update_temp: 
        with open('temp_progress.txt', 'a+') as progress_file: 
            progress_file.writelines(f'rot_ang{rot_ang:>{10}.3f}    ... propagation start ... ' + strftime('%H:%M:%S', localtime()) + ' \n')
    
    for ii in range(np.shape(crystal_rot)[2]): 
        if flag_progress: 
            print('ii = %d / %d' %(ii+1, L), end='\r')
        sample = Array_zeropad_2D(crystal_rot[:,:,ii]*grid_size[2], AddTo=pad_size)
        # attenuation 
        sample_abs = np.exp(- wave_vec * beta * sample)**2
        wf_out *= sample_abs
        # diff_induced attenuation
        if flag_diff_attenu: 
            wf_temp *= 1 - Array_zeropad_2D(att_particle_rot_clean[:,:,ii], AddTo=pad_size)
        # phase delay 
        sample_phd = np.exp(-1j * wave_vec * delta * sample)
        wf_out *= sample_phd
        # propagation 
        wf_out = wavefront_propagator_angspc(wf_out, dist_z=grid_size[2], grid_size=grid_size[:2], 
                                              wave_len=wave_len, pad_size=pad_size, flag_cropout=False)
    return wf_out






# ===========================================================================================
# Below are archived functions for reference
# ===========================================================================================
'''
def CalLat_read_cif(CIF_name='test.cif', CIF_path='', LatticeInfo=False): 
    # Get the current working directory
    retval = os.getcwd()
    
    # Generate the full file path and check if the file exists
    if len(CIF_path) == 0: 
        CIF_path = retval
    if CIF_path[-1] == '/' or CIF_path[-1] == '\\':
        File = CIF_path + CIF_name
    elif '/' in CIF_path: 
        File = CIF_path + '/' + CIF_name
    elif '\\' in CIF_path: 
        File = CIF_path + '\\' + CIF_name
    else: 
        print('>>>>>> Error! Unknown path separator. <<<<<<')
        input('>>>>>> Press any key to quit...')
        return
    if not os.path.isfile(File): 
        print('>>>>>> Error! File does NOT exist. <<<<<<')
        input('>>>>>> Press any key to quit...')
        return
    
    # Read cif file
    cif_data = Crystal.from_cif(File)
    if LatticeInfo: 
        print(cif_data)
    
    return cif_data.lattice_parameters


def CalLat_creat_offset_screw(Crys_coords, ScrewLoc=None, ScrewEdge=None, ScrewDis=None, 
                              Progress=False): 
    ' >>> Introduction <<<
        This function mordifies the Crys_coords (fractional coordiantes of unit cells) to
        creat a screw dislocation. 
        
        Inputs: 
            Crys_coords     (n,3) array, each (n,1) is a raveled axis
            ScrewLoc        (3,) int array, the location of the screw dislocation
            ScrewEdge       (3,) int array, edge of the screw dislocation. 
                                [?,0,0] or [0,?,0] or [0,0,?],  ? is the edge length
            ScrewDis        (3,) int array, magnitude of the dislocation. 
                                [?,0,0] or [0,?,0] or [0,0,?]
        Output: 
            Disloc          (n,3) array, each (n,1) is a raveled axis
    '
    ScrewLoc = np.asarray(ScrewLoc)
    ScrewEdge = np.asarray(ScrewEdge)
    ScrewDis = np.asarray(ScrewDis)
    
    Disloc = np.zeros_like(Crys_coords)
    N,_ = np.shape(Crys_coords)
    edge = np.linalg.norm(ScrewEdge)
    edgeori = ScrewEdge / edge
    ratio = ScrewDis / edge
    
    if np.linalg.norm(ScrewDis) == 0 or np.linalg.norm(ScrewEdge) == 0: 
        return Disloc
    
    if np.nonzero(ScrewDis)[0][0] == 0: 
        idx = 2
    elif np.nonzero(ScrewDis)[0][0] == 1: 
        idx = 0
    elif np.nonzero(ScrewDis)[0][0] == 2: 
        idx = 1
    
    if Progress: 
        print('>>>>>> Start calculating ... ' + strftime('%H:%M:%S', localtime()))
    
    for i in range(N): 
        if Progress: 
            if np.remainder(i, 100) == 0: 
                print('>>>>>> Processing %0.2f %% ... ' %((i+1)/N*100) + 
                      strftime('%H:%M:%S', localtime()), end='\r')
        Pt = Crys_coords[i]
        temp = Dist_point_to_line(Pt, ScrewLoc, vector=ScrewDis)
        foot = temp['foot']
        vect = np.asarray(Pt - foot)
        dist = np.linalg.norm(vect)
        if dist == 0: 
            continue
        elif dist > edge: 
            offset = ScrewDis
        else: 
            offset = ratio * dist
        angle = np.arccos(np.dot(vect/dist, edgeori))
        if vect[idx] < 0: 
            angle = 2*np.pi - angle
        Disloc[i, :] = offset * angle / 2 / np.pi
    
    if Progress: 
        print('>>>>>> Processing 100.00 % ... Done! ' + strftime('%H:%M:%S', localtime()))
    
    return Disloc


def CalLat_RotY_of_Q_old(k=0, Q=[0,0,0], OutputRadian=False):    # Wrong result!  
    ' >>> Instruction <<<
        This function figures out the required rotation of Q around Y axis for the Bragg condition. 
        
        Inputs: 
            k      [A^-1], Wave vector
            Q      (3,) array, the initial Q vector
            
        Math: 
            The incident and diffracted wave vectors are: 
                ki = [0, 0, k] 
                kf = RotY(Delta) * RotX(-Gamma) * ki
                   = [k*cos(Gamma)*sin(Delta), k*sin(Gamma), k*cos(Gamma)*cos(Delta)]
            
            The initial vector Q = [q*cos(Alpha)*sin(BetaI), q*sin(Alpha), q*cos(Alpha)*cos(BetaI)]
            The final vector Q' = [q*cos(Alpha)*sin(BetaF), q*sin(Alpha), q*cos(Alpha)*cos(BetaF)]
            
            Also, there is: kf - ki = Q', i.e.
                    k*cos(Gamma)*sin(Delta) = q*cos(Alpha)*sin(BetaF)
                               k*sin(Gamma) = q*sin(Alpha)
                k*cos(Gamma)*cos(Delta) - k = q*cos(Alpha)*cos(BetaF)
            
            Therefore, 
                    Gamma = arcsin( q/k * sin(Alpha) )
                    BetaF = arccos( -q/k/2 * cos(Alpha)/cos(Gamma) )
                    Delta = arccos( (1 + q/k * cos(Alpha) * cos(BetaF))/cos(Gamma) )
                    
                    The rotation is BetaF - BetaI
    '
    q = np.linalg.norm(Q)
    if q == 0 or k == 0:
        Rotat = 0
        Gamma = 0
        Delta = 0
        return {'Rotation':Rotat, 'Gamma':Gamma, 'Delta':Delta}
    
    BetaI, Alpha = Vector_to_angle(Q, OutputRadian=True)
    
    Gamma = np.arcsin( q/k * np.sin(Alpha) )
    BetaF = np.arccos( -q/k/2 * np.cos(Alpha)/np.cos(Gamma) )
    Delta = np.arccos( (1 + q/k * np.cos(Alpha) * np.cos(BetaF))/np.cos(Gamma) )
    Rotat = BetaF - BetaI
    
    if not OutputRadian: 
        Gamma = np.rad2deg(Gamma)
        Delta = np.rad2deg(Delta)
        Rotat = np.rad2deg(Rotat)
    
    return {'Rotation':Rotat, 'Gamma':Gamma, 'Delta':Delta}


def CalLat_CSF_inFile(q, TempFile=None, PhotonE=None, LattVects=None, Zs=None, 
                      Atoms_coords=None, Crys_coords=None, Disloc=None): 
    q = np.matrix([q]).transpose()
    Lattforfac = CalLat_lattice_structure_factor(q=q, 
                                                 PhotonE=PhotonE, 
                                                 LattVects=LattVects, 
                                                 Zs=Zs,
                                                 Atoms_coords=Atoms_coords, 
                                                 Flag_InputReg=False)
    StruFactor = CalLat_crystal_structure_factor(q=q, 
                                                 Latt_forfac=Lattforfac, 
                                                 LattVects=LattVects,
                                                 Crys_coords=Crys_coords+Disloc, 
                                                 Flag_InputReg=False)
    Result = '%f\t%f\t%f\t%f\n' %(q[0], q[1], q[2], np.abs(StruFactor)**2)
    with open(TempFile, 'a') as File: 
        File.write(Result)


def CalLat_DifPatSimu_MultiNodes(CrySize=[3,3,3], CryShape='cubic', 
                                 ScrewLoc=[0,0,0], ScrewEdge=[2,0,0], ScrewDis =[0,0,0], 
                                 CIF_path=r'/CIFs', CIF_file='Ag.cif', 
                                 Crys_orie=[0,0,0], PhotonE=12,  
                                 Delt_rang=1, Delt_step=0.05, Delt_cent=21.3,
                                 Gamm_rang=1, Gamm_step=0.05, Gamm_cent=45.9, 
                                 Rock_rang=1.2, Rock_step=0.06, Rock_cent=0, 
                                 Rock_axis='X', IsRadian=False, RunSimu=False, 
                                 MultiProcess=False, Processes_count=1, 
                                 SaveResult=False, SaveName='SimuDifPat'): 
    Crystal = CalLat_creat_crystal(L1=CrySize[0], L2=CrySize[1], L3=CrySize[2], Shape=CryShape)
    Crys_coords = Crystal['Coords']
    Crys_datsha = Crystal['Shape']
    Disloc = CalLat_creat_offset_screw(Crys_coords, ScrewLoc=ScrewLoc, ScrewEdge=ScrewEdge, 
                            ScrewDis=ScrewDis, Progress=False)
    result = CalLat_read_cif(CIF_name=CIF_file, CIF_path=CIF_path)
    Diffraction = CalLat_diffraction_vector(Crys_orie=Crys_orie, PhotonE=PhotonE,  
                               Delt_rang=Delt_rang, Delt_step=Delt_step, Delt_cent=Delt_cent,
                               Gamm_rang=Gamm_rang, Gamm_step=Gamm_step, Gamm_cent=Gamm_cent, 
                               Rock_rang=Rock_rang, Rock_step=Rock_step, Rock_cent=Rock_cent, 
                               Rock_axis=Rock_axis, IsRadian=IsRadian)
    DiffVects = Diffraction['DiffVect']
    DiffShape = Diffraction['DataShape']
    dq = Diffraction['dq']
    Ax_gamma = Diffraction['GammaAxis']
    Ax_delta = Diffraction['DeltaAxis']
    Ax_rock = Diffraction['RockAxis']
    print('Array shape [rock, gamma, delta] is ', DiffShape)
    print('dq in [ rock  ] is ', dq[0])
    print('dq in [ gamma ] is ', dq[1])
    print('dq in [ delta ] is ', dq[2])
    
    _,N = np.shape(DiffVects)
    LattVects = result['LattVectors']
    Zs = result['Atoms_Z']
    Atoms_coords = result['Atoms_coords']
    
    if not RunSimu: 
        return
    
    DifPat = np.zeros(N)
    if MultiProcess: 
        pool = Pool(processes=Processes_count)
        par = [(DiffVects[0,0],DiffVects[1,0],DiffVects[2,0])]
        for i in range(N-1): 
            par.append((DiffVects[0,i+1],DiffVects[1,i+1],DiffVects[2,i+1]))
        # for multi-nodes, creat a temp file to record the results
        TempFile = 'temp_0'
        i = 0
        while os.path.isfile(TempFile+'.dat'):  # check if file exists
            i = i + 1
            TempFile = 'temp_%d' %i
        print('>>>>>> Start %d-core multiprocessing ... ' %Processes_count 
              + strftime('%H:%M:%S', localtime()))
        open(TempFile+'.dat', 'w+').close()
        CalLat_CSF_partial = partial(CalLat_CSF_inFile, TempFile=TempFile+'.dat', PhotonE=PhotonE, 
                                     LattVects=LattVects, Zs=Zs, Atoms_coords=Atoms_coords, 
                                     Crys_coords=Crys_coords, Disloc=Disloc)
        DifPat = pool.map(CalLat_CSF_partial, par)
        print('>>>>>> Processing finished! ' + strftime('%H:%M:%S', localtime()))
        pool.terminate()
    else: 
        print('>>>>>> Start calculating ... ' + strftime('%H:%M:%S', localtime()))
        for i in range(N): 
            q = tuple(np.squeeze(np.asarray(DiffVects[:,i])))
            DifPat[i] = CalLat_CSF(q, PhotonE=PhotonE, LattVects=LattVects, Zs=Zs, 
                            Atoms_coords=Atoms_coords, Crys_coords=Crys_coords, Disloc=Disloc)
            if np.remainder(i, 100) == 0: 
                print('>>>>>> Processing %0.2f %% ... ' %((i+1)/N*100) + 
                      strftime('%H:%M:%S', localtime()), end='\r')
        print('>>>>>> Processing 100.00 % ... Done! ' + strftime('%H:%M:%S', localtime()))
    DifPat = np.reshape(DifPat, DiffShape)
    
    if SaveResult: 
        Save_tif((DifPat/1e6).astype('int32'), SaveName+'.tif', OverWrite=True)
        np.save(SaveName+'.npy', DifPat)


def CalLat_CSF_inFile(Q_File, PhotonE=None, LattVects=None, Zs=None, 
                      Atoms_coords=None, Crys_coords=None, Disloc=None): 
    # Read q vectors from file
    DiffVects = np.matrix(np.load(Q_File))
    os.remove(Q_File)
    _,N = np.shape(DiffVects)
    DifPat = np.zeros(N)
    # Calculate diffraction intensity
    for i in range(N): 
        Lattforfac = CalLat_lattice_structure_factor(q=DiffVects[:,i], PhotonE=PhotonE, Zs=Zs, 
                                      LattVects=LattVects, Atoms_coords=Atoms_coords, 
                                      Flag_InputReg=False)
        StruFactor = CalLat_crystal_structure_factor(q=DiffVects[:,i], Crys_coords=Crys_coords+Disloc, 
                                      Latt_forfac=Lattforfac, LattVects=LattVects, 
                                      Flag_InputReg=False)
        DifPat[i] = np.abs(StruFactor)**2
    # Save result to a seperated file
    np.save(Q_File[:-4]+'_Result'+Q_File[-4:], DifPat)


def CalLat_DifPatSimu_MultiNodes(CrySize=[3,3,3], CryShape='cubic', 
                                 ScrewLoc=[0,0,0], ScrewEdge=[2,0,0], ScrewDis =[0,0,0], 
                                 CIF_path=r'/CIFs', CIF_file='Ag.cif', 
                                 Crys_orie=[0,0,0], PhotonE=12,  
                                 Delt_rang=1, Delt_step=0.05, Delt_cent=21.3,
                                 Gamm_rang=1, Gamm_step=0.05, Gamm_cent=45.9, 
                                 Rock_rang=1.2, Rock_step=0.06, Rock_cent=0, 
                                 Rock_axis='X', IsRadian=False, RunSimu=False, 
                                 MultiProcess=False, Processes_count=1, 
                                 SaveResult=False, SaveName='SimuDifPat'): 
    Crystal = CalLat_creat_crystal(L1=CrySize[0], L2=CrySize[1], L3=CrySize[2], Shape=CryShape)
    Crys_coords = Crystal['Coords']
    Crys_datsha = Crystal['Shape']
    print('>>>>>> Crystal data shape is ', Crys_datsha)
    Disloc = CalLat_creat_offset_screw(Crys_coords, ScrewLoc=ScrewLoc, ScrewEdge=ScrewEdge, 
                               ScrewDis=ScrewDis, Progress=False)
    result = CalLat_read_cif(CIF_name=CIF_file, CIF_path=CIF_path)
    Diffraction = CalLat_diffraction_vector(Crys_orie=Crys_orie, PhotonE=PhotonE,  
                               Delt_rang=Delt_rang, Delt_step=Delt_step, Delt_cent=Delt_cent,
                               Gamm_rang=Gamm_rang, Gamm_step=Gamm_step, Gamm_cent=Gamm_cent, 
                               Rock_rang=Rock_rang, Rock_step=Rock_step, Rock_cent=Rock_cent, 
                               Rock_axis=Rock_axis, IsRadian=IsRadian)
    DiffVects = Diffraction['DiffVect']
    DiffShape = Diffraction['DataShape']
    dq = Diffraction['dq']
    Ax_gamma = Diffraction['GammaAxis']
    Ax_delta = Diffraction['DeltaAxis']
    Ax_rock = Diffraction['RockAxis']
    print('>>>>>> Array shape [rock, gamma, delta] is ', DiffShape)
    print('       dq in [ rock  ] is ', dq[0])
    print('       dq in [ gamma ] is ', dq[1])
    print('       dq in [ delta ] is ', dq[2])
    
    _,N = np.shape(DiffVects)
    LattVects = result['LattVectors']
    Zs = result['Atoms_Z']
    Atoms_coords = result['Atoms_coords']
    
    if not RunSimu: 
        print('>>>>>> Switch RunSimu to True to start simulation')
        return
    
    DifPat = np.zeros(N)
    if MultiProcess: 
        # creat a temp file name
        TempFileName = 'temp_A'
        i = 0
        while os.path.isfile(TempFileName+'_%03d.npy' %0):  # check if file exists
            i = i + 1
            TempFileName = 'temp_' + chr(ord('A') + i)
        # for multi-nodes, save DiffVects to several .npy files
        print('>>>>>> Save q-vectors to .npy files ... ' + strftime('%H:%M:%S', localtime()))
        N_seg = int(np.ceil(N/Processes_count))
        for i in range(Processes_count): 
            CurrentFile = TempFileName+'_%03d.npy' %i
            if i == 0: 
                par = [CurrentFile]
            else: 
                par.append(CurrentFile)
            if (i+1) * N_seg <= N: 
                np.save(CurrentFile, DiffVects[:,i*N_seg:(i+1)*N_seg])
            else: 
                np.save(CurrentFile, DiffVects[:,i*N_seg:N])
        # Multiprocessing
        CalLat_CSF_partial = partial(CalLat_CSF_inFile, PhotonE=PhotonE, 
                                     LattVects=LattVects, Zs=Zs, Atoms_coords=Atoms_coords, 
                                     Crys_coords=Crys_coords, Disloc=Disloc)
        pool = Pool(processes=Processes_count)
        print('>>>>>> Start %d-core multiprocessing ... ' %Processes_count 
              + strftime('%H:%M:%S', localtime()))
        DifPat = pool.map(CalLat_CSF_partial, par)
        print('>>>>>> Processing finished! ' + strftime('%H:%M:%S', localtime()))
        pool.terminate()
        # Read and merge result files
        print('>>>>>> Read results from .npy files ... ' + strftime('%H:%M:%S', localtime()))
        for i in range(Processes_count): 
            CurrentFile = TempFileName+'_%03d_Result.npy' %i
            if i == 0:
                DifPat = np.load(CurrentFile)
            else: 
                DifPat = np.append(DifPat, np.load(CurrentFile))
            os.remove(CurrentFile)
    else: 
        print('>>>>>> Start calculating ... ' + strftime('%H:%M:%S', localtime()))
        for i in range(N): 
            q = tuple(np.squeeze(np.asarray(DiffVects[:,i])))
            DifPat[i] = CalLat_CSF(q, PhotonE=PhotonE, LattVects=LattVects, Zs=Zs, 
                            Atoms_coords=Atoms_coords, Crys_coords=Crys_coords, Disloc=Disloc)
            if np.remainder(i, 100) == 0: 
                print('>>>>>> Processing %0.2f %% ... ' %((i+1)/N*100) + 
                      strftime('%H:%M:%S', localtime()), end='\r')
        print('>>>>>> Processing 100.00 % ... Done! ' + strftime('%H:%M:%S', localtime()))
    DifPat = np.reshape(DifPat, DiffShape)
    
    if SaveResult: 
        Save_tif((DifPat/1e6).astype('int32'), SaveName+'.tif', OverWrite=True)
        np.save(SaveName+'.npy', DifPat)


def CalLat_gridsize_estimat(photonE=None, delta=None, gamma=None, drock=None, rockaxis='Y', nrock=None, 
                            nx=None, ny=None, detdist=None, px=None, py=None, disp_info=True): 
    ' >>> Instruction <<< 
        This function calculate grid size in real and reciprocal spaces based on the diffraction geometry. 
        
        Note this is just a ROUGH estimation: 
            Ewald sphere is approximated to a flat plane. 
            Detector surface grid is approximated to a regular grid in reciprocal space. 
        
        Inputs: 
            photonE:     Photon energy in [keV]
            delta:       Detector angle around Lab Y axis
            gamma:       Detector angle around Lab -X axis
            drock:       Step of rocking curve scan
            nrock:       number of rocking steps 
            rockaxis:    Rocking axis. Note drock may be negative for rocking around X. 
            detdist:     Sample-detector distance in [m]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [um]
            nx, ny:      Number of pixels in X(horizontal) and Y(vertical)
            disp_info:   Whether print out all parameters
        
        Output: 
            No output yet. 
    '
    wavelen = 1.2398 / photonE   # [nm]
    wavevec = 2 * np.pi / wavelen   # [nm-1]
    if disp_info: 
        print('photonE: %.3f [keV]' %photonE)
        print('wavelen: %.3f [nm]' %wavelen)
        print('wavevec: %.3f [nm-1]' %wavevec)
        print('')
        print('delta: %.3f [deg]' %delta)
        print('gamma: %.3f [deg] (may need to be negative)' %gamma)
        print('drock: %.3f [deg]' %drock)
        print('')
        print('detdist: %.3f [m]' %(detdist))
        print('px, py: %.1f, %.1f [um]' %(px, py))
        print('')
    
    # Convert units 
    delta = np.deg2rad(delta)
    gamma = np.deg2rad(gamma)
    drock = np.deg2rad(drock)
    detdist *= 1e3   # [mm]
    px *= 1e-3   # [mm]
    py *= 1e-3   # [mm]
    
    # Rotation matrices of kf and detector
    R_delta = Rot_Matrix([0,1,0], delta, IsRadian=True)
    R_gamma = Rot_Matrix([-1,0,0], gamma, IsRadian=True)
    
    # Calculate Q
    ki = wavevec * np.asarray([0, 0, 1]) 
    kf = Rotate_vectors( Rotate_vectors( ki, R_gamma ), R_delta )
    Q = kf - ki   # [nm-1]
    
    # Grid in reciprocal space
    ang_x_step = np.arctan2(px, detdist)   # [rad]
    ang_y_step = np.arctan2(py, detdist)   # [rad]
    ang_r_step = drock   # [rad]
    ang_x_edge = np.arctan2(px * np.floor(nx/2), detdist)   # [rad]
    ang_y_edge = np.arctan2(py * np.floor(ny/2), detdist)   # [rad]
    ang_r_edge = drock * np.floor(nrock/2)   # [rad]
    
    # Rotation matrices of pixels
    R_detx_step = Rot_Matrix(Rotate_vectors(Rotate_vectors([0,-1,0], 
                                                           R_gamma), 
                                            R_delta), 
                             ang_x_step)
    R_dety_step = Rot_Matrix(Rotate_vectors(Rotate_vectors([-1,0,0], 
                                                           R_gamma), 
                                            R_delta), 
                             ang_y_step)
    R_detx_edge = Rot_Matrix(Rotate_vectors(Rotate_vectors([0,-1,0], 
                                                           R_gamma), 
                                            R_delta), 
                             ang_x_edge)
    R_dety_edge = Rot_Matrix(Rotate_vectors(Rotate_vectors([-1,0,0], 
                                                           R_gamma), 
                                            R_delta), 
                             ang_y_edge)
    
    ' dq of each voxel '
    dq_x_step = Rotate_vectors( kf, R_detx_step ) - ki - Q
    dq_y_step = Rotate_vectors( kf, R_dety_step ) - ki - Q
    if rockaxis == 'Y' or rockaxis == 'y': 
        'Crystal is rotated around Y axis. '
        T_rock = Rot_Matrix([0, 1, 0], ang_r_step, IsRadian=True)
    elif rockaxis == 'X' or rockaxis == 'x': 
        'Crystal is rotated around X axis. '
        T_rock = Rot_Matrix([-1, 0, 0], ang_r_step, IsRadian=True)
    else: 
        print('>>>>>> Unknown RockAxis. ')
        input('       Press any key to quit...')
        return
    dq_r_step = np.squeeze(np.asarray(np.matmul(T_rock, Q) - Q))
    
    ' dq of entire grid '
    dq_x_edge = Rotate_vectors( kf, R_detx_edge ) - ki - Q
    dq_y_edge = Rotate_vectors( kf, R_dety_edge ) - ki - Q
    if rockaxis == 'Y' or rockaxis == 'y': 
        'Crystal is rotated around Y axis. '
        T_rock = Rot_Matrix([0, 1, 0], ang_r_edge, IsRadian=True)
    elif rockaxis == 'X' or rockaxis == 'x': 
        'Crystal is rotated around X axis. '
        T_rock = Rot_Matrix([-1, 0, 0], ang_r_edge, IsRadian=True)
    else: 
        print('>>>>>> Unknown RockAxis. ')
        input('       Press any key to quit...')
        return
    dq_r_edge = np.squeeze(np.asarray(np.matmul(T_rock, Q) - Q))
    
    # Grid in real space
    ' dr of entire grid '
    denorm = np.dot(dq_x_step, np.cross(dq_y_step, dq_r_step))
    dr_x_edge = 2 * np.pi * np.cross(dq_y_step, dq_r_step) / denorm
    dr_y_edge = 2 * np.pi * np.cross(dq_r_step, dq_x_step) / denorm
    dr_r_edge = 2 * np.pi * np.cross(dq_x_step, dq_y_step) / denorm
    ' dr of each voxel '
    denorm = np.dot(dq_x_edge, np.cross(dq_y_edge, dq_r_edge))
    dr_x_step = 2 * np.pi * np.cross(dq_y_edge, dq_r_edge) / denorm
    dr_y_step = 2 * np.pi * np.cross(dq_r_edge, dq_x_edge) / denorm
    dr_r_step = 2 * np.pi * np.cross(dq_x_edge, dq_y_edge) / denorm
    
    if disp_info: 
        print('ki: [%.3f, %.3f, %.3f] [nm-1]' %(ki[0], ki[1], ki[2]))
        print('kf: [%.3f, %.3f, %.3f] [nm-1]' %(kf[0], kf[1], kf[2]))
        print('Q:  [%.3f, %.3f, %.3f] [nm-1]' %(Q[0], Q[1], Q[2]))
        print('')
        print('dq_x:     [%.3f, %.3f, %.3f] [um-1]' %(dq_x_step[0]*1e3, dq_x_step[1]*1e3, dq_x_step[2]*1e3))
        print('dq_y:     [%.3f, %.3f, %.3f] [um-1]' %(dq_y_step[0]*1e3, dq_y_step[1]*1e3, dq_y_step[2]*1e3))
        print('dq_rock:  [%.3f, %.3f, %.3f] [um-1]' %(dq_r_step[0]*1e3, dq_r_step[1]*1e3, dq_r_step[2]*1e3))
        print('========== Just a ROUGH estimation ==========')
        print('Reciprocal space: ')
        print('  > Resolution: ')
        print('        dq_x:     %.3f [um-1]' %(np.linalg.norm(dq_x_step)*1e3))
        print('        dq_y:     %.3f [um-1]' %(np.linalg.norm(dq_y_step)*1e3))
        print('        dq_rock:  %.3f [um-1]' %(np.linalg.norm(dq_r_step)*1e3))
        print('  > Field of view: ')
        print('        q_x:      %.3f [um-1] (+/-)' %(np.linalg.norm(dq_x_edge)*1e3))
        print('        q_y:      %.3f [um-1] (+/-)' %(np.linalg.norm(dq_y_edge)*1e3))
        print('        q_rock:   %.3f [um-1] (+/-)' %(np.linalg.norm(dq_r_edge)*1e3))
        print('Real space: ')
        print('  > Resolution: ')
        print('        dr_x:     %.3f [nm], %.3f [nm] (no 2\u03C0)' %(np.linalg.norm(dr_x_step), np.linalg.norm(dr_x_step)/np.pi))
        print('        dr_y:     %.3f [nm], %.3f [nm] (no 2\u03C0)' %(np.linalg.norm(dr_y_step), np.linalg.norm(dr_y_step)/np.pi))
        print('        dr_rock:  %.3f [nm], %.3f [nm] (no 2\u03C0)' %(np.linalg.norm(dr_r_step), np.linalg.norm(dr_r_step)/np.pi))
        print('  > Field of view: ')
        print('        r_x:     %.3f [nm], %.3f [nm] (no 2\u03C0)' %(np.linalg.norm(dr_x_edge), np.linalg.norm(dr_x_edge)/np.pi))
        print('        r_y:     %.3f [nm], %.3f [nm] (no 2\u03C0)' %(np.linalg.norm(dr_y_edge), np.linalg.norm(dr_x_edge)/np.pi))
        print('        r_rock:  %.3f [nm], %.3f [nm] (no 2\u03C0)' %(np.linalg.norm(dr_r_edge), np.linalg.norm(dr_x_edge)/np.pi))
    
    return


def wavefront_propagator_angspc(wf_in, dist_z=5, grid_size=[5,5], wave_len=0.1, 
                                pad_size=[501,501], freq_offset=0, flag_cropout=True): 
    '' >>> Instruction <<< 
        This function propagates 2D wavefront using angular spectrum. 
        See Goodman's Fourier Optics and DOI:/10.1155/2017/7293905 for more details.
        
        Input:
            wf_in           (N, M) array representing a complex wavefront
            dist_z          [nm], propagation distance
            grid_size       [nm, nm], pixel size of the incident 2D wavefront
            wave_len        [nm], wavelength of the incident light
            pad_size        (>=N, >=M), zero padding for fourier transform
            freq_offset     remove phase ramp
            flag_cropout    True: crop output to same size as input
        
        Output: 
            wf_out          (N, M) array, complex wavefront after propagation 
    ''
    # Regularize input and get dimension
    wf_in = np.asarray(wf_in)
    input_x, input_y = np.shape(wf_in)
    # Convert input wavefront to angular specturm via FFT
    grid_x, grid_y = grid_size   # [nm]
    pad_x, pad_y = pad_size
    wf_in_as = sp.fft.ifft2((Array_zeropad_crop_2D(wf_in, AdjTo=[pad_x,pad_y])))   # sp.fft.fftshift
    # Calculate the additional phase
    idx_x = np.arange(pad_x) - pad_x/2 + freq_offset
    idx_y = np.arange(pad_y) - pad_y/2 + freq_offset
    fq_x = 1 / pad_x / grid_x * idx_x   # [nm-1]
    fq_y = 1 / pad_y / grid_y * idx_y   # [nm-1]
    fq_yy, fq_xx = np.meshgrid(fq_y, fq_x)
    ph_term = (np.sqrt(1/wave_len**2 - fq_xx**2 - fq_yy**2)).astype('complex128')
    ph_term *= 2j * np.pi * dist_z 
    # Propagation
    wf_out_as = wf_in_as * np.exp(ph_term)
    wf_out = (sp.fft.fft2(wf_out_as))   # sp.fft.fftshift
    if flag_cropout: 
        wf_out = Array_crop_2D(wf_out, CropTo=(input_x, input_y))
    return wf_out


def wavefront_propagator_angspc(wf_in, dist_z=5, grid_size=[5,5], wave_len=0.1, 
                                pad_size=[501,501], freq_offset=0, flag_cropout=True, 
                                flag_norm=False): 
    '' >>> Instruction <<< 
        This function propagates 2D wavefront using angular spectrum. 
        See Goodman's Fourier Optics and DOI:/10.1155/2017/7293905 for more details.
        
        !!! Note that wf_in dims should be even !!!
        
        Input:
            wf_in           (N, M) array representing a complex wavefront
            dist_z          [nm], propagation distance
            grid_size       [nm, nm], pixel size of the incident 2D wavefront
            wave_len        [nm], wavelength of the incident light
            pad_size        (>=N, >=M), zero padding for fourier transform
            freq_offset     remove phase ramp
            flag_cropout    True: crop output to same size as input
            flag_norm       True: normalize wf_out intensity to wf_in
        
        Output: 
            wf_out          (N, M) array, complex wavefront after propagation 
    ''
    # Regularize input and get dimension
    wf_in = np.asarray(wf_in)
    input_x, input_y = np.shape(wf_in)
    # Convert input wavefront to angular specturm via FFT
    grid_x, grid_y = grid_size   # [nm]
    pad_x, pad_y = pad_size
    wf_in_as = sp.fft.fftshift(sp.fft.fft2((Array_zeropad_crop_2D(wf_in, AdjTo=[pad_x,pad_y])))) 
    # Calculate the additional phase
    idx_x = np.arange(pad_x) - pad_x/2 + freq_offset
    idx_y = np.arange(pad_y) - pad_y/2 + freq_offset
    fq_x = 1 / pad_x / grid_x * idx_x   # [nm-1]
    fq_y = 1 / pad_y / grid_y * idx_y   # [nm-1]
    fq_yy, fq_xx = np.meshgrid(fq_y, fq_x)
    # Angular spectrum phase term
    ph_term = np.sqrt((1/wave_len**2 - fq_xx**2 - fq_yy**2).astype('complex128'))
    ph_term *= 2j * np.pi * dist_z 
    # Eliminate anything outside 
    circ = (np.sqrt( (fq_xx*wave_len)**2 + (fq_yy*wave_len)**2 ) < 1).astype('float')
    wf_in_as *= circ
    ph_term *= circ
    # Propagation
    wf_out_as = wf_in_as * np.exp(ph_term)
    wf_out = sp.fft.ifft2(sp.fft.fftshift(wf_out_as))
    if flag_cropout: 
        wf_out = Array_crop_2D(wf_out, CropTo=(input_x, input_y))
    # Normalization
    if flag_norm: 
        wf_out *= np.sqrt(np.sum( np.abs( wf_in  * np.conjugate(wf_in)  )) / 
                          np.sum( np.abs( wf_out * np.conjugate(wf_out) )) )
    return wf_out


'''


