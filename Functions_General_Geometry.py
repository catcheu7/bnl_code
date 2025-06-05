import numpy as np
from Functions_General_Algebra import *

try: 
    from _trispline import Spline
except: 
    print('Eqtools.trispline() is not available !')

from scipy.interpolate import RectBivariateSpline as rbs
from scipy.interpolate import RegularGridInterpolator as rgi
# from scipy.ndimage import measurements as spmsu
from scipy import ndimage as simg
import scipy.interpolate as sin
from time import time, localtime, strftime
import pyfftw as pf
import numpy.fft as nf
import imutils as imu


def Rot_Matrix(axis, theta, IsRadian=True): 
    ''' >>> Instruction <<< 
        Calculate a generic rotation matrix that makes a rotation of 'theta' around a given 'axis'.

        Inputs:
            'axis'     Rotation axis, format np.asarray([x, y, z])
            'theta'    Rotation angle

        Output: 
            a 3-by-3 array

        In application, assuming that we need to rotate a vector Vi to a new direction Vf, then
            axis   = np.cross(Vi, Vf) / np.linalg.norm(np.cross(Vi, Vf))
            theta  = np.arccos( np.dot(Vi, Vf) / (np.linalg.norm(Vi) * np.linalg.norm(Vf)) )
    '''
    if not IsRadian: 
        theta = np.deg2rad(theta)
    
    if np.linalg.norm(axis) == 0: 
        x,y,z = axis
    else: 
        x,y,z = axis/np.linalg.norm(axis)
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c
    return  np.matrix([[ x*x*C+c,    x*y*C-z*s,  x*z*C+y*s ],
                       [ y*x*C+z*s,  y*y*C+c,    y*z*C-x*s ],
                       [ z*x*C-y*s,  z*y*C+x*s,  z*z*C+c   ]])


def Rotate_vectors(vectors, matrix): 
    ''' >>> INstruction <<<
        Apply rotation matrix to one or multiply vectors. 
        
        Inputs: 
            vectors   (3, n) or (n, 3) array
            matrix    (3, 3) array. From function Rot_Matrix
    '''
    if np.size(np.shape(vectors)) > 2: 
        print(">>>>>> 'vectors' should be a (3, n) or (n, 3) array. ")
        input('>>>>>> Press any key to quit... ')
        return
    if np.shape(vectors)[0] == 3: 
        if np.size(np.shape(vectors)) == 2 and np.shape(vectors)[1] == 3:
            print(">>>>>> 'vectors' is a (3, 3) array. ")
            Flag_square = input(">>>>>> Is the first axis the number of vectors? (Y/N)")
            if Flag_square == 'Y' or Flag_square == 'y': 
                V = vectors.copy()
                Flag_transpose = False
            else: 
                V = np.transpose(vectors)
                Flag_transpose = True
        else: 
            V = vectors.copy()
            Flag_transpose = False
    elif np.shape(vectors)[1] == 3: 
        V = np.transpose(vectors)
        Flag_transpose = True
    else: 
        print(">>>>>> 'vectors' should be a (3, n) or (n, 3) array. ")
        input('>>>>>> Press any key to quit... ')
        return
    M = np.matrix(matrix)
    
    output = np.matmul(M, V)
    if Flag_transpose: 
        output = np.transpose(np.asarray(output))
    else: 
        output = np.asarray(output)
    
    return np.squeeze(output)


def Decomposing_Rot_Matrix(Matrix, OutputRadian=True): 
    ''' >>> Instruction <<< 
        This function decomposes an arbitrary rotation matrix into the three Euler angles. 

        Input: 
            'Matrix'     Rotation matrix, 3-by-3 array

        Output: 
            'Ang_X'      Rotation angle around the X axis, in rad. 
            'Ang_Y'      Rotation angle around the Y axis, in rad. 
            'Ang_Z'      Rotation angle around the Z axis, in rad.

        Matrix = Rot_Z * Rot_Y * Rot_X

        The direction of the rotation is defined by the righthand rule. 
    '''
    M = np.matrix(Matrix)
    
    Ang_X = np.arctan2(M[2,1], M[2, 2])
    Ang_Y = np.arctan2(-M[2, 0], np.sqrt(M[2, 1]**2 + M[2, 2]**2))
    Ang_Z = np.arctan2(M[1, 0], M[0, 0])
    
    if not OutputRadian: 
        Ang_X = np.rad2deg(Ang_X)
        Ang_Y = np.rad2deg(Ang_Y)
        Ang_Z = np.rad2deg(Ang_Z)
    
    return (Ang_X, Ang_Y, Ang_Z)


def Decomposing_Vector(Vector, dv1=[1,0,0], dv2=[0,1,0], dv3=[0,0,1]): 
    ''' >>> Instruction <<<
        This function dscomposes an arbitrary vector into three arbitrary bases. 
        
        Input: 
            Vector        (3,n) array, each (3,1) is a vector that needs to be decomposed.
            dv1/dv2/dv3   Each is a (3,) array. The three bases. 
            
        Output: 
            Idx           (3,n) array, each (3,1) is the coordinates of the vector in bases.
            
        Math: 
            See https://math.stackexchange.com/questions/148199/equation-for-non-orthogonal
            -projection-of-a-point-onto-two-vectors-representing
            
            Let M = [dv1, dv2, dv3], then P = inv( M' * M ) * M'. 
            There is 
                        Y = [a, b, c] = P * Vector, 
                    where 
                        Vector = a * dv1 + b * dv2 + c * dv3
    '''
    # Input Regularization
    if np.size(np.shape(Vector)) > 2: 
        print(">>>>>> Vector's dimension is wrong! Vector should be a (3, n) array. ")
        input('>>>>>> Press any key to quit... ')
        return
    if np.shape(Vector)[0] == 3: 
        V = Vector.copy()
    elif np.shape(Vector)[1] == 3: 
        V = np.transpose(Vector)
    else: 
        print(">>>>>> Vector's size is wrong! Vector should be a (3, n) array. ")
        input('>>>>>> Press any key to quit... ')
        return
    if np.size(dv1) != 3 or np.size(dv2) != 3 or np.size(dv3) != 3: 
        print(">>>>>> dv1/dv2/dv3 should be (3,) array. ")
        input('>>>>>> Press any key to quit... ')
        return
    if np.size(np.shape(dv1)) > 1: 
        dv1 = np.ravel(dv1)
    if np.size(np.shape(dv2)) > 1: 
        dv2 = np.ravel(dv2)
    if np.size(np.shape(dv3)) > 1: 
        dv3 = np.ravel(dv3)
        
    # Decomposition
    M = np.matrix([dv1, dv2, dv3])
    P = np.linalg.inv( np.transpose(M) * M) * np.transpose(M)
    Idx = np.asarray(P * Vector)
    
    if not np.shape(Vector) == np.shape(V): 
        Idx = np.transpose(Idx)
    
    return Idx


def Vector_to_DeltaGamma(Vector, OutputRadian=False): 
    ''' >>> Instruction <<< 
        This function calculate a vector's corresponding delta and gamma angles
        where   
                delta is the angle between vector's XZ projection and Z axis
                gamma is the angle between vector and XZ plane
    '''
    if np.size(Vector) != 3: 
        print(">>>>>> Vector's dimension is wrong! ")
        input('>>>>>> Press any key to quit ... ')
        return
        
    Vector = np.asarray(Vector).reshape((3,))
    
    Vector = Vector / np.linalg.norm(Vector)
    
    Gamma = np.arcsin(Vector[1])
    Delta = np.arctan2(Vector[0], Vector[2])
    
    if OutputRadian: 
        return [Delta, Gamma]
    else: 
        return [np.rad2deg(Delta), np.rad2deg(Gamma)]


def DeltaGamma_to_Vector(delta=0, gamma=0, IsRadian=False): 
    ''' >>> Instruction <<< 
        This function calculate the vector corresponding to certain delta and gamma angles
        where   
                delta is the angle between vector's XZ projection and Z axis
                gamma is the angle between vector and XZ plane
    '''
    if not IsRadian: 
        delta = np.deg2rad(delta)
        gamma = np.deg2rad(gamma)
    
    return Rotate_vectors(Rotate_vectors([0,0,1], 
                                         Rot_Matrix([1,0,0], -gamma, IsRadian=True)), 
                          Rot_Matrix([0,1,0], delta, IsRadian=True))


def Included_Angle(Vector1, Vector2, RotAxis=None, OutputRadian=True): 
    ''' >>> Instruction <<< 
        Calculate the included angle between two arbitrary vectors 
        in Cartesian coordinates. 
        Fuction Rot_Matrix(axis, theta) gives the rotation matix 
        that can rotate Vector1 to Vector2. 
        The resultant 'axis' and 'theta' follow right-hand rule that 
            Vector1 x Vector2 => 'axis', where 'theta' < pi

        Vector1, Vector2      format np.asarray([x, y, z])

        Output: 
            'axis'            Rotation axis
            'theta'           Angle between Vector1 and Vector2

        If RotAxis is provided, the output will be in [0, 2*pi], otherwise [0, pi]
    '''
    Cross = np.cross(Vector1, Vector2)
    Dot = np.dot(Vector1, Vector2)
    
    if np.linalg.norm(Cross) <= 1e-6: 
        Axis = [0, 0, 0]
    else: 
        Axis = Cross / np.linalg.norm(Cross)
    
    if Dot/np.linalg.norm(Vector1)/np.linalg.norm(Vector2) > 1.0: 
        Theta = 0
    else: 
        Theta = np.arccos( Dot/np.linalg.norm(Vector1)/np.linalg.norm(Vector2) )
    
    if RotAxis is not None: 
        if np.dot(RotAxis, Axis) < 0: 
            Theta = 2*np.pi - Theta
    
    if not OutputRadian: 
        Theta = np.rad2deg(Theta)
    return {'axis': Axis, 'theta': Theta}


def Proj_vector_to_plane(Vector, PlaneNorm): 
    ''' >>> Instruction <<< 
        This function projects a vector to a plane in 3D.
    '''
    vect = np.asarray(Vector)
    norm = np.asarray(PlaneNorm) / np.linalg.norm(PlaneNorm)
    return vect - norm * np.dot(vect, norm)


def DeltaGamma_to_2Theta(Delta, Gamma, IsRadian=False): 
    ''' >>> Instruction <<< 
        This function calculate the 2Theta angle based on Delta and Gamma.
        
        Delta:       Detector angle around Lab Y+ axis
        Gamma:       Detector angle around Lab X- axis
    '''
    R_gamma = Rot_Matrix([1, 0, 0], Gamma, IsRadian=IsRadian)
    R_delta = Rot_Matrix([0, 1, 0], Delta, IsRadian=IsRadian)
    R = np.asarray(np.matmul(R_delta, R_gamma))
    
    Vi = np.asarray([0, 0, 1])
    Vf = np.matmul(R, Vi)
    return Included_Angle(Vf, Vi, OutputRadian=IsRadian)['theta']


def Dist_point_to_line(point, line_p1, line_p2=None, vector=None): 
    ''' >>> Instruction <<< 
        This function calculates the distance from one point to a line, where
        the line is defined by two points (line_p1 and line_p2). 
        
        All three inputs should be (3,) arrays
        a = point[0], b = point[1], c = point[2]
        x1 = line_p1[0], y1 = line_p1[1], z1 = line_p1[2]
        x2 = line_p2[0], y2 = line_p2[1], z2 = line_p2[2]
    '''
    point = np.asarray(point)
    line_p1 = np.asarray(line_p1)
    L1 = point - line_p1
    L1_mag = np.linalg.norm(L1)
    if line_p2 is not None: 
        line_p2 = np.asarray(line_p2)
        L2 = line_p2 - line_p1
        L2_mag = np.linalg.norm(L2)
        t = np.dot(L1, L2) / L2_mag
        # foot of perpendicular
        foot = line_p1 + L2/L2_mag*t
    elif vector is not None: 
        vector = np.asarray(vector)/np.linalg.norm(vector)
        t = np.dot(L1, vector)
        # foot of perpendicular
        foot = line_p1 + vector*t
    # distance from point to the line
    dist = np.sqrt(L1_mag**2 - t**2)
    
    return {'dist': dist, 'foot': foot}


def Inter_plane_and_line(Plane, Point1, Point2):
    ''' >>> Instruction <<< 
        The inputs are the plane function, and two points that define the line. 
        The Plane should be a (4,) array; 
        Each point is a (3,) array. 
        
        The output is the coordinate of the intersection point, a (3,) array. 
    '''
    
    # all three inputs should be (3,) arrays
    Point1 = np.asarray(Point1)
    Point2 = np.asarray(Point2)
    
    # Line equation: 
    # (x - Lx1)/(Lx2 - Lx1) = (y - Ly1)/(Ly2 - Ly1) = (z - Lz1)/(Lz2 - Lz1) = t
    
    # Intersection is: 
    t = - ( np.dot(Plane[:3], Point1) + Plane[3] ) / np.dot( Plane[:3], (Point2 - Point1) )
    
    Inter = Point1 + t * (Point2 - Point1)
    
    return Inter


def Line_func(*args): 
    ''' >>> Instruction <<< 
        This function calculates the function of a line in 2D coordinates. 
        
        The inputs should start with a string 'Points' or 'Angle_and_Point' to define the type of input. 
            For 'Points', the rest inputs should contain two points, each is a (2,) array.
            For 'Angle_and_Point', the rest inputs should be one angle in degree, and one point (2,). 
       
        The line equation is ax + by + c = 0
        The output is an array contains a, b, and c. 
    '''
    if args[0] == 'Points': 
        # Two points are args[1] and args[2]
        a = args[2][1] - args[1][1]
        b = args[1][0] - args[2][0]
        c = args[2][0] * args[1][1] - args[1][0] * args[2][1]
    elif args[0] == 'Angle_and_Point': 
        # Angle is args[1], and the point is args[2]
        a = np.sin(np.deg2rad(args[1]))
        b = -np.cos(np.deg2rad(args[1]))
        c = - a * args[2][0] - b * args[2][1]
    else: 
        print('>>>>>> Unknown input string. <<<<<<')
        return
    Line = np.zeros(3)
    Line[:] = [a, b, c]
    return Line


def Line_in_2D(params, ax0, ax1): 
    ''' >>> Instruction <<< 
        This function draws a line in a 2D map. 
        
        Input:
            params          (3,), parameters of line function ax + by + c = 0
            ax0, ax1        axes of the 2D map     
            
        Output: 
            array           2D array with values 1, 0, and -1, where the line is 0
    '''
    YY, XX = np.meshgrid(ax1, ax0)
    temp = params[0] * XX + params[1] * YY + params[2]
    idx = np.zeros_like(XX)
    if   params[0] < 1e-9: 
        idx[np.arange(len(ax0)), np.argmin(np.abs(temp), axis=1)] = 1
    elif params[1] < 1e-9: 
        idx[np.argmin(np.abs(temp), axis=0), np.arange(len(ax1))] = 1
    elif np.abs(params[1]/params[0]) >= 1: 
        idx[np.arange(len(ax0)), np.argmin(np.abs(temp), axis=1)] = 1
    else: 
        idx[np.argmin(np.abs(temp), axis=0), np.arange(len(ax1))] = 1
    array = temp / np.abs(temp)
    array[np.where(idx==1)] = 0
    return array


def Plane_func(*args): 
    ''' >>> Instruction <<< 
        The inputs should start with a string 'Points', 'Vectors', 
        or 'Normal_and_Point' to define the type of input. 
            For 'Points', the rest inputs should contain three points, 
                each is a (3,) array.
            For 'Vectors', the rest inputs should contain two vectors 
                followed by one point, each is a (3,) array. 
            For 'Normal_and_Point', the rest inputs should be one normal 
                vector followed by one point, each is (3,). 
       
        The plane equation is ax + by + cz + d = 0
        The output is an array contains a, b, c, and d. 
    '''
    # Plane equation: ax + by + cz + d = 0
    if args[0] == 'Points': 
        # Three points are args[1], args[2], and args[3]
        Vector_12 = [ args[2][0]-args[1][0], args[2][1]-args[1][1], args[2][2]-args[1][2] ]
        Vector_13 = [ args[3][0]-args[1][0], args[3][1]-args[1][1], args[3][2]-args[1][2] ]
    
        P_normal = np.cross(Vector_12, Vector_13) 
    
        a = P_normal[0] 
        b = P_normal[1] 
        c = P_normal[2]
        d = - a * args[3][0] - b * args[3][1] - c * args[3][2]
    elif args[0] == 'Vectors': 
        # Two vectors args[1] and args[2], and one point args[3]
        P_normal = np.cross(args[1], args[2]) 
    
        a = P_normal[0] 
        b = P_normal[1] 
        c = P_normal[2]
        d = - a * args[3][0] - b * args[3][1] - c * args[3][2]
    elif args[0] == 'Normal_and_Point': 
        # Normal is args[1], and the point is args[2]
        a = args[1][0]
        b = args[1][1]
        c = args[1][2]
        d = - a * args[2][0] - b * args[2][1] - c * args[2][2]
    else: 
        print('>>>>>> Unknown input string. <<<<<<')
        return
    Plane = np.zeros(4)
    Plane[:] = [a, b, c, d]
    return Plane


def Plane_func_lsq(points, display=False): 
    ''' >>> Instruction <<< 
        This function uses least-squares fitting to determine a plane function from 
        a number of points. 
        
        Input 'points' should be a (3, n) array, considered as n points with each point (3,) 
        
        The plane equation is ax + by + cz + d = 0
        Output is an array contains a, b, c, and d. 
    '''
    # Input Regularization
    if np.size(np.shape(points)) > 2 or np.size(np.shape(points)) == 0: 
        print(">>>>>> Input dimension is", str(np.shape(points)), "<<<<<<") 
        print(">>>>>> Input should be a (3, n) array. <<<<<<")
        return
    if np.size(np.shape(points)) == 1: 
        if np.size(points) == 3: 
            pts = np.asarray(points).reshape(3,1)
        else: 
            print(">>>>>> Input dimension is", str(np.shape(points)), "<<<<<<") 
            print(">>>>>> Input should be a (3, n) array. <<<<<<")
            return
    if np.size(np.shape(points)) == 2: 
        if np.shape(points)[1] == 3 and np.shape(points)[0] != 3:
            pts = np.transpose(points)
        elif np.shape(points)[0] == 3: 
            pts = np.asarray(points)
        else: 
            print(">>>>>> Input dimension is", str(np.shape(points)), "<<<<<<") 
            print(">>>>>> Input should be a (3, n) array. <<<<<<")
            return
    
    # define function for fitting
    def func(x, a, b, c, d): 
        return np.abs(a*x[0]+b*x[1]+c*x[2]+d)/np.sqrt(a**2+b**2+c**2)
    
    # perform lsq fitting
    xdata = tuple(map(tuple, pts))
    ydata = np.zeros(np.shape(pts)[1])
    popt, pcov = curve_fit(func, xdata, ydata)
    
    if display: 
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(pts[0,:], pts[1,:], pts[2,:], 
                     cmap=cm.coolwarm,linewidth=1, marker='.',
                     antialiased=True)
        xx, zz = np.meshgrid(np.arange(np.min(pts[0,:]),np.max(pts[0,:])), 
                             np.arange(np.min(pts[2,:]),np.max(pts[2,:])))
        yy = (-f_params[3] - f_params[0]*xx - f_params[2]*zz)/f_params[1]
        ax.plot_surface(xx, yy, zz, alpha=0.2)
    
    return popt


def Dist_point_to_plane(point, plane): 
    ''' >>> Instruction <<< 
        This function calculates the distance from one point to a plane, where
        the plane is defined by [a, b, c, d] as ax + by + cz + d = 0. 
        
        'point' should be a (3,) array, 
        'plane' should be a (4,) array.
        
        x = point[0], y = point[1], z = point[2]
        a = plane[0], b = plane[1], c = plane[2], d = plane[3]
        
        Formulae: 
            The normal vector of the plane is: 
                n = N/|N| = (a, b, c) / sqrt(a^2 + b^2 + c^2)
            For an arbitrary point P(x, y, z), the distance from P to the plane is: 
                Distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2) 
    '''
    dist = (np.abs(plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3]) 
            / np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2))
    return dist
    

def Dist_point_to_sphere(point, sphere): 
    ''' >>> Instruction <<< 
        This function calculates the distance from one point to a sphere, where
        the sphere is defined by [a, b, c, d] as (x-a)^2 + (y-b)^2 + (z-c)^2 = d^2. 
        
        'point' should be a (3,) array, 
        'sphere' should be a (4,) array.
        
        x = point[0], y = point[1], z = point[2]
        a = sphere[0], b = sphere[1], c = sphere[2], d = sphere[3]
        
        Formulae: 
            For an arbitrary point P(x, y, z), the distance from P to the sphere is: 
                Distance = sqrt[(x-a)^2 + (y-b)^2 + (z-c)^2] - d  
    '''
    dist = np.sqrt( (point[0]-sphere[0])**2 + 
                    (point[1]-sphere[1])**2 + 
                    (point[2]-sphere[2])**2 ) - sphere[3]
    return dist


def Generate_plane(Lx,Ly,Nx,Ny,n,d): 
    ''' >>> Instruction <<< 
        Calculate points of a generic plane 

        Arguments:
        - 'Lx' : Plane Length first direction
        - 'Ly' : Plane Length second direction
        - 'Nx' : Number of points, first direction
        - 'Ny' : Number of points, second direction
        - 'n'  : Plane orientation, normal vector
        - 'd'  : distance from the origin
    '''

    x = np.linspace(-Lx/2,Lx/2,Nx)
    y = np.linspace(-Ly/2,Ly/2,Ny)
    # Create the mesh grid, of a XY plane sitting on the origin
    X,Y = np.meshgrid(x,y)
    Z   = np.zeros([Nx,Ny])
    n0  = np.asarray([0,0,1])

    # Rotate plane to the given normal vector
    if any(n0!=n):
        costheta = np.dot(n0,n) / (np.linalg.norm(n0) * np.linalg.norm(n))
        axis     = np.cross(n0,n) / np.linalg.norm(np.cross(n0,n))
        theta    = np.arccos(costheta)
        rotMatrix = Rot_Matrix(axis,theta)
        XYZ = np.vstack([X.flatten(),Y.flatten(),Z.flatten()])
        X,Y,Z = np.asarray(rotMatrix*XYZ).reshape(3,Nx,Ny)

    dVec = (n / np.linalg.norm(n)) * d
    X,Y,Z = X + dVec[0], Y + dVec[1], Z + dVec[2]
    return X,Y,Z


def Generate_sphere(R=1, C=[0, 0, 0], Nu=20, Nv=10): 
    ''' >>> Instruction <<< 
        Calculate points of a generic sphere 

        Arguments:
        - 'R'  : Radius of the sphere
        - 'C'  : Center position of the sphere
        - 'Nu' : Number of points in angle u of the sphere coordinates
        - 'Nv' : Number of points in angle v of the sphere coordinates
    '''
    # Create the mesh grid, of a sphere sitting on the origin
    u, v = np.mgrid[0 : 2*np.pi : (Nu*1j), 0 : np.pi : (Nv*1j)]
    X = R * np.cos(u) * np.sin(v)
    Y = R * np.sin(u) * np.sin(v)
    Z = R * np.cos(v)
    
    # Move the sphere from origin to the defined center
    X,Y,Z = X + C[0], Y + C[1], Z + C[2]
    return X,Y,Z


def Coordinates_translate_3D(Disp, Input): 
    ''' >>> Instruction <<< 
        Input should be a (3, n) array, considered as n points with each point (3,) 
        The same translation will be applied to each point. 
        
        Disp is a (3,) array that defines the 3D displacement we want to apply. 
        
        Output will be the same size array as the Input. 
    '''
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 3:
        In = Input
    elif np.shape(Input)[1] == 3: 
        In = np.transpose(Input)
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
        
    # Perform translation
    if np.shape(In)[1] == 1 or np.size(np.shape(In)) == 1: 
        Trans = np.empty(3)
        Trans[:] = Disp[:]
        Out = np.empty(np.shape(In))
        Out = In + Trans
    else: 
        Trans = np.empty(np.shape(In))
        Trans[0, :] = Disp[0]
        Trans[1, :] = Disp[1]
        Trans[2, :] = Disp[2]
        Out = np.empty(np.shape(In))
        Out = In + Trans
    
    # Output
    if np.shape(Input) != np.shape(Out): 
        Output = np.transpose(Out)
        return Output
    else: 
        return Out


def Coordinates_rotate_3D(Angle, Axis, Input, IsRadian=False): 
    ''' >>> Instruction <<< 
        Input should be a (3, n) array, considered as n vectors with each vector (3,) 
        The same rotation will be applied to each vector. 
        
        Rotate around X (1st axis) : Axis = 0
        Rotate around Y (2nd axis) : Axis = 1
        Rotate around Z (3rd axis) : Axis = 2
        
        Output will be the same size array as the Input. 
    '''
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 3:
        In = Input
    elif np.shape(Input)[1] == 3: 
        In = np.transpose(Input)
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    
    # Generate rotation matrix
    if not IsRadian: 
        Angle = np.deg2rad(Angle)
    
    ca = np.cos(Angle)
    sa = np.sin(Angle)
    Trans = np.zeros((3,3))
    if Axis == 0: 
        Trans[0, 0] = 1
        Trans[1, 1] = ca
        Trans[2, 1] = sa
        Trans[1, 2] = -sa
        Trans[2, 2] = ca
    elif Axis == 1: 
        Trans[0, 0] = ca
        Trans[2, 0] = -sa
        Trans[1, 1] = 1
        Trans[0, 2] = sa
        Trans[2, 2] = ca
    elif Axis == 2: 
        Trans[0, 0] = ca
        Trans[1, 0] = sa
        Trans[0, 1] = -sa
        Trans[1, 1] = ca
        Trans[2, 2] = 1
    
    # Perform rotation
    Out = np.empty(np.shape(In))
    Out = np.matmul(Trans, In)
    
    # Output
    if np.shape(Input) != np.shape(Out): 
        Output = np.transpose(Out)
        return Output
    else: 
        return Out


def Frame_transform_matrix(Origin, Normal): 
    ''' >>> Instruction <<< 
        This function determines the transform matrix from the lab frame to the detector frame. 
        Origin: (3,) array, the origin of the detector frame, i.e. the center of the detector chip; 
        Normal: (3,) array, normal of the detector.
        
        The lab frame is defined as: 
        Origin: O[0, 0, 0], 
        Axes:   A[1, 0, 0], B[0, 1, 0], C[0, 0, 1], 
        Frame:  [A, B, C, O]
        
        The detector frame is defined as: 
        Origin: P[tx, ty, tz], 
        Axes:   X[x_x, x_y, x_z], Y[y_x, y_y, y_z], Z[z_x, z_y, z_z], 
                where Z is the same as the Normal; 
                      X is the intersection of Detector plane and Horizontal plane;
                      Y is Z cross X. 
        Frame:  [X, Y, Z, P]
              
        Output is a dictionary containing four transformation matrices, all are (4, 4) arrays.
        The labels are: 'TransFrame', 'InvTransFrame', 'TransCoord', 'InvTransCoord'.
        
        'M_frame' and 'M_frame_inverse' are the frame transform and inverse transform matrices.
        'M_coord' and 'M_coord_inverse' are the coordinates transform and inverse transform matrices. 
        Coordinates transform matrices are the transposes of frame matrices, respectively.  
        Matrix: M_frame[[m11, m12, m13, 0],
                        [m21, m22, m23, 0],
                        [m31, m32, m33, 0],
                        [m41, m42, m43, 1]]
                M_frame * [A, B, C, O]' = [X, Y, Z, P]' 
    '''
    # Lab frame
    LabX = np.asarray([1, 0, 0])
    LabY = np.asarray([0, 1, 0])
    LabZ = np.asarray([0, 0, 1])
    LabO = np.asarray([0, 0, 0])
    
    # Detector frame
    DetX = np.zeros(3)
    DetY = np.zeros(3)
    DetZ = np.asarray(Normal)
    if np.linalg.norm(DetZ) != 0: 
        DetZ = DetZ / np.linalg.norm(DetZ)   # Make sure the magnitude of the vector is 1
    DetP = np.asarray(Origin)
    
    # Determine DetX
    DetX = np.cross(LabY, DetZ)   # Normal of horizontal plane is LabY
    if np.linalg.norm(DetX) != 0:
        DetX = DetX / np.linalg.norm(DetX)
    # # >>> Rotate Intsec by Phi around DetZ to get DetX
    # Intsec_orth = np.cross(DetZ, Intsec)   # The vector that is orthogonal to both Intsec and DetZ
    # DetX = np.cos(Phi) * Intsec / np.linalg.norm(Intsec) + np.sin(Phi) * Intsec_orth / np.linalg.norm(Intsec_orth)
    # DetX = DetX / np.linalg.norm(DetX)   # Make sure the magnitude of the vector is 1
    
    # Determine DetY
    DetY = np.cross(DetZ, DetX)
    if np.linalg.norm(DetY) != 0: 
        DetY = DetY / np.linalg.norm(DetY)   # Make sure the magnitude of the vector is 1
    
    # Calculate the frame transform matrix
    m11 = np.dot(LabX, DetX)
    m12 = np.dot(LabX, DetY)
    m13 = np.dot(LabX, DetZ)
    m14 = 0
    
    m21 = np.dot(LabY, DetX)
    m22 = np.dot(LabY, DetY)
    m23 = np.dot(LabY, DetZ)
    m24 = 0
    
    m31 = np.dot(LabZ, DetX)
    m32 = np.dot(LabZ, DetY)
    m33 = np.dot(LabZ, DetZ)
    m34 = 0
    
    m41 = DetP[0] - LabO[0]
    m42 = DetP[1] - LabO[1]
    m43 = DetP[2] - LabO[2]
    m44 = 1
    
    M_frame = np.asarray([[m11, m12, m13, m14], 
                          [m21, m22, m23, m24], 
                          [m31, m32, m33, m34], 
                          [m41, m42, m43, m44]])
    # Calculate the frame inverse transform matrix
    m11 = np.dot(DetX, LabX)
    m12 = np.dot(DetX, LabY)
    m13 = np.dot(DetX, LabZ)
    m14 = 0
    
    m21 = np.dot(DetY, LabX)
    m22 = np.dot(DetY, LabY)
    m23 = np.dot(DetY, LabZ)
    m24 = 0
    
    m31 = np.dot(DetZ, LabX)
    m32 = np.dot(DetZ, LabY)
    m33 = np.dot(DetZ, LabZ)
    m34 = 0
    
    m41 = LabO[0] - DetP[0]
    m42 = LabO[1] - DetP[1]
    m43 = LabO[2] - DetP[2]
    m44 = 1
    
    M_frame_inverse = np.asarray([[m11, m12, m13, m14], 
                                  [m21, m22, m23, m24], 
                                  [m31, m32, m33, m34], 
                                  [m41, m42, m43, m44]])
    # Calculate the coordinates transform matrices
    M_coord = np.transpose(M_frame_inverse)
    M_coord_inverse = np.transpose(M_frame)
    
    return {'TransFrame': M_frame, 'InvTransFrame': M_frame_inverse, 
            'TransCoord': M_coord, 'InvTransCoord': M_coord_inverse}


def Cartesian2Polar(Input, OutputRadian=True): 
    ''' >>> Instruction <<< 
        This function transform the Cartesian coordinates of a point to Polar coordinates. 
        Input should be a (2, n) array, representing n of (2,) points. 
        
        Output will be the same size array as the Input. 
        
        The matrix can be found at: 
        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
        
        Definitions: 
        
        In the Cartesian coordinates, (2,) stands for (Axis1, Axis2)
        
        In the Polar coordinates, (2,) stands for (Radius, Angle)
            where: 
            Radius is sqrt( Axis1^2 + Axis2^2 )
            Angle  is the angle from Axis1 to the projection of vector on Axis1-Axis2 plane.
    '''
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 2 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 2:
        In = Input.copy()
    elif np.shape(Input)[1] == 2: 
        In = np.transpose(Input.copy())
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 2 by n array. <<<<<<")
        return
    
    if np.size(np.shape(Input)) == 2: 
        Out = np.empty(np.shape(In))
        Out[0, :] = np.sqrt(In[0, :]**2 + In[1, :]**2)
        Out[1, :] = np.arctan2(In[1, :], In[0, :])
    else: 
        Out = np.empty(np.shape(In))
        Out[0] = np.sqrt(In[0]**2 + In[1]**2)
        Out[1] = np.arctan2(In[1], In[0])
    
    if not OutputRadian: 
        Out[1] = np.rad2deg(Out[1])
    
    # Output
    if np.shape(Input) != np.shape(Out): 
        Output = np.transpose(Out)
        return Output
    else: 
        return Out


def Polar2Cartesian(Input, IsRadian=True): 
    ''' >>> Instruction <<< 
        This function transform the Polar coordinates of a point to Cartesian coordinates. 
        Input should be a (2, n) array, representing n of (2,) points. 
        
        Output will be the same size array as the Input. 
        
        The matrix can be found at: 
        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
        
        Definitions: 
        
        In the Cartesian coordinates, (2,) stands for (Axis1, Axis2)
        
        In the Cylindrical coordinates, (2,) stands for (Radius, Angle)
            where: 
            Radius is sqrt( Axis1^2 + Axis2^2 )
            Angle  is the angle from Axis1 to the projection of vector on Axis1-Axis2 plane.
    '''
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 2 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 2:
        In = Input.copy()
    elif np.shape(Input)[1] == 2: 
        In = np.transpose(Input.copy())
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 2 by n array. <<<<<<")
        return
    
    if not IsRadian: 
        In[1] = np.deg2rad(In[1])
    
    if np.size(np.shape(Input)) == 2: 
        Out = np.empty(np.shape(In))
        Out[0, :] = In[0, :] * np.cos(In[1, :])
        Out[1, :] = In[0, :] * np.sin(In[1, :])
    else: 
        Out = np.empty(np.shape(In))
        Out[0] = In[0] * np.cos(In[1])
        Out[1] = In[0] * np.sin(In[1])
    
    # Output
    if np.shape(Input) != np.shape(Out): 
        Output = np.transpose(Out)
        return Output
    else: 
        return Out


def Cartesian2Cylindrical(Input, OutputRadian=True): 
    ''' >>> Instruction <<< 
        This function transform the Cartesian coordinates of a point to Cylindrical coordinates. 
        Input should be a (3, n) array, representing n of (3,) points. 
        
        Output will be the same size array as the Input. 
        
        The matrix can be found at: 
        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
        
        Definitions: 
        
        In the Cartesian coordinates, (3,) stands for (Axis1, Axis2, Axis3)
        
        In the Cylindrical coordinates, (3,) stands for (Radius, Angle, Height)
            where: 
            Radius is sqrt( Axis1^2 + Axis2^2 )
            Angle  is the angle from Axis1 to the projection of vector on Axis1-Axis2 plane.
            Height is Axis3
    '''
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 3:
        In = Input.copy()
    elif np.shape(Input)[1] == 3: 
        In = np.transpose(Input.copy())
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    
    if np.size(np.shape(Input)) == 2: 
        Out = np.empty(np.shape(In))
        Out[0, :] = np.sqrt(In[0, :]**2 + In[1, :]**2)
        Out[1, :] = np.arctan2(In[1, :], In[0, :])
        Out[2, :] = In[2, :]
    else: 
        Out = np.empty(np.shape(In))
        Out[0] = np.sqrt(In[0]**2 + In[1]**2)
        Out[1] = np.arctan2(In[1], In[0])
        Out[2] = In[2]
    
    if not OutputRadian: 
        Out[1] = np.rad2deg(Out[1])
    
    # Output
    if np.shape(Input) != np.shape(Out): 
        Output = np.transpose(Out)
        return Output
    else: 
        return Out


def Cylindrical2Cartesian(Input, IsRadian=True): 
    ''' >>> Instruction <<< 
        This function transform the Cylindrical coordinates of a point to Cartesian coordinates. 
        Input should be a (3, n) array, representing n of (3,) points. 
        
        Output will be the same size array as the Input. 
        
        The matrix can be found at: 
        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
        
        Definitions: 
        
        In the Cartesian coordinates, (3,) stands for (Axis1, Axis2, Axis3)
        
        In the Cylindrical coordinates, (3,) stands for (Radius, Angle, Height)
            where: 
            Radius is sqrt( Axis1^2 + Axis2^2 )
            Angle  is the angle from Axis1 to the projection of vector on Axis1-Axis2 plane.
            Height is Axis3
    '''
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 3:
        In = Input.copy()
    elif np.shape(Input)[1] == 3: 
        In = np.transpose(Input.copy())
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    
    if not IsRadian: 
        In[1] = np.deg2rad(In[1])
    
    if np.size(np.shape(Input)) == 2: 
        Out = np.empty(np.shape(In))
        Out[0, :] = In[0, :] * np.cos(In[1, :])
        Out[1, :] = In[0, :] * np.sin(In[1, :])
        Out[2, :] = In[2, :]
    else: 
        Out = np.empty(np.shape(In))
        Out[0] = In[0] * np.cos(In[1])
        Out[1] = In[0] * np.sin(In[1])
        Out[2] = In[2]
    
    # Output
    if np.shape(Input) != np.shape(Out): 
        Output = np.transpose(Out)
        return Output
    else: 
        return Out


def Cartesian2Spherical(Input, OutputRadian=True): 
    ''' >>> Instruction <<< 
        This function transform the Cartesian coordinates of a point to Spherical coordinates. 
        Input should be a (3, n) array, representing n of (3,) points. 
        
        Output will be the same size array as the Input. 
        
        The matrix can be found at: 
        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
        
        Definitions: 
        
        In the Cartesian coordinates, (3,) stands for (Axis1, Axis2, Axis3)
        
        In the Spherical coordinates, (3,) stands for (Radius, Angle1, Angle2)
            where: 
            Angle1 is the angle from Axis3 to the vector
            Angle2 is the angle from Axis1 to the projection of vector on Axis1-Axis2 plane.
    '''
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 3:
        In = Input.copy()
    elif np.shape(Input)[1] == 3: 
        In = np.transpose(Input.copy())
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    
    # Convertion
    if np.size(np.shape(Input)) == 2: 
        Out = np.empty(np.shape(In))
        xy = np.sqrt(In[0, :]**2 + In[1, :]**2)
        Out[0, :] = np.sqrt(xy[:]**2 + In[2, :]**2)
        Out[1, :] = np.arctan2(xy[:], In[2, :])
        Out[2, :] = np.arctan2(In[1, :], In[0, :])
    else: 
        Out = np.empty(np.shape(In))
        xy = np.sqrt(In[0]**2 + In[1]**2)
        Out[0] = np.sqrt(xy**2 + In[2]**2)
        Out[1] = np.arctan2(xy, In[2])
        Out[2] = np.arctan2(In[1], In[0])
    
    if not OutputRadian: 
        Out[1] = np.rad2deg(Out[1])
        Out[2] = np.rad2deg(Out[2])
    
    # Output
    if np.shape(Input) != np.shape(Out): 
        Output = np.transpose(Out)
        return Output
    else: 
        return Out


def Spherical2Cartesian(Input, IsRadian=True): 
    ''' >>> Instruction <<< 
        This function transform the Spherical coordinates of a point to Cartesian coordinates. 
        Input should be a (3, n) array, representing n of (3,) points. 
        
        Output will be the same size array as the Input. 
        
        The matrix can be found at: 
        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
        
        Definitions: 
        
        In the Cartesian coordinates, (3,) stands for (Axis1, Axis2, Axis3)
        
        In the Spherical coordinates, (3,) stands for (Radius, Angle1, Angle2)
            where: 
            Angle1 is the angle from Axis3 to the vector
            Angle2 is the angle from Axis1 to the projection of vector on Axis1-Axis2 plane.
    '''
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 3:
        In = Input.copy()
    elif np.shape(Input)[1] == 3: 
        In = np.transpose(Input.copy())
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    
    if not IsRadian: 
        In[1] = np.deg2rad(In[1])
        In[2] = np.deg2rad(In[2])
    
    # Convertion
    if np.size(np.shape(Input)) == 2: 
        Out = np.empty(np.shape(In))
        Out[0, :] = In[0, :] * np.sin(In[1, :]) * np.cos(In[2, :])
        Out[1, :] = In[0, :] * np.sin(In[1, :]) * np.sin(In[2, :])
        Out[2, :] = In[0, :] * np.cos(In[1, :])
    else: 
        Out = np.empty(np.shape(In))
        Out[0] = In[0] * np.sin(In[1]) * np.cos(In[2])
        Out[1] = In[0] * np.sin(In[1]) * np.sin(In[2])
        Out[2] = In[0] * np.cos(In[1])
    
    # Output
    if np.shape(Input) != np.shape(Out): 
        Output = np.transpose(Out)
        return Output
    else: 
        return Out


def Base_transform_matrix_2D(ori_ax1=[0,1], ori_ax2=[1,0], 
                             tar_ax1=[1,0], tar_ax2=[0,1],
                             normalize_bases=False, 
                             ob_to_tb=False): 
    ''' >>> Introduction <<< 
        This function generate coordinate transformation matrix based on origian basis 
        and targeted basis. 
        
        This matrix will be used in interpolation function sp.ndimage.affine_transform().
        For this case, ob_to_tb should be FALSE. 
        
        To display the result in ParaView, notice that ori_ax1/2(tar_ax1/2) should be y/x.
        
        Inputs: 
            ori_ax1/2        Vector 1/2 of the Original Basis. 
            tar_ax1/2        Vector 1/2 of the Targeted Basis. 
            normalize_bases  whether normalize the input bases. 
            ob_to_tb         True:  output ob_to_tb matrix
                             False: output tb_to_ob matrix
        Output: 
            matrix
    '''
    if normalize_bases: 
        ori_ax1 = ori_ax1 / np.linalg.norm(ori_ax1)
        ori_ax2 = ori_ax2 / np.linalg.norm(ori_ax2)
        tar_ax1 = tar_ax1 / np.linalg.norm(tar_ax1)
        tar_ax2 = tar_ax2 / np.linalg.norm(tar_ax2)
    
    # Change-of-basis matrix
    mat_ob = np.asarray([ori_ax1, ori_ax2]).transpose()
    mat_tb = np.asarray([tar_ax1, tar_ax2]).transpose()
    mat_ob_to_tb = np.matmul(np.linalg.inv(mat_tb), mat_ob)
    mat_tb_to_ob = np.matmul(np.linalg.inv(mat_ob), mat_tb)
    if ob_to_tb: 
        matrix = mat_ob_to_tb
    else: 
        matrix = mat_tb_to_ob
    return matrix


def Base_transform_matrix_3D(ori_ax1=[0,0,1], ori_ax2=[0,1,0], ori_ax3=[1,0,0], 
                             tar_ax1=[0,1,0], tar_ax2=[0,0,1], tar_ax3=[1,0,0],
                             normalize_bases=False, 
                             ob_to_tb=False): 
    ''' >>> Introduction <<< 
        This function generate coordinate transformation matrix based on origian basis 
        and targeted basis. 
        
        This matrix will be used in interpolation function sp.ndimage.affine_transform().
        For this case, ob_to_tb should be FALSE. 
        
        To display the result in ParaView, notice that ori_ax1/2/3(tar_ax1/2/3) should be z/y/x.
        
        Inputs: 
            ori_ax1/2/3      Vector 1/2/3 of the Original Basis. 
            tar_ax1/2/3      Vector 1/2/3 of the Targeted Basis. 
            normalize_bases  whether normalize the input bases. 
            ob_to_tb         True:  output ob_to_tb matrix
                             False: output tb_to_ob matrix
        Output: 
            matrix
    '''
    if normalize_bases: 
        ori_ax1 = ori_ax1 / np.linalg.norm(ori_ax1)
        ori_ax2 = ori_ax2 / np.linalg.norm(ori_ax2)
        ori_ax3 = ori_ax3 / np.linalg.norm(ori_ax3)
        tar_ax1 = tar_ax1 / np.linalg.norm(tar_ax1)
        tar_ax2 = tar_ax2 / np.linalg.norm(tar_ax2)
        tar_ax3 = tar_ax3 / np.linalg.norm(tar_ax3)
    
    # Change-of-basis matrix
    mat_ob = np.asarray([ori_ax1, ori_ax2, ori_ax3]).transpose()
    mat_tb = np.asarray([tar_ax1, tar_ax2, tar_ax3]).transpose()
    mat_ob_to_tb = np.matmul(np.linalg.inv(mat_tb), mat_ob)
    mat_tb_to_ob = np.matmul(np.linalg.inv(mat_ob), mat_tb)
    if ob_to_tb: 
        matrix = mat_ob_to_tb
    else: 
        matrix = mat_tb_to_ob
    return matrix


def Base_transform_scipyaffine(array, normalize_bases=False, order=3, 
                               ori_ax1=None, ori_ax2=None, ori_ax3=None, 
                               tar_ax1=None, tar_ax2=None, tar_ax3=None): 
    ''' >>> Introduction <<< 
        This function transforms an array from one grid to another, 
        with different bases. 
        
        To display the result in ParaView, 
            For 2D, notice that ori_ax1/2(tar_ax1/2) should be y/x.
            For 3D, notice that ori_ax1/2/3(tar_ax1/2/3) should be z/y/x.
        
        Inputs: 
            array            original array, 2D or 3D 
            normalize_bases  whether normalize the input bases. 
            order            the order of the spline interpolation for affine. 
            ori_ax1/2/3      vector 1/2/3 of the original basis. 
            tar_ax1/2/3      vector 1/2/3 of the targeted basis. 
            
        Output: 
            new_array        interpolated array
    '''
    # Input regularization
    shape = np.asarray(np.shape(array))   # shape of input array
    dims = np.size(shape)
    new_shape = shape    # shape of output array
    
    ob_1 = np.asarray(ori_ax1).reshape(-1)
    if np.size(ob_1) != dims and np.shape(ob_1)[0] != dims: 
        print('>>>>>> Error! Check the shape of ori_ax1 . ')
        input('>>>>>> Press any key to quit ... ')
        return
    ob_2 = np.asarray(ori_ax2).reshape(-1)
    if np.size(ob_2) != dims and np.shape(ob_2)[0] != dims: 
        print('>>>>>> Error! Check the shape of ori_ax2 . ')
        input('>>>>>> Press any key to quit ... ')
        return
    tb_1 = np.asarray(tar_ax1).reshape(-1)
    if np.size(tb_1) != dims and np.shape(tb_1)[0] != dims: 
        print('>>>>>> Error! Check the shape of tar_ax1 . ')
        input('>>>>>> Press any key to quit ... ')
        return
    tb_2 = np.asarray(tar_ax2).reshape(-1)
    if np.size(tb_2) != dims and np.shape(tb_2)[0] != dims: 
        print('>>>>>> Error! Check the shape of tar_ax2 . ')
        input('>>>>>> Press any key to quit ... ')
        return
    if dims == 3: 
        ob_3 = np.asarray(ori_ax3).reshape(-1)
        if np.size(ob_3) != dims and np.shape(ob_3)[0] != dims: 
            print('>>>>>> Error! Check the shape of ori_ax3 . ')
            input('>>>>>> Press any key to quit ... ')
            return
        tb_3 = np.asarray(tar_ax3).reshape(-1)
        if np.size(tb_3) != dims and np.shape(tb_3)[0] != dims: 
            print('>>>>>> Error! Check the shape of tar_ax3 . ')
            input('>>>>>> Press any key to quit ... ')
            return
    if normalize_bases: 
        ob_1 = ob_1 / np.linalg.norm(ob_1)
        ob_2 = ob_2 / np.linalg.norm(ob_2)
        tb_1 = tb_1 / np.linalg.norm(tb_1)
        tb_2 = tb_2 / np.linalg.norm(tb_2)
        if dims == 3: 
            ob_3 = ob_3 / np.linalg.norm(ob_3)
            tb_3 = tb_3 / np.linalg.norm(tb_3)
    # matrix for transformation 
    if   dims == 2: 
        matrix = Base_transform_matrix_2D(ori_ax1=ob_1, ori_ax2=ob_2, 
                                          tar_ax1=tb_1, tar_ax2=tb_2,
                                          normalize_bases=False, ob_to_tb=False)
    elif dims == 3: 
        matrix = Base_transform_matrix_3D(ori_ax1=ob_1, ori_ax2=ob_2, ori_ax3=ob_3, 
                                          tar_ax1=tb_1, tar_ax2=tb_2, tar_ax3=tb_3,
                                          normalize_bases=False, ob_to_tb=False)
    else: 
        print('>>>>>> Error! Not a 2D or 3D array . ')
        input('>>>>>> Press any key to quit ... ')
        return
    # conduct transformation
    offset = 0.5 * new_shape - np.dot(matrix, 0.5 * shape)
    new_array = sp.ndimage.affine_transform(array, matrix, offset=offset, order=order)
    return new_array


def Base_transform_2D(coords, ob_to_tb=True, 
                      regularize_inputs=True, normalize_bases=False, 
                      ob_1=[0,0,1], ob_2=[0,1,0], 
                      tb_1=[0,1,0], tb_2=[0,0,1]): 
    ''' >>> Introduction <<< 
        This function transforms COORDINATES from one 2D grid to another, 
        with different bases. 
        
        Inputs: 
            coords      (2,n) array. Coordinates on one basis. 
            ob_to_tb    If True, convert coords from ob basis to tb basis; 
                        If False, convert coords from tb basis to ob basis.
            ob_1/2      Vector 1/2 of the Original Basis. 
            tb_1/2      Vector 1/2 of the Targeted Basis. 
            
        Output: 
            new_coords  Converted coords. 
    '''
    # Input regularization
    if regularize_inputs: 
        coords = np.asarray(coords)
        if np.size(np.shape(coords)) != 2: 
            print('>>>>>> Error! Input coords should be 2D array. ')
            input('>>>>>> Press any key to quit ... ')
            return
        if np.size(coords) != 4 and np.shape(coords)[1] == 2: 
            input_coords = coords.transpose()
            # print('>>>>>> Input coords is transposed to [%d, %d]. ' 
            #       %(np.shape(input_coords)[0], np.shape(input_coords)[1]))
            input_transposed = True
        else: 
            input_coords = coords
            input_transposed = False

        ob_1 = np.asarray(ob_1).reshape(-1)
        ob_2 = np.asarray(ob_2).reshape(-1)
        if np.size(ob_1) != 2 and np.shape(ob_1)[0] != 2: 
            print('>>>>>> Error! Check the shape of ob_1 . ')
            input('>>>>>> Press any key to quit ... ')
            return
        if np.size(ob_2) != 2 and np.shape(ob_2)[0] != 2: 
            print('>>>>>> Error! Check the shape of ob_2 . ')
            input('>>>>>> Press any key to quit ... ')
            return
        
        tb_1 = np.asarray(tb_1).reshape(-1)
        tb_2 = np.asarray(tb_2).reshape(-1)
        if np.size(tb_1) != 2 and np.shape(tb_1)[0] != 2: 
            print('>>>>>> Error! Check the shape of tb_1 . ')
            input('>>>>>> Press any key to quit ... ')
            return
        if np.size(tb_2) != 2 and np.shape(tb_2)[0] != 2: 
            print('>>>>>> Error! Check the shape of tb_2 . ')
            input('>>>>>> Press any key to quit ... ')
            return
    
    if normalize_bases: 
        ob_1 = ob_1 / np.linalg.norm(ob_1)
        ob_2 = ob_2 / np.linalg.norm(ob_2)
        tb_1 = tb_1 / np.linalg.norm(tb_1)
        tb_2 = tb_2 / np.linalg.norm(tb_2)
    
    # Change-of-basis matrix
    mat_ob = np.asarray([ob_1, ob_2]).transpose()
    mat_tb = np.asarray([tb_1, tb_2]).transpose()
    mat_ob_to_tb = np.matmul(np.linalg.inv(mat_tb), mat_ob)
    mat_tb_to_ob = np.matmul(np.linalg.inv(mat_ob), mat_tb)
    
    # Convert the coords
    if ob_to_tb: 
        new_coords = np.asarray(np.matmul(mat_ob_to_tb, input_coords))
    else: 
        new_coords = np.asarray(np.matmul(mat_tb_to_ob, input_coords))
    
    if input_transposed: 
        new_coords = new_coords.transpose()
    
    return {'mat_ob':mat_ob, 
            'mat_tb':mat_tb, 
            'mat_ob_to_tb':mat_ob_to_tb, 
            'mat_tb_to_ob':mat_tb_to_ob,
            'new_coords':new_coords}


def Base_transform_3D(coords, ob_to_tb=True, 
                      regularize_inputs=True, normalize_bases=False, 
                      ob_1=[0,0,1], ob_2=[0,1,0], ob_3=[1,0,0], 
                      tb_1=[0,1,0], tb_2=[0,0,1], tb_3=[1,0,0]): 
    ''' >>> Introduction <<< 
        This function transforms COORDINATES from one grid to another, 
        with different bases. 
        
        To display the result in ParaView, notice that ob_1/2/3(tb_1/2/3) are z/y/x.
        
        Inputs: 
            coords      (3,n) array. Coordinates on one basis. 
            ob_to_tb    If True, convert coords from ob basis to tb basis; 
                        If False, convert coords from tb basis to ob basis.
            ob_1/2/3    Vector 1/2/3 of the Original Basis. 
            tb_1/2/3    Vector 1/2/3 of the Targeted Basis. 
            
        Output: 
            new_coords  Converted coords. 
    '''
    # Input regularization
    if regularize_inputs: 
        coords = np.asarray(coords)
        if np.size(np.shape(coords)) != 2: 
            print('>>>>>> Error! Input coords should be 2D array. ')
            input('>>>>>> Press any key to quit ... ')
            return
        if np.size(coords) != 9 and np.shape(coords)[1] == 3: 
            input_coords = coords.transpose()
            # print('>>>>>> Input coords is transposed to [%d, %d]. ' 
            #       %(np.shape(input_coords)[0], np.shape(input_coords)[1]))
            input_transposed = True
        else: 
            input_coords = coords
            input_transposed = False

        ob_1 = np.asarray(ob_1).reshape(-1)
        ob_2 = np.asarray(ob_2).reshape(-1)
        ob_3 = np.asarray(ob_3).reshape(-1)
        if np.size(ob_1) != 3 and np.shape(ob_1)[0] != 3: 
            print('>>>>>> Error! Check the shape of ob_1 . ')
            input('>>>>>> Press any key to quit ... ')
            return
        if np.size(ob_2) != 3 and np.shape(ob_2)[0] != 3: 
            print('>>>>>> Error! Check the shape of ob_2 . ')
            input('>>>>>> Press any key to quit ... ')
            return
        if np.size(ob_3) != 3 and np.shape(ob_3)[0] != 3: 
            print('>>>>>> Error! Check the shape of ob_3 . ')
            input('>>>>>> Press any key to quit ... ')
            return

        tb_1 = np.asarray(tb_1).reshape(-1)
        tb_2 = np.asarray(tb_2).reshape(-1)
        tb_3 = np.asarray(tb_3).reshape(-1)
        if np.size(tb_1) != 3 and np.shape(tb_1)[0] != 3: 
            print('>>>>>> Error! Check the shape of tb_1 . ')
            input('>>>>>> Press any key to quit ... ')
            return
        if np.size(tb_2) != 3 and np.shape(tb_2)[0] != 3: 
            print('>>>>>> Error! Check the shape of tb_2 . ')
            input('>>>>>> Press any key to quit ... ')
            return
        if np.size(tb_3) != 3 and np.shape(tb_3)[0] != 3: 
            print('>>>>>> Error! Check the shape of tb_3 . ')
            input('>>>>>> Press any key to quit ... ')
            return
    
    if normalize_bases: 
        ob_1 = ob_1 / np.linalg.norm(ob_1)
        ob_2 = ob_2 / np.linalg.norm(ob_2)
        ob_3 = ob_3 / np.linalg.norm(ob_3)
        tb_1 = tb_1 / np.linalg.norm(tb_1)
        tb_2 = tb_2 / np.linalg.norm(tb_2)
        tb_3 = tb_3 / np.linalg.norm(tb_3)
    
    # Change-of-basis matrix
    mat_ob = np.asarray([ob_1, ob_2, ob_3]).transpose()
    mat_tb = np.asarray([tb_1, tb_2, tb_3]).transpose()
    mat_ob_to_tb = np.matmul(np.linalg.inv(mat_tb), mat_ob)
    mat_tb_to_ob = np.matmul(np.linalg.inv(mat_ob), mat_tb)
    
    # Convert the coords
    if ob_to_tb: 
        new_coords = np.asarray(np.matmul(mat_ob_to_tb, input_coords))
    else: 
        new_coords = np.asarray(np.matmul(mat_tb_to_ob, input_coords))
    
    if input_transposed: 
        new_coords = new_coords.transpose()
    
    return {'mat_ob':mat_ob, 
            'mat_tb':mat_tb, 
            'mat_ob_to_tb':mat_ob_to_tb, 
            'mat_tb_to_ob':mat_tb_to_ob,
            'new_coords':new_coords}


def Array_threshold(Array, threshold): 
    ''' >>> Introduction <<< 
        This function force pixels that are below threshold to zero. 
    '''
    Array_new = np.where(Array < threshold, 0, Array)
    return Array_new


def Array_size_adjust(array, adj=None, adj_to=None, offset=0, mode='constant', 
                      stat_length=None, constant_values=0, end_values=0, reflect_type=None): 
    ''' >>> Introduction <<< 
        This fuction adjusts the size of a multi-dimensional array. 
        Padding is done by numpy.pad(), see https://numpy.org/doc/stable/reference/generated/numpy.pad.html

        Inputs: 
            array       A numpy.array or list/tuple of rank N (list/tuple will be converted to numpy.array).
            adj         Number of adjustment values to the edges of each axis. 
                        If adj = ((before_1, after_1), ... (before_N, after_N)), each tuple contains adjustments for each axis. 
                        If adj = (before, after) or ((before, after),), it yields same before and after adjustments for each axis. 
                        If adj = (adjustment,) or int, it represents before = after = adjustment for all axes. 
                        Positive/negative adjustment means padding/cropping. 
                
            adj_to      Adjust each axis to the targeted size. 
                        If adj_to = (num_1, ... num_N), it yields targeted size for each axis.
                        If adj_to = (num, ) or int, it represents same size for all axes.
                        Axes shorter than the targeted sizes will be padded. 
                        Axes longer than the targeted sizes will be cropped. 
            
            offset      Used with "adj_to", how to set before/after values based on the targeted size. 
                        If offset = (num_1, ... num_N), it yields offset for each axis.
                        If offset = (num, ) or int, it represents same offset for all axes.
                        When offset = 0, center element of the axis remains the same after adjustment. 
                        When offset > 0, center element of the new axis shifts to "after" direction. 
                        When offset < 0, center element of the new axis shifts to "before" direction. 
                        Default value is 0. 
            
            Parameters for numpy.pad(), see its instruction for detail: 
            
            mode              "constant"(default), 
                              "maximum","mean","median","minimum",
                              "linear_ramp", 
                              "reflect","symmetric",
                              "edge","wrap","empty", <function>
            
            constant_values   Used in ‘constant’. sequence or scalar 
                              The values to set the padded values for each axis.
            
            stat_length       Used in ‘maximum’, ‘mean’, ‘median’, and ‘minimum’. sequence or scalar 
                              Number of values at edge of each axis used to calculate the statistic value.
            
            end_values        Used in ‘linear_ramp’. sequence or scalar 
                              The values used for the ending value of the linear_ramp.
            
            reflect_type      Used in ‘reflect’, and ‘symmetric’. {‘even’(default), ‘odd’} 
                              ‘even’: an unaltered reflection around the edge value. 
                              ‘odd’: subtracting the reflected values from two times the edge value.
        
        Output: 
            array       the numpy.array after size adjustment
    '''
    array = np.asarray(array)   # convert input array to numpy.array
    dims = np.shape(array)   # array shape
    dims_n = np.size(dims)   # array rank
    
    if   adj is not None and adj_to is not None: 
        print('>>>>>> Both "adj" and "adj_to" are defined! Output the original array ... ')
        return array
    elif adj is None and adj_to is None: 
        print('>>>>>> Neither "adj" nor "adj_to" is defined! Output the original array ... ')
        return array
    elif adj is not None and adj_to is None: 
        adj = np.squeeze(np.asarray(adj))
        adj_n = np.size(adj)
        if   np.size(np.shape(adj)) > 2:   # "adj" has a rank > 2 
            print('>>>>>> Rank of "adj" > 2! Output the original array ... ')
            return array
        elif adj_n == 1:   # before = after = adj for all axes 
            adj = int(adj)
            if   adj == 0: 
                return array
            elif adj > 0:   # padding 
                if mode=='constant': 
                    new_array = np.pad(array, pad_width=adj, mode=mode, constant_values=constant_values)
                elif mode in ['maximum','mean','median','minimum']: 
                    new_array = np.pad(array, pad_width=adj, mode=mode, stat_length=stat_length)
                elif mode=='linear_ramp': 
                    new_array = np.pad(array, pad_width=adj, mode=mode, end_values=end_values)
                elif mode in ['reflect','symmetric']: 
                    new_array = np.pad(array, pad_width=adj, mode=mode, reflect_type=reflect_type)
                else: 
                    new_array = np.pad(array, pad_width=adj, mode=mode)
                return new_array
            elif adj < 0:   # cropping
                slc = [slice(-adj, adj)] * dims_n
                new_array = array[tuple(slc)]
                return new_array
            else: 
                print('>>>>>> Unknown "adj". Output the original array ... ')
                return array
        elif adj_n == 2:   # same (before, after) for all axes 
            adj = adj.astype('int')
            new_array = array.copy()
            if np.any(adj > 0):   # padding 
                pad_width = np.zeros_like(adj)
                pad_width[np.where(adj>=0)] = adj[np.where(adj>=0)]
                if mode=='constant': 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, constant_values=constant_values)
                elif mode in ['maximum','mean','median','minimum']: 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, stat_length=stat_length)
                elif mode=='linear_ramp': 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, end_values=end_values)
                elif mode in ['reflect','symmetric']: 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, reflect_type=reflect_type)
                else: 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode)
            if np.any(adj < 0):   # cropping 
                crop_width = np.zeros_like(adj).astype('object')
                crop_width[np.where(adj<0)] = adj[np.where(adj<0)]
                crop_width[0] = np.abs(crop_width[0])   # convert negative "before" to positive 
                crop_width[np.where(crop_width==0)] = None
                slc = [slice(crop_width[0], crop_width[1])] * dims_n
                new_array = new_array[tuple(slc)]
            return new_array
        elif adj_n > 2 and np.mod(adj_n, 2) == 0 and int(adj_n/2) == dims_n:   # (before, after) for each axis 
            adj = adj.astype('int')
            if np.shape(adj)[0] != 2 and np.shape(adj)[1] != 2: 
                print('>>>>>> "adj" has a wrong shape! Output the original array ... ')
                return array
            elif np.shape(adj)[1] != 2: 
                adj = adj.transpose()
            new_array = array.copy()
            if np.any(adj > 0):   # padding 
                pad_width = np.zeros_like(adj)
                pad_width[np.where(adj>=0)] = adj[np.where(adj>=0)]
                if mode=='constant': 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, constant_values=constant_values)
                elif mode in ['maximum','mean','median','minimum']: 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, stat_length=stat_length)
                elif mode=='linear_ramp': 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, end_values=end_values)
                elif mode in ['reflect','symmetric']: 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, reflect_type=reflect_type)
                else: 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode)
            if np.any(adj < 0):   # cropping 
                crop_width = np.zeros_like(adj).astype('object')
                crop_width[np.where(adj<0)] = adj[np.where(adj<0)]
                crop_width[:,0] = np.abs(crop_width[:,0])   # convert negative "before" to positive 
                crop_width[np.where(crop_width==0)] = None
                slc = []
                for ii in range(dims_n): 
                    slc.append(slice(crop_width[ii][0], crop_width[ii][1]))
                new_array = new_array[tuple(slc)]
            return new_array
        else: 
            print('>>>>>> "adj" has a wrong shape/size! Output the original array ... ')
            return array
    elif adj is None and adj_to is not None: 
        adj_to = np.squeeze(np.asarray(adj_to))
        adj_to_n = np.size(adj_to)
        offset = np.squeeze(np.asarray(offset))
        offset_n = np.size(offset)
        if np.size(np.shape(adj_to)) > 1 or np.size(np.shape(offset)) > 1:   # "adj_to"/"offset" has a rank > 1 
            print('>>>>>> Rank of "adj_to"/"offset" > 1! Output the original array ... ')
            return array
        else:  
            if adj_to_n == 1:   # same size for all axes 
                adj_to  = [int(adj_to)] * dims_n
            elif adj_to_n != dims_n: 
                print('>>>>>> "adj_to" has a wrong size! Output the original array ... ')
                return array
            if offset_n == 1:   # same offset for all axes 
                offset = [int(offset)] * dims_n
            elif offset_n != dims_n: 
                print('>>>>>> "offset" has a wrong size! Output the original array ... ')
                return array
            adj = np.zeros((dims_n, 2)).astype('int')
            for ii in range(dims_n):   # calculate (before, after) for each axis
                adj[ii, 0] = int((adj_to[ii] - dims[ii])/2) - int(offset[ii])
                adj[ii, 1] = int( adj_to[ii] - dims[ii] ) - adj[ii, 0]
            new_array = array.copy()
            if np.any(adj > 0):   # padding 
                pad_width = np.zeros_like(adj)
                pad_width[np.where(adj>=0)] = adj[np.where(adj>=0)]
                if mode=='constant': 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, constant_values=constant_values)
                elif mode in ['maximum','mean','median','minimum']: 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, stat_length=stat_length)
                elif mode=='linear_ramp': 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, end_values=end_values)
                elif mode in ['reflect','symmetric']: 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode, reflect_type=reflect_type)
                else: 
                    new_array = np.pad(array, pad_width=pad_width, mode=mode)
            if np.any(adj < 0):   # cropping 
                crop_width = np.zeros_like(adj).astype('object')
                crop_width[np.where(adj<0)] = adj[np.where(adj<0)]
                crop_width[:,0] = np.abs(crop_width[:,0])   # convert negative "before" to positive 
                crop_width[np.where(crop_width==0)] = None
                slc = []
                for ii in range(dims_n): 
                    slc.append(slice(crop_width[ii][0], crop_width[ii][1]))
                new_array = new_array[tuple(slc)]
            return new_array
    
    print('>>>>>> Unforeseen "adj" or "adj_to" ! Output the original array ... ')
    return array


def Array_zeropad_2D(Array, Add=None, AddTo=None, val=0): 
    ''' >>> Introduction <<< 
        Zero padding a 2D numpy array using numpy.pad

        Add = [8, 16, 16, 32] 
            Pad 8/16/16/32 zeros on +X/-X/+Y/-Y, respectively. 

        AddTo = [128, 256]
            Pad the Array to X_dim = 128, Y_dim = 256 
            with almost equally in both directions along each axis. 
    '''
    Array = np.asarray(Array)
    X, Y = np.shape(Array)
    
    if Add is not None and AddTo is None: 
        m1, m2, n1, n2 = np.asarray(Add).astype(int)
        Output = np.pad(Array, ((m1,m2), (n1,n2)), 'constant', 
                        constant_values=[(val, val), (val, val)])
    elif AddTo is not None and Add is None: 
        M, N = np.asarray(AddTo).astype(int)
        if M < X: 
            print('>>>>>> 1st axis of AddTo (%d,) is less then the original (%d,) !' %(M, X))
            print('>>>>>> 1st axis is NOT padded ... ')
            M = X
        if N < Y: 
            print('>>>>>> 2nd axis of AddTo (%d,) is less then the original (%d,) !' %(N, Y))
            print('>>>>>> 2nd axis is NOT padded ... ')
            N = Y
        m1 = int(np.floor((M - X)/2))
        m2 = M - X - m1
        n1 = int(np.floor((N - Y)/2))
        n2 = N - Y - n1
        Output = np.pad(Array, ((m1,m2), (n1,n2)), 'constant', 
                        constant_values=[(val, val), (val, val)])
    else: 
        print('>>>>>> Unknown pad size! Output original array ... ')
        Output = Array
    return Output


def Array_zeropad_3D(Array, Add=None, AddTo=None, val=0): 
    ''' >>> Introduction <<< 
        Zero padding a 3D numpy array using numpy.pad

        Add = [8, 16, 16, 32, 32, 64] 
            Pad 8/16/16/32/32/64 zeros on +X/-X/+Y/-Y/+Z/-Z, respectively. 

        AddTo = [128, 256, 64]
            Pad the Array to X_dim = 128, Y_dim = 256, Z_dim = 64 
            with almost equally in both directions along each axis. 
    '''
    Array = np.asarray(Array)
    X, Y, Z = np.shape(Array)
    
    if Add is not None and AddTo is None: 
        m1, m2, n1, n2, l1, l2 = np.asarray(Add).astype(int)
        Output = np.pad(Array, ((m1,m2), (n1,n2), (l1,l2)), 'constant', 
                        constant_values=[(val, val), (val, val), (val, val)])
    elif AddTo is not None and Add is None: 
        M, N, L = np.asarray(AddTo).astype(int)
        if M < X: 
            print('>>>>>> 1st axis of AddTo (%d,) is less then the original (%d,) !' %(M, X))
            print('>>>>>> 1st axis is NOT padded ... ')
            M = X
        if N < Y: 
            print('>>>>>> 2nd axis of AddTo (%d,) is less then the original (%d,) !' %(N, Y))
            print('>>>>>> 2nd axis is NOT padded ... ')
            N = Y
        if L < Z: 
            print('>>>>>> 3rd axis of AddTo (%d,) is less then the original (%d,) !' %(L, Z))
            print('>>>>>> 3rd axis is NOT padded ... ')
            L = Z
        m1 = int(np.floor((M - X)/2))
        m2 = M - X - m1
        n1 = int(np.floor((N - Y)/2))
        n2 = N - Y - n1
        l1 = int(np.floor((L - Z)/2))
        l2 = L - Z - l1
        Output = np.pad(Array, ((m1,m2), (n1,n2), (l1,l2)), 'constant', 
                        constant_values=[(val, val), (val, val), (val, val)])
    else: 
        print('>>>>>> Unknown pad size! Output original array ... ')
        Output = Array
    return Output


def Array_zeropad_4D(Array, Add=None, AddTo=None, val=0): 
    ''' >>> Introduction <<< 
        Zero padding a 3D numpy array using numpy.pad

        Similar to Array_zeropad_3D().
    '''
    Array = np.asarray(Array)
    X, Y, Z, W = np.shape(Array)
    
    if Add is not None and AddTo is None: 
        m1, m2, n1, n2, l1, l2, p1, p2 = np.asarray(Add).astype(int)
        Output = np.pad(Array, ((m1,m2), (n1,n2), (l1,l2), (p1,p2)), 'constant', 
                        constant_values=[(val, val), (val, val), (val, val), (val, val)])
    elif AddTo is not None and Add is None: 
        M, N, L, P = np.asarray(AddTo).astype(int)
        if M < X: 
            print('>>>>>> 1st axis of AddTo (%d,) is less then the original (%d,) !' %(M, X))
            print('>>>>>> 1st axis is NOT padded ... ')
            M = X
        if N < Y: 
            print('>>>>>> 2nd axis of AddTo (%d,) is less then the original (%d,) !' %(N, Y))
            print('>>>>>> 2nd axis is NOT padded ... ')
            N = Y
        if L < Z: 
            print('>>>>>> 3rd axis of AddTo (%d,) is less then the original (%d,) !' %(L, Z))
            print('>>>>>> 3rd axis is NOT padded ... ')
            L = Z
        if P < W: 
            print('>>>>>> 4th axis of AddTo (%d,) is less then the original (%d,) !' %(P, W))
            print('>>>>>> 4th axis is NOT padded ... ')
            P = W
        m1 = int(np.floor((M - X)/2))
        m2 = M - X - m1
        n1 = int(np.floor((N - Y)/2))
        n2 = N - Y - n1
        l1 = int(np.floor((L - Z)/2))
        l2 = L - Z - l1
        p1 = int(np.floor((P - W)/2))
        p2 = P - W - p1
        Output = np.pad(Array, ((m1,m2), (n1,n2), (l1,l2), (p1,p2)), 'constant', 
                        constant_values=[(val, val), (val, val), (val, val), (val, val)])
    else: 
        print('>>>>>> Unknown pad size! Output original array ... ')
        Output = Array
    return Output


def Array_crop_2D(Array, Crop=None, CropTo=None): 
    ''' >>> Introduction <<< 
        Crop a 2D numpy array

        Crop = [8, 16, 16, 32] 
            Crop 8/16/16/32 zeros on +X/-X/+Y/-Y, respectively. 

        CropTo = [128, 256]
            Crop the Array to X_dim = 128, Y_dim = 256 
            with almost equally in both directions along each axis. 
    '''
    Array = np.asarray(Array)
    X, Y = np.shape(Array)
    
    if Crop is not None and CropTo is None: 
        m1, m2, n1, n2 = np.asarray(Crop).astype(int)
        if m2 == 0: 
            m2 = None
        else: 
            m2 = -m2
        if n2 == 0: 
            n2 = None
        else: 
            n2 = -n2
        Output = Array[m1:m2, n1:n2]
    elif CropTo is not None and Crop is None: 
        M, N = np.asarray(CropTo).astype(int)
        if M > X: 
            print('>>>>>> 1st axis of CropTo (%d,) is larger then the original (%d,) !' %(M,X))
            print('>>>>>> 1st axis is NOT cropped ... ')
            M = X
        if N > Y: 
            print('>>>>>> 2nd axis of CropTo (%d,) is larger then the original (%d,) !' %(N,Y))
            print('>>>>>> 2nd axis is NOT cropped ... ')
            N = Y
        m1 = int(np.floor((X - M)/2))
        m2 = -(X - M - m1)
        if m2 == 0: 
            m2 = None
        n1 = int(np.floor((Y - N)/2))
        n2 = -(Y - N - n1)
        if n2 == 0: 
            n2 = None
        Output = Array[m1:m2, n1:n2]
    else: 
        print('>>>>>> Unknown crop size! Output original array ... ')
        Output = Array
    return Output


def Array_crop_3D(Array, Crop=None, CropTo=None): 
    ''' >>> Introduction <<< 
        Crop a 3D numpy array

        Crop = [8, 16, 16, 32, 32, 64] 
            Crop 8/16/16/32/32/64 zeros on +X/-X/+Y/-Y/+Z/-Z, respectively. 

        CropTo = [128, 256, 64]
            Crop the Array to X_dim = 128, Y_dim = 256, Z_dim = 64 
            with almost equally in both directions along each axis. 
    '''
    Array = np.asarray(Array)
    X, Y, Z = np.shape(Array)
    
    if Crop is not None and CropTo is None: 
        m1, m2, n1, n2, l1, l2 = np.asarray(Crop).astype(int)
        if m2 == 0: 
            m2 = None
        else: 
            m2 = -m2
        if n2 == 0: 
            n2 = None
        else: 
            n2 = -n2
        if l2 == 0: 
            l2 = None
        else: 
            l2 = -l2
        Output = Array[m1:m2, n1:n2, l1:l2]
    elif CropTo is not None and Crop is None: 
        M, N, L = np.asarray(CropTo).astype(int)
        if M > X: 
            print('>>>>>> 1st axis of CropTo (%d,) is larger then the original (%d,) !' %(M,X))
            print('>>>>>> 1st axis is NOT cropped ... ')
            M = X
        if N > Y: 
            print('>>>>>> 2nd axis of CropTo (%d,) is larger then the original (%d,) !' %(N,Y))
            print('>>>>>> 2nd axis is NOT cropped ... ')
            N = Y
        if L > Z: 
            print('>>>>>> 3rd axis of CropTo (%d,) is larger then the original (%d,) !' %(L,Z))
            print('>>>>>> 3rd axis is NOT cropped ... ')
            L = Z
        m1 = int(np.floor((X - M)/2))
        m2 = -(X - M - m1)
        if m2 == 0: 
            m2 = None
        n1 = int(np.floor((Y - N)/2))
        n2 = -(Y - N - n1)
        if n2 == 0: 
            n2 = None
        l1 = int(np.floor((Z - L)/2))
        l2 = -(Z - L - l1)
        if l2 == 0: 
            l2 = None
        Output = Array[m1:m2, n1:n2, l1:l2]
    else: 
        print('>>>>>> Unknown crop size! Output original array ... ')
        Output = Array
    return Output


def Array_crop_4D(Array, Crop=None, CropTo=None): 
    ''' >>> Introduction <<< 
        Crop a 4D numpy array

        Similar to Array_crop_3D().
    '''
    Array = np.asarray(Array)
    X, Y, Z, W = np.shape(Array)
    
    if Crop is not None and CropTo is None: 
        m1, m2, n1, n2, l1, l2, p1, p2 = np.asarray(Crop).astype(int)
        if m2 == 0: 
            m2 = None
        else: 
            m2 = -m2
        if n2 == 0: 
            n2 = None
        else: 
            n2 = -n2
        if l2 == 0: 
            l2 = None
        else: 
            l2 = -l2
        if p2 == 0: 
            p2 = None
        else: 
            p2 = -p2
        Output = Array[m1:m2, n1:n2, l1:l2, p1:p2]
    elif CropTo is not None and Crop is None: 
        M, N, L, P = np.asarray(CropTo).astype(int)
        if M > X: 
            print('>>>>>> 1st axis of CropTo (%d,) is larger then the original (%d,) !' %(M,X))
            print('>>>>>> 1st axis is NOT cropped ... ')
            M = X
        if N > Y: 
            print('>>>>>> 2nd axis of CropTo (%d,) is larger then the original (%d,) !' %(N,Y))
            print('>>>>>> 2nd axis is NOT cropped ... ')
            N = Y
        if L > Z: 
            print('>>>>>> 3rd axis of CropTo (%d,) is larger then the original (%d,) !' %(L,Z))
            print('>>>>>> 3rd axis is NOT cropped ... ')
            L = Z
        if P > W: 
            print('>>>>>> 4th axis of CropTo (%d,) is larger then the original (%d,) !' %(P,W))
            print('>>>>>> 4th axis is NOT cropped ... ')
            P = W
        m1 = int(np.floor((X - M)/2))
        m2 = -(X - M - m1)
        if m2 == 0: 
            m2 = None
        n1 = int(np.floor((Y - N)/2))
        n2 = -(Y - N - n1)
        if n2 == 0: 
            n2 = None
        l1 = int(np.floor((Z - L)/2))
        l2 = -(Z - L - l1)
        if l2 == 0: 
            l2 = None
        p1 = int(np.floor((W - P)/2))
        p2 = -(W - P - p1)
        if p2 == 0: 
            p2 = None
        Output = Array[m1:m2, n1:n2, l1:l2, p1:p2]
    else: 
        print('>>>>>> Unknown crop size! Output original array ... ')
        Output = Array
    return Output


def Array_zeropad_crop_2D(Array, Adj=None, AdjTo=None): 
    ''' >>> Introduction <<< 
        Adjust the size of a 2D array, using 'Array_crop_2D()' and 'Array_zeropad_2D()'.
    '''
    Array = np.asarray(Array)
    X, Y = np.shape(Array)
    
    if Adj is not None and AdjTo is None: 
        Output = Array
        Adj = np.asarray(Adj)
        Add = np.zeros_like(Adj)
        Add[np.where(Adj>=0)] = Adj[np.where(Adj>=0)]
        Crop = np.zeros_like(Adj)
        Crop[np.where(Adj<0)] = Adj[np.where(Adj<0)]
        
        Output = Array_zeropad_2D(Output, Add=Add, AddTo=None)
        Output = Array_crop_2D(Output, Crop=-Crop, CropTo=None)
    elif AdjTo is not None and Adj is None: 
        Output = Array
        M, N = np.asarray(AdjTo).astype(int)
        if M < X: 
            m1 = int(np.floor((X - M)/2))
            m2 = X - M - m1
            Output = Array_crop_2D(Output, Crop=[m1,m2,0,0], CropTo=None)
        else: 
            m1 = int(np.floor((M - X)/2))
            m2 = M - X - m1
            Output = Array_zeropad_2D(Output, Add=[m1,m2,0,0], AddTo=None)
        if N < Y: 
            n1 = int(np.floor((Y - N)/2))
            n2 = Y - N - n1
            Output = Array_crop_2D(Output, Crop=[0,0,n1,n2], CropTo=None)
        else: 
            n1 = int(np.floor((N - Y)/2))
            n2 = N - Y - n1
            Output = Array_zeropad_2D(Output, Add=[0,0,n1,n2], AddTo=None)
    else:
        print('>>>>>> Unknown pad/crop size! Output original array ... ')
        Output = Array
    return Output


def Array_zeropad_crop_3D(Array, Adj=None, AdjTo=None): 
    ''' >>> Introduction <<< 
        Adjust the size of a 3D array, using 'Array_crop_3D()' and 'Array_zeropad_3D()'.
    '''
    Array = np.asarray(Array)
    X, Y, Z = np.shape(Array)
    
    if Adj is not None and AdjTo is None: 
        Output = Array
        Adj = np.asarray(Adj)
        Add = np.zeros_like(Adj)
        Add[np.where(Adj>=0)] = Adj[np.where(Adj>=0)]
        Crop = np.zeros_like(Adj)
        Crop[np.where(Adj<0)] = Adj[np.where(Adj<0)]
        
        Output = Array_zeropad_3D(Output, Add=Add, AddTo=None)
        Output = Array_crop_3D(Output, Crop=-Crop, CropTo=None)
    elif AdjTo is not None and Adj is None: 
        Output = Array
        M, N, L = np.asarray(AdjTo).astype(int)
        if M < X: 
            m1 = int(np.floor((X - M)/2))
            m2 = X - M - m1
            Output = Array_crop_3D(Output, Crop=[m1,m2,0,0,0,0], CropTo=None)
        else: 
            m1 = int(np.floor((M - X)/2))
            m2 = M - X - m1
            Output = Array_zeropad_3D(Output, Add=[m1,m2,0,0,0,0], AddTo=None)
        if N < Y: 
            n1 = int(np.floor((Y - N)/2))
            n2 = Y - N - n1
            Output = Array_crop_3D(Output, Crop=[0,0,n1,n2,0,0], CropTo=None)
        else: 
            n1 = int(np.floor((N - Y)/2))
            n2 = N - Y - n1
            Output = Array_zeropad_3D(Output, Add=[0,0,n1,n2,0,0], AddTo=None)
        if L < Z: 
            l1 = int(np.floor((Z - L)/2))
            l2 = Z - L - l1
            Output = Array_crop_3D(Output, Crop=[0,0,0,0,l1,l2], CropTo=None)
        else: 
            l1 = int(np.floor((L - Z)/2))
            l2 = L - Z - l1
            Output = Array_zeropad_3D(Output, Add=[0,0,0,0,l1,l2], AddTo=None)
    else:
        print('>>>>>> Unknown crop size! Output original array ... ')
        Output = Array
    return Output


def Array_zeropad_crop_4D(Array, Adj=None, AdjTo=None): 
    ''' >>> Introduction <<< 
        Adjust the size of a 4D array, using 'Array_crop_4D()' and 'Array_zeropad_4D()'.
    '''
    Array = np.asarray(Array)
    X, Y, Z, W = np.shape(Array)
    
    if Adj is not None and AdjTo is None: 
        Output = Array
        Adj = np.asarray(Adj)
        Add = np.zeros_like(Adj)
        Add[np.where(Adj>=0)] = Adj[np.where(Adj>=0)]
        Crop = np.zeros_like(Adj)
        Crop[np.where(Adj<0)] = Adj[np.where(Adj<0)]
        
        Output = Array_zeropad_4D(Output, Add=Add, AddTo=None)
        Output = Array_crop_4D(Output, Crop=-Crop, CropTo=None)
    elif AdjTo is not None and Adj is None: 
        Output = Array
        M, N, L, P = np.asarray(AdjTo).astype(int)
        if M < X: 
            m1 = int(np.floor((X - M)/2))
            m2 = X - M - m1
            Output = Array_crop_4D(Output, Crop=[m1,m2,0,0,0,0,0,0], CropTo=None)
        else: 
            m1 = int(np.floor((M - X)/2))
            m2 = M - X - m1
            Output = Array_zeropad_4D(Output, Add=[m1,m2,0,0,0,0,0,0], AddTo=None)
        if N < Y: 
            n1 = int(np.floor((Y - N)/2))
            n2 = Y - N - n1
            Output = Array_crop_4D(Output, Crop=[0,0,n1,n2,0,0,0,0], CropTo=None)
        else: 
            n1 = int(np.floor((N - Y)/2))
            n2 = N - Y - n1
            Output = Array_zeropad_4D(Output, Add=[0,0,n1,n2,0,0,0,0], AddTo=None)
        if L < Z: 
            l1 = int(np.floor((Z - L)/2))
            l2 = Z - L - l1
            Output = Array_crop_4D(Output, Crop=[0,0,0,0,l1,l2,0,0], CropTo=None)
        else: 
            l1 = int(np.floor((L - Z)/2))
            l2 = L - Z - l1
            Output = Array_zeropad_4D(Output, Add=[0,0,0,0,l1,l2,0,0], AddTo=None)
        if P < W: 
            p1 = int(np.floor((W - P)/2))
            p2 = W - P - p1
            Output = Array_crop_4D(Output, Crop=[0,0,0,0,0,0,p1,p2], CropTo=None)
        else: 
            p1 = int(np.floor((P - W)/2))
            p2 = P - W - p1
            Output = Array_zeropad_4D(Output, Add=[0,0,0,0,0,0,p1,p2], AddTo=None)
    else:
        print('>>>>>> Unknown crop size! Output original array ... ')
        Output = Array
    return Output


def Array_binning(Array, Bin=[2,2,2]): 
    ''' >>> Introduction <<< 
        Binning an Array upto three dimensional.

        The standard way to bin a large array to a smaller one by averaging is 
        to reshape it into a higher dimension and then take the means over the 
        appropriate new axes. The following function does this, assuming that 
        each dimension of the new shape is a factor of the corresponding dimension 
        in the old one.

        refer to: 
        https://scipython.com/blog/binning-a-2d-array-in-numpy/
    '''
    Array = np.asarray(Array)
    dims  = np.shape(Array)
    M = len(dims)
    if not len(dims) == len(Bin):
        print(">>>>>> 'Bin' must have same number of dimensions as the array ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    
    if np.asarray(np.nonzero(np.remainder(dims, Bin))).size != 0: 
        print(">>>>>> The new dimensions are not factors of the old dimensions ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    
    if np.array_equal(Bin, np.ones(M)): 
        return Array
    elif M == 1: 
        return Array.reshape(dims[0]//Bin[0], Bin[0]).sum(1)
    elif M == 2:
        return Array.reshape(dims[0]//Bin[0], Bin[0], 
                             dims[1]//Bin[1], Bin[1]).sum(3).sum(1)
    elif M == 3: 
        return Array.reshape(dims[0]//Bin[0], Bin[0], 
                             dims[1]//Bin[1], Bin[1], 
                             dims[2]//Bin[2], Bin[2]).sum(5).sum(3).sum(1)


def Array_shift_int(Array, Shift=[0,0,0]): 
    ''' >>> Introduction <<< 
        Circularly shift a N-D array. 
        Input: 
            Shift:      (n,) list, nD shifts in pixels
                    e.g. Shift[1] = 2 means Array is shifted 2 pixels along axis=1
    '''
    Array = np.asarray(Array)
    dims = np.shape(Array)
    if not len(dims) == len(Shift):
        print(">>>>>> 'Shift' must have same number of dimensions as the array! ")
        input('>>>>>> Press any key to quit ... ')
        return
    Shift = np.asarray(Shift).astype('int')
    '''
    if   len(Shift) == 1: 
        shiftedarr = np.roll(Array, shift=tuple(Shift))
    elif len(Shift) == 2: 
        shiftedarr = np.roll(Array, shift=tuple(Shift), axis=(0,1))
    elif len(Shift) == 3: 
        shiftedarr = np.roll(Array, shift=tuple(Shift), axis=(0,1,2))
    elif len(Shift) == 4: 
        shiftedarr = np.roll(Array, shift=tuple(Shift), axis=(0,1,2,3))
    '''
    return np.roll(Array, shift=tuple(Shift))


def Array_shift_dft(Array, Shift=[0,0,0]): 
    ''' >>> Introduction <<< 
        Circularly shift a N-D array using scipy.ndimage.fourier_shift.  
        Input: 
            Shift:      (n,) list, nD shifts in pixels
                    e.g. Shift[1] = 2 means Array is shifted 2 pixels along axis=1
    '''
    Array = np.asarray(Array)
    dims = np.shape(Array)
    if not len(dims) == len(Shift):
        print(">>>>>> 'Shift' must have same number of dimensions as the array! ")
        input('>>>>>> Press any key to quit ... ')
        return
    return simg.fourier_shift(Array, shift=Shift)


def Array_centering(Array, Method='peak', Progress=False): 
    ''' >>> Introduction <<< 
        Cenering a multi-dimensional array using FFT shift

        The center is determined using center of mass 'COM' or the peak pixel 'Peak'
    '''
    Array = np.asarray(Array)
    dims = np.shape(Array)
    
    if Method.lower() == 'com': 
        Center = np.round(simg.center_of_mass(Array))
    elif Method.lower() == 'peak': 
        Center = np.unravel_index(np.argmax(Array), np.shape(Array))
    else: 
        print('>>>>>> Unknown Method!')
        return None
    
    Shift = np.round(np.asarray(dims)/2) - np.asarray(Center)
    if Progress: 
        print('>>>>>> Array shape is ', dims)
        print('>>>>>> Center is ', Center)
        print('>>>>>> Shift is ', Shift)
    Centeredarr = Array_shift_int(Array, Shift=Shift)
    
    return Centeredarr


def Array_rot_2D(Array, Rot, IsRadian=True): 
    ''' >>> Introduction <<< 
        Rotates a 2-dimensional array using FT method

        see DOI:10.1109/83.784442, DOI:10.1109/ACSSC.1996.600840
    
    '''
    Array = np.asarray(Array)
    if not IsRadian: 
        Rot = np.deg2rad(Rot)
    
    # Generate index grid
    dims = np.shape(Array)
    r=[]
    for d in dims:
        r.append(slice(int(np.ceil(-d/2.)), int(np.ceil(d/2.)), None))
    idxgrid = np.mgrid[r]
    
    t_1 = time()
    # ========= 1st shear =========
    # scipy does normalized ffts!
    ftarr = nf.fft(Array, axis=0)
    Shift = idxgrid[1]*np.tan(-Rot/2)
    ftarr *= np.exp(-2j*np.pi/float(dims[0]) 
                    * nf.fftshift(idxgrid[0]) * Shift)
    Shearedarr = np.abs(nf.ifft(ftarr, axis=0))
    
    # ========= 2nd shear =========
    ftarr = nf.fft(Shearedarr, axis=1)
    Shift = idxgrid[0]*np.sin(Rot)
    ftarr *= np.exp(-2j*np.pi/float(dims[1]) 
                    * nf.fftshift(idxgrid[1]) * Shift)
    Shearedarr = np.abs(nf.ifft(ftarr, axis=1))
    
    # ========= 3rd shear =========
    ftarr = nf.fft(Shearedarr, axis=0)
    Shift = idxgrid[1]*np.tan(-Rot/2)
    ftarr *= np.exp(-2j*np.pi/float(dims[0]) 
                    * nf.fftshift(idxgrid[0]) * Shift)
    Shearedarr = np.abs(nf.ifft(ftarr, axis=0))
    
    t_2 = time()
    print(t_2-t_1)
    
    return Shearedarr


def Array_rot_3D(Array, Rot=[0, 0, 0], Seq='012', IsRadian=True, Progress=False): # pyfftw 
    ''' >>> Introduction <<< 
        Rotates a 3-dimensional array using FT method

        see DOI:10.1109/83.784442, DOI:10.1109/ACSSC.1996.600840

        Input: 
            Rotation:   (3,) list, 3D rotations in rad
                    e.g. Rotation[0] = 0.1 means Array is rotated 0.1 rad around axis=0
            Progress:   True will display the time stamps of each step
    '''
    Array = np.asarray(Array)
    Rot = np.asarray(Rot)
    if not IsRadian: 
        Rot = np.deg2rad(Rot)
    
    dims = np.shape(Array)
    if not len(dims) == len(Rot):
        print(">>>>>> 'Rot' must have same number of dimensions as the array! ")
        input('>>>>>> Press any key to quit ... ')
        return
    # ====== Creat 3D meshgrid ======
    if Progress: 
        print('>>>>>> Creat the meshgrid ... ' + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    r=[]
    for d in dims:
        r.append(slice(int(np.ceil(-d/2.)), int(np.ceil(d/2.)), None))
    idxgrid = np.mgrid[r]
    # ====== Perform the rotation ======
    if Progress: 
        print('>>>>>> Rotate the array ... ' + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    idxdims = np.asarray(range(len(dims)))
    Shearedarr = Array.copy()
    if Seq == '012': 
        RotOrd = idxdims
    elif Seq == '210': 
        RotOrd = idxdims[::-1]
    else: 
        RotOrd = idxdims
    
    for d in RotOrd:   # Reverse the sequence of rotations
        if d == 0: 
            m, n = [1, 2]
            if Progress: 
                print('>>>>>> Rotate axis 0 ... ' + strftime('%H:%M:%S', localtime()))
                t_3 = time()
        elif d == 1: 
            m, n = [2, 0]
            if Progress: 
                print('>>>>>> Rotate axis 1 ... ' + strftime('%H:%M:%S', localtime()))
                t_4 = time()
        elif d == 2: 
            m, n = [0, 1]
            if Progress: 
                print('>>>>>> Rotate axis 2 ... ' + strftime('%H:%M:%S', localtime()))
                t_5 = time()
        else: 
            print('>>>>>> Unknown axis index! ')
            input('>>>>>> Press any key to quit ... ')
            return
        if Rot[d] == 0:   # Skip this axis if no rotation
            continue        
        # ========= 1st shear =========
        # scipy does normalized ffts!
        ftarr = (pf.builders.fft(Shearedarr, axis=m))()
        Shift = idxgrid[n]*np.tan(-Rot[d]/2)
        ftarr *= np.exp(-2j*np.pi/float(dims[m]) 
                        * nf.fftshift(idxgrid[m]) * Shift)
        Shearedarr = np.abs((pf.builders.ifft(ftarr, axis=m))())
        # ========= 2nd shear =========
        ftarr = (pf.builders.fft(Shearedarr, axis=n))()
        Shift = idxgrid[m]*np.sin(Rot[d])
        ftarr *= np.exp(-2j*np.pi/float(dims[n]) 
                        * nf.fftshift(idxgrid[n]) * Shift)
        Shearedarr = np.abs((pf.builders.ifft(ftarr, axis=n))())
        # ========= 3rd shear =========
        ftarr = (pf.builders.fft(Shearedarr, axis=m))()
        Shift = idxgrid[n]*np.tan(-Rot[d]/2)
        ftarr *= np.exp(-2j*np.pi/float(dims[m]) 
                        * nf.fftshift(idxgrid[m]) * Shift)
        Shearedarr = np.abs((pf.builders.ifft(ftarr, axis=m))())
    
    if Progress: 
        t_6 = time()
        print('>>>>>> Summary: ')
        print('           Meshgrid took %0.6f sec.' %(t_2 - t_1))
        print('           Rotate axis 0 took %0.6f sec.' %(t_4 - t_3))
        print('           Rotate axis 1 took %0.6f sec.' %(t_5 - t_4))
        print('           Rotate axis 2 took %0.6f sec.' %(t_6 - t_5))
        print('           Rotation took %0.6f sec in total.' %(t_6 - t_2))
    
    return Shearedarr


def Array_rot_imu_3D(Array, Rot=[0, 0, 0], IsRadian=False, Progress=False): 
    ''' >>> Introduction <<< 
        Rotates a 3-dimensional array using imutils.rotate()
        
        Input: 
            Array:      (n, m, l), input array 
            Rot:        (3,), rotation in DEGREE
                        !! Note: Rot should contain only one non-zero number !!
                        Rot=[0, 5, 0] means Array is rotated 5 degree around axis=1
    '''
    Array = np.asarray(Array)
    Rot = np.asarray(Rot)
    if IsRadian: 
        Rot = np.rad2deg(Rot)
    rot_ax = Rot!=0
    if np.sum(rot_ax) != 1: 
        print('>>>>>> Rot must contain only one non-zero number !')
        return None
    N = np.dot(rot_ax, np.shape(Array))
    array_rot = np.zeros_like(Array)
    if   rot_ax[0] == 1: 
        for ii in range(N): 
            # if Progress: 
            #     if np.remainder(ii, 20) == 0: 
            #         print('Rotating %d / %d ...' %(ii, N), end='\r' )
            array_rot[ii, :, :] = imu.rotate(Array[ii, :, :], Rot[0])
    elif rot_ax[1] == 1: 
        for ii in range(N): 
            # if Progress: 
            #     if np.remainder(ii, 20) == 0: 
            #         print('Rotating %d / %d ...' %(ii, N), end='\r' )
            array_rot[:, ii, :] = imu.rotate(Array[:, ii, :], Rot[1])
    elif rot_ax[2] == 1: 
        for ii in range(N): 
            # if Progress: 
            #     if np.remainder(ii, 20) == 0: 
            #         print('Rotating %d / %d ...' %(ii, N), end='\r' )
            array_rot[:, :, ii] = imu.rotate(Array[:, :, ii], Rot[2])
    return array_rot


def Array_Offset(Array, ArrayRef, Method='CC', DisplayError=False): 
    ''' >>> Introduction <<< 
        Determine the offset between two multi-dimensional arrays using 
           Center of Mass (Method = 'COM' or 'com')
        or
           Cross-correlation (Method = 'CC' or 'cc')

        ArrayRef is the reference. 
        The output is the offset of Array from ArrayRef. 
    '''
    # Input regularization
    Ref = np.asarray(ArrayRef)
    Reg = np.asarray(Array)
    if not np.shape(Ref) == np.shape(Reg): 
        print(">>>>>> The two arrays must have the same shape! ")
        input('>>>>>> Press any key to quit ... ')
        return
    Ref = np.where(np.isnan(Ref),0,Ref)
    Reg = np.where(np.isnan(Reg),0,Reg)
    
    if Method.lower() == 'cc':
        Ref = nf.fftn(Ref)
        Reg = nf.fftn(Reg)

        CrosCorr = nf.ifftn(Ref*np.conj(Reg))
        Dims = np.asarray(np.shape(CrosCorr))
        Magn = np.abs(CrosCorr)
        CCMax = np.max(Magn)

        Ref2 = np.sum(np.abs(Ref)**2) / np.size(Ref)
        Reg2 = np.sum(np.abs(Reg)**2) / np.size(Reg)
        Err = (np.abs(1 - CCMax**2 / (Ref2*Reg2) ))**0.5

        Offset = np.asarray(np.unravel_index(Magn.argmax(), Dims))
    
    elif Method.lower() == 'com':
        Dims = np.asarray(np.shape(Ref))
        RefCOM = np.asarray(simg.center_of_mass(Ref))
        RegCOM = np.asarray(simg.center_of_mass(Reg))
        
        Offset = np.asarray(np.round(RefCOM - RegCOM)).astype(int)
        Err = np.linalg.norm(RefCOM - RegCOM - Offset)
    
    else: 
        print('>>>>>> Unknown Method!')
        Offset = np.zeros(len(Dims)).astype(int)
        Error = 0
    
    Offset = np.where(Offset>=Dims/2, Offset-Dims, Offset)
    
    if DisplayError:
        print(">>>>>> Offset is %s, with Error %0.3f" %(Offset, Err))
    return Offset


def Array_CorrCoef(Array1, Array2): 
    '''Calculate the Pearson correlation coefficient between Array1 and Array2'''
    return np.corrcoef(Array1.reshape(-1), Array2.reshape(-1))[0][1]


def Array_align(Array, ArrayRef, Method='CC', DisplayError=False): 
    '''
    Align Array to ArrayRef using 
       Center of Mass (Method = 'COM' or 'com')
    or
       Cross-correlation (Method = 'CC' or 'cc')
    
    ArrayRef is the reference. 
    The output is the shifted Array 
    '''
    Offset = Array_Offset(Array, ArrayRef, Method=Method, DisplayError=DisplayError)
    ArrayShifted = Array_shift_int(Array, Offset)
    return ArrayShifted


def Array_radial_integration_2D(array, center=None, ax_step=[1,1], segment=1, 
                                method='com', flag_average=False, Progress=False): 
    ''' >>> Instruction <<< 
        This function peforms radial integration on a 2-dimension array.
        Using numpy.histogram to sort voxels in the array. 
        
        Inputs: 
            array           2D array 
            center          (2,), index of the center voxel
                            If None, search center using defined method
            ax_step         (2,), step sizes of 3 axes
            segment         size of segment of distance-to-center when sorting voxels
            method          'max' or 'com', method for searching center voxel
            flag_average    If True, output the mean value of each ring
                            If False, output the sum of each ring
            
        Output: 
            result          1D array (N, ), integration result
            bins            1D array (N+1, ), bins of the histogram
    '''
    # Regularize inputs 
    array = np.asarray(array)
    if center is None: 
        if method.lower() == 'com': 
            center = np.asarray(simg.center_of_mass(array))
        elif method.lower() == 'max': 
            center = np.unravel_index(np.argmax(array), np.shape(array))
        else: 
            print(">>>>>> Unknown searching method ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    else: 
        center = np.asarray(center)
    
    # Calculate distance-to-center 
    dims = np.shape(array)
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    y3, x3 = np.meshgrid(y, x)
    r = np.sqrt( ( (x3 - center[0]) * ax_step[0] )**2 + 
                 ( (y3 - center[1]) * ax_step[1] )**2 )
    r_max = np.amax(r)
    # Calculate sum of each segment 
    radial_sum, r_bins  = np.histogram(r, weights=array, bins=int(r_max/segment))
    norm_factor, r_bins = np.histogram(r, bins=int(r_max/segment))
    
    if Progress: 
        print('array shape: [%d, %d] ' %(dims[0], dims[1]) )
        print('center:      [%d, %d] ' %(center[0], center[1]) )
        print('r_max:       %0.3f ' %r_max)
        print('result size: %d ' %np.shape(radial_sum))
        print('r_bins size: %d ' %np.shape(r_bins))
    
    if flag_average: 
        result = np.divide(radial_sum, norm_factor) 
    else: 
        result = radial_sum
    return {'result': result, 'bins': r_bins}


def Array_radial_integration_3D(array, center=None, ax_step=[1,1,1], segment=1, 
                                method='com', flag_average=False, Progress=False): 
    ''' >>> Instruction <<< 
        This function peforms radial integration on a 3-dimension array.
        Using numpy.histogram to sort voxels in the array. 
        
        Inputs: 
            array           3D array 
            center          (3,), index of the center voxel
                            If None, search center using defined method
            ax_step         (3,), step sizes of 3 axes
            segment         size of segment of distance-to-center when sorting voxels
            method          'max' or 'com', method for searching center voxel
            flag_average    If True, output the mean value of each ring
                            If False, output the sum of each ring
            
        Output: 
            result          1D array (N, ), integration result
            bins            1D array (N+1, ), bins of the histogram
    '''
    # Regularize inputs 
    array = np.asarray(array)
    if center is None: 
        if method.lower() == 'com': 
            center = np.asarray(simg.center_of_mass(array))
        elif method.lower() == 'max': 
            center = np.unravel_index(np.argmax(array), np.shape(array))
        else: 
            print(">>>>>> Unknown searching method ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    else: 
        center = np.asarray(center)
    
    # Calculate distance-to-center 
    dims = np.shape(array)
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    y3, z3, x3 = np.meshgrid(y, z, x)
    r = np.sqrt( ( (x3 - center[0]) * ax_step[0] )**2 + 
                 ( (y3 - center[1]) * ax_step[1] )**2 + 
                 ( (z3 - center[2]) * ax_step[2] )**2 )
    r_max = np.amax(r)
    # Calculate sum of each segment 
    radial_sum, r_bins  = np.histogram(r, weights=array, bins=int(r_max/segment))
    norm_factor, r_bins = np.histogram(r, bins=int(r_max/segment))
    
    if Progress: 
        print('array shape: [%d, %d, %d] ' %(dims[0], dims[1], dims[2]))
        print('center:      [%d, %d, %d] ' %(center[0], center[1], center[2]))
        print('r_max:       %0.3f ' %r_max)
        print('result size: %d ' %np.shape(radial_sum))
        print('r_bins size: %d ' %np.shape(r_bins))
    
    if flag_average: 
        result = np.divide(radial_sum, norm_factor) 
    else: 
        result = radial_sum
    return {'result': result, 'bins': r_bins}


def Array_occurrence_pos(data, occ_val=1, axis=0, last_occ=False, invalid_val=0): 
    ''' >>> Instruction <<< 
        This function finds the first/last elements with a certain value
        along a certain axis of any array. 
        
        Inputs: 
            data            n-D array (n>=1)
            occ_val         occurrence value, default is 1
            axis            search along the specified axis
            last_occ        True: find the last occurrence
                            False: find the first occurrence
            invalid_val     value for space without first/last occurrence 
        
        Output: 
            idx_arr         (n-1)-D array
    '''
    mask = data==occ_val
    if last_occ: 
        val = data.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    else: 
        val = mask.argmax(axis=axis)
    idx_arr = np.where(mask.any(axis=axis), val, invalid_val)
    return idx_arr


def Array_occurrence_neg(data, non_occ_val=0, axis=0, last_occ=False, invalid_val=0): 
    ''' >>> Instruction <<< 
        This is the original 'Array_occurrence' function. 
        This function finds the first/last non-zero(or other value) elements 
        along a certain axis of any array. 
        
        Inputs: 
            data            n-D array (n>=1)
            non_occ_val     non-occurrence value, default is 0
            axis            search along the specified axis
            last_occ        True: find the last occurrence
                            False: find the first occurrence
            invalid_val     value for space without first/last occurrence 
        
        Output: 
            idx_arr         (n-1)-D array
    '''
    mask = data!=non_occ_val
    if last_occ: 
        val = data.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    else: 
        val = mask.argmax(axis=axis)
    idx_arr = np.where(mask.any(axis=axis), val, invalid_val)
    return idx_arr


def Array_interpolation_2D_spline(data, new_coords, 
                                  ori_ax_1=None, ori_ax_2=None, Progress=False): 
    ''' >>> Instruction <<< 
        This function interpolates a 2D array from a regular grid to an arbritrary coordinate, 
        using RectBivariateSpline from SciPy.interpolate. 
        
        Inputs: 
            data            2D array 
            new_coords      (2,n**2) array, targeted coordinates
            ori_ax_1/2      Monotonic axes of the original grid (in strictly ascending order)
        
        Output: 
            data_new        Interpolated 2D (n, n) array 
    '''
    # Input regularization
    if Progress: 
        print('>>>>>> Input regularization ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    dims = np.asarray(np.shape(data))
    if len(dims) == 2: 
        N_elem = int(dims[0]*dims[1])
    else: 
        print('>>>>>> The shape of "array" is not correct! ')
        input('>>>>>> Press any key to quit...')
        return
    
    if np.shape(new_coords)[1] == 2 and np.size(new_coords) != 4: 
        new_coords = np.transpose(new_coords)
        # print('>>>>>> Input coords is transposed to [%d, %d]. ' 
        #       %(np.shape(new_coords)[0], np.shape(new_coords)[1]))
    
    # Generate axes, if necessary
    if Progress: 
        print('>>>>>> Generate axes ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    if ori_ax_1 is None: 
        ori_ax_1 = np.arange(dims[0]) - int((dims[0]-1)/2)
    if ori_ax_2 is None: 
        ori_ax_2 = np.arange(dims[1]) - int((dims[1]-1)/2)
    
    # Interpolation
    if Progress: 
        print('>>>>>> Interpolate array ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
    
    f = rbs(ori_ax_1, ori_ax_2, data)
    data_new = f(np.ravel(new_coords[0,:]), 
                 np.ravel(new_coords[1,:]), grid=False)
    data_new = np.reshape(data_new, dims)
    
    if Progress: 
        print('>>>>>> Interpolation finished.    ' 
              + strftime('%H:%M:%S', localtime()))
        t_4 = time()
        
        plt.figure(figsize=(6.5, 3))
        plt.subplot(121)
        plt.imshow(data)
        plt.title('Original data')
        
        plt.subplot(122)
        plt.imshow(data_new)
        plt.title('Interpolated data')
        
        plt.tight_layout()
    
    return data_new


def Array_interpolation_2D_rgi(data, new_coords, method='cubic', 
                               ori_ax_1=None, ori_ax_2=None, target_dims=None, 
                               PadSize=None, PadIncr=[5,5], Progress=False): 
    ''' >>> Instruction <<< 
        This function interpolates a 2D array from grid coordinates to arbritrary coordinates, 
        using scipy.interpolate.RegularGridInterpolator 
        
        Inputs: 
            data            2D array 
            new_coords      (2,n) array, targeted coordinates
            method          “linear”, “nearest”, “slinear”, “cubic”, and “quintic”
            ori_ax_1/2      Monotonic axes of the original coordinates
            PadSize         (2,) size of the padded array.
            PadIncr         (2,) increaments of the padding in three dimension. 
        
        Output: 
            data_new        Interpolated 3D array 
    '''
    # Input regularization
    if Progress: 
        print('>>>>>> Input regularization ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    PadIncr = np.asarray(PadIncr)
    dims = np.asarray(np.shape(data))
    if len(dims) == 2: 
        N_elem = int(dims[0]*dims[1])
    else: 
        print('>>>>>> The shape of "array" is not correct! ')
        input('>>>>>> Press any key to quit...')
        return
    
    if np.shape(new_coords)[1] == 2 and np.size(new_coords) != 4: 
        new_coords = np.transpose(new_coords)
    
    # Generate axes, if necessary
    if Progress: 
        print('>>>>>> Generate variables ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    if ori_ax_1 is None: 
        ori_ax_1 = np.arange(dims[0]) - int((dims[0]-1)/2)
    if ori_ax_2 is None: 
        ori_ax_2 = np.arange(dims[1]) - int((dims[1]-1)/2)
    
    # Interpolation
    if Progress: 
        print('>>>>>> Calculate coordinates in original basis ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
        print('       Original array size is ', dims)
        
    if (PadSize is None) or (not PadSize >= np.shape(data)):
        dims_new = dims.copy()
    else: 
        dims_new = np.asarray(PadSize)
    
    if target_dims is None: 
        target_dims = dims.copy()
    
    Err_flag = True
    while Err_flag: 
        data_pad = Array_zeropad_2D(data, AddTo=dims_new)
        if ori_ax_1 is None: 
            ori_ax_1 = np.arange(dims_new[0]) - int((dims_new[0]-1)/2)
        else: 
            ori_ax_1 = Extend_1D_array(ori_ax_1, addto_elm=dims_new[0])
        if ori_ax_2 is None: 
            ori_ax_2 = np.arange(dims_new[1]) - int((dims_new[1]-1)/2)
        else: 
            ori_ax_2 = Extend_1D_array(ori_ax_2, addto_elm=dims_new[1])
        
        # Define an interpolating function from data
        interp_function = rgi((ori_ax_1, ori_ax_2), data_pad, method=method)
        
        # Evaluate the interpolating function on new matrix
        try: 
            data_new = interp_function(new_coords.transpose())
            if Progress: 
                print('')
                print('       Padded array size is   ', np.asarray(np.shape(data_pad)))
            data_new = np.reshape(data_new, target_dims)
            Err_flag = False
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', padding ', dims_new-dims, end='\r')
            if ErrM.startswith('One of the requested xi is out of bounds'): 
                Ax_idx = int(ErrM.split(' ')[-1])
                dims_new[Ax_idx] = dims_new[Ax_idx] + PadIncr[Ax_idx]
            else: 
                print(ErrM)
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
    
    if Progress: 
        print('>>>>>> Interpolation finished.    ' 
              + strftime('%H:%M:%S', localtime()))
        print('       Output array size is   ', np.asarray(np.shape(data_new)))
        
        t_4 = time()
        
        plt.figure(figsize=(6.5, 3))
        plt.subplot(121)
        plt.plot(np.ravel(new_coords[0]), alpha=0.5)
        plt.plot(ori_ax_1, alpha=0.5)
        plt.title('Axis 1')
        
        plt.subplot(122)
        plt.plot(np.ravel(new_coords[1]), alpha=0.5)
        plt.plot(ori_ax_2, alpha=0.5)
        plt.title('Axis 2')
        
        plt.tight_layout()
    
    return data_new


def Array_interpolation_3D_spline(data, new_coords, 
                                  ori_ax_1=None, ori_ax_2=None, ori_ax_3=None, 
                                  PadSize=None, PadIncr=[5,5,5], Progress=False): 
    ''' >>> Instruction <<< 
        This function interpolates a 3D array from grid coordinates to arbritrary coordinates, 
        using tricubic spline from python module eqtools. 
        
        To display the result in ParaView, using: 
            Array_to_VTK(data_new.transpose(2, 1, 0).astype('float'), filename='***.vtk')
        
        Inputs: 
            data            3D array 
            new_coords      (3,n) array, targeted coordinates
            ori_ax_1/2/3    Monotonic axes of the original coordinates
            PadSize         (3,) size of the padded array.
            PadIncr         (3,) increaments of the padding in three dimension. 
        
        Output: 
            data_new        Interpolated 3D array 
    '''
    # Input regularization
    if Progress: 
        print('>>>>>> Input regularization ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    PadIncr = np.asarray(PadIncr)
    dims = np.asarray(np.shape(data))
    if len(dims) == 3: 
        N_elem = int(dims[0]*dims[1]*dims[2])
    else: 
        print('>>>>>> The shape of "array" is not correct! ')
        input('>>>>>> Press any key to quit...')
        return
    
    if np.shape(new_coords)[1] == 3 and np.size(new_coords) != 9: 
        new_coords = np.transpose(new_coords)
    
    # Generate axes, if necessary
    if Progress: 
        print('>>>>>> Generate variables ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    if ori_ax_1 is None: 
        ori_ax_1 = np.arange(dims[0]) - int((dims[0]-1)/2)
    if ori_ax_2 is None: 
        ori_ax_2 = np.arange(dims[1]) - int((dims[1]-1)/2)
    if ori_ax_3 is None: 
        ori_ax_3 = np.arange(dims[2]) - int((dims[2]-1)/2)
    
    # Interpolation
    if Progress: 
        print('>>>>>> Calculate coordinates in original basis ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
        print('       Original array size is ', dims)
        
    Err_flag = True
    if (PadSize is None) or (not PadSize >= np.shape(data)):
        dims_new = dims.copy()
    else: 
        dims_new = np.asarray(PadSize)
    
    Flip_ax_1 = False
    Flip_ax_2 = False
    Flip_ax_3 = False
    
    while Err_flag: 
        data_pad = Array_zeropad_3D(data, AddTo=dims_new)
        if ori_ax_1 is None: 
            ori_ax_1 = np.arange(dims_new[0]) - int((dims_new[0]-1)/2)
        else: 
            ori_ax_1 = np.round(Extend_1D_array(ori_ax_1, addto_elm=dims_new[0]))
        if ori_ax_2 is None: 
            ori_ax_2 = np.arange(dims_new[1]) - int((dims_new[1]-1)/2)
        else: 
            ori_ax_2 = np.round(Extend_1D_array(ori_ax_2, addto_elm=dims_new[1]))
        if ori_ax_3 is None: 
            ori_ax_3 = np.arange(dims_new[2]) - int((dims_new[2]-1)/2)
        else: 
            ori_ax_3 = np.round(Extend_1D_array(ori_ax_3, addto_elm=dims_new[2]))
        
        # Check if the axes are monotonic
        try: 
            if Flip_ax_1: 
                ori_ax_1 = np.flip(ori_ax_1)
                data_pad = np.flip(data_pad, axis=0)
            if Flip_ax_2: 
                ori_ax_2 = np.flip(ori_ax_2)
                data_pad = np.flip(data_pad, axis=1)
            if Flip_ax_3: 
                ori_ax_3 = np.flip(ori_ax_3)
                data_pad = np.flip(data_pad, axis=2)
            TriSpline_function = Spline(ori_ax_1, ori_ax_2, ori_ax_3, data_pad)
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', flipped ... ')
            if ErrM.endswith('is not monotonic'): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    if Flip_ax_1 is True: 
                        print('>>>>>> Flip_ax_1 is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_ax_1 = True
                if Ax == 'y': 
                    if Flip_ax_2 is True: 
                        print('>>>>>> Flip_ax_2 is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_ax_2 = True
                if Ax == 'z': 
                    if Flip_ax_3 is True: 
                        print('>>>>>> Flip_ax_3 is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_ax_3 = True
                continue
            else: 
                print(ErrM)
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
        
        # Check if the ranges are too narrow
        try: 
            data_new = TriSpline_function.ev(np.ravel(new_coords[0]), 
                                             np.ravel(new_coords[1]), 
                                             np.ravel(new_coords[2]))
            if Progress: 
                print('')
                print('       Padded array size is   ', np.asarray(np.shape(data_pad)))
            data_new = np.reshape(data_new, dims)
            Err_flag = False
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', padding ', dims_new-dims, end='\r')
            if ErrM.endswith('exceeds bounds of interpolation grid '): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'z': 
                    dims_new[0] = dims_new[0] + PadIncr[0]
                if Ax == 'y': 
                    dims_new[1] = dims_new[1] + PadIncr[1]
                if Ax == 'x': 
                    dims_new[2] = dims_new[2] + PadIncr[2]
            else: 
                print(ErrM)
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
    
    if Progress: 
        print('>>>>>> Interpolation finished.    ' 
              + strftime('%H:%M:%S', localtime()))
        print('       Output array size is   ', np.asarray(np.shape(data_new)))
        print('       Flip_ax_1 = %d ' %Flip_ax_1)
        print('       Flip_ax_2 = %d ' %Flip_ax_2)
        print('       Flip_ax_3 = %d ' %Flip_ax_3)
        
        t_4 = time()
        
        plt.figure(figsize=(7.5, 3))
        plt.subplot(131)
        plt.plot(np.ravel(new_coords[0]), alpha=0.5)
        plt.plot(ori_ax_1, alpha=0.5)
        plt.title('Axis 1')
        
        plt.subplot(132)
        plt.plot(np.ravel(new_coords[1]), alpha=0.5)
        plt.plot(ori_ax_2, alpha=0.5)
        plt.title('Axis 2')
        
        plt.subplot(133)
        plt.plot(np.ravel(new_coords[2]), alpha=0.5)
        plt.plot(ori_ax_3, alpha=0.5)
        plt.title('Axis 3')
        
        plt.tight_layout()
    
    return data_new


def Array_interpolation_3D_rgi(data, new_coords, method='cubic', target_dims=None, 
                               ori_ax_1=None, ori_ax_2=None, ori_ax_3=None, 
                               PadSize=None, PadIncr=[5,5,5], Progress=False): 
    ''' >>> Instruction <<< 
        This function interpolates a 3D array from grid coordinates to arbritrary coordinates, 
        using scipy.interpolate.RegularGridInterpolator 
        
        To display the result in ParaView, using: 
            Array_to_VTK(data_new.transpose(2, 1, 0).astype('float'), filename='***.vtk')
        
        Inputs: 
            data            3D array 
            new_coords      (3,n) array, targeted coordinates
            method          “linear”, “nearest”, “slinear”, “cubic”, and “quintic”
            ori_ax_1/2/3    Monotonic axes of the original coordinates
            PadSize         (3,) size of the padded array.
            PadIncr         (3,) increaments of the padding in three dimension. 
        
        Output: 
            data_new        Interpolated 3D array 
    '''
    # Input regularization
    if Progress: 
        print('>>>>>> Input regularization ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    PadIncr = np.asarray(PadIncr)
    dims = np.asarray(np.shape(data))
    if len(dims) == 3: 
        N_elem = int(dims[0]*dims[1]*dims[2])
    else: 
        print('>>>>>> The shape of "array" is not correct! ')
        input('>>>>>> Press any key to quit...')
        return
    
    if np.shape(new_coords)[1] == 3 and np.size(new_coords) != 9: 
        new_coords = np.transpose(new_coords)
    
    # Generate axes, if necessary
    if Progress: 
        print('>>>>>> Generate variables ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    if ori_ax_1 is None: 
        ori_ax_1 = np.arange(dims[0]) - int((dims[0]-1)/2)
    if ori_ax_2 is None: 
        ori_ax_2 = np.arange(dims[1]) - int((dims[1]-1)/2)
    if ori_ax_3 is None: 
        ori_ax_3 = np.arange(dims[2]) - int((dims[2]-1)/2)
    
    # Interpolation
    if Progress: 
        print('>>>>>> Calculate coordinates in original basis ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
        print('       Original array size is ', dims)
        
    if (PadSize is None) or (not PadSize >= np.shape(data)):
        dims_new = dims.copy()
    else: 
        dims_new = np.asarray(PadSize)
    
    if target_dims is None: 
        target_dims = dims.copy()
    
    Err_flag = True
    while Err_flag: 
        data_pad = Array_zeropad_3D(data, AddTo=dims_new)
        if ori_ax_1 is None: 
            ori_ax_1 = np.arange(dims_new[0]) - int((dims_new[0]-1)/2)
        else: 
            ori_ax_1 = Extend_1D_array(ori_ax_1, addto_elm=dims_new[0])
        if ori_ax_2 is None: 
            ori_ax_2 = np.arange(dims_new[1]) - int((dims_new[1]-1)/2)
        else: 
            ori_ax_2 = Extend_1D_array(ori_ax_2, addto_elm=dims_new[1])
        if ori_ax_3 is None: 
            ori_ax_3 = np.arange(dims_new[2]) - int((dims_new[2]-1)/2)
        else: 
            ori_ax_3 = Extend_1D_array(ori_ax_3, addto_elm=dims_new[2])
        
        # Define an interpolating function from data
        interp_function = rgi((ori_ax_1, ori_ax_2, ori_ax_3), data_pad, method=method)
        
        # Evaluate the interpolating function on new matrix
        try: 
            data_new = interp_function(new_coords.transpose())
            if Progress: 
                print('')
                print('       Padded array size is   ', np.asarray(np.shape(data_pad)))
            data_new = np.reshape(data_new, target_dims)
            Err_flag = False
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', padding ', dims_new-dims, end='\r')
            if ErrM.startswith('One of the requested xi is out of bounds'): 
                Ax_idx = int(ErrM.split(' ')[-1])
                dims_new[Ax_idx] = dims_new[Ax_idx] + PadIncr[Ax_idx]
            else: 
                print(ErrM)
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
    
    if Progress: 
        print('>>>>>> Interpolation finished.    ' 
              + strftime('%H:%M:%S', localtime()))
        print('       Output array size is   ', np.asarray(np.shape(data_new)))
        
        t_4 = time()
        
        plt.figure(figsize=(7.5, 3))
        plt.subplot(131)
        plt.plot(np.ravel(new_coords[0]), alpha=0.5)
        plt.plot(ori_ax_1, alpha=0.5)
        plt.title('Axis 1')
        
        plt.subplot(132)
        plt.plot(np.ravel(new_coords[1]), alpha=0.5)
        plt.plot(ori_ax_2, alpha=0.5)
        plt.title('Axis 2')
        
        plt.subplot(133)
        plt.plot(np.ravel(new_coords[2]), alpha=0.5)
        plt.plot(ori_ax_3, alpha=0.5)
        plt.title('Axis 3')
        
        plt.tight_layout()
    
    return data_new


def Array_phase_ramp_removal(data, upsample=5, com_offset=0.0, flag_intshift=False): 
    ''' >>> Instruction <<< 
        This function remove phase ramp of a complex array by recentering in reciprocal space. 
        
        Inputs: 
            data            n-D array 
            upsample        real space zeropadding ratio
            com_offset      tweak center of mass, usually keep it zero. 
            flag_intshift   if True, reciprocal array is shifted to int pixels
        
        Output: 
            data_new        n-D array after phase ramp removal 
    '''
    data = np.asarray(data)
    dims = np.asarray(np.shape(data))
    new_dims = (dims * upsample).astype('int')
    befores = np.floor( (new_dims - dims)/2 ).astype('int')
    afters  = np.ceil(  (new_dims - dims)/2 ).astype('int')
    data_new = np.pad(data, np.concatenate(([befores], [afters]), axis=0).transpose() )
    data_new = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data_new)))
    com = np.asarray(simg.center_of_mass(np.abs(data_new)**4)) + com_offset
    off_cen = dims/2 - com + befores
    off_cen /= upsample
    del data_new
    temp = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data)))
    if flag_intshift: 
        temp = np.roll(temp, shift=tuple(np.round(off_cen).astype('int')))
    else: 
        temp = np.fft.ifftn(simg.fourier_shift(np.fft.fftn(temp), shift=off_cen))
    data_new = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(temp)))
    data_new = np.abs(data_new) * np.exp(1j * (np.angle(data_new) - np.angle(data_new[tuple((dims/2).astype('int'))])) )
    return data_new


def Array_line_coords_2D(arrdims=[50,60], line_dim=100, line_dist=0.0, line_angle=35.0, nebor_rang=2.0, 
                         flag_plot=False, idx_step=7, flag_debug=False): 
    ''' >>> Instruction <<< 
        This function calclates the 2D coordinates of a line in a 2D grid. 
        Notice: 
            1) Assuming the 2D grid is equally spaced in all dimensions
            2) Assuming the 2D grid is centered at [0, 0]
            3) Assuming the 1D line has the same spacing as the 2D grid
            4) Line is defined by the distance from the grid center and the tilting angle
        
        Inputs: 
            arrdims         dimensions of the 2D grid 
            line_dim        dimension of the line
            line_dist       distance from the grid center to the line
            line_angle      [deg], line tilting angle
            nebor_rang      define the range of grid points around the slice
            flag_plot       if True, plot the line over the 2D image
            idx_step        step when plot grid points in 3D
        
        Output: 
            slice_coords    (n, 2) array, 2D coords of the line
            slice_idx       (n, ) array, idx of pts on the slice that are inside the 3D grid
            slice_nebors    (m, 3) array, grid points around the slice
            
    '''
    # input regulation
    line_dim   = int(line_dim)
    line_dist  = float(line_dist)
    line_angle = float(line_angle)
    nebor_rang = float(nebor_rang)
    if flag_debug: 
        print('line_dist : ', line_dist)
        print('line_angle: ', line_angle, ' [deg]')
    
    # Generate grid coordinates 
    x = np.arange(arrdims[0]) - int(arrdims[0]/2)   # x axis of the grid
    y = np.arange(arrdims[1]) - int(arrdims[1]/2)   # y axis of the grid
    y2, x2 = np.meshgrid(y, x)   # 2D meshgrid
    if flag_debug: 
        print('----------------')
        print('x:  ', np.shape(x))
        print('y:  ', np.shape(y))
        print('x2: ', np.shape(x2))
        print('y2: ', np.shape(y2))
    grid_coords = np.stack((x2.ravel(), y2.ravel()), axis=0).transpose()
    N = np.max(np.shape(grid_coords))
    # 2D base of the slice 
    grid_unit_x = np.asarray([1,0])
    grid_unit_y = np.asarray([0,1])
    line_unit = np.asarray([np.cos(np.deg2rad(line_angle)), 
                            np.sin(np.deg2rad(line_angle))])
    # Generate line coordinates
    lx = np.arange(line_dim) - int(line_dim/2)   # axis of the line
    line_coords = np.matmul(np.asarray([lx]).transpose(), np.asarray([line_unit]))
    if flag_debug: 
        print('----------------')
        print('grid_coords: ', np.shape(grid_coords))
        print('line_unit:   ', np.shape(line_unit), line_unit)
        print('lx :         ', np.shape(lx))
        print('line_coords: ', np.shape(line_coords))
    line_coords[:,0] += np.sin(np.deg2rad(line_angle)) * line_dist
    line_coords[:,1] -= np.cos(np.deg2rad(line_angle)) * line_dist
    # identify points inside the grid 
    idx = np.where( (line_coords[:,0]>=np.amin(x)) & (line_coords[:,0]<=np.amax(x)) & 
                    (line_coords[:,1]>=np.amin(y)) & (line_coords[:,1]<=np.amax(y))  )
    line_idx = np.zeros(np.max(np.shape(line_coords)))
    line_idx[np.squeeze(idx)] = 1
    
    # line functions
    cent_proj = np.asarray([ np.sin(np.deg2rad(line_angle)), 
                            -np.cos(np.deg2rad(line_angle))]) * (line_dist)
    if flag_debug: 
        print('----------------')
        print('cent_proj:    ', cent_proj)
        print('line:         ', Line_func('Angle_and_Point', line_angle, cent_proj))
        
    cent_proj = np.asarray([ np.sin(np.deg2rad(line_angle)), 
                            -np.cos(np.deg2rad(line_angle))]) * (line_dist - nebor_rang)
    line_1  = Line_func('Angle_and_Point', line_angle, cent_proj)
    pixel_side_1 = (  line_1[0] * grid_coords[:, 0] 
                    + line_1[1] * grid_coords[:, 1] 
                    + line_1[2] )
    if flag_debug: 
        print('----------------')
        print('cent_proj:    ', cent_proj)
        print('line_1:       ', line_1)
        print('pixel_side_1: ', np.shape(pixel_side_1), pixel_side_1)
    cent_proj = np.asarray([ np.sin(np.deg2rad(line_angle)), 
                            -np.cos(np.deg2rad(line_angle))]) * (line_dist + nebor_rang)
    line_2  = Line_func('Angle_and_Point', line_angle, cent_proj)
    pixel_side_2 = (  line_2[0] * grid_coords[:, 0] 
                    + line_2[1] * grid_coords[:, 1] 
                    + line_2[2] )
    if flag_debug: 
        print('----------------')
        print('cent_proj:    ', cent_proj)
        print('line_2:       ', line_2)
        print('pixel_side_2: ', np.shape(pixel_side_2), pixel_side_2)
    # identify neighbors
    line_nebors_idx = np.zeros(N)
    idx = np.squeeze(np.where(pixel_side_1 * pixel_side_2 <= 0))
    line_nebors_idx[idx] = 1
    line_nebors = grid_coords[idx, :]
    if flag_debug: 
        print('----------------')
        print('idx :            ', np.shape(idx))
        print('line_nebors_idx: ', np.shape(line_nebors_idx))
        print('line_nebors:     ', np.shape(line_nebors))
    if flag_plot: 
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(111)
        ax1.scatter(line_coords[:,0], line_coords[:,1], 
                    linewidth=0, marker='.',alpha=0.5, antialiased=True)
        ax1.scatter(grid_coords[1:-1:idx_step, 0], 
                    grid_coords[1:-1:idx_step, 1], 
                    linewidth=0, marker='.',alpha=0.2, antialiased=True)
        ax1.scatter(line_coords[np.where(line_idx),0], 
                    line_coords[np.where(line_idx),1], 
                    linewidth=0, marker='.',alpha=0.8, antialiased=True)
        ax1.imshow((pixel_side_1 * pixel_side_2 / 
                    np.abs(pixel_side_1 * pixel_side_2)).reshape(arrdims).transpose(), 
                   extent=[np.amin(x), np.amax(x), np.amin(y), np.amax(y)],
                   origin='lower', alpha=0.2)
        ax1_fov = np.max(arrdims)
        ax1.set_xlim([-ax1_fov/2*1.1, ax1_fov/2*1.1])
        ax1.set_ylim([-ax1_fov/2*1.1, ax1_fov/2*1.1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.tight_layout()
    return {'line_coords': line_coords, 'line_idx': line_idx, 
            'line_nebors': line_nebors, 'line_nebors_idx': line_nebors_idx}


def Array_slice_coords_3D(arrdims=[50,60,70], slice_dims=[100,100], slice_dist=0.0, slice_norm=[1,0,0], nebor_rang=2.0, 
                          flag_nebor=False, flag_plot=False, idx_step=7, flag_debug=False): 
    ''' >>> Instruction <<< 
        This function calclates the 3D coordinates of a 2D slice in a 3D grid. 
        Notice: 
            1) Assuming the 3D grid is equally spaced in all dimensions
            2) Assuming the 3D grid is centered at [0, 0, 0]
            3) Assuming the 2D slice has the same spacing as the 3D grid
            4) Slice is defined by the distance from the grid center and the normal
        
        Inputs: 
            arrdims         dimensions of the 3D grid 
            slice_dims      dimensions of the 2D slice
            slice_dist      distance from the grid center to the slice
            slice_norm      normal direction of the slice
            nebor_rang      define the range of grid points around the slice
            flag_nebor      if True, identify grid points around the slice
            flag_plot       if True, plot the slice in grid
            idx_step        step when plot grid points in 3D
        
        Output: 
            slice_coords    (n, 3) array, 3D coords of the slice
            slice_idx       (n, ) array, idx of pts on the slice that are inside the 3D grid
            slice_nebors    (m, 3) array, grid points around the slice
            
    '''
    # input regulation
    slice_dims = np.asarray(slice_dims)
    slice_norm = np.asarray(slice_norm) / np.linalg.norm(slice_norm)
    slice_dist = float(slice_dist)
    nebor_rang = float(nebor_rang)
    if flag_debug: 
        print('slice_norm: ', slice_norm)
    
    # Generate grid coordinates 
    x = np.arange(arrdims[0]) - int(arrdims[0]/2)   # x axis of the grid
    y = np.arange(arrdims[1]) - int(arrdims[1]/2)   # y axis of the grid
    z = np.arange(arrdims[2]) - int(arrdims[2]/2)   # z axis of the grid
    y3, z3, x3 = np.meshgrid(y, z, x)   # 3D meshgrid
    grid_coords = np.stack((x3.ravel(), y3.ravel()), axis=0)
    grid_coords = np.concatenate((grid_coords, [z3.ravel()]), axis=0).transpose()
    N = np.max(np.shape(grid_coords))
    # 3D base of the slice 
    grid_unit_x = np.asarray([1,0,0])
    grid_unit_y = np.asarray([0,1,0])
    grid_unit_z = np.asarray([0,0,1])
    temp_unit_x = grid_unit_x - slice_norm * np.dot(grid_unit_x, slice_norm)
    temp_unit_y = grid_unit_y - slice_norm * np.dot(grid_unit_y, slice_norm)
    temp_unit_z = grid_unit_z - slice_norm * np.dot(grid_unit_z, slice_norm)
    if flag_debug: 
        print('temp_unit_x: ', temp_unit_x)
        print('temp_unit_y: ', temp_unit_y)
        print('temp_unit_z: ', temp_unit_z)
    if   np.linalg.norm(temp_unit_x) != 0: 
        slice_unit_x = temp_unit_x
    elif np.linalg.norm(temp_unit_y) != 0: 
        slice_unit_x = temp_unit_y
    elif np.linalg.norm(temp_unit_z) != 0:
        slice_unit_x = temp_unit_z
    else: 
        print('>>>>>> Error! Unknown slice_unit_vector ')
        input('>>>>>> Press any key to quit...')
        return
    slice_unit_x /= np.linalg.norm(slice_unit_x)
    slice_unit_y = np.cross(slice_norm, slice_unit_x)
    slice_unit_y /= np.linalg.norm(slice_unit_y)
    if flag_debug: 
        print('slice_unit_x: ', slice_unit_x)
        print('slice_unit_y: ', slice_unit_y)
    # Generate slice coordinates
    sx = np.arange(slice_dims[0]) - int(slice_dims[0]/2)   # x axis of the slice
    sy = np.arange(slice_dims[1]) - int(slice_dims[1]/2)   # y axis of the slice
    sy2, sx2 = np.meshgrid(sy, sx)   # 2D meshgrid
    slice_coords = np.matmul(np.asarray([sx2.ravel()]).transpose(), np.asarray([slice_unit_x]))\
                 + np.matmul(np.asarray([sy2.ravel()]).transpose(), np.asarray([slice_unit_y]))
    slice_coords[:,0] += (slice_norm * slice_dist)[0]
    slice_coords[:,1] += (slice_norm * slice_dist)[1]
    slice_coords[:,2] += (slice_norm * slice_dist)[2]
    # identify points inside the grid 
    idx = np.where( (slice_coords[:,0]>=np.amin(x)) & (slice_coords[:,0]<=np.amax(x)) & 
                    (slice_coords[:,1]>=np.amin(y)) & (slice_coords[:,1]<=np.amax(y)) & 
                    (slice_coords[:,2]>=np.amin(z)) & (slice_coords[:,2]<=np.amax(z))   )
    slice_idx = np.zeros(np.max(np.shape(slice_coords)))
    slice_idx[np.squeeze(idx)] = 1
    if flag_nebor: 
        # plane functions
        cent_proj = slice_norm * (slice_dist - nebor_rang)
        plane_1  = Plane_func('Normal_and_Point', slice_norm, cent_proj)
        voxel_side_1 = (  plane_1[0] * grid_coords[:, 0] 
                        + plane_1[1] * grid_coords[:, 1] 
                        + plane_1[2] * grid_coords[:, 2] 
                        + plane_1[3] )
        cent_proj = slice_norm * (slice_dist + nebor_rang)
        plane_2  = Plane_func('Normal_and_Point', slice_norm, cent_proj)
        voxel_side_2 = (  plane_2[0] * grid_coords[:, 0] 
                        + plane_2[1] * grid_coords[:, 1] 
                        + plane_2[2] * grid_coords[:, 2] 
                        + plane_2[3] )
        # identify neighbors
        slice_nebors_idx = np.zeros(N)
        idx = np.squeeze(np.where(voxel_side_1 * voxel_side_2 <= 0))
        slice_nebors_idx[idx] = 1
        slice_nebors = grid_coords[idx, :]
    if flag_plot: 
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(elev=-10, azim=0)
        surf_slice = ax1.scatter3D(slice_coords[:,0], 
                                   slice_coords[:,1], 
                                   slice_coords[:,2], 
                                   linewidth=0, marker='.',alpha=0.5, antialiased=True)
        surf_grid = ax1.scatter3D(grid_coords[1:-1:idx_step, 0], 
                                  grid_coords[1:-1:idx_step, 1], 
                                  grid_coords[1:-1:idx_step, 2], 
                                  linewidth=0, marker='.',alpha=0.2, antialiased=True)
        # ax1.legend([surf_inc, surf_ext], ['incident surf', 'exit surf'], numpoints = 1)
        ax1_fov = np.max(arrdims)
        ax1.set_xlim3d([-ax1_fov/2*1.3, ax1_fov/2*1.3])
        ax1.set_ylim3d([-ax1_fov/2*1.3, ax1_fov/2*1.3])
        ax1.set_zlim3d([-ax1_fov/2,     ax1_fov/2])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.tight_layout()
    if flag_nebor: 
        return {'slice_coords': slice_coords, 'slice_idx': slice_idx, 
                'slice_nebors': slice_nebors, 'slice_nebors_idx': slice_nebors_idx}
    return {'slice_coords': slice_coords, 'slice_idx': slice_idx}


def Array_arbitrary_line_2D_rbf(data, line_dim=100, line_dist=0.0, line_angle=35.0, nebor_rang=2.0, 
                                interp_seg=1, kernel='linear', smoothing=0, 
                                flag_plot=False, flag_debug=False, flag_info=False):
    ''' >>> Instruction <<< 
        This function calclates the 2D coordinates of a line in a 2D grid, 
        using scipy.interpolate.RBFInterpolator. 
        
        Inputs: 
            data            2D array 
            line_dim        dimension of the line
            line_dist       distance from the array center to the line
            line_angle      [deg], line tilting angle
            nebor_rang      define the range of array points around the line
            interp_seg      divide the line into segments (each axis) to speed up interpolation
            kernel          scipy RBFInterpolator kernels, including: linear, thin_plate_spline, cubic, 
                            quintic, multiquadric, inverse_multiquadric, inverse_quadratic, gaussian.
            smoothing       scipy RBFInterpolator smoothing parameter, default is 0. 
            flag_plot       if True, plot the line in grid
            flag_debug      if True, skip interpolation
            flag_info       if True, display running time
        
        Output: 
            arb_line        1D line
    '''
    # input regulation
    data = np.asarray(data)
    dims = np.shape(data)
    line_dim   = int(line_dim)
    line_dist  = float(line_dist)
    line_angle = float(line_angle)
    nebor_rang = float(nebor_rang)
    if (np.sin(np.deg2rad(line_angle)) == 0.0) or (np.cos(np.deg2rad(line_angle)) == 0.0): 
        print('>>>>>> Do not require interpolation. ')
        input('>>>>>> Press any key to quit...')
        return
    if flag_debug: 
        print('line_dist : ', line_dist)
        print('line_angle: ', line_angle, ' [deg]')
    # get coords of the slice
    temp = Array_line_coords_2D(arrdims=dims, nebor_rang=nebor_rang, 
                                line_dim=line_dim, line_dist=line_dist, line_angle=line_angle, 
                                flag_plot=False, idx_step=1, flag_debug=False)
    line_coords = temp['line_coords']
    line_idx = temp['line_idx']
    line_nebors = temp['line_nebors']
    line_nebors_idx = temp['line_nebors_idx']
    del temp
    nebor_data = data[np.where(line_nebors_idx.reshape(dims))]
    if flag_debug: 
        print('line_coords     dims: ', np.shape(line_coords))
        print('line_idx        dims: ', np.shape(line_idx))
        print('line_nebors     dims: ', np.shape(line_nebors))
        print('line_nebors_idx dims: ', np.shape(line_nebors_idx))
        print('data size:            ', np.size(data))
        print('nebor_data size:      ', np.shape(nebor_data))
    # interpolation
    if not flag_debug: 
        arb_line = np.zeros(line_dim).ravel()
        temp_idx_lin = np.squeeze(np.where(line_idx))
        line_x_limits = [np.min(line_coords[temp_idx_lin,0]), np.max(line_coords[temp_idx_lin,0])]
        line_x_seg    = np.squeeze(np.diff(line_x_limits) / interp_seg)
        line_y_limits = [np.min(line_coords[temp_idx_lin,1]), np.max(line_coords[temp_idx_lin,1])]
        line_y_seg    = np.squeeze(np.diff(line_y_limits) / interp_seg)
        del temp_idx_lin
        counter = 0
        line_idx = line_idx==1   # convert slice idx from 1/0 to True/Flase
        if flag_info: 
            print(' Start ... ' + strftime('%H:%M:%S', localtime()), end='\n')
        for ii in range(interp_seg): 
            for jj in range(interp_seg): 
                counter += 1
                if flag_info: 
                    print(' Interpolating %d / %d ... ' %(counter, interp_seg**2) + strftime('%H:%M:%S', localtime()), end='\r')
                temp_idx_lin = np.where( ( line_coords[:,0]>=(line_x_limits[0] +  ii   *line_x_seg) ) & 
                                         ( line_coords[:,0]<=(line_x_limits[0] + (ii+1)*line_x_seg) ) & 
                                         ( line_coords[:,1]>=(line_y_limits[0] +  jj   *line_y_seg) ) & 
                                         ( line_coords[:,1]<=(line_y_limits[0] + (jj+1)*line_y_seg) ) & line_idx )
                temp_idx_lin = np.squeeze(temp_idx_lin)
                if np.size(temp_idx_lin)==0: 
                    continue
                temp_idx_neb = np.where( ( line_nebors[:,0]>=(line_x_limits[0] +  ii   *line_x_seg - nebor_rang) ) & 
                                         ( line_nebors[:,0]<=(line_x_limits[0] + (ii+1)*line_x_seg + nebor_rang) ) & 
                                         ( line_nebors[:,1]>=(line_y_limits[0] +  jj   *line_y_seg - nebor_rang) ) & 
                                         ( line_nebors[:,1]<=(line_y_limits[0] + (jj+1)*line_y_seg + nebor_rang) )  )
                temp_idx_neb = np.squeeze(temp_idx_neb)
                if np.size(temp_idx_neb)==0: 
                    continue
                fun = sin.RBFInterpolator(line_nebors[temp_idx_neb,:], nebor_data[temp_idx_neb], smoothing=smoothing, kernel=kernel)
                arb_line[temp_idx_lin] = fun(line_coords[temp_idx_lin,:])
        if flag_info: 
            print(' Finished interpolation %d / %d ... ' %(counter, interp_seg**2) + strftime('%H:%M:%S', localtime()), end='\n')
    # plot and output
    if flag_plot: 
        nebors = line_nebors_idx.reshape((dims))
        plt.figure(figsize=(6,6))
        plt.imshow(nebors) 
        plt.imshow(data, alpha=0.5) 
        plt.colorbar()
        plt.tight_layout() 
    if not flag_debug: 
        return arb_line
    return


def Array_arbitrary_slice_3D_rgi(data, slice_dims=[100,100], slice_dist=0.0, slice_norm=[1,0,0], nebor_rang=2.0, 
                                 interp_seg=10, method='linear', flag_plot=False, flag_debug=False, flag_info=False): 
    ''' >>> Instruction <<< 
        This function calclates the 3D coordinates of a 2D slice in a 3D grid, 
        using scipy.interpolate.RegularGridInterpolator
        
        Inputs: 
            data            3D array 
            slice_dims      dimensions of the 2D slice
            slice_dist      distance from the array center to the slice
            slice_norm      normal direction of the slice
            nebor_rang      define the range of array points around the slice
            interp_seg      divide 2D slice into segments (each axis) to speed up interpolation
            method          scipy RegularGridInterpolator methods, including: 
                                'linear', 'nearest', 'slinear', 'cubic', 'quintic' and 'pchip'.
            flag_plot       if True, plot the slice in grid
            flag_debug      if True, skip interpolation
            flag_info       if True, display running time
        
        Output: 
            arb_slice       2D slice
    '''
    # input regulation
    data = np.asarray(data)
    dims = np.shape(data)
    slice_dims = np.asarray(slice_dims)
    slice_norm = np.asarray(slice_norm) / np.linalg.norm(slice_norm)
    slice_dist = float(slice_dist)
    nebor_rang = float(nebor_rang)
    if np.array_equal(slice_norm, [1,0,0]) or np.array_equal(slice_norm, [0,1,0]) or np.array_equal(slice_norm, [0,0,1]): 
        print('>>>>>> Do not require interpolation. ')
        input('>>>>>> Press any key to quit...')
        return
    if flag_debug: 
        print('slice_norm: ', slice_norm)
    # get coords of the slice
    temp = Array_slice_coords_3D(arrdims=dims, nebor_rang=nebor_rang, 
                                 slice_dims=slice_dims, slice_dist=slice_dist, slice_norm=slice_norm, 
                                 flag_nebor=True, flag_plot=False, idx_step=7, flag_debug=False)
    slice_coords = temp['slice_coords']
    slice_idx = temp['slice_idx']
    slice_nebors = temp['slice_nebors']
    slice_nebors_idx = temp['slice_nebors_idx']
    del temp
    nebor_data = data.ravel()[np.where(slice_nebors_idx)]
    if flag_debug: 
        print('slice_coords     dims: ', np.shape(slice_coords))
        print('slice_idx        dims: ', np.shape(slice_idx))
        print('data size:             ', np.size(data))
    # interpolation
    if not flag_debug: 
        arb_slice = np.zeros(slice_dims).ravel()
        temp_idx_slc = np.squeeze(np.where(slice_idx))
        slice_x_limits = [np.min(slice_coords[temp_idx_slc,0]), np.max(slice_coords[temp_idx_slc,0])]
        slice_x_seg    = np.squeeze(np.diff(slice_x_limits) / interp_seg)
        slice_y_limits = [np.min(slice_coords[temp_idx_slc,1]), np.max(slice_coords[temp_idx_slc,1])]
        slice_y_seg    = np.squeeze(np.diff(slice_y_limits) / interp_seg)
        slice_z_limits = [np.min(slice_coords[temp_idx_slc,2]), np.max(slice_coords[temp_idx_slc,2])]
        slice_z_seg    = np.squeeze(np.diff(slice_z_limits) / interp_seg)
        del temp_idx_slc
        counter = 0
        slice_idx = slice_idx==1   # convert slice idx from 1/0 to True/Flase
        if flag_info: 
            print(' Start ... ' + strftime('%H:%M:%S', localtime()), end='\n')
        for ii in range(interp_seg): 
            for jj in range(interp_seg): 
                for kk in range(interp_seg): 
                    counter += 1
                    if flag_info: 
                        print(' Interpolating %d / %d ... ' %(counter, interp_seg**3) + strftime('%H:%M:%S', localtime()), end='\r')
                    temp_idx_slc = np.where( ( slice_coords[:,0]>=(slice_x_limits[0] +  ii   *slice_x_seg) ) & 
                                             ( slice_coords[:,0]<=(slice_x_limits[0] + (ii+1)*slice_x_seg) ) & 
                                             ( slice_coords[:,1]>=(slice_y_limits[0] +  jj   *slice_y_seg) ) & 
                                             ( slice_coords[:,1]<=(slice_y_limits[0] + (jj+1)*slice_y_seg) ) & 
                                             ( slice_coords[:,2]>=(slice_z_limits[0] +  kk   *slice_z_seg) ) & 
                                             ( slice_coords[:,2]<=(slice_z_limits[0] + (kk+1)*slice_z_seg) ) & slice_idx )
                    temp_idx_slc = np.squeeze(temp_idx_slc)
                    if np.size(temp_idx_slc)==0: 
                        continue
                    temp_data_x = np.arange(int(np.floor(slice_x_limits[0] +  ii   *slice_x_seg - nebor_rang)), 
                                            int(np.ceil (slice_x_limits[0] + (ii+1)*slice_x_seg + nebor_rang)))
                    temp_data_y = np.arange(int(np.floor(slice_y_limits[0] +  jj   *slice_y_seg - nebor_rang)), 
                                            int(np.ceil (slice_y_limits[0] + (jj+1)*slice_y_seg + nebor_rang)))
                    temp_data_z = np.arange(int(np.floor(slice_z_limits[0] +  kk   *slice_z_seg - nebor_rang)), 
                                            int(np.ceil (slice_z_limits[0] + (kk+1)*slice_z_seg + nebor_rang)))
                    if (np.size(temp_data_x)<=1) or (np.size(temp_data_y)<=1) or (np.size(temp_data_z)<=1): 
                        continue
                    fun = sin.RegularGridInterpolator((temp_data_x, temp_data_y, temp_data_z), 
                                                      data[(temp_data_x[0] + int(dims[0]/2)):(temp_data_x[-1] + int(dims[0]/2 +1)), 
                                                           (temp_data_y[0] + int(dims[1]/2)):(temp_data_y[-1] + int(dims[1]/2 +1)), 
                                                           (temp_data_z[0] + int(dims[2]/2)):(temp_data_z[-1] + int(dims[2]/2 +1))], 
                                                      method=method)
                    arb_slice[temp_idx_slc] = fun(slice_coords[temp_idx_slc,:])
        if flag_info: 
            print(' Finished interpolation %d / %d ... ' %(counter, interp_seg**3) + strftime('%H:%M:%S', localtime()), end='\n')
    # plot and output
    if flag_plot: 
        nebors = slice_nebors_idx.reshape((dims[2],dims[1],dims[0])).transpose(2,1,0)
        plt.figure(figsize=(12,3.5))
        plt.subplot(131) 
        plt.imshow(data[int(dims[0]/2),:,:], cmap="bwr") 
        plt.imshow(nebors[int(dims[0]/2),:,:], alpha=0.5) 
        plt.tight_layout() 
        plt.subplot(132) 
        plt.imshow(data[:,int(dims[1]/2),:], cmap="bwr") 
        plt.imshow(nebors[:,int(dims[1]/2),:], alpha=0.5) 
        plt.tight_layout()
        plt.subplot(133) 
        plt.imshow(data[:,:,int(dims[2]/2)], cmap="bwr") 
        plt.imshow(nebors[:,:,int(dims[2]/2)], alpha=0.5) 
        plt.tight_layout() 
        plt.colorbar()
    if not flag_debug: 
        return arb_slice.reshape(slice_dims)
    return


def Array_arbitrary_slice_3D_rbf(data, slice_dims=[100,100], slice_dist=0.0, slice_norm=[1,0,0], nebor_rang=2.0, 
                                 interp_seg=10, kernel='linear', smoothing=0, 
                                 flag_plot=False, flag_debug=False, flag_info=False): 
    ''' >>> Instruction <<< 
        This function calclates the 3D coordinates of a 2D slice in a 3D grid, 
        using scipy.interpolate.RBFInterpolator 
        
        Inputs: 
            data            3D array 
            slice_dims      dimensions of the 2D slice
            slice_dist      distance from the array center to the slice
            slice_norm      normal direction of the slice
            nebor_rang      define the range of array points around the slice
            interp_seg      divide 2D slice into segments (each axis) to speed up interpolation
            kernel          scipy RBFInterpolator kernels, including: linear, thin_plate_spline, cubic, 
                            quintic, multiquadric, inverse_multiquadric, inverse_quadratic, gaussian.
            smoothing       scipy RBFInterpolator smoothing parameter, default is 0. 
            flag_plot       if True, plot the slice in grid
            flag_debug      if True, skip interpolation
            flag_info       if True, display running time
        
        Output: 
            arb_slice       2D slice
    '''
    # input regulation
    data = np.asarray(data)
    dims = np.shape(data)
    slice_dims = np.asarray(slice_dims)
    slice_norm = np.asarray(slice_norm) / np.linalg.norm(slice_norm)
    slice_dist = float(slice_dist)
    nebor_rang = float(nebor_rang)
    if np.array_equal(slice_norm, [1,0,0]) or np.array_equal(slice_norm, [0,1,0]) or np.array_equal(slice_norm, [0,0,1]): 
        print('>>>>>> Do not require interpolation. ')
        input('>>>>>> Press any key to quit...')
        return
    if flag_debug: 
        print('slice_norm: ', slice_norm)
    # get coords of the slice
    temp = Array_slice_coords_3D(arrdims=dims, nebor_rang=nebor_rang, 
                                 slice_dims=slice_dims, slice_dist=slice_dist, slice_norm=slice_norm, 
                                 flag_nebor=True, flag_plot=False, idx_step=7, flag_debug=False)
    slice_coords = temp['slice_coords']
    slice_idx = temp['slice_idx']
    slice_nebors = temp['slice_nebors']
    slice_nebors_idx = temp['slice_nebors_idx']
    del temp
    # nebor_data = data[np.where(slice_nebors_idx.reshape((dims[2],dims[1],dims[0])).transpose(2,1,0))]
    nebor_data = data.ravel()[np.ravel_multi_index(np.unravel_index(np.squeeze(np.where(slice_nebors_idx)), 
                                                                    np.shape(data),order='F'), 
                                                   np.shape(data), order='C')]
    if flag_debug: 
        print('slice_coords     dims: ', np.shape(slice_coords))
        print('slice_idx        dims: ', np.shape(slice_idx))
        print('slice_nebors     dims: ', np.shape(slice_nebors))
        print('slice_nebors_idx dims: ', np.shape(slice_nebors_idx))
        print('data size:             ', np.size(data))
        print('nebor_data size:       ', np.shape(nebor_data))
    # interpolation
    if not flag_debug: 
        arb_slice = np.zeros(slice_dims).ravel()
        temp_idx_slc = np.squeeze(np.where(slice_idx))
        slice_x_limits = [np.min(slice_coords[temp_idx_slc,0]), np.max(slice_coords[temp_idx_slc,0])]
        slice_x_seg    = np.squeeze(np.diff(slice_x_limits) / interp_seg)
        slice_y_limits = [np.min(slice_coords[temp_idx_slc,1]), np.max(slice_coords[temp_idx_slc,1])]
        slice_y_seg    = np.squeeze(np.diff(slice_y_limits) / interp_seg)
        slice_z_limits = [np.min(slice_coords[temp_idx_slc,2]), np.max(slice_coords[temp_idx_slc,2])]
        slice_z_seg    = np.squeeze(np.diff(slice_z_limits) / interp_seg)
        del temp_idx_slc
        counter = 0
        slice_idx = slice_idx==1   # convert slice idx from 1/0 to True/Flase
        if flag_info: 
            print(' Start ... ' + strftime('%H:%M:%S', localtime()), end='\n')
        for ii in range(interp_seg): 
            for jj in range(interp_seg): 
                for kk in range(interp_seg): 
                    counter += 1
                    if flag_info: 
                        print(' Interpolating %d / %d ... ' %(counter, interp_seg**3) + strftime('%H:%M:%S', localtime()), end='\r')
                    temp_idx_slc = np.where( ( slice_coords[:,0]>=(slice_x_limits[0] +  ii   *slice_x_seg) ) & 
                                             ( slice_coords[:,0]<=(slice_x_limits[0] + (ii+1)*slice_x_seg) ) & 
                                             ( slice_coords[:,1]>=(slice_y_limits[0] +  jj   *slice_y_seg) ) & 
                                             ( slice_coords[:,1]<=(slice_y_limits[0] + (jj+1)*slice_y_seg) ) & 
                                             ( slice_coords[:,2]>=(slice_z_limits[0] +  kk   *slice_z_seg) ) & 
                                             ( slice_coords[:,2]<=(slice_z_limits[0] + (kk+1)*slice_z_seg) ) & slice_idx )
                    temp_idx_slc = np.squeeze(temp_idx_slc)
                    if np.size(temp_idx_slc)==0: 
                        continue
                    temp_idx_neb = np.where( ( slice_nebors[:,0]>=(slice_x_limits[0] +  ii   *slice_x_seg - nebor_rang) ) & 
                                             ( slice_nebors[:,0]<=(slice_x_limits[0] + (ii+1)*slice_x_seg + nebor_rang) ) & 
                                             ( slice_nebors[:,1]>=(slice_y_limits[0] +  jj   *slice_y_seg - nebor_rang) ) & 
                                             ( slice_nebors[:,1]<=(slice_y_limits[0] + (jj+1)*slice_y_seg + nebor_rang) ) & 
                                             ( slice_nebors[:,2]>=(slice_z_limits[0] +  kk   *slice_z_seg - nebor_rang) ) & 
                                             ( slice_nebors[:,2]<=(slice_z_limits[0] + (kk+1)*slice_z_seg + nebor_rang) )  )
                    temp_idx_neb = np.squeeze(temp_idx_neb)
                    if np.size(temp_idx_neb)==0: 
                        continue
                    fun = sin.RBFInterpolator(slice_nebors[temp_idx_neb,:], nebor_data[temp_idx_neb], smoothing=smoothing, kernel=kernel)
                    arb_slice[temp_idx_slc] = fun(slice_coords[temp_idx_slc,:])
        if flag_info: 
            print(' Finished interpolation %d / %d ... ' %(counter, interp_seg**3) + strftime('%H:%M:%S', localtime()), end='\n')
    # plot and output
    if flag_plot: 
        nebors = slice_nebors_idx.reshape((dims[2],dims[1],dims[0])).transpose(2,1,0)
        plt.figure(figsize=(12,3.5))
        plt.subplot(131) 
        plt.imshow(data[int(dims[0]/2),:,:], cmap="bwr") 
        plt.imshow(nebors[int(dims[0]/2),:,:], alpha=0.5) 
        plt.tight_layout() 
        plt.subplot(132) 
        plt.imshow(data[:,int(dims[1]/2),:], cmap="bwr") 
        plt.imshow(nebors[:,int(dims[1]/2),:], alpha=0.5) 
        plt.tight_layout()
        plt.subplot(133) 
        plt.imshow(data[:,:,int(dims[2]/2)], cmap="bwr") 
        plt.imshow(nebors[:,:,int(dims[2]/2)], alpha=0.5) 
        plt.tight_layout() 
        plt.colorbar()
    if not flag_debug: 
        return arb_slice.reshape(slice_dims)
    return








# ======================================================================================================
# Functions saved for record.
# ======================================================================================================
def Frame_transform_matrix_Rot_nVec(V_original=np.zeros((3,1)), V_final=np.zeros((3,1)), Type='Signed'): 
    ''' >>> Instruction <<< 
        This function determines the rotation matrix from the one frame to another frame.
       To determine the orientation of a Cartesian coordinates, 2 noncollinear vectors 
       are required. 
       Assume the coordinates follow righthand rule. 
       
       Inputs: 
       V_original   (3, n) array, the original vectors
       V_final      (3, n) array, the final vectors
       Type         'Signed': The fitting will match the sign of the vector
                    'Unsigned': The fitting doesn't distinguish the symmetric vectors, 
                                i.e. [1, 0, 1] is same as [-1, 0, 1] etc. 
              
       Output: 
       Matrix   Rotation matrix
    '''
    # Regularize the input vectors
    V_original = np.asarray(V_original)
    V_final = np.asarray(V_final)
    
    if np.size(np.shape(V_original)) > 2 or np.size(np.shape(V_final)) > 2: 
        print(">>>>>> Input vectors' dimensions are wrong! <<<<<<")
        print(">>>>>> The Input should be a (3, n) array. <<<<<<")
        return
    
    if np.shape(V_original)[0] == 3:
        Vo = V_original
    elif np.shape(V_original)[1] == 3: 
        Vo = np.transpose(V_original)
    else: 
        print(">>>>>> V_original's size is wrong! <<<<<<")
        print(">>>>>> The V_original should be a 3 by n array. <<<<<<")
        return
    
    if np.shape(V_final)[0] == 3:
        Vf = V_final
    elif np.shape(V_final)[1] == 3: 
        Vf = np.transpose(V_final)
    else: 
        print(">>>>>> V_final's size is wrong! <<<<<<")
        print(">>>>>> The V_final should be a 3 by n array. <<<<<<")
        return
    
    # Normalize the input vectors
    Vo = Vo/np.linalg.norm(Vo, axis=0)
    Vf = Vf/np.linalg.norm(Vf, axis=0)
    
    # ===========================================================================
    # Define the functions for fitting
    if Type == 'Signed':
        def Rotation(Input, *Rot):
            '''
            'Input':     (3,) array, the original vector
            'Rot':       (9,) array, the rotation matrix.  

            '''
            # Rotation matrices
            M = [[ Rot[0], Rot[1], Rot[2] ], 
                 [ Rot[3], Rot[4], Rot[5] ], 
                 [ Rot[6], Rot[7], Rot[8] ]]

            M = np.matrix(M)

            return np.asarray(np.matmul(M, Input)).reshape(-1)
    elif Type == 'Unsigned': 
        def Rotation(Input, *Rot):
            '''
            'Input':     (3,) array, the original vector
            'Rot':       (9,) array, the rotation matrix.  

            '''
            # Rotation matrices
            M = [[ Rot[0], Rot[1], Rot[2] ], 
                 [ Rot[3], Rot[4], Rot[5] ], 
                 [ Rot[6], Rot[7], Rot[8] ]]

            M = np.matrix(M)

            return np.abs( np.asarray(np.matmul(M, Input)).reshape(-1) )
    # ===========================================================================
    # Fitting
    Init_vals = [1, 0, 0, 
                 0, 1, 0, 
                 0, 0, 1]    # Initial value of Rotation Matrix 
    Para_bounds = ([-1, -1, -1, -1, -1, -1, -1, -1, -1], 
                   [ 1,  1,  1,  1,  1,  1,  1,  1,  1])
            
    try: 
        popt,pcov = curve_fit(Rotation, Vo, Vf.reshape(-1), bounds=Para_bounds, p0=Init_vals)
    except RuntimeError: 
        print('>>>>>> Optimal parameters not found !!!')
        popt = Init_vals
        
    M = np.matrix(np.asarray(popt).reshape((3,3)))
    err = np.sum(np.square(Rotation(Vo, *popt) - Vf.reshape(-1)))
    
    return {'Matrix': M, 'Error': err}









'''
def Lab2Det(Input, Matrix, Phi): 
    ' This function transform the coordinates of a point from lab frame to detector frame. 
        Input should be a (3, n) array, representing n of (3,) points. 
        Matrix is the coordinates transform matrix, a (4, 4) array. 
        Phi is an angle in degree, which is the detector rotation around its normal. 
        
        Output will be the same size array as the Input. 
        
        A vector V can be writen as 
                V = [a1 a2 a3 0] * [v1 v2 v3 P0]' = a_v * f
                where a_v = [a1 a2 a3 0] are the coordinates, f = [v1 v2 v3 P0] is the frame.
        A point P can be writen as
                P = [a1 a2 a3 1] * [v1 v2 v3 P0]' = a_p * f
               
        To transform the coordinates of a point Q between lab frame and detector frame: 
                Q_lab = [a1 a2 a3 1] * [Lab1 Lab2 Lab3 LabO]
                Q_det = [b1 b2 b3 1] * [Det1 Det2 Det3 DetP]
            there are: 
                [Det1 Det2 Det3 DetP] = M_frame * [Lab1 Lab2 Lab3 LabO]
                [Lab1 Lab2 Lab3 LabO] = M_frame_inverse * [Det1 Det2 Det3 DetP]
            and
                [b1 b2 b3 1] = M_coord * [a1 a2 a3 1]
                [a1 a2 a3 1] = M_coord_inverse * [b1 b2 b3 1] 
    '
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 3:
        In = Input
    elif np.shape(Input)[1] == 3: 
        In = np.transpose(Input)
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    
    # Expand the input (3, n) array to (4, n)
    In = np.append(In, np.ones((1, np.shape(In)[1])), 0)
    
    # Transform from lab frame to Det frame
    Out = np.matmul(Matrix, In)
    
    # Perform the rotation around Det normal. 
    # i.e. rotate the coordinates in Det frame by Phi around Z axis
    Out = Coordinates_rotate_3D(Phi, 2, Out[:3, :])
    Output = Out
    
    # Output
    if np.shape(Input) != np.shape(Output): 
        Output = np.transpose(Output)
    
    return Output


def Det2Lab(Input, Matrix, Phi): 
    ' This function transform the coordinates of a point from detector frame to lab frame. 
        Input should be a (3, n) array, representing n of (3,) points. 
        Matrix is the coordinates transform matrix, a (4, 4) array. 
        Phi is an angle in degree, which is the detector rotation around its normal. 
        !!! Note:  
                1) In this function, the rotation is done BEFORE the tranformation.
                2) The Phi here should be the negative of the Phi used in function Lab2Det()
        
        Output will be the same size array as the Input. 
        
        A vector V can be writen as 
                V = [a1 a2 a3 0] * [v1 v2 v3 P0]' = a_v * f
                where a_v = [a1 a2 a3 0] are the coordinates, f = [v1 v2 v3 P0] is the frame.
        A point P can be writen as
                P = [a1 a2 a3 1] * [v1 v2 v3 P0]' = a_p * f
               
        To transform the coordinates of a point Q between lab frame and detector frame: 
                Q_lab = [a1 a2 a3 1] * [Lab1 Lab2 Lab3 LabO]
                Q_det = [b1 b2 b3 1] * [Det1 Det2 Det3 DetP]
            there are: 
                [Det1 Det2 Det3 DetP] = M_frame * [Lab1 Lab2 Lab3 LabO]
                [Lab1 Lab2 Lab3 LabO] = M_frame_inverse * [Det1 Det2 Det3 DetP]
            and
                [b1 b2 b3 1] = M_coord * [a1 a2 a3 1]
                [a1 a2 a3 1] = M_coord_inverse * [b1 b2 b3 1]
        
        To simplify the calculation, in this function we will tailor the Matrix to (3, 3).
    '
    # Input Regularization
    if np.size(np.shape(Input)) > 2: 
        print(">>>>>> Input vector/coordinates' dimension is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    if np.shape(Input)[0] == 3:
        In = Input
    elif np.shape(Input)[1] == 3: 
        In = np.transpose(Input)
    else: 
        print(">>>>>> Input vector/coordinates' size is wrong! <<<<<<")
        print(">>>>>> The Input should be a 3 by n array. <<<<<<")
        return
    
    # Perform the rotation around Det normal. 
    # i.e. rotate the coordinates in Det frame by Phi around Z axis
    Out = Coordinates_rotate_3D(Phi, 2, In)
    
    # Expand the input (3, n) array to (4, n)
    Out = np.append(Out, np.ones((1, np.shape(Out)[1])), 0)
    
    # Transform from lab frame to Det frame
    Out = np.matmul(Matrix, Out)
    Output = Out[:3, :]
    
    # Output
    if np.shape(Input) != np.shape(Output): 
        Output = np.transpose(Output)
    
    return Output


def Array_shift_1D(Array, Shift=0): # pyfftw  
    '' >>> Introduction <<< 
        Ross' function that shifts a 1-dimensional array using FT method
        Input: 
            Shift:      float number, 1D shift in pixels
            
    ''
    Array = np.asarray(np.squeeze(Array))
    dims = np.shape(Array)
    if not len(dims) == 1:
        print(">>>>>> Array should be 1-D ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    if Shift == 0:    # Skip this axis if no shift
        return Array
    # ====== FFT ======
    ftarr = sp.fftpack.fft(Array)
    # ====== Creat 3D meshgrid ======
    d = dims[0]
    r = slice(int(np.ceil(-d/2.)), int(np.ceil(d/2.)), None)
    idxgrid = np.mgrid[r]
    # ====== Perform the shift ======
    ftarr *= np.exp(-1j*2*np.pi*Shift*sp.fftpack.fftshift(idxgrid)/float(d))
    # ====== IFFT ======
    shiftedarr = np.abs( sp.fftpack.ifft(ftarr) )
    
    return shiftedarr


def Array_shift_2D(Array, Shift=[0, 0], Progress=False): # pyfftw  
    '' >>> Introduction <<< 
        Ross' function that shifts a 2-dimensional array using FT method
        Input: 
            Shift:      (2,) list, 2D shifts in pixels
                    e.g. Shift[1] = 2 means Array is shifted 2 pixels along axis=1
            Progress:   True will display the time stamps of each step
    ''
    Array = np.asarray(Array)
    dims = np.shape(Array)
    if not len(dims) == len(Shift):
        print(">>>>>> 'Shift' must have same number of dimensions as the array! ")
        input('>>>>>> Press any key to quit ... ')
        return
    # ====== FFT ======
    if Progress: 
        print('>>>>>> FFT the array ... ' + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    ftarr = sp.fftpack.fft2(Array)
    # ====== Creat 3D meshgrid ======
    if Progress: 
        print('>>>>>> Creat the meshgrid ... ' + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    r=[]
    for d in dims:
        r.append(slice(int(np.ceil(-d/2.)), int(np.ceil(d/2.)), None))
    idxgrid = np.mgrid[r]
    # ====== Perform the shift ======
    if Progress: 
        print('>>>>>> Transform the array ... ' + strftime('%H:%M:%S', localtime()))
        t_3 = time()
    for d in range(len(dims)):
        if Shift[d] == 0:    # Skip this axis if no shift
            continue
        ftarr *= np.exp(-1j*2*np.pi*Shift[d]*sp.fftpack.fftshift(idxgrid[d])/float(dims[d]))
    # ====== IFFT ======
    if Progress: 
        print('>>>>>> IFFT the transformed array ... ' + strftime('%H:%M:%S', localtime()))
        t_4 = time()
    shiftedarr = np.abs( sp.fftpack.ifft2(ftarr) )
    
    if Progress: 
        t_5 = time()
        print('>>>>>> Summary: ')
        print('           FFT took %0.6f sec.' %(t_2 - t_1))
        print('           Meshgrid took %0.6f sec.' %(t_3 - t_2))
        print('           Transform took %0.6f sec.' %(t_4 - t_3))
        print('           iFFT took %0.6f sec.' %(t_5 - t_4))
        print('           Shift took %0.6f sec in total.' %(t_5 - t_1))
    return shiftedarr


def Array_shift_3D(Array, Shift=[0, 0, 0], Progress=False): # pyfftw  
    '' >>> Introduction <<< 
        Ross' function that shifts a multi-dimensional array using FT method
        Input: 
            Shift:      (n,) list, nD shifts in pixels
                    e.g. Shift[1] = 2 means Array is shifted 2 pixels along axis=1
            Progress:   True will display the time stamps of each step
    ''
    Array = np.asarray(Array)
    dims = np.shape(Array)
    if not len(dims) == len(Shift):
        print(">>>>>> 'Shift' must have same number of dimensions as the array! ")
        input('>>>>>> Press any key to quit ... ')
        return
    # ====== FFT ======
    if Progress: 
        print('>>>>>> FFT the array ... ' + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    ftarr = (pf.builders.fftn(Array))()
    # ====== Creat 3D meshgrid ======
    if Progress: 
        print('>>>>>> Creat the meshgrid ... ' + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    r=[]
    for d in dims:
        r.append(slice(int(np.ceil(-d/2.)), int(np.ceil(d/2.)), None))
    idxgrid = np.mgrid[r]
    # ====== Perform the shift ======
    if Progress: 
        print('>>>>>> Transform the array ... ' + strftime('%H:%M:%S', localtime()))
        t_3 = time()
    for d in range(len(dims)):
        if Shift[d] == 0:    # Skip this axis if no shift
            continue
        ftarr *= np.exp(-1j*2*np.pi*Shift[d]*nf.fftshift(idxgrid[d])/float(dims[d]))
    # ====== IFFT ======
    if Progress: 
        print('>>>>>> IFFT the transformed array ... ' + strftime('%H:%M:%S', localtime()))
        t_4 = time()
    shiftedarr = np.abs((pf.builders.ifftn(ftarr))() )
    
    if Progress: 
        t_5 = time()
        print('>>>>>> Summary: ')
        print('           FFT took %0.6f sec.' %(t_2 - t_1))
        print('           Meshgrid took %0.6f sec.' %(t_3 - t_2))
        print('           Transform took %0.6f sec.' %(t_4 - t_3))
        print('           iFFT took %0.6f sec.' %(t_5 - t_4))
        print('           Shift took %0.6f sec in total.' %(t_5 - t_1))
    return shiftedarr


'''


