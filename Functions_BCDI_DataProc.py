import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time, localtime, strftime
import pyfftw as pf
import numpy.fft as nf
from scipy.ndimage import measurements as spmsu
import scipy.interpolate as sin
from Functions_General_Geometry import *
from Functions_Crystallography import *

# from _trispline import Spline


def Track_sample_movement(ScanNum=[1, 2], Pixel=0.055, SpecFile='', 
                          DataPath='', DataFileHead='', 
                          Display=False, ShowThetaScan=False, SaveResult=False):
    '''
    Coordinates of 34-ID-C: Lab_Y: Downstream, Lab_Z: Vertical, Lab_X: Inboard
    For this code, Z is Downstream, Y is Vertical, X is Outboard
    
    Therefore, when reading the spec file, 
            Z should read  'Lab_Y'
            Y should read  'Lab_Z'
            X should read -'Lab_X'
    
    ==========================================================================
    
    The outputs 'Delta' and 'Gamma'are the detector positions.
    
    The output 'Pos' is a (n,9) array: 
        [:,0] and [:,1] are the x, y positions of the crystal
        [:,2] is theta
        [:,3:6] are the Q vector of the COM, no theta change considered
        [:,6:9] are the Q vector of the COM, involving theta change
    
    Note: 
        Gamma is negative as it is rotation around outboard direction.
    '''
    # Get the current working directory
    retval = os.getcwd()
    # Verify spec file
    if not os.path.isfile(SpecFile): 
        print('>>>>>> Error! Spec file does NOT exist. ')
        input('>>>>>> Press any key to quit ... ')
        return
    # -----------------------------------------------------------
    # Read infomation from spec file
    N_scans = np.size(ScanNum)

    Theta = np.zeros(N_scans)
    X_cen = np.zeros(N_scans)
    Y_cen = np.zeros(N_scans)
    Delta = np.zeros(N_scans)
    Gamma = np.zeros(N_scans)
    d_th  = np.zeros(N_scans)
    CamDist = np.zeros(N_scans)
    PhotonE = np.zeros(N_scans)

    for i in range(N_scans):
        print('>>>>>> Reading SpecFile %03d/%03d ...' %(i+1, N_scans), end='\r' )
        Handle = spec.SPECFile(SpecFile)[ScanNum[i]-1]
        Theta[i] =  Handle.init_motor_pos['INIT_MOPO_Theta']
        X_cen[i] = -Handle.init_motor_pos['INIT_MOPO_Lab_X']
        Y_cen[i] =  Handle.init_motor_pos['INIT_MOPO_Lab_Z']
        Delta[i] =  Handle.init_motor_pos['INIT_MOPO_Delta']
        Gamma[i] =  Handle.init_motor_pos['INIT_MOPO_Gamma']
        CamDist[i] =  Handle.init_motor_pos['INIT_MOPO_camdist']
        PhotonE[i] =  Handle.init_motor_pos['INIT_MOPO_Energy']

        Command = Handle.command.split('  ')
        d_th[i] = (float(Command[-3]) - float(Command[-4])) / float(Command[-2])
    print('>>>>>> Reading SpecFile %03d/%03d ... Done' %(i+1, N_scans))
    # -----------------------------------------------------------
    # Get the center-of-mass of the diffraction peak 
    COM = np.zeros((N_scans, 3))
    
    for i in range(N_scans): 
        Datafile = DataPath + DataFileHead + '%04d.tif' % ScanNum[i]
        # Verify Data file
        if not os.path.isfile(Datafile): 
            print('>>>>>> Error! Data file does NOT exist. ')
            print('>>>>>> Check ' + Datafile)
            continue
        print('>>>>>> Processing Data %03d/%03d ...' %(i+1, N_scans), end='\r' )
        im = tif.imread(Datafile)
        COM[i, :] = np.array(spmsu.center_of_mass(im))[:]
    print('>>>>>> Processing Data %03d/%03d ... Done' %(i+1, N_scans))
    N_th, N_ver, N_hor = np.shape(im)
    # -----------------------------------------------------------
    # Get the position array of the scans
    Pos = np.zeros((N_scans, 9))

    # X and Y positions of the crystal
    Pos[:,0] = X_cen[:]
    Pos[:,1] = Y_cen[:]

    # Calculate Q vectors
    for i in range(N_scans):
        WaveLen = 12.398 / PhotonE[i]   # [A]
        WaveVec = 2 * np.pi / WaveLen   # [A^-1]
        
        Pos[i,2] = Theta[i] - (np.floor(N_th/2) - COM[i,0]) * d_th[i]
        # gamma and delta of the COM
        d_gamma = Pixel / CamDist[i]
        d_delta = Pixel / CamDist[i]
        gamma = -Gamma[i] - (np.floor(N_ver/2) - COM[i,1]) * d_gamma
        delta =  Delta[i] - (np.floor(N_hor/2) - COM[i,2]) * d_delta
        # Q vector of the COM
        k_i = np.array([0, 0, WaveVec])
        k_f = Coordinates_rotate_3D(delta, 1, Coordinates_rotate_3D(gamma, 0, k_i))
        Q = k_f - k_i
        Pos[i,3:6] = Q[:]

        # change of the theta of the COM
        theta = Pos[i,2] - Pos[0,2]
        Pos[i,6:9] = Coordinates_rotate_3D(-theta, 1, Q)[:]
    # -----------------------------------------------------------
    # Plot and Output
    if Display:
        # Q drift in 3D ( No Theta change ) 
        fig = plt.figure()
        ax = Axes3D(fig)
        # Plot and label the direct vectors
        for i in range(N_scans):
            ax.scatter(Pos[i,5], Pos[i,3], Pos[i,4], color='r')
            if np.remainder(i, 3) == 0: 
                ax.text(Pos[i,5], Pos[i,3], Pos[i,4], '%d' %(ScanNum[i]))
            if i > 0: 
                ax.plot([Pos[i-1,5], Pos[i,5]], 
                        [Pos[i-1,3], Pos[i,3]], 
                        [Pos[i-1,4], Pos[i,4]], color='k')
        # Plot the Theta scan of the 'middle' scan
        temp_th_steps = 3
        if ShowThetaScan: 
            for j in range(temp_th_steps): 
                theta = (np.floor(temp_th_steps/2) - j) * d_th[int(np.floor(N_scans/2))]
                temp_Q = Coordinates_rotate_3D(theta, 1, Pos[int(np.floor(N_scans/2)),3:6])
                ax.scatter(temp_Q[2], temp_Q[0], temp_Q[1], color='b')

        # Show the image and adjust the aspect ratio
        plt.show()
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.view_init(60, 181)
        # ------------------------------------------------
        # Crystal Rotation in 3D ( Theta change involved ) 
        fig = plt.figure()
        ax = Axes3D(fig)
        # Plot and label the direct vectors
        for i in range(N_scans):
            ax.scatter(Pos[i,8], Pos[i,6], Pos[i,7], color='r')
            if np.remainder(i, 3) == 0: 
                ax.text(Pos[i,8], Pos[i,6], Pos[i,7], '%d' %(ScanNum[i]))
            if i > 0: 
                ax.plot([Pos[i-1,8], Pos[i,8]], 
                        [Pos[i-1,6], Pos[i,6]], 
                        [Pos[i-1,7], Pos[i,7]], color='k')
        # Plot the Theta scan of 569
        temp_th_steps = N_th
        if ShowThetaScan: 
            for j in range(temp_th_steps): 
                theta = (np.floor(temp_th_steps/2) - j) * d_th[int(np.floor(N_scans/2))]
                temp_Q = Coordinates_rotate_3D(theta, 1, Pos[int(np.floor(N_scans/2)),6:9])
                ax.scatter(temp_Q[2], temp_Q[0], temp_Q[1], color='b')
        # Show the image and adjust the aspect ratio
        plt.show()
        # ax.set_aspect(20)
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.view_init(60, 181)
        # ------------------------------------------------
        # Crystal Drift
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot and label the direct vectors
        for i in range(N_scans):
            ax.scatter(Pos[i,0], Pos[i,1], color='r')
            if np.remainder(i, 3) == 0: 
                ax.text(Pos[i,0], Pos[i,1], '%d' %(ScanNum[i]))
            if i > 0: 
                ax.plot([Pos[i-1,0], Pos[i,0]], 
                        [Pos[i-1,1], Pos[i,1]], color='k')
        # Show the image and adjust the aspect ratio
        plt.show()
        # ax.set_aspect(20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    if SaveResult: 
        os.chdir(SavePath)
        Savefile = 'Q_positions_%d-%d_dth-%d-udeg.npy' % (ScanNum[0],ScanNum[-1],d_th[0]*1e6)
        # Verify Save file
        if os.path.isfile(Savefile): 
            print('>>>>>> Error! Save File already exists. ')
            print('>>>>>> Check ' + Savefile)
            overwrite = input('>>>>>> Overwrite? (y/n)')
            if overwrite.lower() == 'y':
                np.save(Savefile, Pos)
                print('>>>>>> File has been overwritten! ')
            else:
                print('>>>>>> Result has not been saved! ')
        else: 
            np.save(Savefile, Pos)
            print('>>>>>> Result saved to ' + Savefile)
        os.chdir(retval)
    
    return {'Pos':Pos, 'Delta':Delta, 'Gamma':Gamma, 'd_th':d_th}


def Scan_align_combin(ScanNum=[1,2], ReadHead='', SaveHead='', threshold=0, 
                      AlignMethod='CC', SaveAligned=False, Progress=False): 
    # Get the current working directory
    retval = os.getcwd()
    
    # Align all scans to the first one
    N_scans = np.size(ScanNum)
    
    im_ref = None
    for i in range(N_scans): 
        if Progress: 
            print('>>>>>> Aligning Scans %d/%d ' %(i+1, N_scans))
        Datafile = ReadHead + '%04d.tif' % ScanNum[i]
        # Verify Data file
        if not os.path.isfile(Datafile): 
            print('>>>>>> Error! Data file does NOT exist. ')
            print('>>>>>> Check ' + Datafile)
            continue
        im = tif.imread(Datafile)
        im = Array_threshold(im, threshold)
        if im_ref is None:
            im_ref = im.copy()
            im_sum = im.copy()
        else: 
            Offset = Array_Offset(im_ref, im, Method=AlignMethod, DisplayError=Progress)
            im = Array_shift(im, Offset)
            im_sum = im_sum + im
        
        if SaveAligned: 
            Savefile = SaveHead + '%04d.tif' % ScanNum[i]
            # Verify Save file
            if os.path.isfile(Savefile): 
                print('>>>>>> Error! Save File already exists. ')
                print('>>>>>> Check ' + Savefile)
                overwrite = input('>>>>>> Overwrite? (y/n)')
                if not (overwrite == 'y' or overwrite == 'Y'):
                    print('>>>>>> File has not been saved! <<<<<<')
                    continue
            tif.imsave(Savefile, im.astype(int))
    if Progress: 
        print('>>>>>> Done. ')
    return im_sum


def Array_chi_square_error(array, reference): 
    ''' >>> Instruction <<< 
        This function calculates the chi-squared error metric between two arrays. 
        
        * Definitions of inputs: 
            array:       array that needs to be evaluated 
            reference:   reference array
    '''
    if np.shape(array) != np.shape(reference): 
        print('>>>>>> Two input arrays have different shapes !!! ')
        input('>>>>>> Press any key to quit...')
        return
    diff = np.sum(np.square(array - reference))
    norm = np.sum(np.square(reference))
    error = diff / norm
    return error


def Array_reciprocal_space_voxel(Array, Lambda=None, PhotonE=None, dRock=0, Rock_axis='Y', 
                                 detdist=500, px=0.055, py=0.055, IsRadian=True): 
    ''' >>> Instruction <<<
        
        !!! Duplicate from Jesse Clark's Matlab code. Do NOT use. !!!
        
        Inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            dRock:       Step of rocking curve scan
            Rock_axis:   Axis of rocking curve scan
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, dYaw and dPitch are in [rad]; otherwise, [deg]
        
        Output: 
            [Vz, Vy, Vx]:   3-dimentional voxel size in [A-1]
    '''
    
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        dRock = np.deg2rad(dRock)
    
    if Lambda is None: 
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    # Voxel size in X and Y  
    Vx = 2*np.pi * px/detdist / Lambda
    Vy = 2*np.pi * py/detdist / Lambda
    
    # Voxel size in Z  
    if not (Rock_axis == 'Y' or Rock_axis == 'y' or Rock_axis == 'X' or Rock_axis == 'x'):
        print('>>>>>> Please check Rock_axis. ')
        input('>>>>>> Press any key to quit...')
        return
    
    Vz = 2*np.pi * dRock / Lambda
        
    return [Vz, Vy, Vx]


def Array_coords_det2lab_SC(Array, Lambda=None, PhotonE=None, Delta=0, Gamma=0, 
                            dYaw=0, dPitch=0, detdist=500, px=0.055, py=0.055, 
                            IsRadian=True, Progress=False):
    ''' >>> Instruction <<<   
        This function transforms the orthogonal coordinates in Detector space to
        the coordinates in Lab space, using several transformation between Cartesian 
        and Spherical coordinates. 
        
        !!! The output is the Lab space coordinates !!!
        !!! The transformation of Array is not done in this function !!!
        
        
        * The transformation is in reciprocal space. 
        
        * The transformation of Array is done using fuction Array_transform_det2lab().
        
        * Pitch, Yaw, and Roll are defined as rotation around X, Y, and Z, respectively.

        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.

        * The unit of the output coordinates is the [nm-1] in reciprocal space. 
        
        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.

        * Definitions of inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            dYaw:        Step of rocking curve scan around Y axis
            dPitch:      Step of rocking curve scan around X axis
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, all the angles are in [rad]; otherwise, [deg]

        * Detector is at the direct beam when both Delta and Gamma are zero. 
        * Delta and Gamma should use the angles of the center of the Array.
        * One and only one of dYaw and dPitch should be ZERO. 
    '''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dYaw = np.deg2rad(dYaw)
        dPitch = np.deg2rad(dPitch)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    
    # Constants  
    ki = 2 * np.pi / Lambda   # [A-1]
    
    # Generate centered meshgrids
    if Progress: 
        print('>>>>>> Creat the meshgrid ... ' + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    X = np.asarray(range(Nx)) - np.floor(Nx/2)
    Y = np.asarray(range(Ny)) - np.floor(Ny/2)
    Z = np.asarray(range(Nz)) - np.floor(Nz/2)
    
    X2, Y2 = np.meshgrid(X, Y) # 2D meshgrid representing the detector pixels
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the data
    
    ''' >>> Math <<<  
        The detector pixels are in an 2D array with coordinates X2*px, Y2*py

        The Ewald sphere is a sphere with a center at [X=0, Y=0, Z=-ki] and a radius ki
            where the magnitude of ki is 2*np.pi/Lambda 

        Therefore, in a spherical coordinates centered at [X=0, Y=0, Z=-ki], the coordinates 
        of the 2D array are: 
            [R, DeltaAxis, GammaAxis] => [R=|ki|, X2*px/detdist, Y2*py/detdist]

        To include the rotation of crystal (dYaw or dPitch), the spherical coordinates needs 
        to be transfered to another spherical coordiantes centered at [X=0, Y=0, Z=0].
            This is done by Transform from the frist Spherical to Cartesian 
                         => Translate from Z=-ki to Z=0 in Cartesian 
                         => Transform from Cartesian to the second Spherical

        Here, the two functions Cartesian2Spherical() and Spherical2Cartesian() are used.
        
        !!!!!! Note !!!!!! 
        Definitions in the two functions are different: 
            For the two functions: 
            In the Cartesian coordinates, (3,) stands for (Axis1, Axis2, Axis3)
            In the Spherical coordinates, (3,) stands for (Radius, Angle1, Angle2)
            where: 
            Angle1 is the angle from Axis3 to the vector
            Angle2 is the angle from Axis1 to the vector's projection on Axis1-Axis2 plane, 
                   i.e. Angle2 stands for the rotation around Axis3
            
            As an example: 
                Since Gamma is the angle from XZ plane to the vector, and Delta 
                is the angle from the Z axis to the XZ projection of the vector, 
                sending [R, pi/2-Gamma, Delta] to function Spherical2Cartesian()
                will get [Z, X, Y]
                
                Similarly, sending [Y, Z, X] to function Cartesian2Spherical() will 
                get [R, pi/2-Alpha, Beta], where Alpha is the angle from YZ plane to 
                the vector, and Beta is the angle from the Y axis to the YZ projection
                of the vector.
            
            Therefore, axes need to be reordered when using those two functions.
    '''
    # Transform detector pixels to Ewald sphere
    if Progress: 
        print('       Project detector to Ewald sphere ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    DeltaAxis = np.arctan2(X2.ravel() * px, detdist) + Delta   # [rad]
    GammaAxis = np.arctan2(Y2.ravel() * py, detdist) + Gamma   # [rad]
    
    Spherical = np.empty((3, int(Ny*Nx)))   # Pixels in the 1st spherical coordinates
    Spherical[0, :] = np.abs(ki) # [nm-1]
    Spherical[1, :] = np.pi/2 - GammaAxis
    Spherical[2, :] = DeltaAxis
    
    # Transform Ewald sphere to Cartesian in Reciprocal space, unit is [nm-1]
    if Progress: 
        print('       1st Spherical to Cartesian ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
    
    Cartesian = Spherical2Cartesian(Spherical, IsRadian=True)
    '''Note: Cartesian[0] is Z axis, Cartesian[1] is X axis, Cartesian[2] is Y axis'''
    
    # Translate Cartesian from Z=-ki to Z=0
    if Progress: 
        print('       Translation in Cartesian ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_4 = time()
    
    Cartesian[0] = Cartesian[0] - ki
    
    # Transform Cartesian to the 2nd spherical coordinates
    if Progress: 
        print('       Cartesian to Spherical ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_5 = time()
        
    if dPitch == 0: 
        ''' Crystal is rotated around Y axis. 
            Therfore, it is easier to have Angle2 defined as rotation around Y axis.
            The current Cartesian is already ordered as [Z, X, Y]. No change required.
        '''
        d_angle = dYaw
        Spherical = Cartesian2Spherical(Cartesian, OutputRadian=True)
    elif dYaw == 0:
        ''' Crystal is rotated around X axis. 
            Therfore, it is easier to have Angle2 defined as rotation around X axis, 
            i.e. it is better to order Cartesian as [Y, Z, X]. 
            
            The current Cartesian is already ordered as [Z, X, Y]. Reordering required.
        '''
        d_angle = dPitch
        NewOrder = np.array([2, 0, 1])
        Spherical = Cartesian2Spherical(Cartesian[NewOrder, :], OutputRadian=True)
    else: 
        print('>>>>>> Please check dYaw and dPitch. ')
        input('>>>>>> Press any key to quit...')
        return
    
    # Generate the 3D array in spherical coordinates
    if Progress: 
        print('       Generate 3D array ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_6 = time()
    
    Rho    = np.repeat(Spherical[0,:].reshape((Ny, Nx))[np.newaxis, :, :], Nz, axis=0)
    Angle1 = np.repeat(Spherical[1,:].reshape((Ny, Nx))[np.newaxis, :, :], Nz, axis=0)
    Angle2 = np.repeat(Spherical[2,:].reshape((Ny, Nx))[np.newaxis, :, :], Nz, axis=0)
    Angle2 = Angle2 + Z3 * d_angle
    
    del Spherical
    Spherical = np.stack((Rho.ravel(), Angle1.ravel()), axis=0)
    Spherical = np.concatenate((Spherical, [Angle2.ravel()]), axis=0)
    
    # Transform the 3D array from spherical coordinates to Cartesian coordinates
    if Progress: 
        print('       2nd Spherical to Cartesian ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_7 = time()
    
    del Cartesian
    Cartesian = Spherical2Cartesian(Spherical, IsRadian=True)
    if dPitch == 0: 
        '''Cartesian is ordered as [Z, X, Y]'''
        X3_lab = Cartesian[1].reshape((Nz, Ny, Nx))
        Y3_lab = Cartesian[2].reshape((Nz, Ny, Nx))
        Z3_lab = Cartesian[0].reshape((Nz, Ny, Nx))
    elif dYaw == 0: 
        '''Cartesian is ordered as [Y, Z, X]'''
        X3_lab = Cartesian[2].reshape((Nz, Ny, Nx))
        Y3_lab = Cartesian[0].reshape((Nz, Ny, Nx))
        Z3_lab = Cartesian[1].reshape((Nz, Ny, Nx))
    
    if Progress: 
        t_8 = time()
        print('>>>>>> Summary: ')
        print('           Creating meshgrid took %0.6f sec.' %(t_2 - t_1))
        print('           Projecting detector to Ewald sphere took %0.6f sec.' %(t_3 - t_2))
        print('           1st Spherical to Cartesian (2D) took %0.6f sec.' %(t_4 - t_3))
        print('           Translation in Cartesian took %0.6f sec.' %(t_5 - t_4))
        print('           Cartesian to Spherical took %0.6f sec.' %(t_6 - t_5))
        print('           Generating 3D array took %0.6f sec.' %(t_7 - t_6))
        print('           2nd Spherical to Cartesian (3D) took %0.6f sec.' %(t_8 - t_7))
        print('           Coordinates calculation took %0.6f sec in total.' %(t_8 - t_1))
    
    return {'X': X3_lab, 'Y': Y3_lab, 'Z': Z3_lab}


def Array_generate_coords_det(Array, Lambda=None, PhotonE=None, Voxel=None, 
                              Delta=0, Gamma=0, dRock=0, Rock_axis='Y', detdist=500,  
                              px=0.055, py=0.055, IsRadian=False, OutputRadian=True): 
    ''' >>> Instruction <<< 
        This function assumes an orthogonal coordinates in the detector frame, 
        and calculate the delta, gamma, and theta of each voxel. 
        
        * Assume that the center voxel of the array has the Q vector corresponding to 
            (Delta, Gamma), and Yaw/Pitch=0 is at the center of the scan range. 
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.

        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.

        * Definitions of inputs: 
            Lambda:       Wavelength in [A]
            PhotonE:      Photon energy in [keV]
            Voxel:        Reciprocal space voxel size, [Vz, Vy, Vx].
                          If None, use Array_reciprocal_space_voxel() to calculate. 
            Delta:        Detector angle around Lab Y+ axis
            Gamma:        Detector angle around Lab X- axis
            dRock:        Step of rocking curve scan
            Rock_axis:    Axis of rocking curve scan
            detdist:      Sample-detector distance in [mm]
            px, py:       X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:     If True, all the angles are in [rad]; otherwise, [deg]
            OutputRadian: Unit of the output coordinates. Ture: [rad], False: [deg]

        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dRock = np.deg2rad(dRock)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    
    # Constants
    ki = 2 * np.pi / Lambda   # [A-1]
    
    if not (Rock_axis == 'Y' or Rock_axis == 'y' or Rock_axis == 'X' or Rock_axis == 'x'):
        print('>>>>>> Please check Rock_axis. ')
        input('>>>>>> Press any key to quit...')
        return
    
    # Generate centered meshgrids    
    X = np.asarray(range(Nx)) - np.floor(Nx/2)
    Y = np.asarray(range(Ny)) - np.floor(Ny/2)
    Z = np.asarray(range(Nz)) - np.floor(Nz/2)
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the data
    
    # Calculate the Delta, Gamma, and Theta values for each pixel
    Delta_1D = np.arctan2(X * px, detdist) + Delta   # [rad]
    Gamma_1D = np.arctan2(Y * py, detdist) + Gamma   # [rad]
    Theta_1D = Z * dRock   # [rad]
    
    Delta_det = np.arctan2(X3 * px, detdist) + Delta   # [rad]
    Gamma_det = np.arctan2(Y3 * py, detdist) + Gamma   # [rad]
    Theta_det = Z3 * dRock   # [rad]
    
    if not OutputRadian: 
        Delta_det = np.rad2deg(Delta_det)
        Gamma_det = np.rad2deg(Gamma_det)
        Theta_det = np.rad2deg(Theta_det)
    
    return {'Delta': Delta_det, 'Gamma': Gamma_det, 'Theta': Theta_det, 
            'Delta_1D': Delta_1D, 'Gamma_1D': Gamma_1D, 'Theta_1D': Theta_1D}


def Array_generate_coords_det_NoPix(Array, Lambda=None, PhotonE=None, Voxel=None, 
                                    Delta=0, Gamma=0, Delt_step=0.03, Gamm_step=0.03, 
                                    Rock_step=0.068, Rock_axis='Y', IsRadian=False, 
                                    OutputRadian=True): 
    ''' >>> Instruction <<<
        This function assumes an orthogonal coordinates in the detector frame, 
        and calculate the delta, gamma, and theta of each voxel. 
        !!! This is for the simulated data that use angular steps instead of pixels !!!
        
        * Assume that the center voxel of the array has the Q vector corresponding to 
            (Delta, Gamma), and Rock=0 is at the center of the scan range. 
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.

        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.

        * Definitions of inputs: 
            Lambda:       Wavelength in [A]
            PhotonE:      Photon energy in [keV]
            Voxel:        Reciprocal space voxel size, [Vz, Vy, Vx].
                          If None, use Array_reciprocal_space_voxel() to calculate. 
            Delta:        Detector angle around Lab Y+ axis
            Gamma:        Detector angle around Lab X- axis
            Delt_step:     Step of Delta angles
            Gamm_step:     Step of Gamma angles
            Rock_step:     Step of rocking curve scan
            Rock_axis:     Axis of rocking curve scan
            IsRadian:     If True, all the angles are in [rad]; otherwise, [deg]
            OutputRadian: Unit of the output coordinates. Ture: [rad], False: [deg]

        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        Delt_step = np.deg2rad(Delt_step)
        Gamm_step = np.deg2rad(Gamm_step)
        Rock_step = np.deg2rad(Rock_step)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    
    # Constants
    ki = 2 * np.pi / Lambda   # [A-1]
    
    if not (Rock_axis == 'Y' or Rock_axis == 'y' or Rock_axis == 'X' or Rock_axis == 'x'): 
        print('>>>>>> Please check Rock_axis. ')
        input('>>>>>> Press any key to quit...')
        return
    
    # Generate centered meshgrids    
    X = np.asarray(range(Nx)) - np.floor(Nx/2)
    Y = np.asarray(range(Ny)) - np.floor(Ny/2)
    Z = np.asarray(range(Nz)) - np.floor(Nz/2)
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the data
    
    # Calculate the Delta, Gamma, and Theta values for each pixel
    Delta_1D = X * Delt_step + Delta   # [rad]
    Gamma_1D = Y * Gamm_step + Gamma   # [rad]
    Theta_1D = Z * Rock_step   # [rad]
    
    Delta_det = X3 * Delt_step + Delta   # [rad]
    Gamma_det = Y3 * Gamm_step + Gamma   # [rad]
    Theta_det = Z3 * Rock_step   # [rad]
    
    if not OutputRadian: 
        Delta_det = np.rad2deg(Delta_det)
        Gamma_det = np.rad2deg(Gamma_det)
        Theta_det = np.rad2deg(Theta_det)
    
    return {'Delta': Delta_det, 'Gamma': Gamma_det, 'Theta': Theta_det, 
            'Delta_1D': Delta_1D, 'Gamma_1D': Gamma_1D, 'Theta_1D': Theta_1D}


def Array_generate_coords_lab_recip(Array, Lambda=None, PhotonE=None, Voxel=None, 
                                    Delta=0, Gamma=0, dRock=0, Rock_axis='Y', detdist=500,  
                                    px=0.055, py=0.055, IsRadian=True):
    ''' >>> Instruction <<< 
        This function assumes an orthogonal coordinates in lab frame of reciprocal space, 
        and calculate the qx, qy, and qz of each voxel. 

        * Assume that the center voxel of the array has the Q vector corresponding to 
            (Delta, Gamma), and Rock=0 is at the center of the scan range. 
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.

        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.

        * Definitions of inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Voxel:       Reciprocal space voxel size, [Vz, Vy, Vx].
                         If None, use Array_reciprocal_space_voxel() to calculate. 
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            dRock:       Step of rocking curve scan
            Rock_axis:   Axis of rocking curve scan
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, all the angles are in [rad]; otherwise, [deg]
        
        * The unit of the output coordinates is [A^-1]

        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dRock = np.deg2rad(dRock)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    
    if Voxel is None: 
        # Voxel = Array_reciprocal_space_voxel(Array, Lambda=Lambda, dYaw=dYaw, dPitch=dPitch,
        #                                      detdist=detdist, px=px, py=py, IsRadian=True)
        if not (Rock_axis == 'Y' or Rock_axis == 'y' or Rock_axis == 'X' or Rock_axis == 'x'):
            print('>>>>>> Check Rock_axis. ')
            input('>>>>>> Press any key to quit...')
            return
        temp_dq = CalLat_dq(Lambda=Lambda, Delta=Delta, Gamma=Gamma, dRock=dRock, 
                            RockAxis=Rock_Axis, detdist=detdist, px=px, py=py, IsRadian=True)
        Voxel = np.array([temp_dq['q_rock'], temp_dq['q_gamma'], temp_dq['q_delta']])
    Vz, Vy, Vx = np.abs(Voxel)
    
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    # Constants  
    WaveVec = 2 * np.pi / Lambda   # [A-1]
    k_i = np.array([0, 0, WaveVec])
    k_f = Coordinates_rotate_3D(-Gamma, 0, k_i, IsRadian=True)
    k_f = Coordinates_rotate_3D( Delta, 1, k_f, IsRadian=True)
    Q_0 = k_f - k_i   # The Q vector of the center voxel, [A-1]
    qx_0, qy_0, qz_0 = Q_0
    
    # Generate centered meshgrids
    X = (np.asarray(range(Nx)) - np.floor(Nx/2)) * Vx + qx_0
    Y = (np.asarray(range(Ny)) - np.floor(Ny/2)) * Vy + qy_0
    Z = (np.asarray(range(Nz)) - np.floor(Nz/2)) * Vz + qz_0
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the Lab space voxels
    
    return {'Qx': X3, 'Qy': Y3, 'Qz': Z3, 
            'Qx_1D': X, 'Qy_1D': Y, 'Qz_1D': Z}


def Array_generate_coords_lab_recip_NoPix(Array, Lambda=None, PhotonE=None, Voxel=None, 
                                          Delta=0, Gamma=0, Delt_step=0.03, Gamm_step=0.03, 
                                          Rock_step=0.068, Rock_axis='Y', IsRadian=True):
    ''' >>> Instruction <<< 
        This function assumes an orthogonal coordinates in lab frame of reciprocal space, 
        and calculate the qx, qy, and qz of each voxel. 
        !!! This is for the simulated data that use angular steps instead of pixels !!!
        
        * Assume that the center voxel of the array has the Q vector corresponding to 
            (Delta, Gamma), and Rock=0 is at the center of the scan range. 
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.

        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.

        * Definitions of inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Voxel:       Reciprocal space voxel size, [Vz, Vy, Vx].
                         If None, use Array_reciprocal_space_voxel() to calculate. 
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            Delt_step:     Step of Delta angles
            Gamm_step:     Step of Gamma angles
            Rock_step:     Step of rocking curve scan
            Rock_axis:     Axis of rocking curve scan
            IsRadian:    If True, all the angles are in [rad]; otherwise, [deg]
        
        * The unit of the output coordinates is [A^-1]

        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        Delt_step = np.deg2rad(Delt_step)
        Gamm_step = np.deg2rad(Gamm_step)
        Rock_step = np.deg2rad(Rock_step)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    
    if Voxel is None: 
        # Voxel = Array_reciprocal_space_voxel(Array, Lambda=Lambda, dYaw=dYaw, dPitch=dPitch,
        #                                      detdist=detdist, px=px, py=py, IsRadian=True)
        if not (Rock_axis == 'Y' or Rock_axis == 'y' or Rock_axis == 'X' or Rock_axis == 'x'): 
            print('>>>>>> Check Rock_axis. ')
            input('>>>>>> Press any key to quit...')
            return
        temp_dq = CalLat_dq(Lambda=Lambda, Delta=Delta, Gamma=Gamma, 
                            dDelta=Delt_step, dGamma=Gamm_step, 
                            dRock=Rock_step, RockAxis=Rock_axis, IsRadian=True)
        Voxel = np.array([temp_dq['q_rock'], temp_dq['q_gamma'], temp_dq['q_delta']])
    Vz, Vy, Vx = np.abs(Voxel)
    
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    # Constants  
    WaveVec = 2 * np.pi / Lambda   # [A-1]
    k_i = np.array([0, 0, WaveVec])
    k_f = Coordinates_rotate_3D(-Gamma, 0, k_i, IsRadian=True)
    k_f = Coordinates_rotate_3D( Delta, 1, k_f, IsRadian=True)
    Q_0 = k_f - k_i   # The Q vector of the center voxel, [A-1]
    qx_0, qy_0, qz_0 = Q_0
    
    # Generate centered meshgrids
    X = (np.asarray(range(Nx)) - np.floor(Nx/2)) * Vx + qx_0
    Y = (np.asarray(range(Ny)) - np.floor(Ny/2)) * Vy + qy_0
    Z = (np.asarray(range(Nz)) - np.floor(Nz/2)) * Vz + qz_0
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the Lab space voxels
    
    return {'Qx': X3, 'Qy': Y3, 'Qz': Z3, 
            'Qx_1D': X, 'Qy_1D': Y, 'Qz_1D': Z}


def Array_generate_coords_crys_recip(Array, Lambda=None, PhotonE=None, Voxel=None, Crys_orie=[0, 0, 0], 
                                     Delta=0, Gamma=0, dRock=0, Rock_axis='Y', detdist=500,  
                                     px=0.055, py=0.055, IsRadian=True):
    ''' >>> Instruction <<< 
        This function assumes an orthogonal coordinates in crystal frame of reciprocal space, 
        and calculate the qx, qy, and qz of each voxel. 

        * Assume that the center voxel of the array has the Q vector corresponding to 
            (Delta, Gamma), and Rock=0 is at the center of the scan range. 
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.

        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.

        * Definitions of inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Voxel:       Reciprocal space voxel size, [Vz, Vy, Vx].
                         If None, use Array_reciprocal_space_voxel() to calculate. 
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            dRock:       Step of rocking curve scan
            Rock_axis:   Axis of rocking curve scan
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, all the angles are in [rad]; otherwise, [deg]
        
        * The unit of the output coordinates is [A^-1]

        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dRock = np.deg2rad(dRock)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    
    if Voxel is None: 
        # Voxel = Array_reciprocal_space_voxel(Array, Lambda=Lambda, dYaw=dYaw, dPitch=dPitch,
        #                                      detdist=detdist, px=px, py=py, IsRadian=True)
        if not (Rock_axis == 'Y' or Rock_axis == 'y' or Rock_axis == 'X' or Rock_axis == 'x'):
            print('>>>>>> Check Rock_axis. ')
            input('>>>>>> Press any key to quit...')
            return
        temp_dq = CalLat_dq(Lambda=Lambda, Delta=Delta, Gamma=Gamma, dRock=dRock, 
                            RockAxis=Rock_Axis, detdist=detdist, px=px, py=py, IsRadian=True)
        Voxel = np.array([temp_dq['q_rock'], temp_dq['q_gamma'], temp_dq['q_delta']])
    Vz, Vy, Vx = np.abs(Voxel)
    
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    # Constants  
    WaveVec = 2 * np.pi / Lambda   # [A-1]
    k_i = np.array([0, 0, WaveVec])
    k_f = Coordinates_rotate_3D(-Gamma, 0, k_i, IsRadian=True)
    k_f = Coordinates_rotate_3D( Delta, 1, k_f, IsRadian=True)
    Q_0 = k_f - k_i   # The Q vector of the center voxel, [A-1]
    
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[2], 2, Q_0,  IsRadian=True)
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[1], 1, Bragg_crys, IsRadian=True)
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[0], 0, Bragg_crys, IsRadian=True)
    qx_0, qy_0, qz_0 = Bragg_crys
    
    # Generate centered meshgrids
    X = (np.asarray(range(Nx)) - np.floor(Nx/2)) * Vx + qx_0
    Y = (np.asarray(range(Ny)) - np.floor(Ny/2)) * Vy + qy_0
    Z = (np.asarray(range(Nz)) - np.floor(Nz/2)) * Vz + qz_0
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the Lab space voxels
    
    return {'Qx': X3, 'Qy': Y3, 'Qz': Z3, 
            'Qx_1D': X, 'Qy_1D': Y, 'Qz_1D': Z}


def Array_generate_coords_crys_recip_NoPix(Array, Lambda=None, PhotonE=None, Voxel=None, 
                                           Crys_orie=[0, 0, 0], Delta=0, Gamma=0, Delt_step=0.03, Gamm_step=0.03, 
                                           Rock_step=0.068, Rock_axis='Y', IsRadian=True):
    ''' >>> Instruction <<< 
        This function assumes an orthogonal coordinates in crystal frame of reciprocal space, 
        and calculate the qx, qy, and qz of each voxel. 
        !!! This is for the simulated data that use angular steps instead of pixels !!!

        * Assume that the center voxel of the array has the Q vector corresponding to 
            (Delta, Gamma), and Rock=0 is at the center of the scan range. 
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.

        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.

        * Definitions of inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Voxel:       Reciprocal space voxel size, [Vz, Vy, Vx].
                         If None, use Array_reciprocal_space_voxel() to calculate. 
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            Delt_step:   Step of Delta angles
            Gamm_step:   Step of Gamma angles
            Rock_step:   Step of rocking curve scan
            Rock_axis:   Axis of rocking curve scan
            IsRadian:    If True, all the angles are in [rad]; otherwise, [deg]
        
        * The unit of the output coordinates is [A^-1]

        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        Delt_step = np.deg2rad(Delt_step)
        Gamm_step = np.deg2rad(Gamm_step)
        Rock_step = np.deg2rad(Rock_step)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    
    if Voxel is None: 
        # Voxel = Array_reciprocal_space_voxel(Array, Lambda=Lambda, dYaw=dYaw, dPitch=dPitch,
        #                                      detdist=detdist, px=px, py=py, IsRadian=True)
        if not (Rock_axis == 'Y' or Rock_axis == 'y' or Rock_axis == 'X' or Rock_axis == 'x'):
            print('>>>>>> Check Rock_axis. ')
            input('>>>>>> Press any key to quit...')
            return
        temp_dq = CalLat_dq(Lambda=Lambda, Delta=Delta, Gamma=Gamma, 
                            dDelta=Delt_step, dGamma=Gamm_step, 
                            dRock=Rock_step, RockAxis=Rock_axis, IsRadian=True)
        Voxel = np.array([temp_dq['q_rock'], temp_dq['q_gamma'], temp_dq['q_delta']])
    Vz, Vy, Vx = np.abs(Voxel)
    
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    # Constants  
    WaveVec = 2 * np.pi / Lambda   # [A-1]
    k_i = np.array([0, 0, WaveVec])
    k_f = Coordinates_rotate_3D(-Gamma, 0, k_i, IsRadian=True)
    k_f = Coordinates_rotate_3D( Delta, 1, k_f, IsRadian=True)
    Q_0 = k_f - k_i   # The Q vector of the center voxel, [A-1]
    
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[2], 2, Q_0,  IsRadian=True)
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[1], 1, Bragg_crys, IsRadian=True)
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[0], 0, Bragg_crys, IsRadian=True)
    qx_0, qy_0, qz_0 = Bragg_crys
    
    # Generate centered meshgrids
    X = (np.asarray(range(Nx)) - np.floor(Nx/2)) * Vx + qx_0
    Y = (np.asarray(range(Ny)) - np.floor(Ny/2)) * Vy + qy_0
    Z = (np.asarray(range(Nz)) - np.floor(Nz/2)) * Vz + qz_0
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the Lab space voxels
    
    return {'Qx': X3, 'Qy': Y3, 'Qz': Z3, 
            'Qx_1D': X, 'Qy_1D': Y, 'Qz_1D': Z}


def Array_generate_coords_real(Array, Lambda=None, PhotonE=None, Voxel=None, 
                               Delta=0, Gamma=0, dRock=0, Rock_axis='Y', detdist=500,  
                               px=0.055, py=0.055, IsRadian=True):
    ''' >>> Instruction <<< 
        This function assumes an orthogonal coordinates in real space, 
        and calculate the rx, ry, and rz of each voxel. 
        
        * Delta, Gamma, and Rock are used to calculate the voxel size only. The array is aligned 
            to the axes of the real space coordinates, either in lab frame or crystal frame.
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.
        
        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.
        
        * Definitions of inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Voxel:       Real space voxel size, [Vz, Vy, Vx].
                         If None, use Array_reciprocal_space_voxel() to calculate. 
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            dRock:       Step of rocking curve scan
            Rock_axis:   Axis of rocking curve scan
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, all the angles are in [rad]; otherwise, [deg]
        
        * The unit of the output coordinates is [A]

        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dRock = np.deg2rad(dRock)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE   # [A]
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    
    if Voxel is None: 
        # Voxel = Array_reciprocal_space_voxel(Array, Lambda=Lambda, dYaw=dYaw, dPitch=dPitch,
        #                                      detdist=detdist, px=px, py=py, IsRadian=True)
        if not (Rock_axis == 'Y' or Rock_axis == 'y' or Rock_axis == 'X' or Rock_axis == 'x'):
            print('>>>>>> Check Rock_axis. ')
            input('>>>>>> Press any key to quit...')
            return
        temp_dq = CalLat_dq(Lambda=Lambda, Delta=Delta, Gamma=Gamma, dRock=dRock, 
                            RockAxis=Rock_Axis, detdist=detdist, px=px, py=py, IsRadian=True)
        Voxel = np.array([temp_dq['r_rock'], temp_dq['r_gamma'], temp_dq['r_delta']])
    Vz, Vy, Vx = np.abs(Voxel)
    
    # Generate centered meshgrids
    X = (np.asarray(range(Nx)) - np.floor(Nx/2)) * Vx
    Y = (np.asarray(range(Ny)) - np.floor(Ny/2)) * Vy
    Z = (np.asarray(range(Nz)) - np.floor(Nz/2)) * Vz
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the Lab space voxels
    
    return {'Rx': X3, 'Ry': Y3, 'Rz': Z3, 
            'Rx_1D': X, 'Ry_1D': Y, 'Rz_1D': Z}


def Array_generate_coords_real_NoPix(Array, Lambda=None, PhotonE=None, Voxel=None, 
                                     Delta=0, Gamma=0, Delt_step=0.03, Gamm_step=0.03, 
                                     Rock_step=0.068, Rock_axis='Y', IsRadian=True):
    ''' >>> Instruction <<< 
        This function assumes an orthogonal coordinates in real space, 
        and calculate the rx, ry, and rz of each voxel. 
        
        * Delta, Gamma, and Rock are used to calculate the voxel size only. The array is aligned 
            to the axes of the real space coordinates, either in lab frame or crystal frame.
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.
        
        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.
        
        * Definitions of inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Voxel:       Real space voxel size, [Vz, Vy, Vx].

            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            Delt_step:     Step of Delta angles
            Gamm_step:     Step of Gamma angles
            Rock_step:     Step of rocking curve scan
            Rock_axis:     Axis of rocking curve scan
            IsRadian:    If True, all the angles are in [rad]; otherwise, [deg]
        
        * The unit of the output coordinates is [A]

        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        Delt_step = np.deg2rad(Delt_step)
        Gamm_step = np.deg2rad(Gamm_step)
        Rock_step = np.deg2rad(Rock_step)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE   # [A]
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    
    if Voxel is None: 
        # Voxel = Array_reciprocal_space_voxel(Array, Lambda=Lambda, dYaw=dYaw, dPitch=dPitch,
        #                                      detdist=detdist, px=px, py=py, IsRadian=True)
        if not (Rock_axis == 'Y' or Rock_axis == 'y' or Rock_axis == 'X' or Rock_axis == 'x'):
            print('>>>>>> Check Rock_axis. ')
            input('>>>>>> Press any key to quit...')
            return
        temp_dq = CalLat_dq(Lambda=Lambda, Delta=Delta, Gamma=Gamma, 
                            dDelta=Delt_step, dGamma=Gamm_step, 
                            dRock=Rock_step, RockAxis=Rock_axis, IsRadian=True)
        Voxel = np.array([temp_dq['r_rock'], temp_dq['r_gamma'], temp_dq['r_delta']])
    Vz, Vy, Vx = np.abs(Voxel)
    
    # Generate centered meshgrids
    X = (np.asarray(range(Nx)) - np.floor(Nx/2)) * Vx
    Y = (np.asarray(range(Ny)) - np.floor(Ny/2)) * Vy
    Z = (np.asarray(range(Nz)) - np.floor(Nz/2)) * Vz
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the Lab space voxels
    
    return {'Rx': X3, 'Ry': Y3, 'Rz': Z3, 
            'Rx_1D': X, 'Ry_1D': Y, 'Rz_1D': Z}


def Array_coords_recip2real(QxCoords=None, QyCoords=None, QzCoords=None, 
                            dq1=None, dq2=None, dq3=None, Q_bragg=None): 
    ''' >>> Instruction <<<
        This function converts reciprocal space coordinates to real space coordinates. 
        
        * Definitions of inputs: 
            QxCoords:    X coordinates of voxels in reciprocal space
            QyCoords:    Y coordinates of voxels in reciprocal space
            QzCoords:    Z coordinates of voxels in reciprocal space
            dq1/dq2/dq3: Bases of reciprocal space. 
            
        * Definitions of outputs: 
            RxCoords:    X coordinates of voxels in real space
            RyCoords:    Y coordinates of voxels in real space
            RzCoords:    Z coordinates of voxels in real space    
    '''
    # Input regularization
    if QxCoords is None or QyCoords is None or QzCoords is None: 
        print('>>>>>> Lack of coordinates! ')
        input('>>>>>> Press any key to quit...')
        return
    
    dimsX = np.shape(QxCoords)
    dimsY = np.shape(QyCoords)
    dimsZ = np.shape(QzCoords)
    
    if not (np.array_equal(dimsX, dimsY) and 
            np.array_equal(dimsY, dimsZ) and
            np.array_equal(dimsZ, dimsX) ):
        print('>>>>>> The shapes of Qx, Qy, Qz coords are not identical! ')
        input('>>>>>> Press any key to quit...')
        return
    Nz, Ny, Nx = dimsX
    N_elem = int(Nz*Ny*Nx)
    
    if np.linalg.norm(dq1)<1e-7 or np.linalg.norm(dq2)<1e-7 or np.linalg.norm(dq3)<1e-7: 
        print('>>>>>> dq is too small !!! ')
        print('       dq1 magnitude is %f' %np.linalg.norm(dq1) )
        print('       dq2 magnitude is %f' %np.linalg.norm(dq2) )
        print('       dq3 magnitude is %f' %np.linalg.norm(dq3) )
        Continue_flag = input('>>>>>> Continue? (Y/N)')
        if Continue_flag == 'N' or Continue_flag == 'n': 
            return
    
    Qx = QxCoords - Q_bragg[0]
    Qy = QyCoords - Q_bragg[1]
    Qz = QzCoords - Q_bragg[2]
    
    Q = np.stack((Qx.ravel(), Qy.ravel()), axis=0)
    Q = np.concatenate((Q, [Qz.ravel()]), axis=0)
    
    # Calculate real space bases
    denorm = np.dot(dq1, np.cross(dq2, dq3))
    da1 = 2 * np.pi * np.cross(dq2, dq3) / denorm 
    da2 = 2 * np.pi * np.cross(dq3, dq1) / denorm 
    da3 = 2 * np.pi * np.cross(dq1, dq2) / denorm 
    
    # Calcualte real space coordinates
    Idx = Decomposing_Vector(Q, dq1, dq2, dq3)
    R = np.matrix([da1, da2, da3]) * np.matrix(Idx)
    
    RxCoords = np.reshape(np.array(R[0, :]), (Nz, Ny, Nx))
    RyCoords = np.reshape(np.array(R[1, :]), (Nz, Ny, Nx))
    RzCoords = np.reshape(np.array(R[2, :]), (Nz, Ny, Nx))
    
    return {'RxCoords': RxCoords, 'RyCoords': RyCoords, 'RzCoords': RzCoords}


def Array_coords_real2recip(RxCoords=None, RyCoords=None, RzCoords=None, 
                            da1=None, da2=None, da3=None, Q_bragg=None): 
    ''' >>> Instruction <<<
        This function converts real space coordinates to reciprocal space coordinates. 
        
        * Definitions of inputs: 
            RxCoords:    X coordinates of voxels in real space
            RyCoords:    Y coordinates of voxels in real space
            RzCoords:    Z coordinates of voxels in real space
            da1/da2/da3: Bases of real space. 
            
        * Definitions of outputs: 
            QxCoords:    X coordinates of voxels in reciprocal space
            QyCoords:    Y coordinates of voxels in reciprocal space
            QzCoords:    Z coordinates of voxels in reciprocal space    
    '''
    # Input regularization
    if RxCoords is None or RyCoords is None or RzCoords is None: 
        print('>>>>>> Lack of coordinates! ')
        input('>>>>>> Press any key to quit...')
        return
    
    dimsX = np.shape(RxCoords)
    dimsY = np.shape(RyCoords)
    dimsZ = np.shape(RzCoords)
    
    if not (np.array_equal(dimsX, dimsY) and 
            np.array_equal(dimsY, dimsZ) and
            np.array_equal(dimsZ, dimsX) ):
        print('>>>>>> The shapes of Rx, Ry, Rz coords are not identical! ')
        input('>>>>>> Press any key to quit...')
        return
    Nz, Ny, Nx = dimsX
    N_elem = int(Nz*Ny*Nx)
    
    if np.linalg.norm(da1)<1e-5 or np.linalg.norm(da2)<1e-5 or np.linalg.norm(da3)<1e-5: 
        print('>>>>>> da is too small !!! ')
        print('       da1 magnitude is %f' %np.linalg.norm(da1) )
        print('       da2 magnitude is %f' %np.linalg.norm(da2) )
        print('       da3 magnitude is %f' %np.linalg.norm(da3) )
        Continue_flag = input('>>>>>> Continue? (Y/N)')
        if Continue_flag == 'N' or Continue_flag == 'n': 
            return
    
    R = np.stack((RxCoords.ravel(), RyCoords.ravel()), axis=0)
    R = np.concatenate((R, [RzCoords.ravel()]), axis=0)
    
    # Calculate real space bases
    denorm = np.dot(da1, np.cross(da2, da3))
    dq1 = 2 * np.pi * np.cross(da2, da3) / denorm 
    dq2 = 2 * np.pi * np.cross(da3, da1) / denorm 
    dq3 = 2 * np.pi * np.cross(da1, da2) / denorm 
    
    # Calcualte real space coordinates
    Idx = Decomposing_Vector(R, da1, da2, da3)
    Q = np.matrix([dq1, dq2, dq3]) * np.matrix(Idx)
    
    Qx = np.reshape(np.array(Q[0, :]), (Nz, Ny, Nx))
    Qy = np.reshape(np.array(Q[1, :]), (Nz, Ny, Nx))
    Qz = np.reshape(np.array(Q[2, :]), (Nz, Ny, Nx))
    
    QxCoords = Qx + Q_bragg[0]
    QyCoords = Qy + Q_bragg[1]
    QzCoords = Qz + Q_bragg[2]
    
    return {'QxCoords': QxCoords, 'QyCoords': QyCoords, 'QzCoords': QzCoords}


def Array_coords_det2lab_recip(DeltaCoords=None, GammaCoords=None, ThetaCoords=None, 
                               Lambda=None,PhotonE=None,Rock_axis='Y',Progress=False):
    ''' >>> Instruction <<<   
        This function transforms coordinates from Detector frame to Lab frame, in reciprocal 
        space. 
        i.e. Assume an orthoginal 3D array (with uniform spacing in each dimension) in 
             detector frame of reciprocal space. 
             The output is the q-vectors of voxels in lab frame of reciprocal space. 
        
        !!! The output is the Lab frame coordinates !!!
        !!! The transformation of Array is not done in this function !!!
        
        * The transformation is in reciprocal space. 
        
        * The transformation of Array is done using interpolation tools (e.g. eqtools).
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.

        * The unit of the output coordinates is [A^-1]. 
        
        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.

        * Definitions of inputs: 
            DeltaCoords: delta values of the voxels, from Array_generate_coords_det()
            GammaCoords: gamma values of the voxels, from Array_generate_coords_det()
            ThetaCoords: theta values of the voxels, from Array_generate_coords_det()
            Rock_axis:   Axis of rocking curve scan

        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    if DeltaCoords is None or GammaCoords is None or ThetaCoords is None: 
        print('>>>>>> Lack of coordinates! ')
        input('>>>>>> Press any key to quit...')
        return
    
    dimsD = np.shape(DeltaCoords)
    dimsG = np.shape(GammaCoords)
    dimsT = np.shape(ThetaCoords)
    
    if not (np.array_equal(dimsD, dimsG) and 
            np.array_equal(dimsG, dimsT) and
            np.array_equal(dimsT, dimsD) ):
        print('>>>>>> The shapes of Delta, Gamma, Theta coords are not identical! ')
        input('>>>>>> Press any key to quit...')
        return
    Nz, Ny, Nx = dimsD
    N_elem = int(Nz*Ny*Nx)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    # Constants
    ki = 2 * np.pi / Lambda   # [A-1]
    
    DeltaAxis = DeltaCoords.ravel()
    GammaAxis = GammaCoords.ravel()
    ThetaAxis = ThetaCoords.ravel()
    
    ''' >>> Math <<<  
        Assume the coordinate of a voxel in lab space is [Qx, Qy, Qz], 
        the cooresponding coordinate in detector space is [Delta, Gamma, Theta]. 
        
        Here, Delta and Gamma define the position of this voxel on the detector, 
        while Theta means AFTER rotating the vector by Theta, it satifies the 
        momentum transfer kf - ki = Q, where: 
                ki = [0, 0, k], in Cartesian
                kf = [k, gamma, delta] in Spherical
                
        Therefore, we have: 
                Rot(Theta) * [Qx, Qy, Qz] = [k*cos(gamma)*sin(delta), 
                                             k*sin(gamma), 
                                             k*(cos(gamma)*cos(delta)-1)]
            where, 
                RotY() = [ cos,    0,  sin, 
                             0,    1,    0, 
                          -sin,    0,  cos ]
                RotX() = [   1,    0,    0, 
                             0,  cos, -sin, 
                             0,  sin,  cos ]
        
        For transform from Detector to Lab, Theta is known. 
        Therefore, 
                [Qx, Qy, Qz] = Rot(-Theta) * [...]
    '''
    
    # Coordinates transformation from detector to lab
    if Progress: 
        print('>>>>>> Transform coords to Lab space ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    if Rock_axis == 'Y' or Rock_axis == 'y': 
        '''Crystal is rotated around Y axis. '''
        Qx3 = ki * (np.cos(ThetaAxis)* np.cos(GammaAxis)*np.sin(DeltaAxis) + 
                    np.sin(ThetaAxis)*(np.cos(GammaAxis)*np.cos(DeltaAxis)-1))
        Qy3 = ki *  np.sin(GammaAxis)
        Qz3 = ki * (np.cos(ThetaAxis)*(np.cos(GammaAxis)*np.cos(DeltaAxis)-1) - 
                    np.sin(ThetaAxis)* np.cos(GammaAxis)*np.sin(DeltaAxis))
    elif Rock_axis == 'X' or Rock_axis == 'x': 
        '''Crystal is rotated around X axis. '''
        Qx3 = ki *  np.cos(GammaAxis)* np.sin(DeltaAxis)
        Qy3 = ki * (np.cos(ThetaAxis)* np.sin(GammaAxis) - 
                    np.sin(ThetaAxis)*(np.cos(GammaAxis)*np.cos(DeltaAxis)-1))
        Qz3 = ki * (np.sin(ThetaAxis)* np.sin(GammaAxis) + 
                    np.cos(ThetaAxis)*(np.cos(GammaAxis)*np.cos(DeltaAxis)-1))
    else: 
        print('>>>>>> Please check Rock_axis. ')
        input('>>>>>> Press any key to quit...')
        return
    
    Qx3 = Qx3.reshape((Nz, Ny, Nx))
    Qy3 = Qy3.reshape((Nz, Ny, Nx))
    Qz3 = Qz3.reshape((Nz, Ny, Nx))
    
    if Progress: 
        t_2 = time()
        print('>>>>>> Coordinates calculation took %0.6f sec in total.' %(t_2 - t_1))
    
    return {'Qx': Qx3, 'Qy': Qy3, 'Qz': Qz3}


def Array_coords_lab2det_recip(QxCoords=None, QyCoords=None, QzCoords=None, 
                               Lambda=None,PhotonE=None,Rock_axis='Y',Progress=False):
    ''' >>> Instruction <<<   
        This function transforms coordinates from Lab frame to Detector frame in reciprocal 
        space. 
        i.e. Assume an orthoginal 3D array (with uniform spacing in each dimension) in  
             lab frame of reciprocal space. 
             The output is the angular positions of voxels in detector frame of reciprocal space. 
        
        !!! The output is the Detector frame coordinates !!!
        !!! The transformation of Array is not done in this function !!!
        
        * The transformation is in reciprocal space. 
        
        * The transformation of Array is done using interpolation tools (e.g. eqtools).
        
        * Pitch, Yaw, and Roll are defined as rotation around X, Y, and Z, respectively.

        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.

        * The unit of the output coordinates is [rad]. 
        
        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation.

        * Definitions of inputs: 
            QxCoords:    qx values of the voxels, from Array_generate_coords_lab_recip()
            QyCoords:    qy values of the voxels, from Array_generate_coords_lab_recip()
            QzCoords:    qz values of the voxels, from Array_generate_coords_lab_recip()
            Rock_axis:   Axis of rocking curve scan
            
        * Detector is at the direct beam when both Delta and Gamma are zero. 
    '''
    # Input regularization
    if QxCoords is None or QyCoords is None or QzCoords is None: 
        print('>>>>>> Lack of coordinates! ')
        input('>>>>>> Press any key to quit...')
        return
    
    dimsX = np.shape(QxCoords)
    dimsY = np.shape(QyCoords)
    dimsZ = np.shape(QzCoords)
    
    if not (np.array_equal(dimsX, dimsY) and 
            np.array_equal(dimsY, dimsZ) and
            np.array_equal(dimsZ, dimsX) ):
        print('>>>>>> The shapes of Qx, Qy, Qz coords are not identical! ')
        input('>>>>>> Press any key to quit...')
        return
    Nz, Ny, Nx = dimsX
    N_elem = int(Nz*Ny*Nx)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    # Constants
    ki = 2 * np.pi / Lambda   # [A-1]
    
    Qx3 = QxCoords.ravel()
    Qy3 = QyCoords.ravel()
    Qz3 = QzCoords.ravel()
    Qmag2 = Qx3**2+Qy3**2+Qz3**2 # |Q|^2
    
    ''' >>> Math <<<  
        Assume the coordinate of a voxel in lab space is [Qx, Qy, Qz], 
        the cooresponding coordinate in detector space is [Delta, Gamma, Theta]. 
        
        Here, Delta and Gamma define the position of this voxel on the detector, 
        while Theta means AFTER rotating the vector by Theta, it satifies the 
        momentum transfer kf - ki = Q, where: 
                ki = [0, 0, k], in Cartesian
                kf = [k, gamma, delta] in Spherical
                
        Therefore, we have: 
                Rot(Theta) * [Qx, Qy, Qz] = [k*cos(gamma)*sin(delta), 
                                             k*sin(gamma), 
                                             k*(cos(gamma)*cos(delta)-1)]
            where, 
                RotY() = [ cos,    0,  sin, 
                             0,    1,    0, 
                          -sin,    0,  cos ]
                RotX() = [   1,    0,    0, 
                             0,  cos, -sin, 
                             0,  sin,  cos ]
        
        For Rotation around Y axis, after being rotated Theta, 
            the final vector Q' = RotY(Theta) * Q
                                = [ qx * cos(Theta) + qz * sin(Theta), 
                                    qy, 
                                   -qx * sin(Theta) + qz * cos(Theta) ]
            which gives
                    k*cos(Gamma)*sin(Delta) =  qx * cos(Theta) + qz * sin(Theta)
                               k*sin(Gamma) =  qy
                k*cos(Gamma)*cos(Delta) - k = -qx * sin(Theta) + qz * cos(Theta)
            
            therefore, 
                    Gamma = arcsin( qy/k )
                    Delta = arccos( (1-q**2/k**2/2) / cos(Gamma) )
                    Theta = arccos( k*(qx*cos(Gamma)*sin(Delta) + qz*(cos(Gamma)*cos(Delta)-1))
                                    / (qx**2 + qz**2) )
        
        For Rotation around X axis, after being rotated Theta, 
            the final vector Q' = RotX(-Theta) * Q
                                = [ qx, 
                                    qy * cos(Theta) + qz * sin(Theta), 
                                   -qy * sin(Theta) + qz * cos(Theta) ]
            which gives
                    k*cos(Gamma)*sin(Delta) =  qx
                               k*sin(Gamma) =  qy * cos(Theta) + qz * sin(Theta)
                k*cos(Gamma)*cos(Delta) - k = -qy * sin(Theta) + qz * cos(Theta)
            
            therefore, 
                    Delta = arctan( qx/k / (1-q**2/k**2/2) )
                    Gamma = arccos( qx / k / sin(Delta) )
                    Theta = arccos( (qy*k*sin(Gamma) + qz*k*(cos(Gamma)*cos(Delta)-1)) 
                                    / (qy**2 + qz**2) )
    '''
    
    # Coordinates transformation from detector to lab
    if Progress: 
        print('>>>>>> Transform coords to Detector space ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    if Rock_axis == 'Y' or Rock_axis == 'y': 
        '''Crystal is rotated around Y axis. '''
        Gamma3 = np.arcsin(Qy3 / ki)
        Delta3 = np.arccos((1 - Qmag2/2/ki**2) / np.cos(Gamma3))
        Theta3 = -np.arcsin(-ki/(Qx3**2+Qz3**2)*(Qx3*(1-np.cos(Gamma3)*np.cos(Delta3))
                                               +Qz3*np.cos(Gamma3)*np.sin(Delta3)))
        # Correct the sign of Delta
        Delta3[np.where(Qx3<0)] *= -1
    elif Rock_axis == 'X' or Rock_axis == 'x': 
        '''Crystal is rotated around X axis. '''
        Delta3 = np.arctan2(Qx3/ki, 1 - Qmag2/2/ki**2)
        Gamma3 = np.arccos(Qx3/ki/np.sin(Delta3))
        Theta3 = -np.arcsin(-ki/(Qy3**2+Qz3**2)*(Qy3*(np.cos(Gamma3)*np.cos(Delta3)-1)
                                               -Qz3*np.sin(Gamma3)))
        # Theta3 = np.arccos(ki*(Qy3*np.sin(Gamma3) + Qz3*(np.cos(Gamma3)*np.cos(Delta3)-1)) 
        #                             / (Qy3**2 + Qz3**2) )
        # Correct the sign of Gamma
        Gamma3[np.where(Qy3<0)] *= -1
    else: 
        print('>>>>>> Please check Rock_axis. ')
        input('>>>>>> Press any key to quit...')
        return
    
    Delta3 = Delta3.reshape((Nz, Ny, Nx))
    Gamma3 = Gamma3.reshape((Nz, Ny, Nx))
    Theta3 = Theta3.reshape((Nz, Ny, Nx))
    
    if Progress: 
        t_2 = time()
        print('>>>>>> Coordinates calculation took %0.6f sec in total.' %(t_2 - t_1))
    
    return {'Delta': Delta3, 'Gamma': Gamma3, 'Theta': Theta3}


def Array_coords_lab2crys_recip(QxCoords=None, QyCoords=None, QzCoords=None, Delta=0, Gamma=0, 
                                Crys_orie=[0, 0, 0], Lambda=None, PhotonE=None, 
                                IsRadian=True, Progress=False): 
    ''' >>> Instruction <<<
        This function converts the coordinates from lab frame to crystal frame, by rotating 
        the array around the center of the Bragg peak. 
        
        !!! The transformation of the array is not done in this function !!!
        !!! See below for the definition of the outputs !!!
        
        * Definitions of inputs: 
            QxCoords:    qx values of the voxels, from Array_generate_coords_lab_*()
            QyCoords:    qy values of the voxels, from Array_generate_coords_lab_*()
            QzCoords:    qz values of the voxels, from Array_generate_coords_lab_*()
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            Crys_orie    Crystal orientation in LAB FRAME
                         (3,) the EXTRICSIC rotation around X, Y, Z axes
        
        * Notice that QxCoords/QyCoords/QzCoords are centered at the origin of reciprocal space, 
            in lab frame.
        
        * Definitions of outputs: 
            Qx_lab:      qx values of the voxels, in lab frame, centered at Bragg peak. 
            Qy_lab:      qy values of the voxels, in lab frame, centered at Bragg peak.
            Qz_lab:      qz values of the voxels, in lab frame, centered at Bragg peak.
            Qx_crys:     qx values of the voxels, in crystal frame, centered at Bragg peak. 
            Qy_crys:     qy values of the voxels, in crystal frame, centered at Bragg peak.
            Qz_crys:     qz values of the voxels, in crystal frame, centered at Bragg peak.
                !!! Qx/Qy/Qz_lab/crys are related to the center of Bragg peak !!!
            Bragg_crys:  Vector of the Bragg peak in crystal frame. 
            Bragg_lab:   Vector of the Bragg peak in lab frame. 
        
        * Make sure using the same Delta/Gamma for generating lab coords. 
        
        * The transformation of Array is done using interpolation tools (e.g. eqtools).
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.
        
        * The unit of the output coordinates is [A^-1]
        
        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation. 
        
        * The rotation of crystal can be considered either globally or locally. 
            For global rotation, the Bragg peak's Q vector is also rotated;  
            For local  rotation, only the 3D array is rotated around its center. 
        
        * Qx/y/zCoords are 3D meshgrids aligned to the X/Y/Z axes of the reciprocal space in lab frame, 
            where the origin is at the origin of reciprocal space. 
          Qx/y/z_lab are 3D meshgrids aligned to the X/Y/Z axes of the reciprocal space in lab frame, 
            where the origin is at Bragg peak. 
          Qx/y/z_crys are rotated meshgrids that takes account of crystal rotations locally, 
            where the origin is at Bragg peak. 
          
        * For interpolation between lab frame and crystal frame, simply use
            [Qx_lab,  Qy_lab,  Qz_lab] and [Qx_crys, Qy_crys, Qz_crys] to interpolate. 
          For interpolation from detector frame to lab/crystal frame, should use
            Bragg_lab + [Qx_lab, Qy_lab, Qz_lab] for lab frame, and 
            Bragg_lab + [Qx_crys, Qy_crys, Qz_crys] for crystal frame. 
    '''
    # Input regularization
    if QxCoords is None or QyCoords is None or QzCoords is None: 
        print('>>>>>> Lack of coordinates! ')
        input('>>>>>> Press any key to quit...')
        return
    
    dimsX = np.shape(QxCoords)
    dimsY = np.shape(QyCoords)
    dimsZ = np.shape(QzCoords)
    
    if not (np.array_equal(dimsX, dimsY) and 
            np.array_equal(dimsY, dimsZ) and
            np.array_equal(dimsZ, dimsX) ):
        print('>>>>>> The shapes of Qx, Qy, Qz coords are not identical! ')
        input('>>>>>> Press any key to quit...')
        return
    Nz, Ny, Nx = dimsX
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        Crys_orie = np.deg2rad(Crys_orie)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE   # [A]
    
    # Constants
    WaveVec = 2 * np.pi / Lambda   # [A^-1]
    k_i = np.array([0, 0, WaveVec])
    k_f = Coordinates_rotate_3D(-Gamma, 0, k_i, IsRadian=True)
    k_f = Coordinates_rotate_3D( Delta, 1, k_f, IsRadian=True)
    Bragg_lab = k_f - k_i   # The Q vector of the Bragg peak in lab frame, [A-1]
    qx_0, qy_0, qz_0 = Bragg_lab
    
    # Bragg peak in crystal frame
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[2], 2, Bragg_lab,  IsRadian=True)
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[1], 1, Bragg_crys, IsRadian=True)
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[0], 0, Bragg_crys, IsRadian=True)
    
    # Vectors related to the center of the Bragg peak, in lab frame
    ''' Note: 
            The inputs Qx/Qy/QzCoords are regular coordinates in lab frame, 
            centered at the origin of reciprocal space.
            So it is used as the coordinates in lab frame, centered at the Bragg peak, 
            after substacting the Q-vector of Bragg peak in lab frame. 
    '''
    Qs_lab = np.array([QxCoords.ravel() - qx_0, 
                       QyCoords.ravel() - qy_0, 
                       QzCoords.ravel() - qz_0])
    
    # Transform to crystal frame
    RotX = Rot_Matrix([1,0,0], -Crys_orie[0], IsRadian=True)
    RotY = Rot_Matrix([0,1,0], -Crys_orie[1], IsRadian=True)
    RotZ = Rot_Matrix([0,0,1], -Crys_orie[2], IsRadian=True)
    
    Qs_crys = np.array(np.matmul(RotX, np.matmul(RotY, np.matmul(RotZ, Qs_lab))))
    
    return {'Qx_lab':  Qs_lab[0].reshape((Nz, Ny, Nx)), 
            'Qy_lab':  Qs_lab[1].reshape((Nz, Ny, Nx)), 
            'Qz_lab':  Qs_lab[2].reshape((Nz, Ny, Nx)), 
            'Qx_crys': Qs_crys[0].reshape((Nz, Ny, Nx)), 
            'Qy_crys': Qs_crys[1].reshape((Nz, Ny, Nx)), 
            'Qz_crys': Qs_crys[2].reshape((Nz, Ny, Nx)), 
            'Bragg_crys':Bragg_crys, 'Bragg_lab':Bragg_lab}


def Array_coords_crys2lab_recip(QxCoords=None, QyCoords=None, QzCoords=None, Delta=0, Gamma=0, 
                                Crys_orie=[0, 0, 0], Lambda=None, PhotonE=None, 
                                IsRadian=True, Progress=False): 
    ''' >>> Instruction <<< 
        This function converts the coordinates from crystal frame to lab frame, by rotating 
        the array around the center of the Bragg peak. 
        
        !!! The transformation of the array is not done in this function !!!
        !!! See below for the definition of the outputs !!!
        
        * Definitions of inputs: 
            QxCoords:    qx values of the voxels, from Array_generate_coords_lab_*()
            QyCoords:    qy values of the voxels, from Array_generate_coords_lab_*()
            QzCoords:    qz values of the voxels, from Array_generate_coords_lab_*()
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            Crys_orie    Crystal orientation in LAB FRAME
                         (3,) the EXTRICSIC rotation around X, Y, Z axes
        
        * Notice that QxCoords/QyCoords/QzCoords are centered at the origin of reciprocal space, 
            in crystal frame. 
        
        * Definitions of outputs: 
            Qx_lab:      qx values of the voxels, in lab frame, centered at Bragg peak. 
            Qy_lab:      qy values of the voxels, in lab frame, centered at Bragg peak.
            Qz_lab:      qz values of the voxels, in lab frame, centered at Bragg peak.
            Qx_crys:     qx values of the voxels, in crystal frame, centered at Bragg peak. 
            Qy_crys:     qy values of the voxels, in crystal frame, centered at Bragg peak.
            Qz_crys:     qz values of the voxels, in crystal frame, centered at Bragg peak.
                !!! Qx/Qy/Qz_lab/crys are related to the center of Bragg peak !!!
            Bragg_crys:  Vector of the Bragg peak center in crystal frame. 
            Bragg_lab:   Vector of the Bragg peak center in lab frame. 
        
        * Make sure using the same Delta/Gamma for generating lab coords. 
        
        * The transformation of Array is done using interpolation tools (e.g. eqtools).
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.
        
        * The unit of the output coordinates is [A^-1]
        
        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation. 
        
        * The rotation of crystal can be considered either globally or locally. 
            For global rotation, the Bragg peak's Q vector is also rotated;  
            For local  rotation, only the 3D array is rotated around its center. 
        
        * Qx/y/zCoords are 3D meshgrids aligned to the X/Y/Z axes of the reciprocal space in crystal frame, 
            where the origin is at the origin of reciprocal space. 
          Qx/y/z_crys are 3D meshgrids aligned to the X/Y/Z axes of the reciprocal space in crystal frame, 
            where the origin is at Bragg peak. 
          Qx/y/z_lab are rotated meshgrids that takes account of crystal rotations locally, 
            where the origin is at Bragg peak. 
        
        * For interpolation between lab frame and crystal frame, simply use
            [Qx_lab,  Qy_lab,  Qz_lab] and [Qx_crys, Qy_crys, Qz_crys] to interpolate. 
          For interpolation from lab/crystal frame to detector frame, should use
            Bragg_lab + [Qx_lab, Qy_lab, Qz_lab] for lab frame, and 
            Bragg_lab + [Qx_crys, Qy_crys, Qz_crys] for crystal frame.  
    '''
    # Input regularization
    if QxCoords is None or QyCoords is None or QzCoords is None: 
        print('>>>>>> Lack of coordinates! ')
        input('>>>>>> Press any key to quit...')
        return
    
    dimsX = np.shape(QxCoords)
    dimsY = np.shape(QyCoords)
    dimsZ = np.shape(QzCoords)
    
    if not (np.array_equal(dimsX, dimsY) and 
            np.array_equal(dimsY, dimsZ) and
            np.array_equal(dimsZ, dimsX) ):
        print('>>>>>> The shapes of Qx, Qy, Qz coords are not identical! ')
        input('>>>>>> Press any key to quit...')
        return
    Nz, Ny, Nx = dimsX
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        Crys_orie = np.deg2rad(Crys_orie)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE   # [A]
    
    # Constants
    WaveVec = 2 * np.pi / Lambda   # [A^-1]
    k_i = np.array([0, 0, WaveVec])
    k_f = Coordinates_rotate_3D(-Gamma, 0, k_i, IsRadian=True)
    k_f = Coordinates_rotate_3D( Delta, 1, k_f, IsRadian=True)
    Bragg_lab = k_f - k_i   # The Q vector of the Bragg peak in lab frame, [A-1]
    qx_0, qy_0, qz_0 = Bragg_lab
    
    # Bragg peak in crystal frame
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[2], 2, Bragg_lab,  IsRadian=True)
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[1], 1, Bragg_crys, IsRadian=True)
    Bragg_crys = Coordinates_rotate_3D(-Crys_orie[0], 0, Bragg_crys, IsRadian=True)
    
    # Vectors related to the origin of reciprocal space, in crystal frame
    ''' Note: 
            The inputs Qx/Qy/QzCoords are regular coordinates in crystal frame, 
            centered at the origin of reciprocal space.
            So it is used as the coordinates in crystal frame, centered at Bragg peak, 
            after substacting the Q-vector of Bragg peak in crystal frame. 
    '''
    Qs_crys = np.array([QxCoords.ravel() - Bragg_crys[0], 
                        QyCoords.ravel() - Bragg_crys[1], 
                        QzCoords.ravel() - Bragg_crys[2]])
    
    # Transform to crystal frame
    RotX = Rot_Matrix([1,0,0], Crys_orie[0], IsRadian=True)
    RotY = Rot_Matrix([0,1,0], Crys_orie[1], IsRadian=True)
    RotZ = Rot_Matrix([0,0,1], Crys_orie[2], IsRadian=True)
    
    Qs_lab = np.array(np.matmul(RotZ, np.matmul(RotY, np.matmul(RotX, Qs_crys))))
    
    return {'Qx_lab':  Qs_lab[0].reshape((Nz, Ny, Nx)), 
            'Qy_lab':  Qs_lab[1].reshape((Nz, Ny, Nx)), 
            'Qz_lab':  Qs_lab[2].reshape((Nz, Ny, Nx)), 
            'Qx_crys': Qs_crys[0].reshape((Nz, Ny, Nx)), 
            'Qy_crys': Qs_crys[1].reshape((Nz, Ny, Nx)), 
            'Qz_crys': Qs_crys[2].reshape((Nz, Ny, Nx)), 
            'Bragg_crys':Bragg_crys, 'Bragg_lab':Bragg_lab}


def Array_coords_lab2crys_real(Rx_lab=None, Ry_lab=None, Rz_lab=None, Crys_orie=[0, 0, 0], 
                               IsRadian=True, Progress=False): 
    ''' >>> Instruction <<< 
        This function converts the coordinates from lab frame to crystal frame in real space, 
        by rotating the array around the origin of the real space coordinates. 
        
        !!! The transformation of the array is not done in this function !!!
        !!! See below for the definition of the outputs !!!
        
        * Definitions of inputs: 
            Rx_lab:      rx values of the voxels, from Array_generate_coords_real_*()
            Ry_lab:      ry values of the voxels, from Array_generate_coords_real_*()
            Rz_lab:      rz values of the voxels, from Array_generate_coords_real_*()
            Crys_orie:   Crystal orientation in LAB FRAME
                         (3,) the EXTRICSIC rotation around X, Y, Z axes
        
        * Definitions of outputs: 
            Rx_crys:     rx values of the voxels, in crystal frame. 
            Ry_crys:     ry values of the voxels, in crystal frame.
            Rz_crys:     rz values of the voxels, in crystal frame.
        
        * The transformation of Array is done using interpolation tools (e.g. eqtools).
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.
        
        * The unit of the output coordinates is [A].
        
        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation. 
        
        * Rx/y/z_lab are 3D meshgrids aligned to the X/Y/Z axes of the real space in lab frame. 
          Rx/y/z_crys are rotated meshgrids that takes account of crystal rotations. 
    '''
    # Input regularization
    if Rx_lab is None or Ry_lab is None or Rz_lab is None: 
        print('>>>>>> Lack of coordinates! ')
        input('>>>>>> Press any key to quit...')
        return
    
    dimsX = np.shape(Rx_lab)
    dimsY = np.shape(Ry_lab)
    dimsZ = np.shape(Rz_lab)
    
    if not (np.array_equal(dimsX, dimsY) and 
            np.array_equal(dimsY, dimsZ) and
            np.array_equal(dimsZ, dimsX) ):
        print('>>>>>> The length of Rx, Ry, Rz coords are not identical! ')
        input('>>>>>> Press any key to quit...')
        return
    Nz, Ny, Nx = dimsX
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Crys_orie = np.deg2rad(Crys_orie)
    
    Rs_lab = np.array([Rx_lab.ravel(), Ry_lab.ravel(), Rz_lab.ravel()])
    
    # Transform to crystal frame
    RotX = Rot_Matrix([1,0,0], -Crys_orie[0], IsRadian=True)
    RotY = Rot_Matrix([0,1,0], -Crys_orie[1], IsRadian=True)
    RotZ = Rot_Matrix([0,0,1], -Crys_orie[2], IsRadian=True)
    
    Rs_crys = np.array(np.matmul(RotX, np.matmul(RotY, np.matmul(RotZ, Rs_lab))))
    
    return {'Rx_crys': Rs_crys[0].reshape((Nz, Ny, Nx)), 
            'Ry_crys': Rs_crys[1].reshape((Nz, Ny, Nx)), 
            'Rz_crys': Rs_crys[2].reshape((Nz, Ny, Nx)) }


def Array_coords_crys2lab_real(Rx_crys=None, Ry_crys=None, Rz_crys=None, Crys_orie=[0, 0, 0], 
                               IsRadian=True, Progress=False): 
    ''' >>> Instruction <<< 
        This function converts the coordinates from crystal frame to lab frame in real space, 
        by rotating the array around the origin of the real space coordinates. 
        
        !!! The transformation of the array is not done in this function !!!
        !!! See below for the definition of the outputs !!!
        
        * Definitions of inputs: 
            Rx_crys:     rx values of the voxels, from Array_generate_coords_real_*()
            Ry_crys:     ry values of the voxels, from Array_generate_coords_real_*()
            Rz_crys:     rz values of the voxels, from Array_generate_coords_real_*()
            Crys_orie:   Crystal orientation in LAB FRAME
                         (3,) the EXTRICSIC rotation around X, Y, Z axes
        
        * Definitions of outputs: 
            Rx_lab:      rx values of the voxels, in lab frame. 
            Ry_lab:      ry values of the voxels, in lab frame.
            Rz_lab:      rz values of the voxels, in lab frame.
        
        * The transformation of Array is done using interpolation tools (e.g. eqtools).
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.
        
        * The unit of the output coordinates is [A].
        
        * Lab coordinates are defined as: 
            +X => Outboard; +Y => Upward; +Z => Beam propagation. 
        
        * Rx/y/z_crys are 3D meshgrids aligned to the X/Y/Z axes of the real space in crystal frame. 
          Rx/y/z_lab are rotated meshgrids that takes account of crystal rotations. 
    '''
    # Input regularization
    if Rx_crys is None or Ry_crys is None or Rz_crys is None: 
        print('>>>>>> Lack of coordinates! ')
        input('>>>>>> Press any key to quit...')
        return
    
    dimsX = np.shape(Rx_crys)
    dimsY = np.shape(Ry_crys)
    dimsZ = np.shape(Rz_crys)
    
    if not (np.array_equal(dimsX, dimsY) and 
            np.array_equal(dimsY, dimsZ) and
            np.array_equal(dimsZ, dimsX) ):
        print('>>>>>> The length of Rx, Ry, Rz coords are not identical! ')
        input('>>>>>> Press any key to quit...')
        return
    Nz, Ny, Nx = dimsX
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Crys_orie = np.deg2rad(Crys_orie)
    
    Rs_crys = np.array([Rx_crys.ravel(), Ry_crys.ravel(), Rz_crys.ravel()])
    
    # Transform to crystal frame
    RotX = Rot_Matrix([1,0,0], Crys_orie[0], IsRadian=True)
    RotY = Rot_Matrix([0,1,0], Crys_orie[1], IsRadian=True)
    RotZ = Rot_Matrix([0,0,1], Crys_orie[2], IsRadian=True)
    
    Rs_lab = np.array(np.matmul(RotZ, np.matmul(RotY, np.matmul(RotX, Rs_crys))))
    
    return {'Rx_lab': Rs_lab[0].reshape((Nz, Ny, Nx)), 
            'Ry_lab': Rs_lab[1].reshape((Nz, Ny, Nx)), 
            'Rz_lab': Rs_lab[2].reshape((Nz, Ny, Nx)) }


''' >>> Coordinates transformation from detector frame to crystal frame in reciprocal space <<< 
    
    !!! This is for transforming the DIFFRACTION PATTERN from detector frame to crystal frame. !!!
    !!!       This does NOT consider the reciprocal space to real space transform.       !!!
    
        * Consider a detector-frame, 3D regular array that contains a centered diffraction pattern. 
        
        * Due to the nature of tricubic spline interpolation, it would be better interpolating a regular 
            grid array to an oblique irregular grid array. 
        
        * Therefore, the goal of this coordinates transformations is finding the detector-frame 
            coordinates of an array that is regular in crystal frame. 
        
        * To transform : 
        1)  Zeropad the data array, so that the array in detector frame covers certain Delta, Gamma, 
            and Theta ranges, which are larger than the ranges that the transformed array covers. 
        
        2)  Generate detector-frame coordinates, "DetCoords", using function "Array_generate_coords_det()"
            and the zeropadded array (for size only). 
                    "DetCoords" contain [Delta, Gamma, Theta], in [rad]
        
        3)  Generate lab-frame coordinates, "LabCoords", using function "Array_generate_coords_lab_recip()" 
            and the original array (for size only). 
                    "LabCoords" contian [Qx, Qy, Qz], in [A^-1]
        
        4)  Note that "LabCoords" are not regular in crystal frame. To take the rotation of crystal into 
            account, using function "Array_coords_lab2crys_recip()". 
        
        5)  From step 4), the coordinates that are regular in crystal frame, "CrysCoords", contain
                    [ 'Qx_lab' + 'Bragg_lab'[0], 
                      'Qy_lab' + 'Bragg_lab'[1], 
                      'Qz_lab' + 'Bragg_lab'[2]  ]
            where 'Qx/Qy/Qz_lab' and 'Bragg_lab' are outputs of "Array_coords_lab2crys_recip()". 
        
        6)  Convert "CrysCoords" from lab frame to detector frame, using "Array_coords_lab2det_recip()". 
            The output contain [Delta, Gamma, Theta], in [rad]. 
        
        7)  Interpolation. '''

''' >>> Coordinates transformation from crystal frame to detector frame in reciprocal space <<< 
    
    !!! This is for transforming the DIFFRACTION PATTERN from detector frame to crystal frame. !!!
    !!!       This does NOT consider the reciprocal space to real space transform.       !!!
    
        * The goal of this coordinates transformations is finding a crystal-frame 
            coordinates of an array that is regular in detector frame. 
        
        * To transform : 
        1)  Generate detector-frame coordinates, "DetCoords", using function "Array_generate_coords_det()". 
                    "DetCoords" contain [Delta, Gamma, Theta], in [rad]
                    
        2)  Convert "DetCoords" from detector frame to lab frame, using "Array_coords_det2lab_recip()". 
            The output contain [Qx, Qy, Qz], in [A^-1]. 
        
        3)  To take the rotation of crystal into account, using function "Array_coords_lab2crys_recip()".
            The coordinates that are regular in crystal frame, "CrysCoords", contain
                    [ 'Qx_lab' + 'Bragg_lab'[0], 
                      'Qy_lab' + 'Bragg_lab'[1], 
                      'Qz_lab' + 'Bragg_lab'[2]  ]
            where 'Qx/Qy/Qz_lab' and 'Bragg_lab' are outputs of "Array_coords_lab2crys_recip()". 
        
        
        
        
        1)  Zeropad the data array, so that the array in crystal frame covers certain X, Y, and Z ranges, 
            which are larger than the ranges that the transformed array covers. 
        
        2)  Generate crystal-frame coordinates, "CrysCoords", using function "Array_generate_coords_lab_recip()"
            and the zeropadded array (for size only). 
                    "CrysCoords" contain [Qx, Qy, Qz], in [A^-1]
        
        3)  To take the rotation of crystal into account, using function "Array_coords_lab2crys_recip()".
            The coordinates that are regular in crystal frame, "CrysCoords", contain
                    [ 'Qx_lab' + 'Bragg_lab'[0], 
                      'Qy_lab' + 'Bragg_lab'[1], 
                      'Qz_lab' + 'Bragg_lab'[2]  ]
            where 'Qx/Qy/Qz_lab' and 'Bragg_lab' are outputs of "Array_coords_lab2crys_recip()". 
        
        4)  Generate detector-frame coordinates, "DetCoords", using function "Array_generate_coords_det()". 
                    "DetCoords" contain [Delta, Gamma, Theta], in [rad]
        
        5)  Convert "DetCoords" from detector frame to lab frame, using "Array_coords_det2lab_recip()". 
            The output contain [Qx, Qy, Qz], in [A^-1]. 
        
        6)  Interpolation. '''

''' >>> Coordinate transformation in practice <<< 
    
    In practice, coordinate transformation should be done in two steps: 
    
    1) The first step uses a while{try-except} structure to determine the proper padding size for 
       each peak, calculates the cooridnates, and saves them for the second step. 
       
    2) The second step is in the loop of iterative phasing. It performs the transformations using 
       the coordinates calculated in the first step. 
    
    Given that, the four functions below, namely 
       "Array_det2crys_recip_interpolation_spline"
       "Array_det2crys_recip_interpolation_spline_NoPix"
       "Array_crys2det_recip_interpolation_spline"
       "Array_crys2det_recip_interpolation_spline_NoPix"
    are not useful in the real phasing code. '''


def Array_det2crys_recip_interpolation_spline(Data, Lambda=None, PhotonE=None, Crys_orie=[0,0,0], 
                                              Delta=None, Gamma=None, dRock=0, Rock_axis='Y', 
                                              detdist=None, px=None, py=None, IsRadian=True, 
                                              PadSize=None, PadIncr=[5,5,5], Progress=False): 
    ''' >>> Instruction <<< 
        This function wraps the detector-frame-to-crystal-frame-in-reciprocal-space interpolation 
        using tricubit spline from python module eqtools. 
        This function can transform the DIFFRACTION PATTERN from detector frame to crystal frame. 
        
        Inputs: 
            Data:        A 3D array. 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Crys_orie    Crystal orientation in LAB FRAME
                             (3,) the extrinsic rotation around X, Y, Z axes
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            dRock:       Step of rocking curve scan
            Rock_axis:   Axis of rocking curve scan
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, dYaw and dPitch are in [rad]; otherwise, [deg]
            PadSize:     (3,) Size of the padded array.
            PadIncr:     (3,) Increaments of the padding in three dimension. 
        
        Output: 
            Data_new     The interpolated array. 
    '''
    # Input regularization
    if Progress: 
        print('>>>>>> Input regularization ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    Crys_orie = np.array(Crys_orie)
    PadIncr = np.array(PadIncr)
    
    dims = np.array(np.shape(Data))
    if len(dims) == 3: 
        N_elem = int(dims[0]*dims[1]*dims[2])
    else: 
        print('>>>>>> The shape of "Data" is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    
    if not IsRadian: 
        Crys_orie = np.deg2rad(Crys_orie)
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dRock = np.deg2rad(dRock)
    
    if Lambda is None: 
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. No transformation needed. ')
        return Data
    
    # Coordinates in crystal frame
    if Progress: 
        print('>>>>>> Calculate coordinates in crystal frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    CrysCoords = Array_generate_coords_crys_recip(Data, Lambda=Lambda, PhotonE=PhotonE, Voxel=None, 
                                                  Crys_orie=Crys_orie, Delta=Delta, Gamma=Gamma, 
                                                  dRock=dRock, Rock_axis=Rock_axis, IsRadian=True, 
                                                  detdist=detdist, px=px, py=py)
    # Transform crystal frame coordinates to detector frame
    LabCoords = Array_coords_crys2lab_recip(QxCoords=CrysCoords['Qx'], 
                                            QyCoords=CrysCoords['Qy'], 
                                            QzCoords=CrysCoords['Qz'],  
                                            Delta=Delta, Gamma=Gamma, Crys_orie=Crys_orie, 
                                            Lambda=Lambda, PhotonE=PhotonE, IsRadian=True)
    CrysToDet = Array_coords_lab2det_recip(QxCoords=LabCoords['Qx_lab']+LabCoords['Bragg_lab'][0], 
                                           QyCoords=LabCoords['Qy_lab']+LabCoords['Bragg_lab'][1], 
                                           QzCoords=LabCoords['Qz_lab']+LabCoords['Bragg_lab'][2], 
                                           Lambda=Lambda,PhotonE=PhotonE,Rock_axis=Rock_axis)
    # Coordinates in detector frame
    if Progress: 
        print('>>>>>> Calculate coordinates in detector frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
        print('       Original array size is ', dims)
    
    Err_flag = True
    if (PadSize is None) or (not PadSize >= np.shape(Data)):
        dims_new = dims.copy()
    else: 
        dims_new = np.array(PadSize)
    
    Flip_Delta = False
    Flip_Gamma = False
    Flip_Theta = False
    
    while Err_flag: 
        Data_pad = Array_zeropad(Data, AddTo=dims_new)
        DetCoords = Array_generate_coords_det(Data_pad, Lambda=Lambda, PhotonE=PhotonE, 
                                              Voxel=None, Delta=Delta, Gamma=Gamma, 
                                              dRock=dRock, Rock_axis=Rock_axis, 
                                              detdist=detdist, px=px, py=py, IsRadian=True, 
                                              OutputRadian=True)
        # Check if the axes are monotonic
        try: 
            if Flip_Theta: 
                Theta_1D = np.flip(DetCoords['Theta_1D'])
                Data_pad = np.flip(Data_pad, axis=0)
            else: 
                Theta_1D = DetCoords['Theta_1D']
            if Flip_Gamma: 
                Gamma_1D = np.flip(DetCoords['Gamma_1D'])
                Data_pad = np.flip(Data_pad, axis=1)
            else: 
                Gamma_1D = DetCoords['Gamma_1D']
            if Flip_Delta: 
                Delta_1D = np.flip(DetCoords['Delta_1D'])
                Data_pad = np.flip(Data_pad, axis=2)
            else: 
                Delta_1D = DetCoords['Delta_1D']
            TriSpline_function = Spline(Theta_1D, Gamma_1D, Delta_1D, Data_pad)
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', flipped ... ')
            if ErrM.endswith('is not monotonic'): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    if Flip_Delta is True: 
                        print('>>>>>> Flip_Delta is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Delta = True
                if Ax == 'y': 
                    if Flip_Gamma is True: 
                        print('>>>>>> Flip_Gamma is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Gamma = True
                if Ax == 'z': 
                    if Flip_Theta is True: 
                        print('>>>>>> Flip_Theta is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Theta = True
                continue
            else: 
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
        
        # Check if the ranges are too narrow
        try: 
            Data_new = TriSpline_function.ev(CrysToDet['Theta'].ravel(), 
                                             CrysToDet['Gamma'].ravel(), 
                                             CrysToDet['Delta'].ravel())
            if Progress: 
                print('')
                print('       Padded array size is   ', np.array(np.shape(Data_pad)))
            Data_new = np.reshape(Data_new, dims)
            Err_flag = False
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', padding ', dims_new-dims, end='\r')
            if ErrM.endswith('exceeds bounds of interpolation grid '): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    dims_new[2] = dims_new[2] + PadIncr[2]
                if Ax == 'y': 
                    dims_new[1] = dims_new[1] + PadIncr[1]
                if Ax == 'z': 
                    dims_new[0] = dims_new[0] + PadIncr[0]
            else: 
                print('\n>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
    
    if Progress: 
        print('>>>>>> Interpolation finished.    ' 
              + strftime('%H:%M:%S', localtime()))
        print('       Output array size is   ', np.array(np.shape(Data_new)))
        t_4 = time()
        
        plt.figure(figsize=(7.5, 3))
        plt.subplot(131)
        plt.plot(DetCoords['Delta'].ravel(), alpha=0.5)
        plt.plot(CrysToDet['Delta'].ravel(), alpha=0.5)
        plt.title('Delta')
        
        plt.subplot(132)
        plt.plot(DetCoords['Gamma'].ravel(), alpha=0.5)
        plt.plot(CrysToDet['Gamma'].ravel(), alpha=0.5)
        plt.title('Gamma')
        
        plt.subplot(133)
        plt.plot(DetCoords['Theta'].ravel(), alpha=0.5)
        plt.plot(CrysToDet['Theta'].ravel(), alpha=0.5)
        plt.title('Theta')
        
        plt.tight_layout()
    return {'Data': Data_new, 'PadTo': dims_new}


def Array_det2crys_recip_interpolation_spline_NoPix(Data, Lambda=None, PhotonE=None, Crys_orie=[0,0,0], 
                                                    Delta=None, Delt_step=0.03, 
                                                    Gamma=None, Gamm_step=0.03, 
                                                    Rock_step=0.068, Rock_axis='Y', IsRadian=True, 
                                                    PadSize=None, PadIncr=[5,5,5], Progress=False): 
    ''' >>> Instruction <<< 
        This function wraps the detector-frame-to-crystal-frame-in-reciprocal-space interpolation 
        using tricubit spline from python module eqtools. 
        This function can transform the DIFFRACTION PATTERN from detector frame to crystal frame. 
        !!! This is for the simulated data that use angular steps instead of pixels !!!
        
        Inputs: 
            Data:        A 3D array. 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Crys_orie    Crystal orientation in LAB FRAME
                             (3,) the extrinsic rotation around X, Y, Z axes
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            Delt_step:   Step of Delta angles
            Gamm_step:   Step of Gamma angles
            Rock_step:   Step of rocking curve scan
            Rock_axis:   Axis of rocking curve scan
            IsRadian:    If True, dYaw and dPitch are in [rad]; otherwise, [deg]
            PadSize:     (3,) Size of the padded array.
            PadIncr:     (3,) Increaments of the padding in three dimension. 
        
        Output: 
            Data_new     The interpolated array. 
    '''
    # Input regularization
    if Progress: 
        print('>>>>>> Input regularization ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    Crys_orie = np.array(Crys_orie)
    PadIncr = np.array(PadIncr)
    
    dims = np.array(np.shape(Data))
    if len(dims) == 3: 
        N_elem = int(dims[0]*dims[1]*dims[2])
    else: 
        print('>>>>>> The shape of "Data" is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    
    if not IsRadian: 
        Crys_orie = np.deg2rad(Crys_orie)
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        Delt_step = np.deg2rad(Delt_step)
        Gamm_step = np.deg2rad(Gamm_step)
        Rock_step = np.deg2rad(Rock_step)
    
    if Lambda is None: 
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. No transformation needed. ')
        return Data
    
    # Coordinates in crystal frame
    if Progress: 
        print('>>>>>> Calculate coordinates in crystal frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    CrysCoords = Array_generate_coords_crys_recip_NoPix(Data, Lambda=Lambda, PhotonE=PhotonE, Voxel=None, 
                                                        Crys_orie=Crys_orie, Delta=Delta, Gamma=Gamma, 
                                                        Delt_step=Delt_step, Gamm_step=Gamm_step, 
                                                        Rock_step=Rock_step, Rock_axis=Rock_axis, 
                                                        IsRadian=True)
    # Transform crystal frame coordinates to detector frame
    LabCoords = Array_coords_crys2lab_recip(QxCoords=CrysCoords['Qx'], 
                                            QyCoords=CrysCoords['Qy'], 
                                            QzCoords=CrysCoords['Qz'],  
                                            Delta=Delta, Gamma=Gamma, Crys_orie=Crys_orie, 
                                            Lambda=Lambda, PhotonE=PhotonE, IsRadian=True)
    CrysToDet = Array_coords_lab2det_recip(QxCoords=LabCoords['Qx_lab']+LabCoords['Bragg_lab'][0], 
                                           QyCoords=LabCoords['Qy_lab']+LabCoords['Bragg_lab'][1], 
                                           QzCoords=LabCoords['Qz_lab']+LabCoords['Bragg_lab'][2], 
                                           Lambda=Lambda, PhotonE=PhotonE, Rock_axis=Rock_axis)
    # Coordinates in detector frame
    if Progress: 
        print('>>>>>> Calculate coordinates in detector frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
        print('       Original array size is ', dims)
    
    Err_flag = True
    if (PadSize is None) or (not PadSize >= np.shape(Data)):
        dims_new = dims.copy()
    else: 
        dims_new = np.array(PadSize)
    
    Flip_Delta = False
    Flip_Gamma = False
    Flip_Theta = False
    
    while Err_flag: 
        Data_pad = Array_zeropad(Data, AddTo=dims_new)
        DetCoords = Array_generate_coords_det_NoPix(Data_pad, Lambda=Lambda, PhotonE=PhotonE, 
                                                    Voxel=None, Delta=Delta, Gamma=Gamma, 
                                                    Delt_step=Delt_step, Gamm_step=Gamm_step, 
                                                    Rock_step=Rock_step, Rock_axis=Rock_axis, 
                                                    IsRadian=True, OutputRadian=True)
        # Check if the axes are monotonic
        try: 
            if Flip_Theta: 
                Theta_1D = np.flip(DetCoords['Theta_1D'])
                Data_pad = np.flip(Data_pad, axis=0)
            else: 
                Theta_1D = DetCoords['Theta_1D']
            if Flip_Gamma: 
                Gamma_1D = np.flip(DetCoords['Gamma_1D'])
                Data_pad = np.flip(Data_pad, axis=1)
            else: 
                Gamma_1D = DetCoords['Gamma_1D']
            if Flip_Delta: 
                Delta_1D = np.flip(DetCoords['Delta_1D'])
                Data_pad = np.flip(Data_pad, axis=2)
            else: 
                Delta_1D = DetCoords['Delta_1D']
            TriSpline_function = Spline(Theta_1D, Gamma_1D, Delta_1D, Data_pad)
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', flipped ... ')
            if ErrM.endswith('is not monotonic'): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    if Flip_Delta is True: 
                        print('>>>>>> Flip_Delta is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Delta = True
                if Ax == 'y': 
                    if Flip_Gamma is True: 
                        print('>>>>>> Flip_Gamma is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Gamma = True
                if Ax == 'z': 
                    if Flip_Theta is True: 
                        print('>>>>>> Flip_Theta is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Theta = True
                continue
            else: 
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
        
        # Check if the ranges are too narrow
        try: 
            Data_new = TriSpline_function.ev(CrysToDet['Theta'].ravel(), 
                                             CrysToDet['Gamma'].ravel(), 
                                             CrysToDet['Delta'].ravel())
            if Progress: 
                print('')
                print('       Padded array size is   ', np.array(np.shape(Data_pad)))
            Data_new = np.reshape(Data_new, dims)
            Err_flag = False
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', padding ', dims_new-dims, end='\r')
            if ErrM.endswith('exceeds bounds of interpolation grid '): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    dims_new[2] = dims_new[2] + PadIncr[2]
                if Ax == 'y': 
                    dims_new[1] = dims_new[1] + PadIncr[1]
                if Ax == 'z': 
                    dims_new[0] = dims_new[0] + PadIncr[0]
            else: 
                print('\n>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
    
    if Progress: 
        print('>>>>>> Interpolation finished.    ' 
              + strftime('%H:%M:%S', localtime()))
        print('       Output array size is   ', np.array(np.shape(Data_new)))
        t_4 = time()
        
        plt.figure(figsize=(7.5, 3))
        plt.subplot(131)
        plt.plot(DetCoords['Delta'].ravel(), alpha=0.5)
        plt.plot(CrysToDet['Delta'].ravel(), alpha=0.5)
        plt.title('Delta')
        
        plt.subplot(132)
        plt.plot(DetCoords['Gamma'].ravel(), alpha=0.5)
        plt.plot(CrysToDet['Gamma'].ravel(), alpha=0.5)
        plt.title('Gamma')
        
        plt.subplot(133)
        plt.plot(DetCoords['Theta'].ravel(), alpha=0.5)
        plt.plot(CrysToDet['Theta'].ravel(), alpha=0.5)
        plt.title('Theta')
        
        plt.tight_layout()
    return {'Data': Data_new, 'PadTo': dims_new}


def Array_crys2det_recip_interpolation_spline(Data, Lambda=None, PhotonE=None, Crys_orie=[0,0,0], 
                                              Delta=None, Gamma=None, dRock=0, Rock_axis='Y', 
                                              detdist=None, px=None, py=None, IsRadian=True, 
                                              PadSize=None, PadIncr=[5,5,5], Progress=False): 
    ''' >>> Instruction <<< 
        This function wraps the crystal-frame-to-detector-frame-in-reciprocal-space interpolation 
        using tricubit spline from python module eqtools. 
        This function can transform the DIFFRACTION PATTERN from crystal frame to detector frame. 
        
        Inputs: 
            Data:        The 3D diffraction pattern. 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Crys_orie    Crystal orientation in LAB FRAME
                             (3,) the extrinsic rotation around X, Y, Z axes
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            dRock:       Step of rocking curve scan
            Rock_axis:   Axis of rocking curve scan
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, Delta, Gamma, dRock, Crys_orie are in [rad]; Otherwise, [deg]
            PadSize:     (3,) Size of the padded array.
            PadIncr:     (3,) Increaments of the padding in three dimension. 
        
        Output: 
            Data_new     The interpolated 3D diffracton pattern. 
    '''
    # Input regularization
    if Progress: 
        print('>>>>>> Input regularization ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    Crys_orie = np.array(Crys_orie)
    PadIncr = np.array(PadIncr)
    
    dims = np.array(np.shape(Data))
    if len(dims) == 3: 
        N_elem = int(dims[0]*dims[1]*dims[2])
    else: 
        print('>>>>>> The shape of "Data" is not correct! ')
        input('>>>>>> Press any key to quit...')
        return
    
    if not IsRadian: 
        Crys_orie = np.deg2rad(Crys_orie)
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dRock = np.deg2rad(dRock)
    
    if Lambda is None: 
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. No transformation needed. ')
        return Data
    
    # Coordinates in detector frame
    if Progress: 
        print('>>>>>> Calculate coordinates in detector frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    DetCoords = Array_generate_coords_det(Data, Lambda=Lambda, PhotonE=PhotonE, 
                                          Voxel=None, Delta=Delta, Gamma=Gamma, 
                                          dRock=dRock, Rock_axis=Rock_axis, 
                                          detdist=detdist, px=px, py=py, IsRadian=True, 
                                          OutputRadian=True)
    
    # Transform detector frame coordinates to crystal frame
    LabCoords = Array_coords_det2lab_recip(DeltaCoords=DetCoords['Delta'], 
                                           GammaCoords=DetCoords['Gamma'], 
                                           ThetaCoords=DetCoords['Theta'], 
                                           Lambda=Lambda, PhotonE=PhotonE, 
                                           Rock_axis=Rock_axis, Progress=False)
    DetToCrys = Array_coords_lab2crys_recip(QxCoords=LabCoords['Qx'], 
                                            QyCoords=LabCoords['Qy'], 
                                            QzCoords=LabCoords['Qz'],  
                                            Delta=Delta, Gamma=Gamma, Crys_orie=Crys_orie, 
                                            Lambda=Lambda, PhotonE=PhotonE, IsRadian=True)
    
    # Coordinates in crystal frame
    if Progress: 
        print('>>>>>> Calculate coordinates in crystal frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
        print('       Original array size is ', dims)
    
    Err_flag = True
    if (PadSize is None) or (not PadSize >= np.shape(Data)):
        dims_new = dims.copy()
    else: 
        dims_new = np.array(PadSize)
    
    Flip_X = False
    Flip_Y = False
    Flip_Z = False
    
    while Err_flag: 
        Data_pad = Array_zeropad(Data, AddTo=dims_new)
        CrysCoords = Array_generate_coords_crys_recip(Data_pad, Lambda=Lambda, PhotonE=PhotonE, 
                                                      Voxel=None, Crys_orie=Crys_orie,
                                                      Delta=Delta, Gamma=Gamma, dRock=dRock, 
                                                      Rock_axis=Rock_axis, detdist=detdist, 
                                                      px=px, py=py, IsRadian=True, 
                                                      OutputRadian=True)
        # Check if the axes are monotonic
        try: 
            if Flip_Z: 
                Qz_1D = np.flip(CrysCoords['Qz_1D'])
                Data_pad = np.flip(Data_pad, axis=0)
            else: 
                Qz_1D = CrysCoords['Qz_1D']
            if Flip_Y: 
                Qy_1D = np.flip(CrysCoords['Qy_1D'])
                Data_pad = np.flip(Data_pad, axis=1)
            else: 
                Qy_1D = CrysCoords['Qy_1D']
            if Flip_X: 
                Qx_1D = np.flip(CrysCoords['Qx_1D'])
                Data_pad = np.flip(Data_pad, axis=2)
            else: 
                Qx_1D = CrysCoords['Qx_1D']
            TriSpline_function = Spline(Qz_1D, Qy_1D, Qx_1D, Data_pad)
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', flipped ... ')
            if ErrM.endswith('is not monotonic'): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    if Flip_X is True: 
                        print('>>>>>> Flip_X is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_X = True
                if Ax == 'y': 
                    if Flip_Y is True: 
                        print('>>>>>> Flip_Y is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Y = True
                if Ax == 'z': 
                    if Flip_Z is True: 
                        print('>>>>>> Flip_Z is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Z = True
                continue
            else: 
                print(ErrM)
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
        
        # Check if the ranges are too narrow
        try: 
            Data_new = TriSpline_function.ev(np.ravel(DetToCrys['Qz_crys']+DetToCrys['Bragg_crys'][2]), 
                                             np.ravel(DetToCrys['Qy_crys']+DetToCrys['Bragg_crys'][1]), 
                                             np.ravel(DetToCrys['Qx_crys']+DetToCrys['Bragg_crys'][0]))
            if Progress: 
                print('')
                print('       Padded array size is   ', np.array(np.shape(Data_pad)))
            Data_new = np.reshape(Data_new, dims)
            Err_flag = False
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', padding ', dims_new-dims, end='\r')
            if ErrM.endswith('exceeds bounds of interpolation grid '): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    dims_new[2] = dims_new[2] + PadIncr[2]
                if Ax == 'y': 
                    dims_new[1] = dims_new[1] + PadIncr[1]
                if Ax == 'z': 
                    dims_new[0] = dims_new[0] + PadIncr[0]
            else: 
                print(ErrM)
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
    
    if Progress: 
        print('>>>>>> Interpolation finished.    ' 
              + strftime('%H:%M:%S', localtime()))
        print('       Output array size is   ', np.array(np.shape(Data_new)))
        t_4 = time()
        
        plt.figure(figsize=(7.5, 3))
        plt.subplot(131)
        plt.plot(CrysCoords['Qx'].ravel(), alpha=0.5)
        plt.plot(np.ravel(DetToCrys['Qx_crys']+DetToCrys['Bragg_crys'][0]), alpha=0.5)
        plt.title('Qx')
        
        plt.subplot(132)
        plt.plot(CrysCoords['Qy'].ravel(), alpha=0.5)
        plt.plot(np.ravel(DetToCrys['Qy_crys']+DetToCrys['Bragg_crys'][1]), alpha=0.5)
        plt.title('Qy')
        
        plt.subplot(133)
        plt.plot(CrysCoords['Qz'].ravel(), alpha=0.5)
        plt.plot(np.ravel(DetToCrys['Qz_crys']+DetToCrys['Bragg_crys'][2]), alpha=0.5)
        plt.title('Qz')
        
        plt.tight_layout()
    return {'Data': Data_new, 'PadTo': dims_new}


def Array_crys2det_recip_interpolation_spline_NoPix(Data, Lambda=None, PhotonE=None, Crys_orie=[0,0,0], 
                                                    Delta=None, Delt_step=0.03, 
                                                    Gamma=None, Gamm_step=0.03, 
                                                    Rock_step=0.068, Rock_axis='Y', IsRadian=True, 
                                                    PadSize=None, PadIncr=[5,5,5], Progress=False): 
    ''' >>> Instruction <<< 
        This function wraps the crystal-frame-to-detector-frame-in-reciprocal-space interpolation 
        using tricubit spline from python module eqtools. 
        This function can transform the DIFFRACTION PATTERN from crystal frame to detector frame. 
        !!! This is for the simulated data that use angular steps instead of pixels !!!
        
        Inputs: 
            Data:        The 3D diffraction pattern. 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Crys_orie    Crystal orientation in LAB FRAME
                             (3,) the extrinsic rotation around X, Y, Z axes
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            dRock:       Step of rocking curve scan
            Rock_axis:   Axis of rocking curve scan
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, Delta, Gamma, dRock, Crys_orie are in [rad]; Otherwise, [deg]
            PadSize:     (3,) Size of the padded array.
            PadIncr:     (3,) Increaments of the padding in three dimension. 
        
        Output: 
            Data_new     The interpolated 3D diffracton pattern. 
    '''
    # Input regularization
    if Progress: 
        print('>>>>>> Input regularization ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    Crys_orie = np.array(Crys_orie)
    PadIncr = np.array(PadIncr)
    
    dims = np.array(np.shape(Data))
    if len(dims) == 3: 
        N_elem = int(dims[0]*dims[1]*dims[2])
    else: 
        print('>>>>>> The shape of "Data" is not correct! ')
        input('>>>>>> Press any key to quit...')
        return
    
    if not IsRadian: 
        Crys_orie = np.deg2rad(Crys_orie)
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        Delt_step = np.deg2rad(Delt_step)
        Gamm_step = np.deg2rad(Gamm_step)
        Rock_step = np.deg2rad(Rock_step)
    
    if Lambda is None: 
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. No transformation needed. ')
        return Data
    
    # Coordinates in detector frame
    if Progress: 
        print('>>>>>> Calculate coordinates in detector frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    DetCoords = Array_generate_coords_det_NoPix(Data, Lambda=Lambda, PhotonE=PhotonE, 
                                                Voxel=None, Delta=Delta, Gamma=Gamma, 
                                                Delt_step=Delt_step, Gamm_step=Gamm_step, 
                                                Rock_step=Rock_step, Rock_axis=Rock_axis, 
                                                IsRadian=True, OutputRadian=True)
    
    # Transform detector frame coordinates to crystal frame
    LabCoords = Array_coords_det2lab_recip(DeltaCoords=DetCoords['Delta'], 
                                           GammaCoords=DetCoords['Gamma'], 
                                           ThetaCoords=DetCoords['Theta'], 
                                           Lambda=Lambda, PhotonE=PhotonE, 
                                           Rock_axis=Rock_axis, Progress=False)
    DetToCrys = Array_coords_lab2crys_recip(QxCoords=LabCoords['Qx'], 
                                            QyCoords=LabCoords['Qy'], 
                                            QzCoords=LabCoords['Qz'],  
                                            Delta=Delta, Gamma=Gamma, Crys_orie=Crys_orie, 
                                            Lambda=Lambda, PhotonE=PhotonE, IsRadian=True)
    
    # Coordinates in crystal frame
    if Progress: 
        print('>>>>>> Calculate coordinates in crystal frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
        print('       Original array size is ', dims)
    
    Err_flag = True
    if (PadSize is None) or (not PadSize >= np.shape(Data)):
        dims_new = dims.copy()
    else: 
        dims_new = np.array(PadSize)
    
    Flip_X = False
    Flip_Y = False
    Flip_Z = False
    
    while Err_flag: 
        Data_pad = Array_zeropad(Data, AddTo=dims_new)
        CrysCoords = Array_generate_coords_crys_recip_NoPix(Data_pad, Lambda=Lambda, PhotonE=PhotonE, 
                                                            Voxel=None, Crys_orie=Crys_orie, 
                                                            Delta=Delta, Delt_step=Delt_step, 
                                                            Gamma=Gamma, Gamm_step=Gamm_step, 
                                                            Rock_step=Rock_step, Rock_axis=Rock_axis, 
                                                            IsRadian=True)
        # Check if the axes are monotonic
        try: 
            if Flip_Z: 
                Qz_1D = np.flip(CrysCoords['Qz_1D'])
                Data_pad = np.flip(Data_pad, axis=0)
            else: 
                Qz_1D = CrysCoords['Qz_1D']
            if Flip_Y: 
                Qy_1D = np.flip(CrysCoords['Qy_1D'])
                Data_pad = np.flip(Data_pad, axis=1)
            else: 
                Qy_1D = CrysCoords['Qy_1D']
            if Flip_X: 
                Qx_1D = np.flip(CrysCoords['Qx_1D'])
                Data_pad = np.flip(Data_pad, axis=2)
            else: 
                Qx_1D = CrysCoords['Qx_1D']
            TriSpline_function = Spline(Qz_1D, Qy_1D, Qx_1D, Data_pad)
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', flipped ... ')
            if ErrM.endswith('is not monotonic'): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    if Flip_X is True: 
                        print('>>>>>> Flip_X is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_X = True
                if Ax == 'y': 
                    if Flip_Y is True: 
                        print('>>>>>> Flip_Y is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Y = True
                if Ax == 'z': 
                    if Flip_Z is True: 
                        print('>>>>>> Flip_Z is already True !')
                        input('>>>>>> Press any key to quit...')
                        return
                    Flip_Z = True
                continue
            else: 
                print(ErrM)
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
        
        # Check if the ranges are too narrow
        try: 
            Data_new = TriSpline_function.ev(np.ravel(DetToCrys['Qz_crys']+DetToCrys['Bragg_crys'][2]), 
                                             np.ravel(DetToCrys['Qy_crys']+DetToCrys['Bragg_crys'][1]), 
                                             np.ravel(DetToCrys['Qx_crys']+DetToCrys['Bragg_crys'][0]))
            if Progress: 
                print('')
                print('       Padded array size is   ', np.array(np.shape(Data_pad)))
            Data_new = np.reshape(Data_new, dims)
            Err_flag = False
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', padding ', dims_new-dims, end='\r')
            if ErrM.endswith('exceeds bounds of interpolation grid '): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    dims_new[2] = dims_new[2] + PadIncr[2]
                if Ax == 'y': 
                    dims_new[1] = dims_new[1] + PadIncr[1]
                if Ax == 'z': 
                    dims_new[0] = dims_new[0] + PadIncr[0]
            else: 
                print(ErrM)
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
    
    if Progress: 
        print('>>>>>> Interpolation finished.    ' 
              + strftime('%H:%M:%S', localtime()))
        print('       Output array size is   ', np.array(np.shape(Data_new)))
        t_4 = time()
        
        plt.figure(figsize=(7.5, 3))
        plt.subplot(131)
        plt.plot(CrysCoords['Qx'].ravel(), alpha=0.5)
        plt.plot(np.ravel(DetToCrys['Qx_crys']+DetToCrys['Bragg_crys'][0]), alpha=0.5)
        plt.title('Qx')
        
        plt.subplot(132)
        plt.plot(CrysCoords['Qy'].ravel(), alpha=0.5)
        plt.plot(np.ravel(DetToCrys['Qy_crys']+DetToCrys['Bragg_crys'][1]), alpha=0.5)
        plt.title('Qy')
        
        plt.subplot(133)
        plt.plot(CrysCoords['Qz'].ravel(), alpha=0.5)
        plt.plot(np.ravel(DetToCrys['Qz_crys']+DetToCrys['Bragg_crys'][2]), alpha=0.5)
        plt.title('Qz')
        
        plt.tight_layout()
    return {'Data': Data_new, 'PadTo': dims_new}








# Old functions, saved here for reference
'''
def Gaussian_fitting(data, n=1, p0=[[1, 0, 1], [1, 0, 1]], ManualFit=False, PlotResult=False): 
    ''
    Multi-gaussian fitting
    p0 = [[a1, b1, c1], [a2, b2, c2], ...]
    y = a * exp{ -(x-b)**2/(2*c**2) }
    
    n is the number of gaussians for fitting
    p0 is an array of initial parameters
    ''
    # Function for fitting
    def Gaussians(x, *args): 
        N = int(np.size(args)/3)   # N is the number of Gaussians
        params = np.reshape(args, (N,3))
        y = 0
        for i in range(N): 
            y = y + params[i,0] * np.exp( -(x-params[i,1])**2/(2*params[i,2]**2) )
        return y
    # ============================================================
    # Input regularization
    data = np.array(data)
    p0 = np.array(p0)
    
    M = np.size(data)
    data = data.reshape((M,))
    
    N,L = np.shape(p0)   # N is the number of Gaussians
    if L != 3: 
        print('>>>>>> The shape of p0 is (%d, %d). <<<<<< ' %(N, L) )
        print('>>>>>> p0 should be a (n, 3) array. <<<<<< ')
        input('>>>>>> Press Ctrl + C to quit ... ')
        return
    elif N != n: 
        print('>>>>>> The shape of p0 does not match the number of Gaussians. <<<<<< ')
        print('>>>>>> p0 defines %d Gaussians. But n = %d. <<<<<< ' %(N, n) )
        print('>>>>>> Force p0 to match the number of Gaussians. <<<<<< ')
        p0 = np.zeros((n, 3))
        p0[:, 0] = 1
        p0[:, 2] = 1
    # ============================================================
    # Fitting
    X = sp.asarray(range(M))
    if ManualFit:
        popt = p0
    else:
        popt, pcov = curve_fit(Gaussians, X, data, p0=p0)
    
    fitData = Gaussians(X, *popt)
    
    if PlotResult:     
        plt.figure()
        plt.plot(X, data, label='Data')
        plt.plot(X, fitData, label='Fit')
        plt.legend()
    
    return {'Params': np.reshape(popt, (N,3)), 'fitData': fitData}    


def Array_shift_rot_FFT(Array, Shift=[0,0,0], Rotation=[0,0,0], 
                        RotFirst=True, Centering=False, Progress=False): 
    ''
    3D array shift and rotation using FFT. 
    
    Input: 
        Shift:      (3,) list, 3D shifts in pixels
                e.g. Shift[1] = 2 means Array is shifted 2 pixels along axis=1
        Rotation:   (3,) list, 3D rotations in degree
                e.g. Rotation[0] = 30 means Array is rotated 30 deg around axis=0
        RotFirst:   True  means Array is rotated first and then shifted
                    False means Array is shifted first and then rotated
        Centering:  If True, Array is centered via COM first, then rotated, then shifted back.
        Progress:   True will display the time stamps of each step
    
    ''
    # ====== Input regularization ======
    Array = np.array(Array)
    Shift = np.array(Shift)
    Rot = np.array(np.deg2rad(Rotation))
    dims = np.shape(Array)
    if not len(dims) == len(Shift):
        print(">>>>>> 'Shift' must have same number of dimensions as the array! ")
        input('>>>>>> Press any key to quit ... ')
        return
    if not len(dims) == len(Rot):
        print(">>>>>> 'Rotation' must have same number of dimensions as the array! ")
        input('>>>>>> Press any key to quit ... ')
        return
    
    # ====== FFT ======
    if Progress: 
        print('>>>>>> FFT the array ... ' + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    ftarr = sf.fftn(Array)
    
    # ====== Creat 3D meshgrid ======
    if Progress: 
        print('>>>>>> Creat the meshgrid ... ' + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    r=[]
    for d in dims:
        r.append(slice(int(np.ceil(-d/2.)), int(np.ceil(d/2.)), None))
    idxgrid = np.mgrid[r]
    
    # ====== Perform the transformation ======
    if Progress: 
        print('>>>>>> Transform the array ... ' + strftime('%H:%M:%S', localtime()))
        t_3 = time()
    # ------ Shift, if RotFirst is False ------
    if not RotFirst: 
        if Progress: 
            print('           Shifting ... ' + strftime('%H:%M:%S', localtime()))
        for d in range(len(dims)): 
            if Shift[d] == 0:    # Skip this axis if no shift
                continue
            ftarr *= np.exp(-1j*2*np.pi*Shift[d]*sf.fftshift(idxgrid[d])/float(dims[d]))
    # ------ Center the COM, if Centering is True ------
    if Centering: 
        if Progress: 
            print('           Centering ... ' + strftime('%H:%M:%S', localtime()))
        COM = spmsu.center_of_mass(Array)  # COM of the original array
        if not RotFirst:   # If shift has been performed, COM is shifted
            COM = COM + Shift 
        # The 'shift' for centering the COM
        CenterShift = np.floor(np.array(dims)/2) - COM
        # Shift the COM to center
        for d in range(len(dims)):
            if CenterShift[d] == 0:    # Skip this axis if no shift
                continue
            ftarr *= np.exp(-1j*2*np.pi*CenterShift[d]*sf.fftshift(idxgrid[d])/float(dims[d]))
    
    
    
    
    
    # ------ Rotation ------
    if Progress: 
        print('           Rotating ... ' + strftime('%H:%M:%S', localtime()))
    idx_axes = np.asarray(range(len(dims)))
    for d in range(len(dims)):
        if np.remainder(Rot[d], 2*np.pi) == 0:    # Skip this axis if no rotation
            continue
        indice = np.append(idx_axes[:d], idx_axes[d+1:])
        
    
    
    
    
    
    
    # ------ Move COM back, if Centering is True ------
    if Centering: 
        if Progress: 
            print('           Returning ... ' + strftime('%H:%M:%S', localtime()))
        for d in range(len(dims)):
            if CenterShift[d] == 0:    # Skip this axis if no shift
                continue
            ftarr *= np.exp(1j*2*np.pi*CenterShift[d]*sf.fftshift(idxgrid[d])/float(dims[d]))
    # ------ Shift, if RotFirst is True ------
    if RotFirst: 
        if Progress: 
            print('           Shifting ... ' + strftime('%H:%M:%S', localtime()))
        for d in range(len(dims)): 
            if Shift[d] == 0:    # Skip this axis if no shift
                continue
            ftarr *= np.exp(-1j*2*np.pi*Shift[d]*sf.fftshift(idxgrid[d])/float(dims[d]))
    
    # ====== IFFT ======
    if Progress: 
        print('>>>>>> IFFT the transformed array ... ' + strftime('%H:%M:%S', localtime()))
        t_4 = time()
    Array_new = np.abs(sf.ifftn(ftarr))
    
    if Progress: 
        t_5 = time()
        print('>>>>>> Summary: ')
        print('           FFT took %0.6f sec.' %(t_2 - t_1))
        print('           Meshgrid took %0.6f sec.' %(t_3 - t_2))
        print('           Transform took %0.6f sec.' %(t_4 - t_3))
        print('           iFFT took %0.6f sec.' %(t_5 - t_4))
        
    return Array_new


def Array_shift(Array, Shift=[0, 0, 0], Progress=False): # numpy  
    ''
    Ross' function that shifts a multi-dimensional array using FT method
    Input: 
        Shift:      (3,) list, 3D shifts in pixels
                e.g. Shift[1] = 2 means Array is shifted 2 pixels along axis=1
        Progress:   True will display the time stamps of each step
    ''
    Array = np.array(Array)
    dims = np.shape(Array)
    if not len(dims) == len(Shift):
        print(">>>>>> 'Shift' must have same number of dimensions as the array! ")
        input('>>>>>> Press any key to quit ... ')
        return
    # ====== FFT ======
    # scipy does normalized ffts!
    if Progress: 
        print('>>>>>> FFT the array ... ' + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    ftarr = nf.fftn(Array)
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
    shiftedarr = np.abs(nf.ifftn(ftarr))
    
    if Progress: 
        t_5 = time()
        print('>>>>>> Summary: ')
        print('           FFT took %0.6f sec.' %(t_2 - t_1))
        print('           Meshgrid took %0.6f sec.' %(t_3 - t_2))
        print('           Transform took %0.6f sec.' %(t_4 - t_3))
        print('           iFFT took %0.6f sec.' %(t_5 - t_4))
        print('           Shift took %0.6f sec in total.' %(t_5 - t_1))
    return shiftedarr


def Array_rot_3D(Array, Rot=[0, 0, 0], Seq='012', IsRadian=True, Progress=False): # numpy  
    ''
    Rotates a 3-dimensional array using FT method
    
    see DOI:10.1109/83.784442, DOI:10.1109/ACSSC.1996.600840
    
    Input: 
        Rotation:   (3,) list, 3D rotations in rad
                e.g. Rotation[0] = 0.1 means Array is rotated 0.1 rad around axis=0
        Progress:   True will display the time stamps of each step
    ''
    Array = np.array(Array)
    Rot = np.array(Rot)
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
        ftarr = nf.fft(Shearedarr, axis=m)
        Shift = idxgrid[n]*np.tan(-Rot[d]/2)
        ftarr *= np.exp(-2j*np.pi/float(dims[m]) 
                        * nf.fftshift(idxgrid[m]) * Shift)
        Shearedarr = np.abs(nf.ifft(ftarr, axis=m))
        # ========= 2nd shear =========
        ftarr = nf.fft(Shearedarr, axis=n)
        Shift = idxgrid[m]*np.sin(Rot[d])
        ftarr *= np.exp(-2j*np.pi/float(dims[n]) 
                        * nf.fftshift(idxgrid[n]) * Shift)
        Shearedarr = np.abs(nf.ifft(ftarr, axis=n))
        # ========= 3rd shear =========
        ftarr = nf.fft(Shearedarr, axis=m)
        Shift = idxgrid[n]*np.tan(-Rot[d]/2)
        ftarr *= np.exp(-2j*np.pi/float(dims[m]) 
                        * nf.fftshift(idxgrid[m]) * Shift)
        Shearedarr = np.abs(nf.ifft(ftarr, axis=m))
    
    if Progress: 
        t_6 = time()
        print('>>>>>> Summary: ')
        print('           Meshgrid took %0.6f sec.' %(t_2 - t_1))
        print('           Rotate axis 0 took %0.6f sec.' %(t_4 - t_3))
        print('           Rotate axis 1 took %0.6f sec.' %(t_5 - t_4))
        print('           Rotate axis 2 took %0.6f sec.' %(t_6 - t_5))
        print('           Rotation took %0.6f sec in total.' %(t_6 - t_2))
    
    return Shearedarr


def Array_rot_3D(Array, Rot=[0, 0, 0], Seq='012', IsRadian=True, Progress=False): # scipy  
    ''
    Rotates a 3-dimensional array using FT method
    
    see DOI:10.1109/83.784442, DOI:10.1109/ACSSC.1996.600840
    
    Input: 
        Rotation:   (3,) list, 3D rotations in rad
                e.g. Rotation[0] = 0.1 means Array is rotated 0.1 rad around axis=0
        Progress:   True will display the time stamps of each step
    ''
    Array = np.array(Array)
    Rot = np.array(Rot)
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
        ftarr = sf.fft(Shearedarr, axis=m)
        Shift = idxgrid[n]*np.tan(-Rot[d]/2)
        ftarr *= np.exp(-2j*np.pi/float(dims[m]) 
                        * sf.fftshift(idxgrid[m]) * Shift)
        Shearedarr = np.abs(sf.ifft(ftarr, axis=m))
        # ========= 2nd shear =========
        ftarr = sf.fft(Shearedarr, axis=n)
        Shift = idxgrid[m]*np.sin(Rot[d])
        ftarr *= np.exp(-2j*np.pi/float(dims[n]) 
                        * sf.fftshift(idxgrid[n]) * Shift)
        Shearedarr = np.abs(sf.ifft(ftarr, axis=n))
        # ========= 3rd shear =========
        ftarr = sf.fft(Shearedarr, axis=m)
        Shift = idxgrid[n]*np.tan(-Rot[d]/2)
        ftarr *= np.exp(-2j*np.pi/float(dims[m]) 
                        * sf.fftshift(idxgrid[m]) * Shift)
        Shearedarr = np.abs(sf.ifft(ftarr, axis=m))
    
    if Progress: 
        t_6 = time()
        print('>>>>>> Summary: ')
        print('           Meshgrid took %0.6f sec.' %(t_2 - t_1))
        print('           Rotate axis 0 took %0.6f sec.' %(t_4 - t_3))
        print('           Rotate axis 1 took %0.6f sec.' %(t_5 - t_4))
        print('           Rotate axis 2 took %0.6f sec.' %(t_6 - t_5))
        print('           Rotation took %0.6f sec in total.' %(t_6 - t_2))
    
    return Shearedarr


def Array_RossMT_det2lab(Array, Lambda=None, PhotonE=None, Voxel=None, 
                         Delta=0, Gamma=0, dYaw=0, dPitch=0, detdist=500,  
                         px=0.055, py=0.055, IsRadian=True, Progress=False):
    '' >>> Instruction <<<   
        Coordinates transformation using matrix from Ross Harder's code
    ''
    
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dYaw = np.deg2rad(dYaw)
        dPitch = np.deg2rad(dPitch)
    
    if Lambda is None:
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    Lambda = Lambda / 10   # Convert [A] to [nm]
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. Please use the proper code. ')
        input('>>>>>> Press any key to quit...')
        return
    # ==============================================================
    dx = 1/Nx
    dy = 1/Ny
    dz = 1/Nz
    
    tth = Delta
    gam = Gamma
    dth = dYaw
    print('tth ', tth)
    print('gam ', gam)
    print('dth ', dth)
    
    dpx = px/detdist
    dpy = py/detdist
    print('dpx ', dpx)
    print('dpy ', dpy)
    
    dQdpx = np.zeros(3)
    dQdpy = np.zeros(3)
    dQdth = np.zeros(3)
    
    dQdpx[0] = - np.cos(tth) * np.cos(gam)
    dQdpx[1] = 0.0
    dQdpx[2] = + np.sin(tth) * np.cos(gam)
    
    dQdpy[0] = + np.sin(tth) * np.sin(gam)
    dQdpy[1] = - np.cos(gam)
    dQdpy[2] = + np.cos(tth) * np.sin(gam)
    
    dQdth[0] = - np.cos(tth) * np.cos(gam) + 1.0
    dQdth[1] = 0.0
    dQdth[2] = + np.sin(tth) * np.cos(gam)
    
    print('')
    print('dQdpx ', dQdpx)
    print('dQdpy ', dQdpy)
    print('dQdth ', dQdth)
    
    ''
    New XH version
    dQdpx(1) = cos(tth);
    dQdpx(2) = 0.0;
    dQdpx(3) = -sin(tth);
     
    dQdpy(1) = sin(tth)*sin(gam);
    dQdpy(2) = -cos(gam);
    dQdpy(3) = cos(tth)*sin(gam);
    
    dQdth(1) = cos(tth)*cos(gam)-1.0;
    dQdth(2) = 0.0;
    dQdth(3) = -sin(tth)*cos(gam);
    ''
    
    Astar = np.zeros(3)
    Bstar = np.zeros(3)
    Cstar = np.zeros(3)
    
    Astar[0] = (2*np.pi/Lambda) * dpx * dQdpx[0]
    Astar[1] = (2*np.pi/Lambda) * dpx * dQdpx[1]
    Astar[2] = (2*np.pi/Lambda) * dpx * dQdpx[2]
    
    Bstar[0] = (2*np.pi/Lambda) * dpy * dQdpy[0]
    Bstar[1] = (2*np.pi/Lambda) * dpy * dQdpy[1]
    Bstar[2] = (2*np.pi/Lambda) * dpy * dQdpy[2]
    
    Cstar[0] = (2*np.pi/Lambda) * dth * dQdth[0]
    Cstar[1] = (2*np.pi/Lambda) * dth * dQdth[1]
    Cstar[2] = (2*np.pi/Lambda) * dth * dQdth[2]
    
    Astar = np.array(Astar)
    Bstar = np.array(Bstar)
    Cstar = np.array(Cstar)
    print('')
    print('Astar ', Astar)
    print('Bstar ', Bstar)
    print('Cstar ', Cstar)
    
    denom   = np.dot(Astar, np.cross(Bstar,Cstar))
    Axdenom = np.cross(Bstar, Cstar)
    Bxdenom = np.cross(Cstar, Astar)
    Cxdenom = np.cross(Astar, Bstar)
    print('')
    print('denom ', denom)
    print('Axdenom ', Axdenom)
    print('Bxdenom ', Bxdenom)
    print('Cxdenom ', Cxdenom)
    
    A = np.zeros(3)
    B = np.zeros(3)
    C = np.zeros(3)
    
    A[0] = 2 * np.pi * Axdenom[0] / denom
    A[1] = 2 * np.pi * Axdenom[1] / denom
    A[2] = 2 * np.pi * Axdenom[2] / denom
    
    B[0] = 2 * np.pi * Bxdenom[0] / denom
    B[1] = 2 * np.pi * Bxdenom[1] / denom
    B[2] = 2 * np.pi * Bxdenom[2] / denom
    
    C[0] = 2 * np.pi * Cxdenom[0] / denom
    C[1] = 2 * np.pi * Cxdenom[1] / denom
    C[2] = 2 * np.pi * Cxdenom[2] / denom


    Astarmag = np.dot(Astar, Astar)
    Astarmag = np.sqrt(Astarmag)
    Bstarmag = np.dot(Bstar, Bstar)
    Bstarmag = np.sqrt(Bstarmag)
    Cstarmag = np.dot(Cstar, Cstar)
    Cstarmag = np.sqrt(Cstarmag)

    Amag = np.dot(A, A)
    Amag = np.sqrt(Amag)
    Bmag = np.dot(B, B)
    Bmag = np.sqrt(Bmag)
    Cmag = np.dot(C, C)
    Cmag = np.sqrt(Cmag)

    T = np.array([[A[0], B[0], C[0]], 
                  [A[1], B[1], C[1]], 
                  [A[2], B[2], C[2]]])
    # ==============================================================
    X = (np.asarray(range(Nx)) - np.floor(Nx/2))
    Y = (np.asarray(range(Ny)) - np.floor(Ny/2))
    Z = (np.asarray(range(Nz)) - np.floor(Nz/2))
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid representing the Lab space voxels
    
    OriCoords = np.stack((X3.ravel(), Y3.ravel()), axis=0)
    OriCoords = np.concatenate((OriCoords, [Z3.ravel()]), axis=0)
    
    Output = np.matmul(T, OriCoords)
    
    return Output


def Array_resize_interpolate(Array, Coords=None, Voxel=[1,1,1], ReductRatio=0.66, 
                             Equal3D=True, Method='linear', Progress=False):  
    ' >>> Instruction <<<   
        This function resize an Array based on the Coords and Voxel size, using interpolation. 
        
        * Definitions of inputs: 
            Array:       3D array
            Coords:      Cartesian coordinates. Take the result of Array_coords_det2lab().
            Voxel:       Step sizes in 3D 
            ReductRatio: Maximum reduction allowed in the field of view 
                         to prevent cutting the object
            IsRadian:    If True, all the angles are in [rad]; otherwise, [deg]
            Equal3D:     If True, Voxel size will be modified to a cubic shape. 
            Method:      'linear': Scipy's griddata
                         'Sibson': 
        
        * Array axes use the typical definition of a tif stack in ImageJ
            i.e. 1st axis is the frame, 2nd and 3rd are the Y and X axes of each frame.
        
        * Coords is a dictionary including 'X', 'Y', and 'Z', each has the same shape as Array.
            If Coords is None, a 3D meshgrid is used as the coordinates. 
        
        * Voxel will be used to define the voxel of the output array. 
            (1) Make sure that Voxel has the same sequence as Array
                    i.e. Ordered as [Z-size, Y-size, X-size]
            (2) The output array does not necessarily have this voxel size. 
                    The size is also affected by ReductRatio. 
        
        * If ReductRatio is None, the new voxel size is simply the min of Voxel.
    '
    # Input regularization
    Array = np.array(Array)
    dims  = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    Vz, Vy, Vx = Voxel
    
    # Creat meshgrid
    if Progress: 
        print('>>>>>> Creat the meshgrid ... ' + strftime('%H:%M:%S', localtime()))
        t_1 = time()
        
    X = np.asarray(range(Nx)) - np.floor(Nx/2)
    Y = np.asarray(range(Ny)) - np.floor(Ny/2)
    Z = np.asarray(range(Nz)) - np.floor(Nz/2)
    
    Y3, Z3, X3 = np.meshgrid(Y, Z, X) # 3D meshgrid
    
    # The original coordinates
    if Progress: 
        print('>>>>>> Get original and new coordinates ... ' + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    if Coords is None: 
        X3_ori = X3.ravel()
        Y3_ori = Y3.ravel()
        Z3_ori = Z3.ravel()
    else: 
        X3_ori = Coords['X'].ravel() - np.mean(Coords['X'])
        Y3_ori = Coords['Y'].ravel() - np.mean(Coords['Y'])
        Z3_ori = Coords['Z'].ravel() - np.mean(Coords['Z'])
    
    X3_ori_min = np.min(X3_ori)
    X3_ori_max = np.max(X3_ori)
    Y3_ori_min = np.min(Y3_ori)
    Y3_ori_max = np.max(Y3_ori)
    Z3_ori_min = np.min(Z3_ori)
    Z3_ori_max = np.max(Z3_ori)
    print('>>>>>> Original X is in [%g, %g].' %(X3_ori_min, X3_ori_max))
    print('>>>>>> Original Y is in [%g, %g].' %(Y3_ori_min, Y3_ori_max))
    print('>>>>>> Original Z is in [%g, %g].' %(Z3_ori_min, Z3_ori_max))
    
    # Determine the new coordinates
    if Vx * Nx > X3_ori_max - X3_ori_min: 
        Vx = (X3_ori_max - X3_ori_min) / Nx
    if Vy * Ny > Y3_ori_max - Y3_ori_min: 
        Vy = (Y3_ori_max - Y3_ori_min) / Ny
    if Vz * Nz > Z3_ori_max - Z3_ori_min: 
        Vz = (Z3_ori_max - Z3_ori_min) / Nz
    
    if Equal3D: 
        Vs = np.min(Voxel)
        if not ReductRatio is None: 
            CurRatio = np.min([Vs/Vz, Vs/Vy, Vs/Vx])
            if CurRatio < ReductRatio: 
                Vs = Vs * ReductRatio / CurRatio
        X3_new = X3.ravel() * Vs
        Y3_new = Y3.ravel() * Vs
        Z3_new = Z3.ravel() * Vs
    else: 
        X3_new = X3.ravel() * Vx
        Y3_new = Y3.ravel() * Vy
        Z3_new = Z3.ravel() * Vz
    print('>>>>>> New X is in [%.3g, %.3g].' %(np.min(X3_new), np.max(X3_new)))
    print('>>>>>> New Y is in [%.3g, %.3g].' %(np.min(Y3_new), np.max(Y3_new)))
    print('>>>>>> New Z is in [%.3g, %.3g].' %(np.min(Z3_new), np.max(Z3_new)))
    
    # Interpolation
    if Progress: 
        print('>>>>>> Interpolate and resample the array ... ' + strftime('%H:%M:%S', localtime()))
        t_3 = time()
    
    Array_new = sin.griddata((Z3_ori,Y3_ori,X3_ori), Array.ravel(), 
                              (Z3_new,Y3_new,X3_new), method='nearest')
    # Output 
    if Progress: 
        t_4 = time()
        print('>>>>>> Summary: ')
        print('           Creating meshgrid took %0.6f sec.' %(t_2 - t_1))
        print('           Generating coordinates took %0.6f sec.' %(t_3 - t_2))
        print('           Interpolation took %0.6f sec.' %(t_4 - t_3))
        print('           Coordinates transformation took %0.6f sec in total.' %(t_4 - t_1))
    
    if Equal3D: 
        print('>>>>>> The final voxel size is x = y = z = %.3g' %Vs)
        Voxel = [Vs, Vs, Vs]
    else: 
        print('>>>>>> The final voxel size is x = %.3g, y = %.3g, z = %.3g.' %(Vx,Vy,Vz))
        Voxel = [Vz, Vy, Vx]
    
    return {'Array': Array_new.reshape((Nz,Ny,Nx)), 'Voxel size': Voxel}


def Array_det2crys_interpolation_spline(Data, Lambda=None, PhotonE=None, Crys_orie=[0,0,0], 
                                        Delta=None, Gamma=None, dYaw=0, dPitch=0, 
                                        detdist=None, px=None, py=None, IsRadian=True, 
                                        PadSize=None, PadIncr=[5,5,5], Progress=False): 
    ' >>> Instruction <<< 
        This function wraps the detector-frame-to-crystal-frame interpolation using 
        tricubit spline from python module eqtools. 
        
        Inputs: 
            Data:        A 3D array. 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            Crys_orie    Crystal orientation in LAB FRAME
                             (3,) the extrinsic rotation around X, Y, Z axes
            Delta:       Detector angle around Lab Y+ axis
            Gamma:       Detector angle around Lab X- axis
            dYaw:        Step of rocking curve scan around Y axis
            dPitch:      Step of rocking curve scan around X axis
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, dYaw and dPitch are in [rad]; otherwise, [deg]
            PadSize:     (3,) Size of the padded array.
            PadIncr:     (3,) Increaments of the padding in three dimension. 
        
        Output: 
            Data_new     The interpolated array. 
    '
    # Input regularization
    if Progress: 
        print('>>>>>> Input regularization ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_1 = time()
    
    Crys_orie = np.array(Crys_orie)
    PadIncr = np.array(PadIncr)
    
    dims = np.array(np.shape(Data))
    if len(dims) == 3: 
        N_elem = int(dims[0]*dims[1]*dims[2])
    else: 
        print('>>>>>> The shape of "Data" is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    
    if not IsRadian: 
        Crys_orie = np.deg2rad(Crys_orie)
        Delta = np.deg2rad(Delta)
        Gamma = np.deg2rad(Gamma)
        dYaw = np.deg2rad(dYaw)
        dPitch = np.deg2rad(dPitch)
    
    if Lambda is None: 
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    
    if Delta == 0 and Gamma == 0: 
        print('>>>>>> Forward scattering geometry. No transformation needed. ')
        return Data
    
    # Coordinates in crystal frame
    if Progress: 
        print('>>>>>> Calculate coordinates in crystal frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_2 = time()
    
    LabCoords = Array_generate_coords_lab(Data, Lambda=Lambda, PhotonE=PhotonE, Voxel=None, 
                                          Delta=Delta, Gamma=Gamma, dYaw=dYaw, dPitch=dPitch, 
                                          detdist=detdist, px=px, py=py, IsRadian=True)
    CrysCoords = Array_coords_lab2crys(QxCoords=LabCoords['Qx'], 
                                       QyCoords=LabCoords['Qy'], 
                                       QzCoords=LabCoords['Qz'],  
                                       Delta=Delta, Gamma=Gamma, Crys_orie=Crys_orie, 
                                       Lambda=Lambda, PhotonE=PhotonE, IsRadian=True)
    CrysToDet = Array_coords_lab2det(QxCoords=CrysCoords['Qx_lab']+CrysCoords['Bragg_lab'][0], 
                                     QyCoords=CrysCoords['Qy_lab']+CrysCoords['Bragg_lab'][1], 
                                     QzCoords=CrysCoords['Qz_lab']+CrysCoords['Bragg_lab'][2], 
                                     Lambda=Lambda,PhotonE=PhotonE,dYaw=dYaw,dPitch=dPitch)
    
    # Coordinates in detector frame
    if Progress: 
        print('>>>>>> Calculate coordinates in detector frame ... ' 
              + strftime('%H:%M:%S', localtime()))
        t_3 = time()
        print('       Original array size is ', dims)
    
    Err_flag = True
    if (PadSize is None) or (not PadSize >= np.shape(Data)):
        dims_new = dims.copy()
    else: 
        dims_new = np.array(PadSize)
    while Err_flag: 
        try: 
            Data_pad = Array_zeropad(Data, AddTo=dims_new)
            DetCoords = Array_generate_coords_det(Data_pad, Lambda=Lambda, PhotonE=PhotonE, 
                                                  Voxel=None, Delta=Delta, Gamma=Gamma, 
                                                  dYaw=dYaw, dPitch=dPitch, detdist=detdist,  
                                                  px=px, py=py, IsRadian=True, 
                                                  OutputRadian=True)
            TriSpline_function = Spline(DetCoords['Theta_1D'], 
                                        DetCoords['Gamma_1D'], 
                                        DetCoords['Delta_1D'], Data_pad)
            Data_new = TriSpline_function.ev(CrysToDet['Theta'].ravel(), 
                                             CrysToDet['Gamma'].ravel(), 
                                             CrysToDet['Delta'].ravel())
            if Progress: 
                print('')
                print('       Padded array size is   ', np.array(np.shape(Data_pad)))
                
            Data_new = np.reshape(Data_new, dims)
            Err_flag = False
        except Exception as err_message: 
            ErrM = str(err_message)
            if Progress: 
                print('       ' + ErrM + ', padding ', dims_new-dims, end='\r')
            if ErrM.endswith('exceeds bounds of interpolation grid '): 
                Ax = ErrM.split(' ')[0]
                if Ax == 'x': 
                    dims_new[2] = dims_new[2] + PadIncr[2]
                if Ax == 'y': 
                    dims_new[1] = dims_new[1] + PadIncr[1]
                if Ax == 'z': 
                    dims_new[0] = dims_new[0] + PadIncr[0]
            else: 
                print('\n>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit...')
                return
    
    if Progress: 
        print('>>>>>> Interpolation finished.    ' 
              + strftime('%H:%M:%S', localtime()))
        print('       Output array size is   ', np.array(np.shape(Data_new)))
        t_4 = time()
        
        plt.figure(figsize=(7.5, 3))
        plt.subplot(131)
        plt.plot(DetCoords['Delta'].ravel(), alpha=0.5)
        plt.plot(CrysToDet['Delta'].ravel(), alpha=0.5)
        plt.title('Delta')
        
        plt.subplot(132)
        plt.plot(DetCoords['Gamma'].ravel(), alpha=0.5)
        plt.plot(CrysToDet['Gamma'].ravel(), alpha=0.5)
        plt.title('Gamma')
        
        plt.subplot(133)
        plt.plot(DetCoords['Theta'].ravel(), alpha=0.5)
        plt.plot(CrysToDet['Theta'].ravel(), alpha=0.5)
        plt.title('Theta')
        
        plt.tight_layout()
    return {'Data': Data_new, 'PadTo': dims_new}


def Array_real_space_voxel(Array, Lambda=None, PhotonE=None, dYaw=0, dPitch=0, 
                           detdist=500, px=0.055, py=0.055, IsRadian=True): 
    '' >>> Instruction <<< 
        
        !!! Duplicate from Jesse Clark's Matlab code. Do NOT use. !!!
        
        Inputs: 
            Lambda:      Wavelength in [A]
            PhotonE:     Photon energy in [keV]
            dYaw:        Step of rocking curve scan around Y axis
            dPitch:      Step of rocking curve scan around X axis
            detdist:     Sample-detector distance in [mm]
            px, py:      X(horizontal) and Y(vertical) size of the detector in [mm]
            IsRadian:    If True, dYaw and dPitch are in [rad]; otherwise, [deg]
        
        Output: 
            [Vz, Vy, Vx]:   3-dimentional voxel size in [A]
    ''
    # Input regularization
    dims = np.shape(Array)
    
    if len(dims) == 2: 
        Ny, Nx = dims
        Nz = 1
    elif len(dims) == 3: 
        Nz, Ny, Nx = dims
    else: 
        print('>>>>>> The shape of Array is not right! ')
        input('>>>>>> Press any key to quit...')
        return
    N_elem = int(Nz*Ny*Nx)
    
    if not IsRadian: 
        dYaw = np.deg2rad(dYaw)
        dPitch = np.deg2rad(dPitch)
    
    if Lambda is None: 
        if PhotonE is None: 
            Lambda = input(u'>>>>>> Input the wave length in [\u212b]')
        else: 
            Lambda = 12.398 / PhotonE
    # Lambda = Lambda / 10   # Convert [A] to [nm]
    
    # Voxel size in X and Y  
    Vx = Lambda * detdist/px / Nx   #  / 2/np.pi
    Vy = Lambda * detdist/py / Ny   #  / 2/np.pi
    
    # Voxel size in Z  
    if dPitch == 0: 
        ' Crystal is rotated around Y axis. '
        d_angle = dYaw
    elif dYaw == 0:
        ' Crystal is rotated around X axis. '
        d_angle = dPitch
    else: 
        print('>>>>>> Please check dYaw and dPitch. ')
        input('>>>>>> Press any key to quit...')
        return
    
    Vz = Lambda / d_angle / Nz   #  / 2/np.pi
        
    return [Vz, Vy, Vx]


'''