import numpy as np
# from skimage.external 
import tifffile as tif
try: 
    from xrayutilities.io import spec
except: 
    print('xrayutilities.io.spec is not available !')
import os
import re
import csv
import scipy.io as si
from PIL import Image


def FileFormat(Value, Target='uint32'): 
    ''' >>> Instruction <<<
        This function is used to figure out the propoer format of int 
        for saving the input 'Value'
        
        Input: 
            Value       The value that needs to be saved. 
                        For an array, use the max/min value. 
            Target      The targeted format. The function will output the 
                        magnification for saving the value in this format. 
        Output: 
            'int'       Format for int. Could be int8, int16, int32, etc.
            'uint'      Format for uint. Could be uint8, uint16, uint32, etc.
            'float'     Format for float. Could be float16, float32, etc.
            'magn'      Magnification required for save the value in the 
                        targeted 
    '''
    if Value < 0: 
        sign = -1
        Value = -Value
    else: 
        sign = 1
    
    whole, dec = str(Value).split('.')
    sig_dig = len(whole) + len(dec)
    whole   = int(whole)
    dec     = int(dec)
    
    form_int   = 8
    form_uint  = 8
    form_float = 16
    
    # bits for int
    if sign == 1: 
        while whole >= 2**(form_int-1):
            form_int = form_int * 2
    if sign == -1: 
        while whole >  2**(form_int-1):
            form_int = form_int * 2
    # bits for uint
    while whole >= 2**(form_uint): 
        form_uint = form_uint * 2
    # bits for float
    if sig_dig > 3: 
        form_float = 32
    if sig_dig > 6: 
        form_float = 64
    if sig_dig > 15: 
        form_float = 128
    
    # Magnification
    form, bits, _ = re.split('(\d+)', Target)
    bits = int(bits)
    if form == 'int': 
        max_value = 2**(bits-1)-1
        min_value = 2**(bits-1)
        if sign == 1: 
            Magn = 1/10**np.ceil(np.log10(whole/max_value))
        if sign == -1: 
            Magn = 1/10**np.ceil(np.log10(whole/min_value))
    elif form == 'uint': 
        max_value = 2**bits-1
        Magn = 1/10**np.ceil(np.log10(whole/max_value))
    elif form == 'float': 
        Magn = 1
    else: 
        print('>>>>>> Unknown targeted format. ')
        input('       Press any key to quit...')
        return
    
    return {'int':form_int, 'uint':form_uint, 'float':form_float, 'magn':Magn}


def Read_tif(File):  
    if not os.path.isfile(File): 
        print('>>>>>> Error! File does NOT exist. ')
        return None
    im = tif.imread(File)
    return im


def Read_tif_PIL(filename='test.tif', filepath=''): 
    File = os.path.join(filepath, filename)
    if not os.path.isfile(File): 
        print('>>>>>> Error! File does NOT exist. ')
        return None
    im = Image.open(File)
    V,H = np.shape(np.array(im))   # Vertical and Horizontal pixels 
    Array = np.zeros((V, H))
    Array[:, :] = np.array(im)[:, :]
    return Array


def Read_tif_from_stack(file, frame):  
    if not os.path.isfile(file): 
        print('>>>>>> Error! File does NOT exist. ')
        return None
    im = tif.imread(file, key=frame)
    return im


def Load_Bin_images(FileName, X_bin=2, Y_bin=2): 
    ''' The standard way to bin a large array to a smaller one by averaging is 
        to reshape it into a higher dimension and then take the means over the 
        appropriate new axes. The following function does this, assuming that 
        each dimension of the new shape is a factor of the corresponding dimension 
        in the old one.
        
        refer to: 
        https://scipython.com/blog/binning-a-2d-array-in-numpy/
        
        Notice: 
        The X and Y axes of the loaded image is switched. 
    '''
    imarray = tif.imread(FileName)
    
    if not np.size(np.shape(imarray)) == 2: 
        print('>>>>>> The file is not a 2D image !')
        input('>>>>>> Press any key to quit...')
        return
    
    m = np.shape(imarray)[0]
    n = np.shape(imarray)[1]
    
    if np.remainder(m, Y_bin) != 0 or np.remainder(n, X_bin) != 0: 
        print('>>>>>> New dimension is not a factor of the old dimension !')
        input('>>>>>> Press any key to quit...')
        return
    
    if not (X_bin == 1 and Y_bin == 1): 
        imarray = imarray.reshape(m//Y_bin, Y_bin, n//X_bin, X_bin).mean(-1).mean(1)
    
    return imarray


def Load_Bin_stack(file, X_bin=2, Y_bin=2, Z_bin=2): 
    ''' >>> Introduction <<<
        This function loads the image from a tiff stack frame-by-frame. During the 
        loading, the images are binned in up to 3D. 
        
        The standard way to bin a large array to a smaller one by averaging is 
        to reshape it into a higher dimension and then take the means over the 
        appropriate new axes. The following function does this, assuming that 
        each dimension of the new shape is a factor of the corresponding dimension 
        in the old one.
        
        refer to: 
        https://scipython.com/blog/binning-a-2d-array-in-numpy/
        
        Notice: 
        The X and Y axes of the loaded image is switched. 
    '''
    if not os.path.isfile(file): 
        print('>>>>>> Error! File does NOT exist. ')
        return None
    
    new_frame = tif.imread(file, key=0)
    
    if not np.size(np.shape(new_frame)) == 2: 
        print('>>>>>> The file is not an image stack !')
        input('>>>>>> Press any key to quit...')
        return
    
    m = np.shape(new_frame)[0]
    n = np.shape(new_frame)[1]
    
    if np.remainder(m, Y_bin) != 0 or np.remainder(n, X_bin) != 0: 
        print('>>>>>> New dimension is not a factor of the old dimension !')
        input('>>>>>> Press any key to quit ...')
        return
    
    temp_array = np.zeros((Z_bin, m//Y_bin, n//X_bin))
    
    Err_flag = True
    frameidx = 0
    Z_count = 0
    flag_new = True
    print('>>>>>> Loading Image %04d ... ' %(frameidx+1), end='\r')
    while Err_flag: 
        if not (X_bin == 1 and Y_bin == 1): 
            new_frame = new_frame.reshape(m//Y_bin, Y_bin, n//X_bin, X_bin).mean(-1).mean(1)
        temp_array[Z_count, :, :] = new_frame[:,:]
        
        if Z_count == Z_bin - 1: 
            Z_count = -1
            temp_array_bin = temp_array.mean(axis=0).reshape((1, m//Y_bin, n//X_bin))
            if flag_new: 
                imarray = temp_array_bin
                flag_new = False
            else: 
                imarray = np.concatenate((imarray, temp_array_bin), axis=0)
        
        frameidx = frameidx + 1
        try:
            new_frame = tif.imread(file, key=frameidx)
            print('>>>>>> Loading Image %04d ... ' %(frameidx+1), end='\r')
            Z_count = Z_count + 1
        except Exception as err_message:
            ErrM = str(err_message)
            if ErrM.startswith('list index out'):
                Err_flag = False
                break
            else: 
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit ...')
                return
        
    if Z_count >= 0: 
        temp_array_bin = temp_array.sum(axis=0).reshape((1, m//Y_bin, n//X_bin))/(Z_count + 1)
        imarray = np.concatenate((imarray, temp_array_bin), axis=0)
    
    print('>>>>>> %d images have been loaded. ' %(frameidx))
    return imarray


def Load_Bin_stack_and_max(file, X_bin=2, Y_bin=2): 
    ''' >>> Introduction <<<
        This function loads the image from a tiff stack frame-by-frame, bins them, 
        and porject the whole stack along Z-axis using the maximum value of each 
        corresponding pixel. 
        
        The standard way to bin a large array to a smaller one by averaging is 
        to reshape it into a higher dimension and then take the means over the 
        appropriate new axes. The following function does this, assuming that 
        each dimension of the new shape is a factor of the corresponding dimension 
        in the old one.
        
        refer to: 
        https://scipython.com/blog/binning-a-2d-array-in-numpy/
        
        Notice: 
        The X and Y axes of the loaded image is switched. 
    '''
    if not os.path.isfile(file): 
        print('>>>>>> Error! File does NOT exist. ')
        return None
    
    imarray = tif.imread(file, key=0)
    
    if not np.size(np.shape(imarray)) == 2: 
        print('>>>>>> The file is not an image stack !')
        input('>>>>>> Press any key to quit...')
        return
    
    m = np.shape(imarray)[0]
    n = np.shape(imarray)[1]
    
    if np.remainder(m, Y_bin) != 0 or np.remainder(n, X_bin) != 0: 
        print('>>>>>> New dimension is not a factor of the old dimension !')
        input('>>>>>> Press any key to quit ...')
        return
    
    if not (X_bin == 1 and Y_bin == 1): 
        imarray = imarray.reshape(m//Y_bin, Y_bin, n//X_bin, X_bin).mean(-1).mean(1)
    
    Err_flag = True
    frameidx = 0
    print('>>>>>> Loading Image %04d ... ' %(frameidx+1), end='\r')
    while Err_flag: 
        frameidx = frameidx + 1
        try:
            new_frame = tif.imread(file, key=frameidx)
            print('>>>>>> Loading Image %04d ... ' %(frameidx+1), end='\r')
        except Exception as err_message:
            ErrM = str(err_message)
            if ErrM.startswith('list index out'):
                Err_flag = False
                break
            else: 
                print('>>>>>> Unknown Error !')
                input('>>>>>> Press any key to quit ...')
                return
        
        if not (X_bin == 1 and Y_bin == 1): 
            new_frame = new_frame.reshape(m//Y_bin, Y_bin, n//X_bin, X_bin).mean(-1).mean(1)
        
        imarray = np.stack((imarray, new_frame)).max(axis=0)
    
    print('>>>>>> %d images have been loaded. ' %(frameidx))
    return imarray


def Save_tif(Array, File, OverWrite=False):  
    if os.path.isfile(File): 
        print('>>>>>> Error! File already exists. ')
        if not OverWrite: 
            print('>>>>>> Array is not saved! ')
            return None
        else: 
            print('>>>>>> Existing file is overwritten! ')
    # tif.imsave(File, Array)
    tif.imsave(File, Array, bigtiff=True)
    return


def Save_tif_PIL(Array, File, OverWrite=False):  
    if os.path.isfile(File): 
        print('>>>>>> Error! File already exists. ')
        if not OverWrite: 
            print('>>>>>> Array is not saved! ')
            return None
        else: 
            print('>>>>>> Existing file is overwritten! ')
    # tif.imsave(File, Array)
    im = Image.fromarray(Array)
    im.save(File)
    return


def Array_to_VTK(Array, filename='temp.vtk', Array2=None, spacing=1.0, org=[0.0,0.0,0.0], 
                 Coords_order=[0,1,2], data_type='float', display=False): 
    
    if not ((data_type is None) or (data_type == '')): 
        Array = np.array(Array).astype(data_type)
    
    Coords_order = np.array(Coords_order)
    org = np.array(org)
    
    Nx, Ny, Nz = np.array(np.shape(Array))[Coords_order]
    orgx, orgy, orgz = org[Coords_order]
    
    lines = ['# vtk DataFile Version 2.0\n',
             'Comment goes here\n', 
             'ASCII\n', 
             '\n'] 
    
    file_format = ['DATASET STRUCTURED_POINTS\n', 
                     'DIMENSIONS %d %d %d\n' % (Nx, Ny, Nz), 
                     '\n', 
                     'ORIGIN %f %f %f\n' % (orgx, orgy, orgz), 
                     'SPACING %f %f %f\n' % (spacing, spacing, spacing), 
                     '\n']
        
    data_format = ['POINT_DATA %d\n' % (Nx*Ny*Nz),
                   'SCALARS amp double\n',
                   'LOOKUP_TABLE default\n',
                   '\n']
    
    fid = open(filename, 'w+')
    fid.writelines(lines)
    fid.writelines(file_format)
    fid.writelines(data_format)

    if display: 
        print(' >>>>>> Writing the 1st array ... ', end='\r')
    for c in range(Nz):
        if display: 
            print(' >>>>>> Writing the 1st array ... %0.1f%% ' %(c/Nz*100), end='\r')
        for b in range(Ny):
            for a in range(Nx):
                temp = '%g ' % (Array[a, b, c])
                fid.writelines(temp)
            fid.writelines('\n')
    if display: 
        print(' >>>>>> Writing the 1st array ... %0.1f%% ' %(100), end='\n')
    
    if not Array2 is None: 
        if not np.shape(Array) == np.shape(Array2): 
            print('>>>>>> Error! Two arrays have different shapes. ')
            input('>>>>>> Press any key to quit...')
            return
        
        data_format = ['FIELD FieldData 1 \n',
                       'phases 1 %d' % (Nx*Ny*Nz),
                       'double \n',
                       '\n']
        
        fid.writelines(data_format)
        
        if display: 
            print(' >>>>>> Writing the 2nd array ... ', end='\r')
        for c in range(Nz):
            if display: 
                print(' >>>>>> Writing the 2nd array ... %0.1f%% ' %(c/Nz*100), end='\r')
            for b in range(Ny):
                for a in range(Nx):
                    temp = '%g ' % (Array2[a, b, c])
                    fid.writelines(temp)
                fid.writelines('\n')
        if display: 
            print(' >>>>>> Writing the 2nd array ... %0.1f%% ' %(100), end='\n')
    
    fid.close()
    return


def Array_coords_to_CSV(x3, y3, z3, scalar, filename='temp.csv', data_type='float'): 
    ''' >>> Instruction <<<
        This function writes points in a 3D coordinates to a csv file. 
        The csv file can be read in ParaView and plotted in 3D
        
        Inputs: 
            x3, y3, z3, scalar    Each is a (N, 1) array
    '''
    
    if not ((data_type is None) or (data_type == '')): 
        x3 = np.array(x3).astype(data_type)
        y3 = np.array(y3).astype(data_type)
        z3 = np.array(z3).astype(data_type)
        scalar = np.array(scalar).astype(data_type)
    
    Nx = np.size(x3)
    Ny = np.size(y3)
    Nz = np.size(z3)
    Ns = np.size(scalar)
    
    if not ( (Nx == Ny) and (Nx == Nz) and (Nx == Ns) ): 
        print('>>>>>> Error! Input coords and scalar have different lengths. ')
        input('>>>>>> Press any key to quit...')
        return
    
    with open(filename, 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x coord','y coord','z coord','scalar'])
        for i in range(Nx):
            if np.remainder(i, 10) == 0: 
                print(' >>>>>> Writing ... %0.1f%% ' %(i/Nx*100), end='\r')
            writer.writerow(['%.4f' %x3[i], 
                             '%.4f' %y3[i], 
                             '%.4f' %z3[i], 
                             '%.4f' %scalar[i] ])
        print(' >>>>>> Writing ... %0.1f%% ' %(100), end='\r')
    return


def Read_mat_file(FileName='test.mat', VarName='array', FilePath=''): 
    ''' >>> Instruction <<<
        This function loads MATLAB '.mat' file to a numpy array.
        
        Input: 
            FileName    Full file name.  
            VarName     Data = si.loadmat(File)[VarName], 
                        i.e., dictionary item name. 
            FilePath    File path, default is current directory. 
            
        Output: 
            Data        Numpy array. 
    '''
    # Get the current working directory
    retval = os.getcwd()
    # Generate the full file path and check if it exists
    if len(FilePath) == 0: 
        FilePath = retval
    File = os.path.join(FilePath, FileName)
    if not os.path.isfile(File): 
        print('>>>>>> Error! File does NOT exist. ')
        input('>>>>>> Press any key to quit...')
        return
    # Read the .mat (or .rec) file
    Data = si.loadmat(File)[VarName]
    return Data


def Save_mat_file(Data, FileName='test.mat', VarName='array', FilePath='', OverWrite=False): 
    ''' >>> Instruction <<<
        This function saves data to MATLAB '.mat' file.
        
        Input: 
            Data        Usually a numpy array.
            FileName    Full file name.  
            VarName     The variable name in this .mat file.  
            FilePath    File path, default is current directory. 
            OverWrite   If True, overwrite the file if it already exists. 
            
        Output: 
            Return 1 is the data is successfully saved. 
    '''
    # Get the current working directory
    retval = os.getcwd()
    # Generate the full file path and check if it exists
    if len(FilePath) == 0: 
        FilePath = retval
    if not os.path.isdir(FilePath):
        print('>>>>>> Error! Directory does NOT exist. ')
        input('>>>>>> Press any key to quit...')
        return
    File = os.path.join(FilePath, FileName)
    if os.path.isfile(File): 
        print('>>>>>> Error! File already exists. ')
        if not OverWrite: 
            print('>>>>>> Array is not saved! ')
            return None
        else: 
            print('>>>>>> Existing file is overwritten! ')
    # Save Data to .mat file
    mdic = {VarName: Data}
    si.savemat(File, mdic)
    return 1






# Below are archived functions for reference
'''
def Array_to_VTK(Array, filename='temp.vtk', Array2=None, spacing=1.0, org=[0.0,0.0,0.0], 
                 data_type='float'): 
    
    if not ((data_type is None) or (data_type == '')): 
        Array = np.array(Array).astype(data_type)
    Nx, Ny, Nz = np.shape(Array)
    orgx, orgy, orgz = org
    
    lines = ['# vtk DataFile Version 2.0\n',
             'Comment goes here\n', 
             'ASCII\n', 
             '\n'] 
    
    file_format = ['DATASET STRUCTURED_POINTS\n', 
                     'DIMENSIONS %d %d %d\n' % (Nx, Ny, Nz), 
                     '\n', 
                     'ORIGIN %f %f %f\n' % (orgx, orgy, orgz), 
                     'SPACING %f %f %f\n' % (spacing, spacing, spacing), 
                     '\n']
        
    data_format = ['POINT_DATA %d\n' % (Nx*Ny*Nz),
                   'SCALARS amp double\n',
                   'LOOKUP_TABLE default\n',
                   '\n']
    
    fid = open(filename, 'w+')
    fid.writelines(lines)
    fid.writelines(file_format)
    fid.writelines(data_format)

    print(' >>>>>> Writing the 1st array ... ', end='\r')
    for c in range(Nz):
        print(' >>>>>> Writing the 1st array ... %0.1f%% ' %(c/Nz*100), end='\r')
        for b in range(Nx):
            for a in range(Ny):
                temp = '%g ' % (Array[b, a, c])
                fid.writelines(temp)
            fid.writelines('\n')
    print(' >>>>>> Writing the 1st array ... %0.1f%% ' %(100), end='\n')
    
    if not Array2 is None: 
        if not np.shape(Array) == np.shape(Array2): 
            print('>>>>>> Error! Two arrays have different shapes. ')
            input('>>>>>> Press any key to quit...')
            return
        
        data_format = ['FIELD FieldData 1 \n',
                       'phases 1 %d' % (Nx*Ny*Nz),
                       'double \n',
                       '\n']
        
        fid.writelines(data_format)
        
        print(' >>>>>> Writing the 2nd array ... ', end='\r')
        for c in range(Nz):
            print(' >>>>>> Writing the 2nd array ... %0.1f%% ' %(c/Nz*100), end='\r')
            for b in range(Nx):
                for a in range(Ny):
                    temp = '%g ' % (Array2[b, a, c])
                    fid.writelines(temp)
                fid.writelines('\n')
        print(' >>>>>> Writing the 2nd array ... %0.1f%% ' %(100), end='\n')
    
    fid.close()
    return






'''
