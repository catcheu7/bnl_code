import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from scipy.stats import norm as nm


def FT_1D(data, stepsize=1, flag_power_of_two=False): 
    ''' >>> Instruction <<< 
        This function performs Fourier Transform on 1D data. 
    '''
    data = np.asarray(data.reshape(-1))
    N = np.size(data)
    if flag_power_of_two: 
        N_fft = int(2**np.ceil(np.log2(N)))
    else: 
        N_fft = N
    fq_stp = 2 * np.pi / N_fft / stepsize
    FFT = sp.fft.fftshift(sp.fft.fft(data, N_fft)/N)
    
    amp = np.abs(FFT)
    ph = np.angle(FFT)
    freq = (np.arange(N_fft) - N_fft/2) * fq_stp
    
    return {'amp':amp, 'ph':ph, 'fq':freq}


def IFT_1D(amp, ph, freq, output_size=100, output_step=1): 
    ''' >>> Instruction <<< 
        This function regenerates 1D data using Fourier Transform result. 
    '''
    N = np.size(amp)
    if (N != np.size(ph)) or (N != np.size(freq)): 
        print(">>>>>> Check size of amp, ph, and freq ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    axis = np.arange(output_size) * output_step
    data = np.zeros(output_size).astype('complex')
    for i in range(N): 
        data += amp[i] * np.exp(1j * (axis * freq[i] + ph[i]) )
    return data


def FT_2D(Data, stepsize=[1,1], flag_power_of_two=False): 
    ''' >>> Instruction <<< 
        This function performs Fourier Transoform on grid 2D data. 
        
        flag_power_of_two: whether pad the array to next power of 2.
        
        Plot results using matplotlib, e.g. :
            plt.imshow(result['amp'], 
                       extent=[np.amin(result['fq']['ax1']), 
                               np.amax(result['fq']['ax1']), 
                               np.amin(result['fq']['ax0']), 
                               np.amax(result['fq']['ax0'])] )
    '''
    Data = np.asarray(Data)
    N, M = np.shape(Data)
    if flag_power_of_two: 
        N_fft = int(2**np.ceil(np.log2(N)))
        M_fft = int(2**np.ceil(np.log2(M)))
    else: 
        N_fft = N
        M_fft = M
    fq_stp_N = 2 * np.pi / N_fft / stepsize[0]
    fq_stp_M = 2 * np.pi / M_fft / stepsize[1]
    FFT = sp.fft.fftshift(sp.fft.fft2(Data, s=[N_fft, M_fft]))/N/M
    
    amp = np.abs(FFT)
    ph = np.angle(FFT)
    freq_N = (np.arange(N_fft) - N_fft/2) * fq_stp_N
    freq_M = (np.arange(M_fft) - M_fft/2) * fq_stp_M
    freq = {'ax0': freq_N, 'ax1': freq_M}
    
    return {'amp':amp, 'ph':ph, 'fq':freq}


def Gaussian_1D(Axis, Amp, Offset, Mu, Sigma): 
    ''' >>> Introduction <<< 
        This function generates a Gaussian curve using Axis. 
    '''
    Axis = np.asarray(Axis)
    M = np.shape(Axis)
    
    if np.size(M) > 1: 
        if np.prod(M) == np.size(Axis): 
            M = np.shape(Axis.ravel())
        else: 
            print(">>>>>> 'Axis' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    
    Data = Amp * np.exp(-(Axis - Mu)**2/2/Sigma**2) + Offset
    return Data


def Gaussian_2D(X, Y, Amp, Offset, MuX, SigmaX, MuY, SigmaY): 
    ''' >>> Introduction <<< 
        This function generates a Gaussian curve using Axis. 
    '''
    X = np.asarray(X)
    M = np.shape(X)
    Y = np.asarray(Y)
    N = np.shape(Y)
    
    if np.size(M) > 1: 
        if np.prod(M) == np.size(X): 
            X = X.ravel()
            M = np.shape(X)
        else: 
            print(">>>>>> 'X' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    if np.size(N) > 1: 
        if np.prod(N) == np.size(Y): 
            Y = Y.ravel()
            N = np.shape(Y)
        else: 
            print(">>>>>> 'Y' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    
    YY, XX = np.meshgrid(Y, X)
    
    Data = Amp * np.exp(-(XX - MuX)**2/2/SigmaX**2) * np.exp(-(YY - MuY)**2/2/SigmaY**2) + Offset
    return Data


def Gaussian_3D(X, Y, Z, Amp, Offset, MuX, SigmaX, MuY, SigmaY, MuZ, SigmaZ): 
    ''' >>> Introduction <<< 
        This function generates a Gaussian curve using Axis. 
    '''
    X = np.asarray(X)
    M = np.shape(X)
    Y = np.asarray(Y)
    N = np.shape(Y)
    Z = np.asarray(Z)
    L = np.shape(Z)
    
    if np.size(M) > 1: 
        if np.prod(M) == np.size(X): 
            X = X.ravel()
            M = np.shape(X)
        else: 
            print(">>>>>> 'X' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    if np.size(N) > 1: 
        if np.prod(N) == np.size(Y): 
            Y = Y.ravel()
            N = np.shape(Y)
        else: 
            print(">>>>>> 'Y' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    if np.size(L) > 1: 
        if np.prod(L) == np.size(Z): 
            Z = Z.ravel()
            L = np.shape(Z)
        else: 
            print(">>>>>> 'Z' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    
    YY, XX, ZZ = np.meshgrid(Y, X, Z)
    
    Data = Amp * (np.exp(-(XX - MuX)**2/2/SigmaX**2) * 
                  np.exp(-(YY - MuY)**2/2/SigmaY**2) * 
                  np.exp(-(ZZ - MuZ)**2/2/SigmaZ**2)) + Offset
    return Data


def Fit_Gaussian_1D(Data, Axis=None, Init_vals=None, Bound_vals=None, Display=False): 
    ''' >>> Introduction <<< 
        * This function fits the Data to a Gaussian function using Axis. 
        
        * Init_vals is the initial guess of the parameters. 
            Init_vals should be a list of [Amp, Offset, Mu, Sigma], where: 
                    Data = Amp * np.exp(-(Axis - Mu)**2/2/Sigma**2) + Offset
        
        * Bound_vals is the bounds of the parameters. 
            Init_vals should be a tuple of two lists [Amp, Offset, Mu, Sigma], i.e.  
                    ([Amp_low, Offset_low, Mu_low, Sigma_low], 
                     [Amp_up , Offset_up , Mu_up , Sigma_up ])
        
        * Output includes: 
            'Params': The parameters of the 1D Gaussian fit [Amp, Offset, Mu, Sigma] 
            'Error':  The sum of squared differences between Data and Fit. 
    '''
    Data = np.asarray(Data)
    if Axis is None: 
        Axis = np.arange(np.size(Data))
    else: 
        Axis = np.asarray(Axis)
    
    N = np.shape(Data)
    M = np.shape(Axis)
    
    if np.size(N) > 1: 
        if np.prod(N) == np.size(Data): 
            Data = Data.reshape(-1)
            N = np.size(Data)
        else: 
            print(">>>>>> 'Data' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    if np.size(M) > 1: 
        if np.prod(M) == np.size(Axis): 
            Axis = Axis.reshape(-1)
            M = np.size(Axis)
        else: 
            print(">>>>>> 'Axis' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    N = np.size(Data)
    M = np.size(Axis)
    if M != N: 
        print(">>>>>> 'Data' and 'Axis' should have the same size ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    
    if Init_vals is None: 
        Init_vals = [np.mean(Data),    # Amp
                     np.mean(Data),    # Offset
                     np.mean(Axis),    # Mu
                     (np.max(Axis) - np.min(Axis))/5]    # Sigma
    
    try: 
        if Bound_vals is None: 
            popt,pcov = curve_fit(Gaussian_1D, Axis, Data, p0=Init_vals)
        else: 
            popt,pcov = curve_fit(Gaussian_1D, Axis, Data, bounds=Bound_vals, p0=Init_vals)
    except RuntimeError: 
        print('>>>>>> Optimal parameters not found !!!')
        popt = Init_vals
    
    Fit  = Gaussian_1D(Axis, *popt)
    err2 = np.sum(np.square( np.divide(Fit - Data, Data) ))
    err  = np.sqrt(err2 / N)
    
    if Display: 
        plt.figure()
        plt.plot(Axis, Data, label='data', marker='o')
        plt.plot(Axis, Fit, label='fit')
        plt.title('Amp: %.2g, Offset: %.2g,\nMu: %.2g, Sigma: %.2g' 
                  %(popt[0],popt[1],popt[2],popt[3]))
        plt.legend()
        
    return {'Params': popt, 'Error': err}


def Fit_Gaussian_2D(Data, X=None, Y=None, Init_vals=None, Bound_vals=None, Display=False, DisplayType='2D'): 
    ''' >>> Introduction <<< 
        * This function fits the Data to a 2D Gaussian function using Axis. 
        
        * Init_vals is the initial guess of the parameters. 
            Init_vals should be a list of [Amp, Offset, MuX, SigmaX, MuY, SigmaY], where: 
                    Data = Amp * np.exp(-(X - MuX)**2/2/SigmaX**2) 
                               * np.exp(-(Y - MuY)**2/2/SigmaY**2) + Offset
        
        * Bound_vals is the bounds of the parameters. 
            Init_vals should be a tuple of two lists [Amp, Mu, Sigma, Offset], i.e.  
                    ([Amp_low, Offset_low, MuX_low, SigmaX_low, MuY_low, SigmaY_low], 
                     [Amp_up , Offset_up , MuX_up , SigmaX_up , MuY_up , SigmaY_up ])
        
        * Output includes: 
            'Params': The parameters of the 1D Gaussian fit [Amp, Offset, MuX, SigmaX, MuY, SigmaY] 
            'Error':  The sum of squared differences between Data and Fit. 
    '''
    Data = np.asarray(Data)
    
    if X is None: 
        X = np.arange(np.shape(Data)[0])
    else: 
        X = np.asarray(X)
    
    if Y is None: 
        Y = np.arange(np.shape(Data)[1])
    else: 
        Y = np.asarray(Y)
    
    M = np.shape(X)
    N = np.shape(Y)
    if np.size(M) > 1: 
        if np.prod(M) == np.size(X): 
            X = X.ravel()
            M = np.size(X)
        else: 
            print(">>>>>> 'X' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    else: 
        M = np.size(X)
    if np.size(N) > 1: 
        if np.prod(N) == np.size(Y): 
            Y = Y.ravel()
            N = np.size(Y)
        else: 
            print(">>>>>> 'Y' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    else: 
        N = np.size(Y)
    
    dims = np.shape(Data)
    if np.size(dims) != 2: 
        print(">>>>>> 'Data' should be a 2D array ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    if dims[0] != M or dims[1] != N:
        print(">>>>>> The shape of 'Data' does not match X or Y ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    
    if Init_vals is None: 
        Init_vals = [np.mean(Data),    # Amp
                     np.mean(Data),    # Offset
                     np.mean(X),    # MuX
                     (np.max(X) - np.min(X))/5,    # SigmaX
                     np.mean(Y),    # MuX
                     (np.max(Y) - np.min(Y))/5]    # SigmaX
    
    def func(Pos, Amp, Offset, MuX, SigmaX, MuY, SigmaY): 
        Data = (Amp * np.exp(-(Pos[0] - MuX)**2/2/SigmaX**2) 
                    * np.exp(-(Pos[1] - MuY)**2/2/SigmaY**2) + Offset)
        return Data
    
    YY, XX = np.meshgrid(Y, X)
    Axes = np.vstack( (XX.ravel(), YY.ravel()) )
    
    try: 
        if Bound_vals is None: 
            popt,pcov = curve_fit(func, Axes, Data.ravel(), p0=Init_vals)
        else: 
            popt,pcov = curve_fit(func, Axis, Data.ravel(), p0=Init_vals, bounds=Bound_vals)
    except RuntimeError: 
        print('>>>>>> Optimal parameters not found !!!')
        popt = Init_vals
    
    Fit  = func(Axes, *popt).reshape(dims)
    err2 = np.sum(np.square( np.divide(Fit - Data, Data) ))
    err  = np.sqrt(err2/N/M)
    
    if Display: 
        if DisplayType == '3D': 
            fig = plt.figure()
            ax = Axes3D(fig)
            for i in range(N*M//500):
                ind = np.unravel_index(i*500, dims)
                ax.scatter(XX[ind], YY[ind], Data[ind], marker='o', alpha=0.25, color='r')
            ax.set_title('Amp: %.2g, Offset:%.2g;\nMuX: %.2g, SigmaX:%.2g;\nMuY: %.2g, SigmaY:%.2g.'
                          %(popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]))
            fig = plt.gca(projection='3d')
            fig.plot_surface(XX, YY, Fit, alpha=0.5)
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(Data.transpose(), cmap='plasma', origin='lower', 
                      extent=(X.min(), X.max(), Y.min(), Y.max()))
            ax.contour(XX, YY, Fit, colors='w', alpha=0.5)
            ax.set_title('Amp: %.2g, Offset:%.2g;\nMuX: %.2g, SigmaX:%.2g;\nMuY: %.2g, SigmaY:%.2g.'
                          %(popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]))
            plt.show()
        
    return {'Params': popt, 'Error': err}


def Fit_Gaussian_3D(Data, X=None, Y=None, Z=None, Init_vals=None, Bound_vals=None, Display=False, DisplayType='2D'): 
    ''' >>> Introduction <<< 
        * This function fits the Data to a 3D Gaussian function using Axis. 
        
        * Init_vals is the initial guess of the parameters. 
            Init_vals should be a list of [Amp, Offset, MuX, SigmaX, MuY, SigmaY, MuZ, SigmaZ], where: 
                    Data = Amp * np.exp(-(X - MuX)**2/2/SigmaX**2) 
                               * np.exp(-(Y - MuY)**2/2/SigmaY**2)
                               * np.exp(-(Z - MuZ)**2/2/SigmaZ**2) + Offset
        
        * Bound_vals is the bounds of the parameters. 
            Init_vals should be a tuple of two lists [Amp, Mu, Sigma, Offset], i.e.  
                    ([Amp_low, Offset_low, MuX_low, SigmaX_low, MuY_low, SigmaY_low, MuZ_low, SigmaZ_low], 
                     [Amp_up , Offset_up , MuX_up , SigmaX_up , MuY_up , SigmaY_up , MuZ_up , SigmaZ_up ])
        
        * Output includes: 
            'Params': The parameters of the 1D Gaussian fit [Amp, Offset, MuX, SigmaX, MuY, SigmaY, MuZ, SigmaZ] 
            'Error':  The sum of squared differences between Data and Fit. 
    '''
    Data = np.asarray(Data)
    
    if X is None: 
        X = np.arange(np.shape(Data)[0])
    else: 
        X = np.asarray(X)
    
    if Y is None: 
        Y = np.arange(np.shape(Data)[1])
    else: 
        Y = np.asarray(Y)
    
    if Z is None: 
        Z = np.arange(np.shape(Data)[2])
    else: 
        Z = np.asarray(Z)
    
    M = np.shape(X)
    N = np.shape(Y)
    L = np.shape(Z)
    if np.size(M) > 1: 
        if np.prod(M) == np.size(X): 
            X = X.ravel()
            M = np.size(X)
        else: 
            print(">>>>>> 'X' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    else: 
        M = np.size(X)
    if np.size(N) > 1: 
        if np.prod(N) == np.size(Y): 
            Y = Y.ravel()
            N = np.size(Y)
        else: 
            print(">>>>>> 'Y' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    else: 
        N = np.size(Y)
    if np.size(L) > 1: 
        if np.prod(L) == np.size(Z): 
            Z = Z.ravel()
            L = np.size(L)
        else: 
            print(">>>>>> 'Z' should be a 1D array ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    else: 
        L = np.size(Z)
    
    dims = np.shape(Data)
    if np.size(dims) != 3: 
        print(">>>>>> 'Data' should be a 3D array ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    if dims[0] != M or dims[1] != N or dims[2] != L:
        print(">>>>>> The shape of 'Data' does not match X, Y, or Z ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    
    if Init_vals is None: 
        Init_vals = [np.mean(Data),    # Amp
                     np.mean(Data),    # Offset
                     np.mean(X),    # MuX
                     (np.max(X) - np.min(X))/5,    # SigmaX
                     np.mean(Y),    # MuX
                     (np.max(Y) - np.min(Y))/5,    # SigmaX
                     np.mean(Z),    # MuX
                     (np.max(Z) - np.min(Z))/5]    # SigmaX
    
    def func(Pos, Amp, Offset, MuX, SigmaX, MuY, SigmaY, MuZ, SigmaZ): 
        Data = (Amp * np.exp(-(Pos[0] - MuX)**2/2/SigmaX**2) 
                    * np.exp(-(Pos[1] - MuY)**2/2/SigmaY**2)
                    * np.exp(-(Pos[2] - MuZ)**2/2/SigmaZ**2) + Offset)
        return Data
    
    YY, XX, ZZ = np.meshgrid(Y, X, Z)
    Axes = np.vstack( (XX.ravel(), YY.ravel(), ZZ.ravel()) )
    
    try: 
        if Bound_vals is None: 
            popt,pcov = curve_fit(func, Axes, Data.ravel(), p0=Init_vals)
        else: 
            popt,pcov = curve_fit(func, Axis, Data.ravel(), p0=Init_vals, bounds=Bound_vals)
    except RuntimeError: 
        print('>>>>>> Optimal parameters not found !!!')
        popt = Init_vals
    
    Fit  = func(Axes, *popt).reshape(dims)
    err2 = np.sum(np.square( np.divide(Fit - Data, Data) ))
    err  = np.sqrt(err2/N/M)
    
    if Display: 
        if DisplayType == '3D': 
            fig = plt.figure()
            ax = Axes3D(fig)
            for i in range(N*M*L//500):
                ind = np.unravel_index(i*500, dims)
                ax.scatter(XX[ind], YY[ind], ZZ[ind], Data[ind], marker='o', alpha=0.25, color='r')
            ax.set_title('Amp: %.2g, Offset:%.2g;\nMuX: %.2g, SigmaX:%.2g;\nMuY: %.2g, SigmaY:%.2g;\nMuZ: %.2g, SigmaZ:%.2g.'
                          %(popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]))
            ''' Need plot isosurface here '''
            # fig = plt.gca(projection='3d')
            # fig.plot_surface(XX, YY, ZZ, Fit, alpha=0.5)
            plt.show()
        else:
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.imshow(Data.transpose(), cmap='plasma', origin='lower', 
            #           extent=(X.min(), X.max(), Y.min(), Y.max()))
            # ax.contour(XX, YY, Fit, colors='w', alpha=0.5)
            # ax.set_title('Amp: %.2g, Offset:%.2g;\nMuX: %.2g, SigmaX:%.2g;\nMuY: %.2g, SigmaY:%.2g.'
            #               %(popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]))
            # plt.show()
            print('Display function not available yet')
        
    return {'Params': popt, 'Error': err}


def Polynomial_1D(axis=None, params=[0,0,0]): 
    ''' >>> Instruction <<< 
        This function generates polynomial based on parameters. 
        
        Inputs: 
            axis                  The coordinates of the points, 1D array n
            params                Polynomial parameters
            Flag_plot_result      If True, plot the result
        
        Output: 
            data                  points of the polynomial curve
    '''
    dims = np.shape(axis)
    if np.size(dims) != 1: 
        print(">>>>>> 'axis' should be a 1D array ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    orders = np.size(params)
    data = np.zeros(dims)
    for i in range(orders): 
        data += params[i] * axis**i
    return data


def Fit_polynomial_1D(data, axis=None, order=2, outlier=None, Flag_remove_outlier=True, Flag_plot_result=False): 
    ''' >>> Instruction <<< 
        This function determines the best-fit using 1st or higher order polynomial 
        over a set of points.
        
        Inputs: 
            data                  The data points, 1D array n
            axis                  The  coordinates of the points, 1D array n
            order                 Order of the polynomial fitting
            outlier               The coords of points that are excluded in the fitting, [[1,2], ...]
            Flag_remove_outlier   If True,  the region defined by 'outlier' is not fitted
                                  If False, ONLY the region defined by 'outlier' is fitted
            Flag_plot_result      If True, plot the fitting result
        
        Outputs: 
            fit                   Fitted curve
            params                Polynomial parameters
    '''
    # Inputs regularization
    dims = np.shape(data)
    if np.size(dims) != 1: 
        print(">>>>>> 'data' should be a 1D array ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    n = dims[0]
    if axis is None: 
        axis = np.linspace(1, n, n)
    else: 
        if not np.size(axis) == n: 
            print(">>>>>> 'axis' size does NOT match 'data' ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    # Remove/Keep outliers
    data_full = data
    axis_full = axis
    if outlier is not None: 
        p = np.shape(outlier)[0]
        for i in range(p): 
            idx_outlier = np.all([axis >= outlier[i][0],axis <= outlier[i][1]], axis=0)
            if Flag_remove_outlier: 
                if i == 0: 
                    idx = ~idx_outlier
                else: 
                    idx = np.all([idx, ~idx_outlier],axis=0)
            else: 
                if i == 0: 
                    idx = idx_outlier
                else:
                    idx = np.any([idx, idx_outlier],axis=0)
        data = data[idx]
        axis = axis[idx]
    
    # Fitting
    if   order == 1: 
        # best-fit linear plane
        A = np.c_[np.ones(np.size(data)), 
                  axis]
        C,_,_,_ = sp.linalg.lstsq(A, data)    # coefficients
        # evaluate it on grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full], C).reshape(dims)
    elif order == 2: 
        # best-fit quadratic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2], C).reshape(dims)
    elif order == 3: 
        # best-fit cubic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3], C).reshape(dims)
    elif order == 4: 
        # best-fit quartic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4], C).reshape(dims)
    elif order == 5: 
        # best-fit quintic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5], C).reshape(dims)
    elif order == 6: 
        # best-fit sextic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6], C).reshape(dims)
    elif order == 7: 
        # best-fit septic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6, 
                  axis**7]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6, 
                         axis_full**7], C).reshape(dims)
    elif order == 8: 
        # best-fit octic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6, 
                  axis**7, 
                  axis**8]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6, 
                         axis_full**7, 
                         axis_full**8], C).reshape(dims)
    elif order == 9: 
        # best-fit nonic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6, 
                  axis**7, 
                  axis**8, 
                  axis**9]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6, 
                         axis_full**7, 
                         axis_full**8, 
                         axis_full**9], C).reshape(dims)
    elif order == 10: 
        # best-fit decic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2, 
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6, 
                  axis**7, 
                  axis**8, 
                  axis**9, 
                  axis**10]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6, 
                         axis_full**7, 
                         axis_full**8, 
                         axis_full**9, 
                         axis_full**10], C).reshape(dims)
    # plot points and fitted curve
    if Flag_plot_result: 
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(111)
        ax.plot(axis_full, Z, alpha=0.2)
        ax.scatter(axis, data, c='r', s=3)
        plt.xlabel('axis')
        plt.ylabel('data')
        # ax.axis('equal')
        plt.tight_layout()
        plt.show()
    return {'fit':Z, 'params': C}


def Fit_polynomial_2D(z_array, x_coords=None, y_coords=None, order=2, 
                           x_outlier=None, y_outlier=None, Flag_remove_outlier=True, 
                           Flag_plot_result=False): 
    ''' >>> Instruction <<< 
        This function determines the best-fit plane/surface (1st or higher order polynomial) 
        over a set of three-dimensional points.
        
        Inputs: 
            z_array               The height values of the points, 2D array m*n
            x_coords              The x coordinates of the points, 1D array n
            y_coords              The y coordinates of the points, 1D array m
            order                 Order of the polynomial fitting
            x_outlier             The x coords of points that are excluded in the fitting, [[1,2], ...]
            y_outlier             The y coords of points that are excluded in the fitting, [[1,2], ...]
            Flag_remove_outlier   If True,  the region defined by x/y_outlier is not fitted
                                  If False, only the region defined by x/y_outlier is fitted
            Flag_plot_result      If True, plot the fitting result
        
        Outputs: 
            fit                   Fitted curve
            params                Polynomial parameters
    '''
    [m, n] = np.shape(z_array)
    if x_coords is None: 
        x_coords = np.linspace(1, n, n)
    else: 
        if not np.size(x_coords) == n: 
            print(">>>>>> 'x_coords' size does NOT match 'z_array' ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    if y_coords is None: 
        y_coords = np.linspace(1, m, m)
    else: 
        if not np.size(y_coords) == n: 
            print(">>>>>> 'y_coords' size does NOT match 'z_array' ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    
    X, Y = np.meshgrid(x_coords, y_coords)
    XX = X.flatten()
    YY = Y.flatten()
    data = z_array.flatten()
    
    # Remove/Keep outliers
    data_full = data
    XX_full = XX
    YY_full = YY
    
    if not (x_outlier is None or y_outlier is None): 
        p = np.shape(x_outlier)[0]
        q = np.shape(y_outlier)[0]
        if not p == q: 
            print(">>>>>> 'x_outlier' does NOT match 'y_outlier' ! ")
            input('>>>>>> Press any key to quit ... ')
            return
        
        for i in range(p): 
            idx_outlier = np.all([np.all([XX >= x_outlier[i][0],XX <= x_outlier[i][1]], axis=0), 
                                  np.all([YY >= y_outlier[i][0],YY <= y_outlier[i][1]], axis=0)],
                                 axis=0) 
            if Flag_remove_outlier: 
                if i == 0: 
                    idx = ~idx_outlier
                else: 
                    idx = np.all([idx, ~idx_outlier],axis=0)
            else: 
                if i == 0: 
                    idx = idx_outlier
                else:
                    idx = np.any([idx, idx_outlier],axis=0)
        data = data[idx]
        XX = XX[idx]
        YY = YY[idx]
    
    # Fitting
    if   order == 1: 
        # best-fit linear plane
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY]
        C,_,_,_ = sp.linalg.lstsq(A, data)    # coefficients
        # evaluate it on grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full], C).reshape(X.shape)
    elif order == 2: 
        # best-fit quadratic curve
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY, 
                  XX**2, 
                  np.multiply(XX, YY), 
                  YY**2]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full, 
                         XX_full**2, 
                         np.multiply(XX_full, YY_full), 
                         YY_full**2], C).reshape(X.shape)
    elif order == 3: 
        # best-fit cubic curve
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY, 
                  XX**2, 
                  np.multiply(XX**1, YY**1), 
                  YY**2,
                  XX**3, 
                  np.multiply(XX**2, YY**1), 
                  np.multiply(XX**1, YY**2), 
                  YY**3]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full, 
                         XX_full**2, 
                         np.multiply(XX_full**1, YY_full**1), 
                         YY_full**2,
                         XX_full**3, 
                         np.multiply(XX_full**2, YY_full**1), 
                         np.multiply(XX_full**1, YY_full**2), 
                         YY_full**3], C).reshape(X.shape)
    elif order == 4: 
        # best-fit quartic curve
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY, 
                  XX**2, 
                  np.multiply(XX**1, YY**1), 
                  YY**2,
                  XX**3, 
                  np.multiply(XX**2, YY**1), 
                  np.multiply(XX**1, YY**2), 
                  YY**3, 
                  XX**4,
                  np.multiply(XX**3, YY**1), 
                  np.multiply(XX**2, YY**2), 
                  np.multiply(XX**1, YY**3), 
                  YY**4]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full, 
                         XX_full**2, 
                         np.multiply(XX_full**1, YY_full**1), 
                         YY_full**2,
                         XX_full**3, 
                         np.multiply(XX_full**2, YY_full**1), 
                         np.multiply(XX_full**1, YY_full**2), 
                         YY_full**3, 
                         XX_full**4,
                         np.multiply(XX_full**3, YY_full**1), 
                         np.multiply(XX_full**2, YY_full**2), 
                         np.multiply(XX_full**1, YY_full**3), 
                         YY_full**4], C).reshape(X.shape)
    elif order == 5: 
        # best-fit quintic curve
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY, 
                  XX**2, 
                  np.multiply(XX**1, YY**1), 
                  YY**2,
                  XX**3, 
                  np.multiply(XX**2, YY**1), 
                  np.multiply(XX**1, YY**2), 
                  YY**3, 
                  XX**4,
                  np.multiply(XX**3, YY**1), 
                  np.multiply(XX**2, YY**2), 
                  np.multiply(XX**1, YY**3), 
                  YY**4, 
                  XX**5, 
                  np.multiply(XX**4, YY**1), 
                  np.multiply(XX**3, YY**2), 
                  np.multiply(XX**2, YY**3), 
                  np.multiply(XX**1, YY**4), 
                  YY**5]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full, 
                         XX_full**2, 
                         np.multiply(XX_full**1, YY_full**1), 
                         YY_full**2,
                         XX_full**3, 
                         np.multiply(XX_full**2, YY_full**1), 
                         np.multiply(XX_full**1, YY_full**2), 
                         YY_full**3, 
                         XX_full**4,
                         np.multiply(XX_full**3, YY_full**1), 
                         np.multiply(XX_full**2, YY_full**2), 
                         np.multiply(XX_full**1, YY_full**3), 
                         YY_full**4, 
                         XX_full**5, 
                         np.multiply(XX_full**4, YY_full**1), 
                         np.multiply(XX_full**3, YY_full**2), 
                         np.multiply(XX_full**2, YY_full**3), 
                         np.multiply(XX_full**1, YY_full**4), 
                         YY_full**5], C).reshape(X.shape)
    elif order == 6: 
        # best-fit sextic curve
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY, 
                  XX**2, 
                  np.multiply(XX**1, YY**1), 
                  YY**2,
                  XX**3, 
                  np.multiply(XX**2, YY**1), 
                  np.multiply(XX**1, YY**2), 
                  YY**3, 
                  XX**4,
                  np.multiply(XX**3, YY**1), 
                  np.multiply(XX**2, YY**2), 
                  np.multiply(XX**1, YY**3), 
                  YY**4, 
                  XX**5, 
                  np.multiply(XX**4, YY**1), 
                  np.multiply(XX**3, YY**2), 
                  np.multiply(XX**2, YY**3), 
                  np.multiply(XX**1, YY**4), 
                  YY**5,
                  XX**6, 
                  np.multiply(XX**5, YY**1), 
                  np.multiply(XX**4, YY**2), 
                  np.multiply(XX**3, YY**3), 
                  np.multiply(XX**2, YY**4), 
                  np.multiply(XX**1, YY**5), 
                  YY**6]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full, 
                         XX_full**2, 
                         np.multiply(XX_full**1, YY_full**1), 
                         YY_full**2,
                         XX_full**3, 
                         np.multiply(XX_full**2, YY_full**1), 
                         np.multiply(XX_full**1, YY_full**2), 
                         YY_full**3, 
                         XX_full**4,
                         np.multiply(XX_full**3, YY_full**1), 
                         np.multiply(XX_full**2, YY_full**2), 
                         np.multiply(XX_full**1, YY_full**3), 
                         YY_full**4, 
                         XX_full**5, 
                         np.multiply(XX_full**4, YY_full**1), 
                         np.multiply(XX_full**3, YY_full**2), 
                         np.multiply(XX_full**2, YY_full**3), 
                         np.multiply(XX_full**1, YY_full**4), 
                         YY_full**5, 
                         XX_full**6, 
                         np.multiply(XX_full**5, YY_full**1), 
                         np.multiply(XX_full**4, YY_full**2), 
                         np.multiply(XX_full**3, YY_full**3), 
                         np.multiply(XX_full**2, YY_full**4), 
                         np.multiply(XX_full**1, YY_full**5), 
                         YY_full**6], C).reshape(X.shape)
    elif order == 7: 
        # best-fit septic curve
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY, 
                  XX**2, 
                  np.multiply(XX**1, YY**1), 
                  YY**2,
                  XX**3, 
                  np.multiply(XX**2, YY**1), 
                  np.multiply(XX**1, YY**2), 
                  YY**3, 
                  XX**4,
                  np.multiply(XX**3, YY**1), 
                  np.multiply(XX**2, YY**2), 
                  np.multiply(XX**1, YY**3), 
                  YY**4, 
                  XX**5, 
                  np.multiply(XX**4, YY**1), 
                  np.multiply(XX**3, YY**2), 
                  np.multiply(XX**2, YY**3), 
                  np.multiply(XX**1, YY**4), 
                  YY**5,
                  XX**6, 
                  np.multiply(XX**5, YY**1), 
                  np.multiply(XX**4, YY**2), 
                  np.multiply(XX**3, YY**3), 
                  np.multiply(XX**2, YY**4), 
                  np.multiply(XX**1, YY**5), 
                  YY**6, 
                  XX**7, 
                  np.multiply(XX**6, YY**1), 
                  np.multiply(XX**5, YY**2), 
                  np.multiply(XX**4, YY**3), 
                  np.multiply(XX**3, YY**4), 
                  np.multiply(XX**2, YY**5), 
                  np.multiply(XX**1, YY**6), 
                  YY**7]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full, 
                         XX_full**2, 
                         np.multiply(XX_full**1, YY_full**1), 
                         YY_full**2,
                         XX_full**3, 
                         np.multiply(XX_full**2, YY_full**1), 
                         np.multiply(XX_full**1, YY_full**2), 
                         YY_full**3, 
                         XX_full**4,
                         np.multiply(XX_full**3, YY_full**1), 
                         np.multiply(XX_full**2, YY_full**2), 
                         np.multiply(XX_full**1, YY_full**3), 
                         YY_full**4, 
                         XX_full**5, 
                         np.multiply(XX_full**4, YY_full**1), 
                         np.multiply(XX_full**3, YY_full**2), 
                         np.multiply(XX_full**2, YY_full**3), 
                         np.multiply(XX_full**1, YY_full**4), 
                         YY_full**5, 
                         XX_full**6, 
                         np.multiply(XX_full**5, YY_full**1), 
                         np.multiply(XX_full**4, YY_full**2), 
                         np.multiply(XX_full**3, YY_full**3), 
                         np.multiply(XX_full**2, YY_full**4), 
                         np.multiply(XX_full**1, YY_full**5), 
                         YY_full**6, 
                         XX_full**7, 
                         np.multiply(XX_full**6, YY_full**1), 
                         np.multiply(XX_full**5, YY_full**2), 
                         np.multiply(XX_full**4, YY_full**3), 
                         np.multiply(XX_full**3, YY_full**4), 
                         np.multiply(XX_full**2, YY_full**5), 
                         np.multiply(XX_full**1, YY_full**6), 
                         YY_full**7], C).reshape(X.shape)
    elif order == 8: 
        # best-fit octic curve
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY, 
                  XX**2, 
                  np.multiply(XX**1, YY**1), 
                  YY**2,
                  XX**3, 
                  np.multiply(XX**2, YY**1), 
                  np.multiply(XX**1, YY**2), 
                  YY**3, 
                  XX**4,
                  np.multiply(XX**3, YY**1), 
                  np.multiply(XX**2, YY**2), 
                  np.multiply(XX**1, YY**3), 
                  YY**4, 
                  XX**5, 
                  np.multiply(XX**4, YY**1), 
                  np.multiply(XX**3, YY**2), 
                  np.multiply(XX**2, YY**3), 
                  np.multiply(XX**1, YY**4), 
                  YY**5,
                  XX**6, 
                  np.multiply(XX**5, YY**1), 
                  np.multiply(XX**4, YY**2), 
                  np.multiply(XX**3, YY**3), 
                  np.multiply(XX**2, YY**4), 
                  np.multiply(XX**1, YY**5), 
                  YY**6, 
                  XX**7, 
                  np.multiply(XX**6, YY**1), 
                  np.multiply(XX**5, YY**2), 
                  np.multiply(XX**4, YY**3), 
                  np.multiply(XX**3, YY**4), 
                  np.multiply(XX**2, YY**5), 
                  np.multiply(XX**1, YY**6), 
                  YY**7, 
                  XX**8, 
                  np.multiply(XX**7, YY**1), 
                  np.multiply(XX**6, YY**2), 
                  np.multiply(XX**5, YY**3), 
                  np.multiply(XX**4, YY**4), 
                  np.multiply(XX**3, YY**5), 
                  np.multiply(XX**2, YY**6), 
                  np.multiply(XX**1, YY**7), 
                  YY**8]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full, 
                         XX_full**2, 
                         np.multiply(XX_full**1, YY_full**1), 
                         YY_full**2,
                         XX_full**3, 
                         np.multiply(XX_full**2, YY_full**1), 
                         np.multiply(XX_full**1, YY_full**2), 
                         YY_full**3, 
                         XX_full**4,
                         np.multiply(XX_full**3, YY_full**1), 
                         np.multiply(XX_full**2, YY_full**2), 
                         np.multiply(XX_full**1, YY_full**3), 
                         YY_full**4, 
                         XX_full**5, 
                         np.multiply(XX_full**4, YY_full**1), 
                         np.multiply(XX_full**3, YY_full**2), 
                         np.multiply(XX_full**2, YY_full**3), 
                         np.multiply(XX_full**1, YY_full**4), 
                         YY_full**5, 
                         XX_full**6, 
                         np.multiply(XX_full**5, YY_full**1), 
                         np.multiply(XX_full**4, YY_full**2), 
                         np.multiply(XX_full**3, YY_full**3), 
                         np.multiply(XX_full**2, YY_full**4), 
                         np.multiply(XX_full**1, YY_full**5), 
                         YY_full**6, 
                         XX_full**7, 
                         np.multiply(XX_full**6, YY_full**1), 
                         np.multiply(XX_full**5, YY_full**2), 
                         np.multiply(XX_full**4, YY_full**3), 
                         np.multiply(XX_full**3, YY_full**4), 
                         np.multiply(XX_full**2, YY_full**5), 
                         np.multiply(XX_full**1, YY_full**6), 
                         YY_full**7, 
                         XX_full**8, 
                         np.multiply(XX_full**7, YY_full**1), 
                         np.multiply(XX_full**6, YY_full**2), 
                         np.multiply(XX_full**5, YY_full**3), 
                         np.multiply(XX_full**4, YY_full**4), 
                         np.multiply(XX_full**3, YY_full**5), 
                         np.multiply(XX_full**2, YY_full**6), 
                         np.multiply(XX_full**1, YY_full**7), 
                         YY_full**8], C).reshape(X.shape)
    elif order == 9: 
        # best-fit nonic curve
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY, 
                  XX**2, 
                  np.multiply(XX**1, YY**1), 
                  YY**2,
                  XX**3, 
                  np.multiply(XX**2, YY**1), 
                  np.multiply(XX**1, YY**2), 
                  YY**3, 
                  XX**4,
                  np.multiply(XX**3, YY**1), 
                  np.multiply(XX**2, YY**2), 
                  np.multiply(XX**1, YY**3), 
                  YY**4, 
                  XX**5, 
                  np.multiply(XX**4, YY**1), 
                  np.multiply(XX**3, YY**2), 
                  np.multiply(XX**2, YY**3), 
                  np.multiply(XX**1, YY**4), 
                  YY**5,
                  XX**6, 
                  np.multiply(XX**5, YY**1), 
                  np.multiply(XX**4, YY**2), 
                  np.multiply(XX**3, YY**3), 
                  np.multiply(XX**2, YY**4), 
                  np.multiply(XX**1, YY**5), 
                  YY**6, 
                  XX**7, 
                  np.multiply(XX**6, YY**1), 
                  np.multiply(XX**5, YY**2), 
                  np.multiply(XX**4, YY**3), 
                  np.multiply(XX**3, YY**4), 
                  np.multiply(XX**2, YY**5), 
                  np.multiply(XX**1, YY**6), 
                  YY**7, 
                  XX**8, 
                  np.multiply(XX**7, YY**1), 
                  np.multiply(XX**6, YY**2), 
                  np.multiply(XX**5, YY**3), 
                  np.multiply(XX**4, YY**4), 
                  np.multiply(XX**3, YY**5), 
                  np.multiply(XX**2, YY**6), 
                  np.multiply(XX**1, YY**7), 
                  YY**8, 
                  XX**9, 
                  np.multiply(XX**8, YY**1), 
                  np.multiply(XX**7, YY**2), 
                  np.multiply(XX**6, YY**3), 
                  np.multiply(XX**5, YY**4), 
                  np.multiply(XX**4, YY**5), 
                  np.multiply(XX**3, YY**6), 
                  np.multiply(XX**2, YY**7), 
                  np.multiply(XX**1, YY**8), 
                  YY**9]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full, 
                         XX_full**2, 
                         np.multiply(XX_full**1, YY_full**1), 
                         YY_full**2,
                         XX_full**3, 
                         np.multiply(XX_full**2, YY_full**1), 
                         np.multiply(XX_full**1, YY_full**2), 
                         YY_full**3, 
                         XX_full**4,
                         np.multiply(XX_full**3, YY_full**1), 
                         np.multiply(XX_full**2, YY_full**2), 
                         np.multiply(XX_full**1, YY_full**3), 
                         YY_full**4, 
                         XX_full**5, 
                         np.multiply(XX_full**4, YY_full**1), 
                         np.multiply(XX_full**3, YY_full**2), 
                         np.multiply(XX_full**2, YY_full**3), 
                         np.multiply(XX_full**1, YY_full**4), 
                         YY_full**5, 
                         XX_full**6, 
                         np.multiply(XX_full**5, YY_full**1), 
                         np.multiply(XX_full**4, YY_full**2), 
                         np.multiply(XX_full**3, YY_full**3), 
                         np.multiply(XX_full**2, YY_full**4), 
                         np.multiply(XX_full**1, YY_full**5), 
                         YY_full**6, 
                         XX_full**7, 
                         np.multiply(XX_full**6, YY_full**1), 
                         np.multiply(XX_full**5, YY_full**2), 
                         np.multiply(XX_full**4, YY_full**3), 
                         np.multiply(XX_full**3, YY_full**4), 
                         np.multiply(XX_full**2, YY_full**5), 
                         np.multiply(XX_full**1, YY_full**6), 
                         YY_full**7, 
                         XX_full**8, 
                         np.multiply(XX_full**7, YY_full**1), 
                         np.multiply(XX_full**6, YY_full**2), 
                         np.multiply(XX_full**5, YY_full**3), 
                         np.multiply(XX_full**4, YY_full**4), 
                         np.multiply(XX_full**3, YY_full**5), 
                         np.multiply(XX_full**2, YY_full**6), 
                         np.multiply(XX_full**1, YY_full**7), 
                         YY_full**8, 
                         XX_full**9, 
                         np.multiply(XX_full**8, YY_full**1), 
                         np.multiply(XX_full**7, YY_full**2), 
                         np.multiply(XX_full**6, YY_full**3), 
                         np.multiply(XX_full**5, YY_full**4), 
                         np.multiply(XX_full**4, YY_full**5), 
                         np.multiply(XX_full**3, YY_full**6), 
                         np.multiply(XX_full**2, YY_full**7), 
                         np.multiply(XX_full**1, YY_full**8), 
                         YY_full**9], C).reshape(X.shape)
    elif order == 10: 
        # best-fit decic curve
        A = np.c_[np.ones(np.size(data)), 
                  XX, 
                  YY, 
                  XX**2, 
                  np.multiply(XX**1, YY**1), 
                  YY**2,
                  XX**3, 
                  np.multiply(XX**2, YY**1), 
                  np.multiply(XX**1, YY**2), 
                  YY**3, 
                  XX**4,
                  np.multiply(XX**3, YY**1), 
                  np.multiply(XX**2, YY**2), 
                  np.multiply(XX**1, YY**3), 
                  YY**4, 
                  XX**5, 
                  np.multiply(XX**4, YY**1), 
                  np.multiply(XX**3, YY**2), 
                  np.multiply(XX**2, YY**3), 
                  np.multiply(XX**1, YY**4), 
                  YY**5,
                  XX**6, 
                  np.multiply(XX**5, YY**1), 
                  np.multiply(XX**4, YY**2), 
                  np.multiply(XX**3, YY**3), 
                  np.multiply(XX**2, YY**4), 
                  np.multiply(XX**1, YY**5), 
                  YY**6, 
                  XX**7, 
                  np.multiply(XX**6, YY**1), 
                  np.multiply(XX**5, YY**2), 
                  np.multiply(XX**4, YY**3), 
                  np.multiply(XX**3, YY**4), 
                  np.multiply(XX**2, YY**5), 
                  np.multiply(XX**1, YY**6), 
                  YY**7, 
                  XX**8, 
                  np.multiply(XX**7, YY**1), 
                  np.multiply(XX**6, YY**2), 
                  np.multiply(XX**5, YY**3), 
                  np.multiply(XX**4, YY**4), 
                  np.multiply(XX**3, YY**5), 
                  np.multiply(XX**2, YY**6), 
                  np.multiply(XX**1, YY**7), 
                  YY**8, 
                  XX**9, 
                  np.multiply(XX**8, YY**1), 
                  np.multiply(XX**7, YY**2), 
                  np.multiply(XX**6, YY**3), 
                  np.multiply(XX**5, YY**4), 
                  np.multiply(XX**4, YY**5), 
                  np.multiply(XX**3, YY**6), 
                  np.multiply(XX**2, YY**7), 
                  np.multiply(XX**1, YY**8), 
                  YY**9, 
                  XX**10, 
                  np.multiply(XX**9, YY**1), 
                  np.multiply(XX**8, YY**2), 
                  np.multiply(XX**7, YY**3), 
                  np.multiply(XX**6, YY**4), 
                  np.multiply(XX**5, YY**5), 
                  np.multiply(XX**4, YY**6), 
                  np.multiply(XX**3, YY**7), 
                  np.multiply(XX**2, YY**8), 
                  np.multiply(XX**1, YY**9), 
                  YY**10]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX_full.shape), 
                         XX_full, 
                         YY_full, 
                         XX_full**2, 
                         np.multiply(XX_full**1, YY_full**1), 
                         YY_full**2,
                         XX_full**3, 
                         np.multiply(XX_full**2, YY_full**1), 
                         np.multiply(XX_full**1, YY_full**2), 
                         YY_full**3, 
                         XX_full**4,
                         np.multiply(XX_full**3, YY_full**1), 
                         np.multiply(XX_full**2, YY_full**2), 
                         np.multiply(XX_full**1, YY_full**3), 
                         YY_full**4, 
                         XX_full**5, 
                         np.multiply(XX_full**4, YY_full**1), 
                         np.multiply(XX_full**3, YY_full**2), 
                         np.multiply(XX_full**2, YY_full**3), 
                         np.multiply(XX_full**1, YY_full**4), 
                         YY_full**5, 
                         XX_full**6, 
                         np.multiply(XX_full**5, YY_full**1), 
                         np.multiply(XX_full**4, YY_full**2), 
                         np.multiply(XX_full**3, YY_full**3), 
                         np.multiply(XX_full**2, YY_full**4), 
                         np.multiply(XX_full**1, YY_full**5), 
                         YY_full**6, 
                         XX_full**7, 
                         np.multiply(XX_full**6, YY_full**1), 
                         np.multiply(XX_full**5, YY_full**2), 
                         np.multiply(XX_full**4, YY_full**3), 
                         np.multiply(XX_full**3, YY_full**4), 
                         np.multiply(XX_full**2, YY_full**5), 
                         np.multiply(XX_full**1, YY_full**6), 
                         YY_full**7, 
                         XX_full**8, 
                         np.multiply(XX_full**7, YY_full**1), 
                         np.multiply(XX_full**6, YY_full**2), 
                         np.multiply(XX_full**5, YY_full**3), 
                         np.multiply(XX_full**4, YY_full**4), 
                         np.multiply(XX_full**3, YY_full**5), 
                         np.multiply(XX_full**2, YY_full**6), 
                         np.multiply(XX_full**1, YY_full**7), 
                         YY_full**8, 
                         XX_full**9, 
                         np.multiply(XX_full**8, YY_full**1), 
                         np.multiply(XX_full**7, YY_full**2), 
                         np.multiply(XX_full**6, YY_full**3), 
                         np.multiply(XX_full**5, YY_full**4), 
                         np.multiply(XX_full**4, YY_full**5), 
                         np.multiply(XX_full**3, YY_full**6), 
                         np.multiply(XX_full**2, YY_full**7), 
                         np.multiply(XX_full**1, YY_full**8), 
                         YY_full**9, 
                         XX_full**10, 
                         np.multiply(XX_full**9, YY_full**1), 
                         np.multiply(XX_full**8, YY_full**2), 
                         np.multiply(XX_full**7, YY_full**3), 
                         np.multiply(XX_full**6, YY_full**4), 
                         np.multiply(XX_full**5, YY_full**5), 
                         np.multiply(XX_full**4, YY_full**6), 
                         np.multiply(XX_full**3, YY_full**7), 
                         np.multiply(XX_full**2, YY_full**8), 
                         np.multiply(XX_full**1, YY_full**9), 
                         YY_full**10], C).reshape(X.shape)
    # plot points and fitted surface
    if Flag_plot_result: 
        fig = plt.figure(figsize=(9,6))
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(XX, YY, data, c='r', s=5)
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        # ax.axis('equal')
        ax.axis('tight')
        plt.show()
    return {'fit':Z, 'params': C}


def Azimuthal_Average_2D(array, bins=None, Find_center=True): 
    ''' >>> Instruction <<< 
        This function does azimuthal average on a 2D image. 
        Input: 
            array           2D array
            bins            Number of points in the 1D result. If None, using the maximum integer distance
            Find_center     If False, center is the middle pixel of the array
                            If True, center is the pixel with the maxmum absolute value
        Output: 
            azi_avg         1D array
            bin_edges       Axis of the 1D array
    '''
    # Input normalization
    array = np.asarray(array)
    dims = np.shape(array)
    if np.size(dims) != 2: 
        print(">>>>>> 'array' should be a 2D array ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    
    # Find the center pixel
    if Find_center: 
        idx = np.argmax(np.abs(array))
        idx = np.unravel_index(idx, dims)
    else: 
        idx = tuple((np.divide(dims, 2)).astype('int'))
    
    # Calculate the distance array and get histogram
    X = np.linspace(0-idx[0], dims[0]-idx[0]-1, dims[0])
    Y = np.linspace(0-idx[1], dims[1]-idx[1]-1, dims[1])
    YY, XX = np.meshgrid(Y, X)
    dist = np.sqrt(XX**2 + YY**2)
    if bins is None: 
        bins = int(np.max(dist))
    hist, bin_edges = np.histogram(dist, bins=bins)
    
    # Galculate average of each bin
    azi_avg = np.zeros(bins)
    for i in range(bins): 
        if i != (bins - 1): 
            azi_avg[i] = np.average(array[np.where((dist >= bin_edges[i]) & (dist < bin_edges[i+1]))])
        else: 
            azi_avg[i] = np.average(array[np.where((dist >= bin_edges[i]) & (dist <= bin_edges[i+1]))])
    return {'azi_avg':azi_avg, 'bin_edges':bin_edges}


def Extend_1D_array(data, axis=None, new_axis=None, order=1, 
                    add_elm=None, addto_elm=None): 
    ''' >>> Introduction <<< 
        This function extends/resamples a 1D array, by polynomially fitting the original array. 
        
        Inputs: 
            data         The original 1D array. 
            axis         Original axis, for resampling the array. 
            new_axis     New axis, for resampling the array. 
            order        Order of polynomial fitting. 
            add_elm      Number of elements adding on BOTH ends. 
            addto_elm    Extend the original array to certain length. 
            
        Output is the new 1D array. 
    '''
    # Inputs regularization
    dims = np.shape(data)
    if np.size(dims) != 1: 
        print(">>>>>> 'data' should be a 1D array ! ")
        input('>>>>>> Press any key to quit ... ')
        return
    n = dims[0]
    if axis is None: 
        axis = np.arange(n) - int((n-1)/2)
    else: 
        if not np.size(axis) == n: 
            print(">>>>>> 'axis' size does NOT match 'data' ! ")
            input('>>>>>> Press any key to quit ... ')
            return
    
    # Obtain new_axis
    if new_axis is None: 
        if   (addto_elm is None) and (add_elm is not None): 
            m = int(n + 2 * add_elm)
            axis_full = np.arange(m) - int((m-1)/2)
        elif (add_elm is None) and (addto_elm is not None): 
            m = addto_elm
            axis_full = np.arange(m) - int((m-1)/2)
        else: 
            print(">>>>>> 'add_elm' OR 'addto_elm' has to be None. ")
            input('>>>>>> Press any key to quit ... ')
            return
    else: 
        axis_full = new_axis
    dims = np.shape(axis_full)
    
    # Fitting
    if   order == 1: 
        # best-fit linear plane
        A = np.c_[np.ones(np.size(data)), 
                  axis]
        C,_,_,_ = sp.linalg.lstsq(A, data)    # coefficients
        # evaluate it on grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full], C).reshape(dims)
    elif order == 2: 
        # best-fit quadratic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2], C).reshape(dims)
    elif order == 3: 
        # best-fit cubic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3], C).reshape(dims)
    elif order == 4: 
        # best-fit quartic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4], C).reshape(dims)
    elif order == 5: 
        # best-fit quintic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5], C).reshape(dims)
    elif order == 6: 
        # best-fit sextic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6], C).reshape(dims)
    elif order == 7: 
        # best-fit septic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6, 
                  axis**7]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6, 
                         axis_full**7], C).reshape(dims)
    elif order == 8: 
        # best-fit octic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6, 
                  axis**7, 
                  axis**8]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6, 
                         axis_full**7, 
                         axis_full**8], C).reshape(dims)
    elif order == 9: 
        # best-fit nonic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2,
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6, 
                  axis**7, 
                  axis**8, 
                  axis**9]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6, 
                         axis_full**7, 
                         axis_full**8, 
                         axis_full**9], C).reshape(dims)
    elif order == 10: 
        # best-fit decic curve
        A = np.c_[np.ones(np.size(data)), 
                  axis, 
                  axis**2, 
                  axis**3, 
                  axis**4, 
                  axis**5,
                  axis**6, 
                  axis**7, 
                  axis**8, 
                  axis**9, 
                  axis**10]
        C,_,_,_ = sp.linalg.lstsq(A, data)
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(axis_full.shape), 
                         axis_full, 
                         axis_full**2, 
                         axis_full**3, 
                         axis_full**4, 
                         axis_full**5, 
                         axis_full**6, 
                         axis_full**7, 
                         axis_full**8, 
                         axis_full**9, 
                         axis_full**10], C).reshape(dims)
    
    return Z





'''
def FT_2D(Data, stepsize=[1,1], mode='same'): 
    '' >>> Instruction <<< 
        This function performs Fourier Transoform on grid 2D data. 
        
        mode: 'same', return results with same size
              'power2'
        
        Plot results using matplotlib, e.g. :
            plt.imshow(result['amp'], 
                       extent=[np.amin(result['fq']['ax1']), 
                               np.amax(result['fq']['ax1']), 
                               np.amin(result['fq']['ax0']), 
                               np.amax(result['fq']['ax0'])] )
    ''
    Data = np.asarray(Data)
    N, M = np.shape(Data)
    
    Freq_max_N = 1 / stepsize[0]
    Freq_max_M = 1 / stepsize[1]
    N_fft = int(2**np.ceil(np.log2(N)))
    M_fft = int(2**np.ceil(np.log2(M)))
    
    Data_new = np.zeros((N_fft, M_fft))
    Data_new[int(np.ceil((N_fft-N)/2)):int(np.ceil((N_fft-N)/2)+N), 
             int(np.ceil((M_fft-M)/2)):int(np.ceil((M_fft-M)/2)+M)] = Data[:, :]
    
    FFT = np.fft.fftshift(np.fft.fft2(Data_new))/N/M
    Freq_N = Freq_max_N/2 * np.linspace(-1,1,int(N_fft+1))
    Freq_M = Freq_max_M/2 * np.linspace(-1,1,int(M_fft+1))
    
    if mode == 'same': 
        FFT_ss = FFT[int(np.ceil((N_fft-N)/2)):int(np.ceil((N_fft-N)/2)+N), 
                     int(np.ceil((M_fft-M)/2)):int(np.ceil((M_fft-M)/2)+M)]
        Ampl = np.abs(FFT_ss)
        Phas = np.angle(FFT_ss)
        Freq = {'ax0': Freq_N[int(np.ceil((N_fft-N)/2)):int(np.ceil((N_fft-N)/2)+N)], 
                'ax1': Freq_M[int(np.ceil((M_fft-M)/2)):int(np.ceil((M_fft-M)/2)+M)]}
    else: 
        Ampl = np.abs(FFT)
        Phas = np.angle(FFT)
        Freq = {'ax0': Freq_N[1:], 'ax1': Freq_M[1:]}
    
    return {'amp':Ampl, 'ph':Phas, 'fq':Freq}





'''
