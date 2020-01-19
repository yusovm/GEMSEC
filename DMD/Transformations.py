import pywt
import numpy as np
import pandas as pd
from numpy import fft
import scipy.signal as signal

class Transformations:
    def __init__(self, angle_data):
        self.data = angle_data.dropna(axis = 0)
        self.hr = None
        self.hrR = None
        self.hrI = None
        self.hc = None
        self.hcR = None
        self.hcI = None
        self.h = None
        self.hR = None
        self.hI = None
        self.fft_real = None
        self.fft_imag = None
        self.wave_A = None
        self.wave_D = None
    
    #Perform a row-wise hilbert transform of the data
    #Parameters:
    # -imag: boolean; determines if only imaginary values of the transform will be returned
    # -real: boolean; determines if only real values of the transform will be returned
    # Caution: Only ONE of the above (real or imag) can be True, otherwise an error will return
    #Returns:
    # -hil: DataFrame; appropriate values from the row-wise hilbert transform of the data
    def hilb_transform_row(self, imag = False, real = False):
        hil=[]
        hilcols= []
        
        if real and imag:
            raise Exception("Please select only EITHER real or imaginary values to display.")
        elif imag:
            for i in range(len(self.data)):
                hil.append(np.imag(signal.hilbert(self.data.iloc[i,:])))
            for c in self.data.columns:
                hilcols.append(c+'_imag')
        elif real:
            for i in range(len(self.data)):
                hil.append(np.real(signal.hilbert(self.data.iloc[i,:])))
            for c in self.data.columns:
                hilcols.append(c+'_real')
        else:
            for i in range(len(self.data)):
                hil.append(np.abs(signal.hilbert(self.data.iloc[i,:])))
            for c in self.data.columns:
                hilcols.append(c+'_abs')
        
        hil=pd.DataFrame(hil,index=self.data.index,columns=self.data.columns)
        hil.columns = hilcols

        if imag:
            self.hrI = hil
        elif real:
            self.hrR = hil
        else:
            self.hr = hil

        return hil
    
    #Perform a column-wise hilbert transform of the data
    #Parameters:
    # -imag: boolean; determines if only imaginary values of the transform will be returned
    # -real: boolean; determines if only real values of the transform will be returned
    # Caution: Only ONE of the above (real or imag) can be True, otherwise an error will return
    #Returns:
    # -hil: DataFrame; appropriate values from the column-wise hilbert transform of the data
    def hilb_transform_col(self, imag = False, real = False):
        hil=pd.DataFrame(index=self.data.index,columns=self.data.columns)
        hilcols = []
        
        if real and imag:
            raise Exception("Please select only EITHER real or imaginary values to display.")
        elif imag:
            for col in self.data.columns:
                hil[col] = np.imag(signal.hilbert(self.data[col].values))
            for c in self.data.columns:
                hilcols.append(c+'_imag')
        elif real:
            for col in self.data.columns:
                hil[col] = np.real(signal.hilbert(self.data[col].values))
            for c in self.data.columns:
                hilcols.append(c+'_real')
        else:
            for col in self.data.columns:
                hil[col] = np.abs(signal.hilbert(self.data[col].values))
            for c in self.data.columns:
                hilcols.append(c+'_abs')

        hil.columns = hilcols

        if imag:
            self.hcI = hil
        elif real:
            self.hcR = hil
        else:
            self.hc = hil
            
        return hil
    
    #Perform row-wise and column-wise hilbert transforms of the data, which are then concatenated together
    #Parameters:
    # -imag: boolean; determines if only imaginary values of the transform will be returned
    # -real: boolean; determines if only real values of the transform will be returned
    # Caution: Only ONE of the above (real or imag) can be True, otherwise an error will return
    #Returns:
    # -hil: DataFrame; appropriate values from the concatenated row-wise and
    #                  column-wise hilbert transforms of the data
    def hilb_transform(self, imag = False, real = False):
        if real and imag:
            raise Exception("Please select only EITHER real or imaginary values to display.")
        elif imag:
            if self.hrI is None:
                htrow = self.hilb_transform_row(imag = True)
            else:
                htrow = self.hrI
            if self.hcI is None:
                htcol = self.hilb_transform_col(imag = True)
            else:
                htcol = self.hcI
        elif real:
            if self.hrR is None:
                htrow = self.hilb_transform_row(real = True)
            else:
                htrow = self.hrR
            if self.hcR is None:
                htcol = self.hilb_transform_col(real = True)
            else:
                htcol = self.hcR
        else:
            if self.hr is None:
                htrow = self.hilb_transform_row()
            else:
                htrow = self.hr
            if self.hc is None:
                htcol = self.hilb_transform_col()
            else:
                htcol = self.hc
        
        hil=pd.concat([htrow, htcol], axis = 1)
        
        if imag:
            self.hI = hil
        elif real:
            self.hR = hil
        else:
            self.h = hil
            
        return hil
    
    #Performs a Fast Fourier Transform on the data
    #Returns:
    # -ft_df_real: DataFrame; real values from the FFT of the data
    # -ft_df_imag: DataFrame; imaginary values from the FFT of the data
    def fourier_transform(self):
        ft_df_real = pd.DataFrame(np.real(fft.fft(self.data)))
        ft_df_imag = pd.DataFrame(np.imag(fft.fft(self.data)))
        
        self.fft_real = ft_df_real
        self.fft_imag = ft_df_imag
        
        return ft_df_real, ft_df_imag
    
    #Performs a Discrete Wavelet Transform (Haar) on the data
    #Returns:
    # -wave_A: array; approximation values from the transform
    # -wave_D: array; detail coefficients for the approximation values
    def wavelet_transform(self):
        wave_A, wave_D = pywt.dwt(self.data, 'haar')
        
        self.wave_A = wave_A
        self.wave_D = wave_D
        
        return wave_A, wave_D
    
    #Performs all above transformations in all variations possible
    #Returns:
    # -T_list: list containing DataFrames/arrays; list of all transformation results in
    #          the following order -- Row-Wise Hilbert, Real Row-Wise Hilbert, Imaginary
    #          Row-Wise Hilbert, Column-Wise Hilbert, Real Column-Wise Hilbert, Imaginary
    #          Column-Wise Hilbert, Concatenated Hilbert, Real Concatenated Hilbert,
    #          Imaginary Concatenated Hilbert, Fourier, Wavelet
    def All_Transforms(self):
        #Row-Wise Hilbert Transforms
        hr = self.hilb_transform_row()
        self.hr = hr
        hrR = self.hilb_transform_row(real = True)
        self.hrR = hrR
        hrI = self.hilb_transform_row(imag = True)
        self.hrI = hrI
        #Column-Wise Hilbert Transforms
        hc = self.hilb_transform_col()
        self.hc = hc
        hcR = self.hilb_transform_col(real = True)
        self.hcR = hcR
        hcI = self.hilb_transform_col(imag = True)
        self.hcI = hcI
        #Concatenated Row/Column-Wise Hilbert Transforms
        h = pd.concat([hr, hc], axis = 1)
        self.h = h
        hR = pd.concat([hrR, hcR], axis = 1)
        self.hR = hR
        hI = pd.concat([hrI, hcI], axis = 1)
        self.hI = hI
        #Fast Fourier Transform
        f = self.fourier_transform()
        self.fft_real = f[0]
        self.fft_imag = f[1]
        #Haar Discrete Wavelet Transform
        w = self.wavelet_transform()
        self.wave_A = w[0]
        self.wave_D = w[1]
        
        T_list = [hr, hrR, hrI, hc, hcR, hcI, h, hR, hI, f, w]
        
        return T_list

#Create a Transformation Object, passing the data as a parameter
#transform_obj = Transformations("data") #Replace "data" with an actual DataFrame object (i.e. angles)
#Perform each desired transformation by calling the appropriate function
#wave = transform_obj.wavelet_transform()
#hil = transform_obj.hilb_transform() #Can pass additional parameters to return only real OR imaginary values
#hil_row = transform_obj.hilb_transform_row() #Can pass additional parameters to return only real OR imaginary values
#hil_col = transform_obj.hilb_transform_col() #Can pass additional parameters to return only real OR imaginary values

#If wanted, can run ALL transforms (with all real/imaginary versions) by just running All_Transforms()
#all_list = transform_obj.All_Transforms()