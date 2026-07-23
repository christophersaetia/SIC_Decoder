# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 12:58:34 2026

@author: saceh
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal, constants
import copy


'''
Single Channel
'''
class Single_Channel_Recorded_Samples_Original:
    def __init__(self, file_name_I_channel, plot_flag = False):
        
        '''
        Read in CSV file into a raw dataframe
        '''
        df_I = pd.read_csv(file_name_I_channel, skiprows = [0,1], dtype=np.float64)

        df_I = df_I.iloc[1:].reset_index(drop = True)
        df_I.columns = ['Time', 'CH1']

        self.ch1_volts = np.array([float(val) for val in df_I['CH1']])
        self.ch1_time_secs = np.array([float(val) for val in df_I['Time']])
        
        if plot_flag == True:
            fig, ax = plt.subplots(layout = 'constrained')
            ax.plot(self.ch1_time_secs, self.ch1_volts, label = 'CH1')
            ax.grid()
            ax.set_xlabel('Time [sec]')
            ax.set_ylabel('Amplitude [volts]')
            ax.set_title('Original Recording', fontweight = 'bold')
            ax.legend()
            plt.show()
            
            # N = len(self.ch1_volts)
            # pilot_fft = 2.0/N * np.abs(fft(self.ch1_volts))
            # freqs = fftfreq(N, self.ch1_time_secs[1]-self.ch1_time_secs[0])
            # fig, ax  = plt.subplots(layout = 'constrained')
            # ax.plot(freqs, pilot_fft)
            # ax.set_xlim([0,1e6])
            
        self.fs = 1 / (self.ch1_time_secs[1] - self.ch1_time_secs[0])
        self.fs_MHz = self.fs/1e6


'''
Filtered Channel - Filtered values
'''
class Filtered_Values:
    def __init__(self, input_vals, fs, f_lp, f_hp = 0, order = 5, sps = 1e6, plot_flag = False, speed_plot_flag = False):
        order = 5
        
        if f_hp != 0:
            sos_bpf = signal.bessel(order, [f_hp, f_lp], btype = 'bandpass', 
                                fs = fs,
                                output='sos')
    
    
            self.vals_filtered = signal.sosfiltfilt(sos_bpf, input_vals)
        elif f_hp == 0:
            sos_lpf = signal.bessel(order, f_lp, btype = 'lowpass', 
                                fs = fs,
                                output='sos')
    
            self.vals_filtered = signal.sosfiltfilt(sos_lpf, input_vals)
            
            
            
        if plot_flag == True:
            fig, ax = plt.subplots(1, 2, layout='constrained', figsize = (10,4))
            ax[0].plot(self.vals_filtered)
            ax[0].grid()
            ax[0].set_xlabel('Sample #')
            ax[0].set_ylabel('Voltage [volts]')
            ax[0].set_title('Filtered Output', fontweight = 'bold')
            

            ax[1].plot(self.vals_filtered[int(1e6) : int((1e6 + 70*sps))])
            ax[1].grid()
            ax[1].set_xlabel('Sample #')
            ax[1].set_ylabel('Voltage [volts]')
            ax[1].set_title('Filtered Output [Zoomed]', fontweight = 'bold')

        if speed_plot_flag == True:
            fig, ax = plt.subplots(1, 2, layout='constrained', figsize = (10,4))
            ax[0].plot(self.vals_filtered[0:int(len(self.vals_filtered)*0.5)])
            ax[0].grid()
            ax[0].set_xlabel('Sample #')
            ax[0].set_ylabel('Voltage [volts]')
            ax[0].set_title('Truncated Filtered Output (for speed)', fontweight = 'bold')
            
            ax[1].plot(self.vals_filtered[int(1e6) : int((1e6 + 70*sps))])
            ax[1].grid()
            ax[1].set_xlabel('Sample #')
            ax[1].set_ylabel('Voltage [volts]')
            ax[1].set_title('Filtered Output [Zoomed]', fontweight = 'bold')
          
'''
////// Baseline Adjustment
'''
class Baseline_Adjustment():
    def __init__(self, input_vals):
        
        amp_median_level = np.median(input_vals)
        adjustment_amount = np.abs(0 - amp_median_level)
        
        self.baseline_adjusted_seq = input_vals - adjustment_amount
        
        fig,ax = plt.subplots(layout='constrained')
        ax.plot(input_vals[0:int(len(input_vals)*0.5)], label = 'original')
        ax.plot(self.baseline_adjusted_seq[0: int(len(self.baseline_adjusted_seq) * 0.5)], label = 'Adjusted')
        ax.legend(loc = 'upper right')
        ax.grid()
        ax.set_title('Baseline Adjusted Waveform [truncated for speed]')
          
'''
////// Match filter and align initial samples
'''
class Framed_Data_Matched_Filtering:
    def __init__(self, input_vals, input_time, x_truth_seq, fs, mod_rate, sps = None, template_comparision = None, plot_flag = False, end_frame_sample_idx = int(20e3), corr_sens = 4):
        
        '''
        Calculate the sps and repeat the x_truth_seq symbols accordingly
        - If SPS and template comparision waveform isn't provided, create them for OOK scheme
        '''
        
        #SPS generation
        if sps != None:
            self.sps = sps
        else:
            self.sps = int(fs/mod_rate) #Default
        
        #Template comparision generation
        if template_comparision.any() != None:
            self.x_truth_waveform = template_comparision
        else:
            self.x_truth_waveform = np.repeat(x_truth_seq, self.sps) #Default
        
        #Input vals and times
        self.input_vals = input_vals[0*self.sps : ]
        self.input_time = input_time[0*self.sps : ]
        
        '''
        Align to the start of a message pilot frame
        '''
        #For aligning received samples to the start
        #End_frame_sample_idx ensures efficient/faster search that won't discard much values
        test_frame_input_vals = input_vals[0*self.sps: int(end_frame_sample_idx)]
        self.pilot_seq_corr = signal.correlate(test_frame_input_vals, self.x_truth_waveform, mode = 'same') /self.sps
        self.pilot_seq_lags = signal.correlation_lags(len(test_frame_input_vals), len(self.x_truth_waveform), mode="same")
        zero_val_index_lags = int(np.where(self.pilot_seq_lags == 0)[0])
        self.pilot_seq_lag_samples_amount = self.pilot_seq_lags[zero_val_index_lags:][np.argmax(self.pilot_seq_corr[zero_val_index_lags:])]
        

 
        #For figuring out symbol values and smoothing
        div_factor = corr_sens
        self.corr = signal.correlate(self.input_vals, np.ones(int(self.sps/div_factor)), mode = 'full') / int(self.sps/div_factor)
        
        
        self.aligned_input_vals = self.input_vals[self.pilot_seq_lag_samples_amount:] #ligned original input
        self.aligned_input_vals_matched_filtered = self.corr[self.pilot_seq_lag_samples_amount:] #Matched filter out
        self.aligned_input_time = self.input_time[self.pilot_seq_lag_samples_amount:]
        
        self.top_peaks_indices_framed,_ = find_peaks(self.aligned_input_vals_matched_filtered, distance=self.sps*0.5)
        self.top_peaks_indices_framed = self.top_peaks_indices_framed[np.where(self.aligned_input_vals_matched_filtered[self.top_peaks_indices_framed] > 0)]
        self.bottom_peaks_indices_framed,_ = find_peaks(-1*self.aligned_input_vals_matched_filtered, distance=self.sps*0.5)
        self.bottom_peaks_indices_framed = self.bottom_peaks_indices_framed[np.where(self.aligned_input_vals_matched_filtered[self.bottom_peaks_indices_framed] <= 0)]
        
        self.combined_peaks = np.sort(np.concatenate((self.top_peaks_indices_framed, self.bottom_peaks_indices_framed)))
        

        self.transition_edges_indices_framed = generate_clocked_transitions(input_vals = self.aligned_input_vals_matched_filtered, 
                                                                       input_time = input_time, 
                                                                       mod_rate = mod_rate, 
                                                                       fs = fs)
        
        #Peak reconstruction that only utilizes one peak per symbol frame
        peak_symbol_vals = []
        for i, edge_start_index in enumerate(self.transition_edges_indices_framed):
            low_bound = edge_start_index
            high_bound = edge_start_index + self.sps
            
            peaks_indices_in_range = self.combined_peaks[np.where((self.combined_peaks >= low_bound) & (self.combined_peaks <= high_bound))[0]]


            if len(peaks_indices_in_range) != 0 :
                peak_symbol_vals.append(np.mean(self.aligned_input_vals_matched_filtered[peaks_indices_in_range]))
            elif len(peaks_indices_in_range) == 0:
                peak_symbol_vals.append(np.mean(self.aligned_input_vals_matched_filtered[low_bound:high_bound]))
        
        self.peak_symbol_vals = np.array(peak_symbol_vals)
        
        
                
        if plot_flag == True:
            start_symbol_index = 0
            stop_symbol_index = len(x_truth_seq)


            
            fig, (ax_truth, ax_unaligned, ax_aligned, ax_reconstructed) = plt.subplots(4,1, layout = 'constrained', sharex = True)
            ax_truth.plot(self.x_truth_waveform[start_symbol_index*self.sps : stop_symbol_index*self.sps])
            ax_truth.grid()
            ax_truth.set_title('Truth Sequence')
            
            
            ax_unaligned.plot(self.input_vals[start_symbol_index*self.sps : stop_symbol_index*self.sps], label = 'Original')
            ax_unaligned.plot(self.corr[start_symbol_index*self.sps : stop_symbol_index*self.sps], label = 'Correlated Out.')
            ax_unaligned.grid()
            ax_unaligned.legend()
            ax_unaligned.set_title('Unaligned Samples')
            
            
            top_peaks_markers_selected_indices = self.top_peaks_indices_framed[np.where((self.top_peaks_indices_framed >= start_symbol_index*self.sps) & (self.top_peaks_indices_framed <= stop_symbol_index*self.sps))[0]]
            bottom_peaks_markers_selected_indices = self.bottom_peaks_indices_framed[np.where((self.bottom_peaks_indices_framed >= start_symbol_index*self.sps) & (self.bottom_peaks_indices_framed <= stop_symbol_index*self.sps))[0]]
            
            ax_aligned.plot(self.input_vals[self.pilot_seq_lag_samples_amount : stop_symbol_index*self.sps], label = 'Original')
            ax_aligned.plot(self.aligned_input_vals_matched_filtered[0 : stop_symbol_index*self.sps], label = 'Matched Filtered Out.')
            ax_aligned.scatter(top_peaks_markers_selected_indices, self.aligned_input_vals_matched_filtered[top_peaks_markers_selected_indices], color = 'green', label = 'Top peaks')
            ax_aligned.scatter(bottom_peaks_markers_selected_indices, self.aligned_input_vals_matched_filtered[bottom_peaks_markers_selected_indices], color = 'red', label = 'Bottom peaks')
            # ax_aligned.scatter(np.arange(0, len(self.peak_symbol_vals)*self.sps, self.sps) + self.sps/2, self.peak_symbol_vals, color = 'brown')
            # ax_aligned.plot(np.repeat(self.peak_symbol_vals, self.sps), color = 'fuchsia', linestyle = '--', label = 'Reconstructed')
            for index in self.transition_edges_indices_framed[start_symbol_index : stop_symbol_index]:
                ax_aligned.axvline(index, linestyle = '--', color = 'pink')
            ax_aligned.set_xlim([0, stop_symbol_index*self.sps])
            ax_aligned.legend()
            ax_aligned.grid()
            ax_aligned.set_title('Aligned and Framed Matched Output')
            
            ax_reconstructed.plot(self.input_vals[self.pilot_seq_lag_samples_amount : stop_symbol_index*self.sps], label = 'Original Received')
            # ax_reconstructed.plot(np.repeat(self.peak_symbol_vals, self.sps), color = 'fuchsia', linestyle = '--', label = 'Reconstructed')
            ax_reconstructed.grid()
            plt.suptitle('Matched Filter Framing Process')


'''
////// LSE Estimation for Single User
'''
class Estimation_Single_User:
    def __init__(self, y_vals, x_truth_vals, skip_2_start_index = 0, plot_flag = False):
        
        self.y_original_vals = y_vals
        self.x_original_truth_vals = x_truth_vals
        self.y_vals = y_vals[skip_2_start_index:]
        

        n = len(self.y_vals)
        repeat_int_amount = int(n/len(x_truth_vals))
        
        self.x_truth_seq = np.tile(x_truth_vals, repeat_int_amount)
        leftover = x_truth_vals[0: int(n % len(x_truth_vals))]
        self.x_truth_seq = np.concatenate((self.x_truth_seq, leftover))

        #LST sq example: https://numpy.org/doc/2.2/reference/generated/numpy.linalg.lstsq.html
        self.A = np.vstack([self.x_truth_seq, np.ones(len(self.x_truth_seq))]).T
        self.m, self.c = np.linalg.lstsq(self.A, self.y_vals, rcond=None)[0]
        
        
        if plot_flag == True:
            fig, ax = plt.subplots(layout = 'constrained')
            ax.plot(self.y_vals, label = 'Measured Received Val. [volts]')
            ax.plot(self.x_truth_seq, label = 'Expected Refl. Coeff.')
            ax.grid()
            
            
            fig, ax = plt.subplots(layout = 'constrained')
            ax.scatter(self.x_truth_seq, self.y_vals, label = 'Original data')
            ax.plot(self.x_truth_seq, self.m*self.x_truth_seq + self.c, label = 'Fitted line')
            ax.legend()
            ax.grid()
            ax.set_xlabel('Expected Refl. Coeff')
            ax.set_ylabel('Measured Received Value [volts]')
        
        
        
        # h_hat = np.linalg.inv(np.transpose(self.x_truth_seq) * self.x_truth_seq)*np.transpose(self.x_truth_seqs)*self.y_vals
        # h_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_truth_vals), x_truth_vals)), np.transpose(x_truth_vals)), y_vals) 
        # self.solve_channel_coeffs_LSE = h_hat
        
'''
////// LSE Estimation for Single  - OOK
'''
class Estimation_Single_User_Differential:
    def __init__(self, y_vals, x_truth_vals, skip_2_start_index = 0, plot_flag = False):
        
        self.y_original_vals = y_vals #Peak value within each symbol
        self.x_original_truth_vals = x_truth_vals
        self.y_vals = y_vals[skip_2_start_index:]
        

        n = len(self.y_vals)
        repeat_int_amount = int(n/len(x_truth_vals))
        
        self.x_truth_seq = np.tile(x_truth_vals, repeat_int_amount)
        leftover = x_truth_vals[0: int(n % len(x_truth_vals))]
        self.x_truth_seq = np.concatenate((self.x_truth_seq, leftover))
        
        
        indices_2_eval = []
        for i, val in enumerate(np.abs(np.diff(self.x_truth_seq))):
            if val != 0:
                indices_2_eval.append(i)
                
        diff_vals = []
        for stored_index in indices_2_eval:
            diff_val_temp = np.abs(self.y_original_vals[stored_index + 1] - self.y_original_vals[stored_index])
            diff_vals.append(diff_val_temp)
            
            
        self.m = np.mean(diff_vals)

'''
////// SIC pilots only checker
'''        
class SIC_Pilots_Checker:
    def __init__(self, SI_input_vals, sps, num_users, thresh_y_vals_arr = None, template_comparision = None, u0_guess_attempts_arr = [0,0], start_align_idx_arr = [0,0], samples_search_amount = int(20e3), detect_bound_index_shift = 0, plot_flag = True, speed_plot_flag = False):
        
        
        self.SI_input_vals = SI_input_vals
        self.sps = sps
        self.pilot_length = len(template_comparision)
        
        #Grab the pilot sequences and align them
        users_pilot_seqs_aligned_arr = []
        
        for i in range(num_users):
            matching_template = template_comparision
            
            #Need end_stop_align_idx to not get rid of too many samples when trying to align for the preamble
            seq_corr, seq_lags, seq_zero_val_index_lags, seq_lag_samples_amount = apply_match_filter(input_vals = copy.deepcopy(self.SI_input_vals)[start_align_idx_arr[i] : start_align_idx_arr[i] + samples_search_amount], template_comparision = matching_template, sps = self.sps)

            users_pilot_seqs_aligned_arr.append(copy.deepcopy(self.SI_input_vals)[ start_align_idx_arr[i] + seq_lag_samples_amount : start_align_idx_arr[i] + seq_lag_samples_amount + self.pilot_length])
                
        self.users_pilot_seqs_aligned_arr = np.array(users_pilot_seqs_aligned_arr)
        
        #Create a deep copy to prevent the original input vals from being modified
        current_input_vals = copy.deepcopy(self.SI_input_vals)
        

        def decode(self, current_input_vals, thresh = None):
            #Set threshhold
            if thresh == None:
                thresh = np.average(current_input_vals) + 0.08*np.abs(np.max(current_input_vals) - np.min(current_input_vals))

            #Assuming the current_input_vals is framed/aligned correctly, draw symbol detection marker boundaries in time
            detection_bounds_idxs = np.arange(0, len(current_input_vals), self.sps)
            
            #Find the indices of the 'peaks' in the data
            #Using these to decode symbols' values and potential
            top_peaks_idxs, bottom_peaks_idxs, combined_peaks_idxs = find_peaks_indices(current_input_vals)
            
            num_detectable_symbs = int(detection_bounds_idxs[-1]/self.sps)
            
            symbs_decoded = []
            #Detect symbol values
            for i, edge_idx in enumerate(detection_bounds_idxs):
    
                if i < num_detectable_symbs:
                    low_bound_edge_idx = edge_idx
                    upper_bound_edge_idx = edge_idx + self.sps
     
                    '''
                    Using peaks for thresh. comparision
                    '''
                    detect_idxs_temp = combined_peaks_idxs[np.where((combined_peaks_idxs >= low_bound_edge_idx) & (combined_peaks_idxs < upper_bound_edge_idx))[0]]
                

                    if len(detect_idxs_temp) == 0:#Need to check
                        symbs_decoded.append(1)
                    elif (any(val > thresh for val in current_input_vals[detect_idxs_temp])):
                        symbs_decoded.append(1)
                    else:
                        symbs_decoded.append(0)
                        
            return thresh, np.array(symbs_decoded), combined_peaks_idxs, detection_bounds_idxs
        
        
        '''
        '''
        #NOTE THIS IS HARD CODED TO ONLY CHECK FOR U1 SINCE THERE'S ONLY TWO TAGS. NEED TO MODIFY FOR IF THERE'S MORE
        original_seq = copy.deepcopy(self.users_pilot_seqs_aligned_arr[1])
        u1_plots = [original_seq]
        u1_new_peaks_plots = []
        
        for guess_attempt in u0_guess_attempts_arr:
            

            top_peaks_indices, bottom_peaks_indices, combined_peaks_idxs = find_peaks_indices(original_seq)
            
            #For subtraction
            new_seq = copy.deepcopy(original_seq)
            


            
            
            #Create detection boundaries
            u1_detection_bounds_idxs = np.arange(0, len(original_seq), self.sps) -  detect_bound_index_shift
            
            #Can't have negative detection index since it wraps around then
            if(u1_detection_bounds_idxs[0] < 0):
                u1_detection_bounds_idxs[0] = 0
            
            num_detectable_symbs = int(u1_detection_bounds_idxs[-1]/self.sps)
            
            
            for i, edge_idx in enumerate(u1_detection_bounds_idxs):
    
                if i < num_detectable_symbs:
                    low_bound_edge_idx = edge_idx
                    upper_bound_edge_idx = edge_idx + self.sps
     
                    '''
                    Using peaks for thresh. comparision
                    '''
                    peaks_detected_idxs_temp = combined_peaks_idxs[np.where((combined_peaks_idxs >= low_bound_edge_idx) & (combined_peaks_idxs < upper_bound_edge_idx))[0]]
                    
                    #EDGE CLEANUP
                    #From left detection edge to next peak
                    if original_seq[peaks_detected_idxs_temp[0]] >= thresh_y_vals_arr[0]:
                        new_seq[low_bound_edge_idx : peaks_detected_idxs_temp[0]] = copy.deepcopy(original_seq)[peaks_detected_idxs_temp[0]] - np.abs(guess_attempt)
                    else:
                        new_seq[low_bound_edge_idx : peaks_detected_idxs_temp[0]] = copy.deepcopy(original_seq)[peaks_detected_idxs_temp[0]]
                    
                    #From right detection edge 
                    if original_seq[peaks_detected_idxs_temp[-1]] >= thresh_y_vals_arr[0]:
                        new_seq[peaks_detected_idxs_temp[-1] : upper_bound_edge_idx] = copy.deepcopy(original_seq)[peaks_detected_idxs_temp[-1]] - np.abs(guess_attempt)
                    else:
                        new_seq[peaks_detected_idxs_temp[-1] : upper_bound_edge_idx] = copy.deepcopy(original_seq)[peaks_detected_idxs_temp[-1]] 

                    #In between peaks
                    if len(peaks_detected_idxs_temp) > 1:
                        for j in range(0, len(peaks_detected_idxs_temp) - 1):
                            
                            if copy.deepcopy(original_seq)[peaks_detected_idxs_temp[j]] >= thresh_y_vals_arr[0]: 
                                new_seq[peaks_detected_idxs_temp[j] : peaks_detected_idxs_temp[j+1]] = copy.deepcopy(original_seq)[peaks_detected_idxs_temp[j] : peaks_detected_idxs_temp[j+1]] - np.abs(guess_attempt)

            


            top_peaks_indices_new_temp, bottom_peaks_indices_new_temp, combined_peaks_idxs_new_temp = find_peaks_indices(new_seq)
            u1_new_peaks_plots.append(combined_peaks_idxs_new_temp)
            
            
            #Add new sequence for plotting
            u1_plots.append(new_seq)
            
            
            
 
        '''
        Plotting
        '''
        
        fig, ax = plt.subplots(len(u1_plots), 1, layout = 'constrained', sharey = True)
        
        start_symbol_idx = 0
        stop_symbol_idx = 300*2
        
        for i, seq in enumerate(u1_plots):
            if i ==0:
                ax[i].plot(original_seq[start_symbol_idx*self.sps : stop_symbol_idx*self.sps], color = 'green')
                ax[i].axhline(y=thresh_y_vals_arr[0], linestyle = '--', color = 'red')
                ax[i].set_title('Original')  
            elif i > 0:
                ax[i].plot(original_seq[start_symbol_idx*self.sps : stop_symbol_idx*self.sps], linestyle = '--', color = 'lightgreen')
                ax[i].plot(seq[start_symbol_idx*self.sps : stop_symbol_idx*self.sps], linewidth = 3)
                
                for peak_idx in combined_peaks_idxs[start_symbol_idx * self.sps : stop_symbol_idx * self.sps]:
                    ax[i].scatter(peak_idx, original_seq[peak_idx], s = 30, color = 'green')
                
                for peak_idx in u1_new_peaks_plots[i-1][start_symbol_idx * self.sps : stop_symbol_idx * self.sps]:
                    ax[i].scatter(peak_idx, seq[peak_idx], s = 30, color = 'orange')
                
                
                
                ax[i].set_title('Guess ' +str(i))
            
            for detection_thresh_val in u1_detection_bounds_idxs:
                ax[i].axvline(x = detection_thresh_val, linestyle = '--', color = 'pink')
         
            
'''
////// SIC Decoder with known channel coeffs
'''    
class SIC_Decoder:
    def __init__(self, SI_input_vals, sps, num_users, users_channel_coeffs_arr, 
                 thresh_y_vals_arr = None, template_comparision = None, start_align_idx_arr = [0,0], 
                 samples_search_amount = int(5e3), detect_bound_index_shift_arr = [0,0], plot_flag = True, speed_plot_flag = False, 
                 median_subtract_flag = False):
        
        num_iter = num_users
        
        self.SI_input_vals = SI_input_vals
        self.sps = sps
        self.users_channel_coeffs_arr = np.array(users_channel_coeffs_arr)
        self.thresh_y_vals_arr = thresh_y_vals_arr
        self.detect_bound_index_shift_arr = detect_bound_index_shift_arr
        

        def decode(self, current_input_vals, thresh_val, detect_bound_index_shift):
            
            #Set threshhold
            if thresh_val == None:
                thresh = np.average(current_input_vals) + 0.08*np.abs(np.max(current_input_vals) - np.min(current_input_vals))
            else:
                thresh = thresh_val
                
            #Assuming the current_input_vals is framed/aligned correctly, draw symbol detection marker boundaries in time
            detection_bounds_idxs = np.arange(0, len(current_input_vals), self.sps) - detect_bound_index_shift
            
            if detection_bounds_idxs[0] < 0:
                detection_bounds_idxs[0] = 0
            
            
            #Find the indices of the 'peaks' in the data
            #Using these to decode symbols' values and potential
            top_peaks_idxs, bottom_peaks_idxs, combined_peaks_idxs = find_peaks_indices(current_input_vals)
            
            num_detectable_symbs = int(detection_bounds_idxs[-1]/self.sps)
            
            symbs_decoded = []
            #Detect symbol values
            for i, edge_idx in enumerate(detection_bounds_idxs):
    
                if i < num_detectable_symbs:
                    low_bound_edge_idx = edge_idx
                    upper_bound_edge_idx = edge_idx + self.sps
                
                    if np.median(current_input_vals[low_bound_edge_idx:upper_bound_edge_idx]) > thresh:
                        symbs_decoded.append(1)
                    else:
                        symbs_decoded.append(0)
                       

                        
            return thresh, np.array(symbs_decoded), combined_peaks_idxs, detection_bounds_idxs
        
        def estimate_iter_channel_state(detection_thresh, current_input_vals, combined_peaks_idxs):
            
            '''
            Method 1  - Constructive Interference
            '''
            # HIGH - 1
            # above_detection_thresh_vals = np.array([val for val in current_input_vals[combined_peaks_idxs] if val >= detection_thresh])
            above_detection_thresh_vals = np.array([val for val in current_input_vals[combined_peaks_idxs] if val >= detection_thresh])
            
            #Want to use the y values close to the threshold
            LSE_one_thresh = 0.1 * (np.max(above_detection_thresh_vals) - np.min(above_detection_thresh_vals)) + np.min(above_detection_thresh_vals)
            ones_y_vals = np.array([val for val in above_detection_thresh_vals if val <= LSE_one_thresh])
            ones_x_vals = np.ones(len(ones_y_vals))
            
            # LOW - 0
            below_detection_thresh_vals = np.array([val for val in current_input_vals[combined_peaks_idxs] if val < detection_thresh])
            LSE_zero_thresh = 0.05 * (np.max(below_detection_thresh_vals) - np.min(below_detection_thresh_vals)) + np.min(below_detection_thresh_vals)
            zeros_y_vals = np.array([val for val in below_detection_thresh_vals if val <= LSE_zero_thresh])            
            zeros_x_vals = np.zeros(len(zeros_y_vals))
            
            
            y_vals = np.concatenate((ones_y_vals, zeros_y_vals))
            x_vals = np.concatenate((ones_x_vals, zeros_x_vals))
            
            m,c = apply_LSE(y_seq = y_vals, x_seq = x_vals)   
            
            return m

        def estimate_channel_marker_peak():
            #CHANGE BASED ON LOCATION OF MARKER 1
            start_idx = detection_bounds_idxs[36]
            end_idx = detection_bounds_idxs[37] 
            
            target_combined_peaks_idxs = combined_peaks_idxs[(combined_peaks_idxs >= start_idx) & (combined_peaks_idxs <= end_idx)]
            target_peaks_vals = current_input_vals[target_combined_peaks_idxs]
            # print(target_peaks_vals)
            # combined_peaks_idxs[np.where((combined_peaks_idxs >= start_idx) & (combined_peaks_idxs <= end_idx))[0][0]-1]
            
            if len(target_peaks_vals) > 0:
                left_peak = target_peaks_vals[0]
                right_peak = target_peaks_vals[-1]
            else:
                left_peak = 0
                right_peak = 0
            
            if iter_attempt == 2:
                left_edge_amount = 0
                right_edge_amount = 0
            else:
                left_edge_amount = np.abs(left_peak - current_input_vals[combined_peaks_idxs[np.where((combined_peaks_idxs >= start_idx) & (combined_peaks_idxs <= end_idx))[0][0]-1]])
                right_edge_amount = np.abs(right_peak - current_input_vals[combined_peaks_idxs[np.where((combined_peaks_idxs >= start_idx) & (combined_peaks_idxs <= end_idx))[0][-1]+1]])
            
            # print(left_peak)
            # print(current_input_vals[combined_peaks_idxs[np.where((combined_peaks_idxs >= start_idx) & (combined_peaks_idxs <= end_idx))[0][0]-1]])
            print('Peak Guess: ' + str(np.min([left_edge_amount, right_edge_amount])))
            print()
            a = np.min([left_edge_amount, right_edge_amount])
            
            return a
        
        
        
        '''
        SIC Process
        '''
        self.sequences_decoded_iterations = []
        self.users_decoded = []
        self.users_decoding_peaks_indices = []
        self.users_decoding_threshes = []
        self.detection_bounds_idxs_arr = []
        
        #Create a deep copy to prevent the original input vals from being modified
        current_input_vals = copy.deepcopy(self.SI_input_vals)
        
        #Order channel state coefs by largest magnitude
        self.users_channel_state_coeffs_ordered = self.users_channel_coeffs_arr[np.argsort(np.abs(self.users_channel_coeffs_arr))[::-1]]
        
        #Iterate
        for iter_attempt in range(num_iter):
            
            matching_template = template_comparision
            
            #Flip if channel coefficient is negative
            if self.users_channel_state_coeffs_ordered[iter_attempt] < 0 and iter_attempt == 0:
                current_input_vals = -1 * copy.deepcopy(current_input_vals)
                    
            if iter_attempt != 0:
                if np.sign(self.users_channel_state_coeffs_ordered[iter_attempt - 1]) != np.sign(self.users_channel_state_coeffs_ordered[iter_attempt]):
                    current_input_vals = -1 * copy.deepcopy(current_input_vals)
            
            #Step 1: Align
            #Need end_stop_align_idx to not get rid of too many samples when trying to align for the preamble
            
            if iter_attempt == 0:
                start_align_idx = start_align_idx_arr[iter_attempt]
                end_stop_align_idx = start_align_idx  + samples_search_amount
            else:
                start_align_idx = start_align_idx_arr[iter_attempt] - start_align_idx_arr[iter_attempt-1]
                end_stop_align_idx = start_align_idx  + samples_search_amount
            
            seq_corr, seq_lags, seq_zero_val_index_lags, seq_lag_samples_amount = apply_match_filter(input_vals = current_input_vals[start_align_idx : end_stop_align_idx], template_comparision = matching_template, sps = self.sps)
            current_input_vals = current_input_vals[start_align_idx + seq_lag_samples_amount : ]
            
                
            #Step 2: Decode
            thresh, symbs_decoded, combined_peaks_idxs, detection_bounds_idxs = decode(self, current_input_vals = current_input_vals, thresh_val = thresh_y_vals_arr[iter_attempt], detect_bound_index_shift=self.detect_bound_index_shift_arr[iter_attempt])
            
            self.users_decoded.append(symbs_decoded)
            self.users_decoding_peaks_indices.append(combined_peaks_idxs)
            self.users_decoding_threshes.append(thresh)
            self.detection_bounds_idxs_arr.append(detection_bounds_idxs)
            
            #For plotting the iterations
            self.sequences_decoded_iterations.append(current_input_vals)
            
            #Step 3: Channel state estimation for particular user
            # a = estimate_iter_channel_state(detection_thresh = thresh, current_input_vals = current_input_vals, combined_peaks_idxs = combined_peaks_idxs)

            '''
            '''
            new_seq = copy.deepcopy(current_input_vals)
            


            
            if (median_subtract_flag == False) and (self.users_channel_state_coeffs_ordered[iter_attempt] != None):
                a = self.users_channel_state_coeffs_ordered[iter_attempt]
            elif median_subtract_flag == True:
                a = np.median(current_input_vals)
                
            num_detectable_symbs = int(detection_bounds_idxs[-1]/self.sps)
            
            for i, edge_idx in enumerate(detection_bounds_idxs):

                if i < num_detectable_symbs and i != len(detection_bounds_idxs)-1:
                    low_bound_edge_idx = edge_idx
                    upper_bound_edge_idx = edge_idx + self.sps
     
                    '''
                    Using peaks for thresh. comparision
                    '''
                    peaks_detected_idxs_temp = combined_peaks_idxs[np.where((combined_peaks_idxs >= low_bound_edge_idx) & (combined_peaks_idxs < upper_bound_edge_idx))[0]]
                    
                    if (len(peaks_detected_idxs_temp) > 0):
                        #EDGE CLEANUP
                        #From left detection edge to next peak
                        if current_input_vals[peaks_detected_idxs_temp[0]] >= thresh:
                            new_seq[low_bound_edge_idx : peaks_detected_idxs_temp[0]] = current_input_vals[peaks_detected_idxs_temp[0]] - np.abs(a)
                        else:
                            new_seq[low_bound_edge_idx : peaks_detected_idxs_temp[0]] = current_input_vals[peaks_detected_idxs_temp[0]]
                        
                        if current_input_vals[peaks_detected_idxs_temp[-1]] >= thresh:
                            new_seq[peaks_detected_idxs_temp[-1] : upper_bound_edge_idx] = current_input_vals[peaks_detected_idxs_temp[-1]] - np.abs(a)
                        else:
                            new_seq[peaks_detected_idxs_temp[-1] : upper_bound_edge_idx] = current_input_vals[peaks_detected_idxs_temp[-1]] 

                    
                    if len(peaks_detected_idxs_temp) > 1:
                        for j in range(0, len(peaks_detected_idxs_temp) - 1):
                            
                            if current_input_vals[peaks_detected_idxs_temp[j]] >= thresh: 
                                new_seq[peaks_detected_idxs_temp[j] : peaks_detected_idxs_temp[j+1]] = current_input_vals[peaks_detected_idxs_temp[j] : peaks_detected_idxs_temp[j+1]] - np.abs(a)
            
            
            
            
            # if iter_attempt == 0:
            #     plt.figure()
            #     plt.plot(current_input_vals[0 : int(0.6e6)], label = 'current')
            #     plt.plot(new_seq[0 : int(0.6e6)], label = 'new seq')
                
            #     for idx in detection_bounds_idxs[0 : int(0.6e6)]:
            #         plt.axvline(x = idx, linestyle = '--', color = 'pink')
                    

            #     plt.title('After First Sutraction')
            #     plt.grid()
            #     plt.legend(loc = 'upper right')
            
            #Set the next iteration vals to the new seq that subtracted out the old user
            current_input_vals = copy.deepcopy(new_seq)
            
            #SAVE LAST LAST  for plotting
            if iter_attempt == num_iter-1:
                self.sequences_decoded_iterations.append(current_input_vals)
        
        
        '''
        Plotting Functions
        '''
        if plot_flag == True:
            start_symbol_idx = 0
            stop_symbol_idx = 200*2
            
            fig, ax = plt.subplots(layout='constrained')
            ax.plot(self.sequences_decoded_iterations[0], label = 'Iteration: 0')
            ax.plot(self.sequences_decoded_iterations[1], label = 'Iteration: 1')
            ax.plot(self.sequences_decoded_iterations[2], label = 'Iteration: Final')

            ax.set_xlim([start_symbol_idx * self.sps, stop_symbol_idx * self.sps])
            ax.set_title('SIC - Subtraction Comparision')
            ax.legend(loc = 'upper right')
            ax.grid()
            
            
            
            
            fig, ax = plt.subplots(num_iter + 1, 1, layout= 'constrained', sharey= True, sharex= True)
            
            for i in range(num_iter + 1):
                if speed_plot_flag == True: 
                    ax[i].plot(self.sequences_decoded_iterations[i][start_symbol_idx * self.sps : stop_symbol_idx * self.sps], label = 'Waveform')

                else:
                    ax[i].plot(self.sequences_decoded_iterations[i], label = 'Waveform')
                
                if i != num_iter:
                    ax[i].scatter(self.users_decoding_peaks_indices[i], self.sequences_decoded_iterations[i][self.users_decoding_peaks_indices[i]], color = 'green', label = 'Decoding Points')
                    ax[i].axhline(y = self.users_decoding_threshes[i], label = 'Decision Thresh.', color = 'red', linestyle = '--')
                    
                    for bound_idx in self.detection_bounds_idxs_arr[i][0:stop_symbol_idx]:
                        ax[i].axvline(x = bound_idx, linestyle = '--', color = 'red')
                    
                
                ax[i].set_xlim([start_symbol_idx * self.sps, stop_symbol_idx * self.sps])
                ax[i].grid()
                ax[i].legend(loc = 'upper right')
                if i != num_iter:
                    ax[i].set_title('Iteration: ' + str(i))
                else:
                    ax[i].set_title('Final Result')         

'''
////// SIC Decoder with known channel coeffs
'''    
class SIC_Decoder_v2:
    def __init__(self, SI_input_vals, sps, num_users, users_channel_coeffs_arr, 
                 thresh_y_vals_arr = None, template_comparision = None, 
                 start_align_idx_arr = [0,0], samples_search_amount = int(5e3), 
                 detect_bound_index_shift_arr = [0,0], plot_flag = True, 
                 speed_plot_flag = True, plot_title = None, median_subtract_flag = False):
        
        num_iter = num_users
        
        self.SI_input_vals = SI_input_vals
        self.sps = sps
        self.users_channel_coeffs_arr = np.array(users_channel_coeffs_arr)
        self.thresh_y_vals_arr = thresh_y_vals_arr
        self.detect_bound_index_shift_arr = detect_bound_index_shift_arr
        

        def decode(self, current_input_vals, thresh_val, detection_bounds_idxs):
            
            #Set threshhold
            if thresh_val == None:
                thresh = np.average(current_input_vals) + 0.08*np.abs(np.max(current_input_vals) - np.min(current_input_vals))
            else:
                thresh = thresh_val
                
            
            
            #Find the indices of the 'peaks' in the data
            #Using these to decode symbols' values and potential
            top_peaks_idxs, bottom_peaks_idxs, combined_peaks_idxs = find_peaks_indices(current_input_vals)
            
            num_detectable_symbs = int(detection_bounds_idxs[-1]/self.sps)
            
            symbs_decoded = []
            symbs_decoded_idxs = []
            
            #Detect symbol values
            for i, edge_idx in enumerate(detection_bounds_idxs):
    
                if i < num_detectable_symbs:
                    low_bound_edge_idx = edge_idx
                    upper_bound_edge_idx = edge_idx + self.sps
                    
                    midway_idx = int((low_bound_edge_idx + upper_bound_edge_idx)/2)
                    symbs_decoded_idxs.append(midway_idx)
                    
                    # if (current_input_vals[low_bound_edge_idx:upper_bound_edge_idx] > thresh).any():
                    #     symbs_decoded.append(1)
                    # else:
                    #     symbs_decoded.append(0)
                        
                    # if (current_input_vals[midway_idx]) > thresh:
                    #     symbs_decoded.append(1)
                    # else:
                    #     symbs_decoded.append(0)
                      
                    if np.median(current_input_vals[low_bound_edge_idx:upper_bound_edge_idx]) > thresh:
                        symbs_decoded.append(1)
                    else:
                        symbs_decoded.append(0)
                        

                        
            return thresh, np.array(symbs_decoded), combined_peaks_idxs, detection_bounds_idxs, np.array(symbs_decoded_idxs)
        

        '''
        SIC Process
        '''
        self.sequences_decoded_iterations = []
        self.users_decoded = []
        self.users_decoded_whole_seq = []
        self.users_decoding_peaks_indices = []
        self.users_decoding_threshes = []
        self.detection_bounds_idxs_arr = []
        self.global_frame_start_idxs_arr = []
        self.flip_true_flags = []
        self.sequences_symbs_decoded_idxs = []
        
        #Create a deep copy to prevent the original input vals from being modified
        current_input_vals = copy.deepcopy(self.SI_input_vals)
        
        #Order channel state coefs by largest magnitude
        self.users_channel_state_coeffs_ordered = self.users_channel_coeffs_arr[np.argsort(np.abs(self.users_channel_coeffs_arr))[::-1]]
        
        #Iterate
        for iter_attempt in range(num_iter):
            
            matching_template = template_comparision
            
            #Flip if channel coefficient is negative
            if self.users_channel_state_coeffs_ordered[iter_attempt] < 0 and iter_attempt == 0:
                current_input_vals = -1 * copy.deepcopy(current_input_vals)
                self.flip_true_flags.append(True)
                
            elif iter_attempt != 0:
                if np.sign(self.users_channel_state_coeffs_ordered[iter_attempt - 1]) != np.sign(self.users_channel_state_coeffs_ordered[iter_attempt]):
                    current_input_vals = -1 * copy.deepcopy(current_input_vals)
                    self.flip_true_flags.append(True)
                else:
                    self.flip_true_flags.append(False)
                    
            else:
               self.flip_true_flags.append(False)
               
               
            #Step 1: Align
            #Need end_stop_align_idx to not get rid of too many samples when trying to align for the preamble
            

            start_align_idx = start_align_idx_arr[iter_attempt]
            end_stop_align_idx = start_align_idx  + samples_search_amount            
            
            
            #To find the start
            seq_corr, seq_lags, seq_zero_val_index_lags, seq_lag_samples_amount = apply_match_filter(input_vals = current_input_vals[start_align_idx : end_stop_align_idx], template_comparision = matching_template, sps = self.sps)
            
            frame_start_idx = start_align_idx + seq_lag_samples_amount
            self.global_frame_start_idxs_arr.append(frame_start_idx)
        
            detection_bounds_idxs_right = np.arange(frame_start_idx, len(current_input_vals), self.sps)
            detection_bounds_idxs_left = np.arange(frame_start_idx, 0, -1*self.sps)
            detection_bounds_idxs = np.sort(np.unique(np.concatenate((detection_bounds_idxs_left, detection_bounds_idxs_right)))) - self.detect_bound_index_shift_arr[iter_attempt]
            
            if detection_bounds_idxs[0] < 0:
                detection_bounds_idxs[0] = 0
            
            #Step 2: Decode
            thresh, symbs_decoded, combined_peaks_idxs, detection_bounds_idxs, symbs_decoded_idxs = decode(self, current_input_vals = current_input_vals, thresh_val = thresh_y_vals_arr[iter_attempt], detection_bounds_idxs=detection_bounds_idxs)
            
            self.users_decoded_whole_seq.append(symbs_decoded)
            self.sequences_symbs_decoded_idxs.append(symbs_decoded_idxs)
            
            
            #For plotting the iterations
            self.sequences_decoded_iterations.append(current_input_vals)
            self.detection_bounds_idxs_arr.append(detection_bounds_idxs)
            self.users_decoding_threshes.append(thresh)
            self.users_decoding_peaks_indices.append(combined_peaks_idxs)
            
            #Only looking at the frame
            self.users_decoded.append(symbs_decoded[int(frame_start_idx/self.sps):])
            

            '''
            '''
            new_seq = copy.deepcopy(current_input_vals)

            #Substraction amount
            if median_subtract_flag == False and self.users_channel_state_coeffs_ordered[iter_attempt] != None:
                a = self.users_channel_state_coeffs_ordered[iter_attempt]
            elif median_subtract_flag == True:
                a = np.median(current_input_vals)
               
            
            #Calculate number of detectable symbols
            num_detectable_symbs = int(detection_bounds_idxs[-1]/self.sps)
            
            for i, edge_idx in enumerate(detection_bounds_idxs):

                if i < num_detectable_symbs and i != len(detection_bounds_idxs)-1:
                    low_bound_edge_idx = edge_idx
                    upper_bound_edge_idx = edge_idx + self.sps
     
                    '''
                    Using peaks for thresh. comparision
                    '''
                    peaks_detected_idxs_temp = combined_peaks_idxs[np.where((combined_peaks_idxs >= low_bound_edge_idx) & (combined_peaks_idxs < upper_bound_edge_idx))[0]]
                    
                    if (len(peaks_detected_idxs_temp) > 0):
                        #EDGE CLEANUP
                        #From left detection edge to next peak
                        if current_input_vals[peaks_detected_idxs_temp[0]] >= thresh:
                            new_seq[low_bound_edge_idx : peaks_detected_idxs_temp[0]] = current_input_vals[peaks_detected_idxs_temp[0]] - np.abs(a)
                        else:
                            new_seq[low_bound_edge_idx : peaks_detected_idxs_temp[0]] = current_input_vals[peaks_detected_idxs_temp[0]]
                        
                        if current_input_vals[peaks_detected_idxs_temp[-1]] >= thresh:
                            new_seq[peaks_detected_idxs_temp[-1] : upper_bound_edge_idx] = current_input_vals[peaks_detected_idxs_temp[-1]] - np.abs(a)
                        else:
                            new_seq[peaks_detected_idxs_temp[-1] : upper_bound_edge_idx] = current_input_vals[peaks_detected_idxs_temp[-1]] 

                    
                    if len(peaks_detected_idxs_temp) > 1:
                        for j in range(0, len(peaks_detected_idxs_temp) - 1):
                            
                            if current_input_vals[peaks_detected_idxs_temp[j]] >= thresh: 
                                new_seq[peaks_detected_idxs_temp[j] : peaks_detected_idxs_temp[j+1]] = current_input_vals[peaks_detected_idxs_temp[j] : peaks_detected_idxs_temp[j+1]] - np.abs(a)
            
            #Set the next iteration vals to the new seq that subtracted out the old user
            current_input_vals = copy.deepcopy(new_seq)
            
            #SAVE LAST LAST  for plotting
            if iter_attempt == num_iter-1:
                self.sequences_decoded_iterations.append(current_input_vals)
        
        
            #Decoding without SIC for comparision
            if iter_attempt == num_users - 1:
                thresh_no_SIC, symbs_decoded_no_SIC, combined_peaks_idxs_no_SIC, detection_bounds_idxs_no_SIC,symbs_decoded_idxs_no_SIC = decode(self, current_input_vals = copy.deepcopy(self.SI_input_vals), thresh_val = thresh_y_vals_arr[iter_attempt], detection_bounds_idxs=detection_bounds_idxs)
                
                self.symbs_decoded_no_SIC_framed = symbs_decoded_no_SIC[int(frame_start_idx/self.sps):]
        
        
        '''
        Plotting Functions
        '''
        if plot_flag == True:
 
 
            fontsize = 16
            tick_fontsize = 12
            linewidth = 3
            fig, ax = plt.subplots(num_iter, 1, layout= 'constrained', sharey= True, sharex= True)
            
            for i in range(num_iter):

                num_disp_symbs = 300
                start_plot_idx = int(self.global_frame_start_idxs_arr[i])
                end_plot_idx = int(self.global_frame_start_idxs_arr[i] + num_disp_symbs*self.sps)
                
                
                if i != num_iter:

                    for j, bound_idx in enumerate(self.detection_bounds_idxs_arr[i][(self.detection_bounds_idxs_arr[i] >= start_plot_idx) & (self.detection_bounds_idxs_arr[i] <= end_plot_idx)] - int(self.global_frame_start_idxs_arr[i])):
                        if j == len(self.detection_bounds_idxs_arr[i][(self.detection_bounds_idxs_arr[i] >= start_plot_idx) & (self.detection_bounds_idxs_arr[i] <= end_plot_idx)] - int(self.global_frame_start_idxs_arr[i]))-1:
                            ax[i].axvline(x = bound_idx, linestyle = '--', color = 'pink', label = 'Symbol Boundary')
                        else:
                            ax[i].axvline(x = bound_idx, linestyle = '--', color = 'pink')
                            # ax[i].text(bound_idx, 0, str(j))
                            
                    # ax[i].scatter(self.users_decoding_peaks_indices[i], self.sequences_decoded_iterations[i][self.users_decoding_peaks_indices[i]], color = 'green', label = 'Decoding Points')
                    ax[i].axhline(y = self.users_decoding_threshes[i], label = 'Decision Thresh.', color = 'red', linestyle = '--', linewidth = linewidth)                            
                            
                if i == 1:
                    if(self.flip_true_flags[i] == True):
                        ax[1].plot(-1*self.sequences_decoded_iterations[i-1][start_plot_idx : end_plot_idx], label = 'Original waveform - Before subtraction', color = 'lightgreen', linewidth = linewidth/2 )
                    else:
                        ax[1].plot(self.sequences_decoded_iterations[i-1][start_plot_idx : end_plot_idx], label = 'Original waveform - Before subtraction', color = 'lightgreen', linewidth = linewidth/2 )
                
                
                if speed_plot_flag == True: 
                    ax[i].plot(self.sequences_decoded_iterations[i][start_plot_idx : end_plot_idx], label = 'Current waveform', linewidth = linewidth)
                else:
                    ax[i].plot(self.sequences_decoded_iterations[i], label = 'Current waveform')
                
                
                
                
                #Plot symbs decode
                # if(len(self.sequences_symbs_decoded_idxs[i]) != 0):
                #     symbs_decoded_idxs_temp = self.sequences_symbs_decoded_idxs[i][(self.sequences_symbs_decoded_idxs[i] >= start_plot_idx) & (self.sequences_symbs_decoded_idxs[i] <= end_plot_idx)]
                #     #Note, need to reindex the start x val to zero
                #     ax[i].scatter(symbs_decoded_idxs_temp-start_plot_idx, self.sequences_decoded_iterations[i][symbs_decoded_idxs_temp], s = 40, color = 'red')
                
                
                
                ax[i].set_xlim([0 , num_disp_symbs*self.sps])
                ax[i].grid()
                ax[i].legend(loc = 'upper right')
                if i != num_iter:
                    ax[i].set_title('Iteration: ' + str(i) + ' -- User ' + str(i), fontsize = fontsize, fontweight = 'bold')
                    ax[i].set_xlabel('Sample Index #', fontsize = fontsize, fontweight = 'bold')
                    ax[i].set_ylabel('Voltage [volts]', fontsize = fontsize, fontweight = 'bold')
                    ax[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
                else:
                    ax[i].set_title('Final Result')    
            
            if plot_title != None:
                plt.suptitle(plot_title + '\n', fontweight = 'bold', fontsize = fontsize)
            else:
                plt.suptitle('SIC Process\n', fontweight = 'bold', fontsize = fontsize)
            
            plt.show()
            
         
'''
////// SIC Decoder with known channel coeffs
'''    
class SIC_Decoder_Blind_Iterations:
    def __init__(self, SI_input_vals, sps, num_users, users_channel_coeffs_arr, thresh_y_vals_arr = None, template_comparision = None, start_align_idx_arr = [0,0], samples_search_amount = int(5e3), detect_bound_index_shift_arr = [0,0], plot_flag = True, speed_plot_flag = False):
        
        num_iter = num_users
        
        self.SI_input_vals = SI_input_vals
        self.sps = sps
        self.users_channel_coeffs_arr = np.array(users_channel_coeffs_arr)
        self.thresh_y_vals_arr = thresh_y_vals_arr
        self.detect_bound_index_shift_arr = detect_bound_index_shift_arr
        

        def decode(self, current_input_vals, thresh_val, detect_bound_index_shift):
            
            #Set threshhold
            if thresh_val == None:
                thresh = np.average(current_input_vals) + 0.08*np.abs(np.max(current_input_vals) - np.min(current_input_vals))
            else:
                thresh = thresh_val
                
            #Assuming the current_input_vals is framed/aligned correctly, draw symbol detection marker boundaries in time
            detection_bounds_idxs = np.arange(0, len(current_input_vals), self.sps) - detect_bound_index_shift
            
            if detection_bounds_idxs[0] < 0:
                detection_bounds_idxs[0] = 0
            
            
            #Find the indices of the 'peaks' in the data
            #Using these to decode symbols' values and potential
            top_peaks_idxs, bottom_peaks_idxs, combined_peaks_idxs = find_peaks_indices(current_input_vals)
            
            num_detectable_symbs = int(detection_bounds_idxs[-1]/self.sps)
            
            symbs_decoded = []
            #Detect symbol values by looking at the middle value in a symbol to compare to a thresh
            for i, edge_idx in enumerate(detection_bounds_idxs):
    
                if i < num_detectable_symbs:
                    low_bound_edge_idx = edge_idx
                    upper_bound_edge_idx = edge_idx + self.sps
                
                    if np.median(current_input_vals[low_bound_edge_idx:upper_bound_edge_idx]) > thresh:
                        symbs_decoded.append(1)
                    else:
                        symbs_decoded.append(0)

                        
            return thresh, np.array(symbs_decoded), combined_peaks_idxs, detection_bounds_idxs
        
        '''
        Channel coeff estimation
        '''
        def estimate_iter_channel_state_LSE(detection_thresh, current_input_vals, combined_peaks_idxs):
            
            '''
            Method 1  - Constructive Interference
            '''
            # HIGH - 1
            # above_detection_thresh_vals = np.array([val for val in current_input_vals[combined_peaks_idxs] if val >= detection_thresh])
            above_detection_thresh_vals = np.array([val for val in current_input_vals[combined_peaks_idxs] if val >= detection_thresh])
            
            #Want to use the y values close to the threshold
            LSE_one_thresh = 0.1 * (np.max(above_detection_thresh_vals) - np.min(above_detection_thresh_vals)) + np.min(above_detection_thresh_vals)
            ones_y_vals = np.array([val for val in above_detection_thresh_vals if val <= LSE_one_thresh])
            ones_x_vals = np.ones(len(ones_y_vals))
            
            # LOW - 0
            below_detection_thresh_vals = np.array([val for val in current_input_vals[combined_peaks_idxs] if val < detection_thresh])
            LSE_zero_thresh = 0.05 * (np.max(below_detection_thresh_vals) - np.min(below_detection_thresh_vals)) + np.min(below_detection_thresh_vals)
            zeros_y_vals = np.array([val for val in below_detection_thresh_vals if val <= LSE_zero_thresh])            
            zeros_x_vals = np.zeros(len(zeros_y_vals))
            
            
            y_vals = np.concatenate((ones_y_vals, zeros_y_vals))
            x_vals = np.concatenate((ones_x_vals, zeros_x_vals))
            
            m,c = apply_LSE(y_seq = y_vals, x_seq = x_vals)   
            
            return m

        def estimate_channel_marker_peak(marker_index_loc, current_input_vals):
            #CHANGE BASED ON LOCATION OF MARKER 1
            start_idx = detection_bounds_idxs[marker_index_loc]
            end_idx = detection_bounds_idxs[marker_index_loc + 1] 
            marker_y_val_median = np.median(current_input_vals[start_idx : end_idx])
            
            left_symbol_median = np.median(current_input_vals[detection_bounds_idxs[marker_index_loc-1] : detection_bounds_idxs[marker_index_loc]])
            right_symbol_median = np.median(current_input_vals[detection_bounds_idxs[marker_index_loc+1]: detection_bounds_idxs[marker_index_loc+2]])
            
            a = np.min([np.abs(marker_y_val_median - left_symbol_median),
                            np.abs(marker_y_val_median - right_symbol_median)])

            print('Peak Guess: ' + str(np.round(a,5)))
            print()
            
            return a

        '''
        SIC Process
        '''
        self.sequences_decoded_iterations = []
        self.users_decoded = []
        self.users_decoded_no_SIC = []
        self.users_decoding_peaks_indices = []
        self.users_decoding_threshes = []
        self.detection_bounds_idxs_arr = []
        self.is_pos_flags = []
        start_index_framed_start = 0 #Starting index of the framed user
        
        #For BER
        self.marker_index_offset_arr = []
        
        #Create a deep copy to prevent the original input vals from being modified
        current_input_vals = copy.deepcopy(self.SI_input_vals)
        
        #Order channel state coefs by largest magnitude
        self.users_channel_state_coeffs_ordered = self.users_channel_coeffs_arr[np.argsort(np.abs(self.users_channel_coeffs_arr))[::-1]]
        
        #Iterate
        for iter_attempt in range(num_iter):
            
            matching_template = template_comparision

            #Step 1: Align
            #Need end_stop_align_idx to not get rid of too many samples when trying to align for the preamble
            
            if iter_attempt == 0:
                start_align_matched_filter_idx = start_align_idx_arr[iter_attempt]
                end_stop_align_matched_filter_idx = start_align_matched_filter_idx  + samples_search_amount
            else:
                start_align_matched_filter_idx = np.abs(start_align_idx_arr[iter_attempt] - start_index_framed_start)
                end_stop_align_matched_filter_idx = start_align_matched_filter_idx  + samples_search_amount
            
            
            #Check to see if need to flip current iteration input values for better matched filtering align,ent
            is_pos_flag_temp = not flipped_flag(input_vals = current_input_vals[start_align_matched_filter_idx  : end_stop_align_matched_filter_idx ], 
                                        template_comparision = matching_template, 
                                        sps = self.sps,
                                        debug_output=False,
                                        plot_flag=False)
            
            self.is_pos_flags.append(is_pos_flag_temp)
            
            
            if iter_attempt == 0 and is_pos_flag_temp == False:
                current_input_vals = -1 * current_input_vals
                current_input_vals_no_SIC = -1 * copy.deepcopy(self.SI_input_vals)
            else:
                if self.is_pos_flags[iter_attempt - 1] != is_pos_flag_temp:
                    current_input_vals = -1 * (current_input_vals)
                    current_input_vals_no_SIC = -1 * copy.deepcopy(self.SI_input_vals)
            
            
            #Align with match filtering
            seq_corr, seq_lags, seq_zero_val_index_lags, seq_lag_samples_amount = apply_match_filter(input_vals = current_input_vals[start_align_matched_filter_idx : end_stop_align_matched_filter_idx ], template_comparision = matching_template, sps = self.sps)
            
            start_index_framed_start = start_align_matched_filter_idx + seq_lag_samples_amount
            current_input_vals = current_input_vals[start_index_framed_start  : ]
            

            #Step 2: Decode
            thresh, symbs_decoded, combined_peaks_idxs, detection_bounds_idxs = decode(self, current_input_vals = current_input_vals, thresh_val = thresh_y_vals_arr[iter_attempt], detect_bound_index_shift=self.detect_bound_index_shift_arr[iter_attempt])

            self.users_decoded.append(symbs_decoded)
            self.users_decoding_peaks_indices.append(combined_peaks_idxs)
            self.users_decoding_threshes.append(thresh)
            self.detection_bounds_idxs_arr.append(detection_bounds_idxs)
            
            #For plotting the iterations
            self.sequences_decoded_iterations.append(current_input_vals)
            
            print(len(current_input_vals))
            print(len(copy.deepcopy(self.SI_input_vals)[start_align_idx_arr[iter_attempt] + seq_lag_samples_amount : ]))
            print()
            
            #For comparision, decode without SIC
            thres_no_SIC, symbs_decoded_no_SIC, combined_peaks_idxs_no_SIC, detection_bounds_idxs_no_SIC = decode(self, 
                                                                                                                  current_input_vals = current_input_vals_no_SIC[start_align_idx_arr[iter_attempt] + seq_lag_samples_amount : ], 
                                                                                                                  thresh_val = thresh_y_vals_arr[iter_attempt], 
                                                                                                                  detect_bound_index_shift=self.detect_bound_index_shift_arr[iter_attempt])
            self.users_decoded_no_SIC.append(symbs_decoded_no_SIC)
            
            #Note, due to the start align index, the matched filtering might cause the first few symbols of the pilot preamble to be missed. Need to account for that
            #Need to search in the middle for any offset
            marker_symbol_index = np.where(symbs_decoded[75:] == 1)[0][0] + 75
            marker_index_offset = marker_symbol_index - 100
            self.marker_index_offset_arr.append(marker_index_offset)
            
            
            #Step 3: Channel state estimation for particular user
            a = estimate_channel_marker_peak(marker_index_loc = 100 + marker_index_offset, 
                                             current_input_vals = current_input_vals)

            if iter_attempt == 0:
                a =-0.045
            
            '''
            '''
            new_seq = copy.deepcopy(current_input_vals)
            
            # a = self.users_channel_state_coeffs_ordered[iter_attempt]
            num_detectable_symbs = int(detection_bounds_idxs[-1]/self.sps)
            
            for i, edge_idx in enumerate(detection_bounds_idxs):

                if i < num_detectable_symbs and i != len(detection_bounds_idxs)-1:
                    low_bound_edge_idx = edge_idx
                    upper_bound_edge_idx = edge_idx + self.sps
     
                    '''
                    Using peaks for thresh. comparision
                    '''
                    peaks_detected_idxs_temp = combined_peaks_idxs[np.where((combined_peaks_idxs >= low_bound_edge_idx) & (combined_peaks_idxs < upper_bound_edge_idx))[0]]
                    
                    if (len(peaks_detected_idxs_temp) > 0):
                        #EDGE CLEANUP
                        #From left detection edge to next peak
                        if current_input_vals[peaks_detected_idxs_temp[0]] >= thresh:
                            new_seq[low_bound_edge_idx : peaks_detected_idxs_temp[0]] = current_input_vals[peaks_detected_idxs_temp[0]] - np.abs(a)
                        else:
                            new_seq[low_bound_edge_idx : peaks_detected_idxs_temp[0]] = current_input_vals[peaks_detected_idxs_temp[0]]
                        
                        if current_input_vals[peaks_detected_idxs_temp[-1]] >= thresh:
                            new_seq[peaks_detected_idxs_temp[-1] : upper_bound_edge_idx] = current_input_vals[peaks_detected_idxs_temp[-1]] - np.abs(a)
                        else:
                            new_seq[peaks_detected_idxs_temp[-1] : upper_bound_edge_idx] = current_input_vals[peaks_detected_idxs_temp[-1]] 

                    
                    if len(peaks_detected_idxs_temp) > 1:
                        for j in range(0, len(peaks_detected_idxs_temp) - 1):
                            
                            if current_input_vals[peaks_detected_idxs_temp[j]] >= thresh: 
                                new_seq[peaks_detected_idxs_temp[j] : peaks_detected_idxs_temp[j+1]] = current_input_vals[peaks_detected_idxs_temp[j] : peaks_detected_idxs_temp[j+1]] - np.abs(a)
            

            
            #Set the next iteration vals to the new seq that subtracted out the old user
            current_input_vals = copy.deepcopy(new_seq)

            #SAVE LAST LAST  for plotting
            if iter_attempt == num_iter-1:
                self.sequences_decoded_iterations.append(current_input_vals)
        
        
        '''
        Plotting Functions
        '''
        if plot_flag == True:
            start_symbol_idx = 0
            stop_symbol_idx = 200*2
            
            fig, ax = plt.subplots(layout='constrained')
            ax.plot(self.sequences_decoded_iterations[0], label = 'Iteration: 0')
            ax.plot(self.sequences_decoded_iterations[1], label = 'Iteration: 1')
            ax.plot(self.sequences_decoded_iterations[2], label = 'Iteration: Final')

            ax.set_xlim([start_symbol_idx * self.sps, stop_symbol_idx * self.sps])
            ax.set_title('SIC - Subtraction Comparision')
            ax.legend(loc = 'upper right')
            ax.grid()
            

            fig, ax = plt.subplots(num_iter + 1, 1, layout= 'constrained', sharey= True, sharex= True)
            
            for i in range(num_iter + 1):
                if speed_plot_flag == True: 
                    ax[i].plot(self.sequences_decoded_iterations[i][start_symbol_idx * self.sps : stop_symbol_idx * self.sps], label = 'Waveform')

                else:
                    ax[i].plot(self.sequences_decoded_iterations[i], label = 'Waveform')
                
                if i != num_iter:
                    ax[i].scatter(self.users_decoding_peaks_indices[i], self.sequences_decoded_iterations[i][self.users_decoding_peaks_indices[i]], color = 'green', label = 'Decoding Points')
                    ax[i].axhline(y = self.users_decoding_threshes[i], label = 'Decision Thresh.', color = 'red', linestyle = '--')
                    
                    for bound_idx in self.detection_bounds_idxs_arr[i][0:stop_symbol_idx]:
                        ax[i].axvline(x = bound_idx, linestyle = '--', color = 'red')
                    
                
                ax[i].set_xlim([start_symbol_idx * self.sps, stop_symbol_idx * self.sps])
                ax[i].grid()
                ax[i].legend(loc = 'upper right')
                if i != num_iter:
                    ax[i].set_title('Iteration: ' + str(i))
                else:
                    ax[i].set_title('Final Result')                   
         
            
class Regular_Thresh_Decoder:
    def __init__(self, input_vals, sps, num_users, thresh_y_vals_arr = None, template_comparision = None, start_align_idx_arr = [0,0], samples_search_amount = int(5e3), detect_bound_index_shift_arr = [0,0], plot_flag = True, speed_plot_flag = False):       
        
        self.input_vals = input_vals
        self.sps = sps
        self.thresh_y_vals_arr = thresh_y_vals_arr
        self.detect_bound_index_shift_arr
        
        def decode(self, current_input_vals, thresh_val, detect_bound_index_shift):
            
            #Set threshhold
            if thresh_val == None:
                thresh = np.average(current_input_vals) + 0.08*np.abs(np.max(current_input_vals) - np.min(current_input_vals))
            else:
                thresh = thresh_val
                
            #Assuming the current_input_vals is framed/aligned correctly, draw symbol detection marker boundaries in time
            detection_bounds_idxs = np.arange(0, len(current_input_vals), self.sps) - detect_bound_index_shift
            
            if detection_bounds_idxs[0] < 0:
                detection_bounds_idxs[0] = 0
            
            
            #Find the indices of the 'peaks' in the data
            #Using these to decode symbols' values and potential
            top_peaks_idxs, bottom_peaks_idxs, combined_peaks_idxs = find_peaks_indices(current_input_vals)
            
            num_detectable_symbs = int(detection_bounds_idxs[-1]/self.sps)
            
            symbs_decoded = []
            #Detect symbol values by looking at the middle value in a symbol to compare to a thresh
            for i, edge_idx in enumerate(detection_bounds_idxs):
    
                if i < num_detectable_symbs:
                    low_bound_edge_idx = edge_idx
                    upper_bound_edge_idx = edge_idx + self.sps
                
                    if np.median(current_input_vals[low_bound_edge_idx:upper_bound_edge_idx]) > thresh:
                        symbs_decoded.append(1)
                    else:
                        symbs_decoded.append(0)

                        
            return thresh, np.array(symbs_decoded), combined_peaks_idxs, detection_bounds_idxs        
         
            
         
            
         
            
         
            
         
            
'''
////// COMMON FUNCTIONS
'''        
        
'''
Generate clocked transitions
'''
def generate_clocked_transitions(input_vals, input_time, mod_rate, fs):
    
    #Detect rising edge
    start_index =  np.where(np.diff(np.sign(input_vals)))[0][0]
    
    symbol_period = 1/mod_rate
    index_step = np.abs(input_time - (input_time[0]+symbol_period)).argmin()   
    transition_edges_indices = np.arange(start_index, len(input_vals), index_step)   
    
    sps = int(fs/mod_rate)
    
    transition_edges_indices_framed = transition_edges_indices - start_index
    
    return transition_edges_indices_framed

'''
Find peaks indices
'''
def find_peaks_indices(input_vals, pos_neg_flag = False):
    
    top_peaks_indices,_ = find_peaks(input_vals)
    bottom_peaks_indices,_ = find_peaks(-1*input_vals)
    
    if pos_neg_flag == True:
        top_peaks_indices = top_peaks_indices[np.where(input_vals[top_peaks_indices] > 0)]         
    else:
        bottom_peaks_indices = bottom_peaks_indices[np.where(input_vals[bottom_peaks_indices] <= 0)]
        
    combined_peaks_indices = np.sort(np.concatenate((top_peaks_indices, bottom_peaks_indices)))
    
    return top_peaks_indices, bottom_peaks_indices, combined_peaks_indices

'''
Apply match filter
'''
def apply_match_filter(input_vals, template_comparision, sps, plot_flag = False):
    
    #For aligning received samples to the start
    seq_corr = signal.correlate(input_vals, template_comparision, mode = 'same') /sps
    seq_lags = signal.correlation_lags(len(input_vals), len(template_comparision), mode="same")
    
    zero_val_index_lags = int(np.where(seq_lags == 0)[0]) #X-axis doesn't start at zero. for simplicity, only look at values above zero
    seq_lag_samples_amount = seq_lags[zero_val_index_lags:][np.argmax(seq_corr[zero_val_index_lags:])]
    
    # seq_lag_samples_amount = seq_lags[np.argmax(seq_corr)]
    
    if plot_flag == True:
        fig, ax  = plt.subplots(layout = 'constrained')
        ax.plot(seq_corr)
        ax.plot(template_comparision)  
        ax.grid()
    
    return seq_corr, seq_lags, zero_val_index_lags, seq_lag_samples_amount


'''
Perform LSE
'''
def apply_LSE(y_seq, x_seq):
    
    A = np.vstack([x_seq, np.ones(len(x_seq))]).T
    m,c = np.linalg.lstsq(A,y_seq, rcond = None)[0]
    
    return m,c


'''
'''
def flipped_flag(input_vals, template_comparision, sps, plot_flag = False, debug_output = False):
    
    original_input = copy.deepcopy(input_vals)
    flipped_input = -1*copy.deepcopy(input_vals)
    
    seq_corr_original, seq_lags_original, zero_val_index_lags_original, seq_lag_samples_amount_original = apply_match_filter(input_vals = original_input, template_comparision = template_comparision, sps =sps)
    original_shifted = original_input[seq_lag_samples_amount_original:]

    seq_corr_flipped, seq_lags_flipped, zero_val_index_lags_flipped, seq_lag_samples_amount_flipped = apply_match_filter(input_vals = flipped_input, template_comparision = template_comparision, sps = int(sps))
    
    # print(seq_lag_samples_amount_flipped)
    flipped_shifted = flipped_input[seq_lag_samples_amount_flipped:] #Original
    # flipped_shifted = flipped_input[int(seq_lag_samples_amount_flipped):]    
    
    seq_corr_original = signal.correlate(original_shifted[0:len(template_comparision)], template_comparision, mode = 'same') /sps
    seq_corr_flipped = signal.correlate(flipped_shifted[0:len(template_comparision)], template_comparision, mode = 'same') /sps
    
    
    originaL_corr_size_comparision = np.min([len(original_shifted), len(template_comparision)])
    flipped_corr_size_comparision = np.min([len(flipped_shifted), len(template_comparision)])
    seq_corr_original = np.corrcoef(original_shifted[0:originaL_corr_size_comparision], template_comparision[0:originaL_corr_size_comparision])
    seq_corr_flipped = np.corrcoef(flipped_shifted[0:flipped_corr_size_comparision], template_comparision[0:flipped_corr_size_comparision])
    
    if plot_flag == True:
        #Plot maker
        fig,ax = plt.subplots(layout = 'constrained')
        ax.plot(original_shifted[0 : len(template_comparision)])
        ax.plot(flipped_shifted[0 : len(template_comparision)])
        ax.plot(template_comparision)
        ax.grid()
    
    # print('Original Corr - ' + str(np.mean(seq_corr_original)))
    # print('Flipped Corr - ' + str(np.mean(seq_corr_flipped)))
    
    if debug_output == True:
        if np.mean(seq_corr_flipped) > np.mean(seq_corr_original):
            print('Flipped')
        else:
            print('Not flipped')
        
    if np.mean(seq_corr_flipped) > np.mean(seq_corr_original):
        return True
    else:
        return False
    
    
def calc_BER(truth_arr, decoded_arr, skip_amount, message_length = None):
    
    if message_length == None:
        message_length = len(truth_arr)
        
    BER = np.mean(truth_arr != decoded_arr[skip_amount : skip_amount+message_length])
    
    comparision_df = pd.DataFrame({'Truth': truth_arr,
                                   'Decoded_Arr': decoded_arr[skip_amount : skip_amount+message_length],
                                   'Decision Arr': truth_arr != decoded_arr[skip_amount : skip_amount+message_length]})
        
    return BER, comparision_df




'''
Old Subtraction Method
'''
            #Crucial process for making sure edges don't leave residues after subtracting
            # for i, idx in enumerate((combined_peaks_idxs)):

            #     if i!=0 and (i < len(combined_peaks_idxs) - 1) and copy.deepcopy(original_seq)[idx] >= thresh_y_vals_arr[0]:
            #         new_seq[combined_peaks_idxs[i-1]:idx] = copy.deepcopy(original_seq)[idx] - np.abs(guess_attempt)

            #         if(copy.deepcopy(original_seq)[combined_peaks_idxs[i+1]] < thresh_y_vals_arr[0]):
            #             new_seq[idx: combined_peaks_idxs[i+1]] = copy.deepcopy(original_seq)[idx] - np.abs(guess_attempt)
                
