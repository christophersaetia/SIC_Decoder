# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:49:19 2026

@author: saceh
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import NOMA_Experiment_utils_v2 as utils
import message_utils as mu
import copy

plt.close('all')

f_hp = int(0)
f_lp = int(4e6)
mod_rate = int(1e6)
balanced_flag = False
#%% Comparision Pilot Template Setup
#Original pilot sequence
comparision_pilot_sequence_orig = mu.u0_pilot_longer_barker_v2[0:]*0.1

if balanced_flag == True:
    #Since the original pilot sequence is balanced, need to convert it to be detected for a OOK decoder 
    #Interleave original sequence with zeros to show where the balanced zeros are added
    comparision_pilot_sequence = np.zeros(len(comparision_pilot_sequence_orig)*2, dtype = comparision_pilot_sequence_orig.dtype)
    comparision_pilot_sequence[::2] = comparision_pilot_sequence_orig
    comparision_pilot_sequence = comparision_pilot_sequence[0:]
else:
    comparision_pilot_sequence = comparision_pilot_sequence_orig

users_channel_coeffs = []
#%% u0 Initial Read
plt.close('all')
print('\nUser 0 Analysis --------')
u0_file = '../Oscope_Recordings/5_5_2026/6dB/Random/NOMA_Test_1MHz_40mV_3ms__u0a0_1.csv'


u0_original_recording = utils.Single_Channel_Recorded_Samples_Original(file_name_I_channel= u0_file,
                                                                    plot_flag = False)

u0_vals = u0_original_recording.ch1_volts
u0_sps = int(np.round(u0_original_recording.fs/mod_rate))

if balanced_flag == True:
    u0_decoder_sps = int(u0_sps/2)
else:
    u0_decoder_sps = u0_sps

#Create a template waveform of the pilot_sequence   
u0_template_pilot_comparision_waveform = np.repeat(comparision_pilot_sequence, u0_decoder_sps)


#%% u0 Filtered and Aligned

#Pass through a band-pass filter
u0_Filtered = utils.Filtered_Values(input_vals = u0_vals, 
                                           fs = u0_original_recording.fs, 
                                           f_lp = f_lp,
                                           f_hp = f_hp,
                                           sps = u0_sps,
                                           plot_flag = False,
                                           speed_plot_flag=True)


#Baseline-correction
u0_Adjusted = utils.Baseline_Adjustment(input_vals=u0_Filtered.vals_filtered)


#Detect if the user has been flipped. Will be used for the input into the user matched filter
start_search_i = int(304e3)
end_search_i = int(start_search_i + 40e3)
u0_flipped_flag =  utils.flipped_flag(input_vals = u0_Adjusted.baseline_adjusted_seq[start_search_i:end_search_i], 
                   template_comparision = u0_template_pilot_comparision_waveform , 
                   sps = u0_decoder_sps)

if(u0_flipped_flag == True):
    u0_framed_matched_input = -1*u0_Adjusted.baseline_adjusted_seq[start_search_i:]
else:
    u0_framed_matched_input = u0_Adjusted.baseline_adjusted_seq[start_search_i:]


u0_Framed_Matched = utils.Framed_Data_Matched_Filtering(input_vals = u0_framed_matched_input, 
                                                        input_time = u0_original_recording.ch1_time_secs[start_search_i:], 
                                                        x_truth_seq = mu.u0_pilot_longer_v1, 
                                                        fs = u0_original_recording.fs, 
                                                        mod_rate = mod_rate,
                                                        sps = u0_decoder_sps,
                                                        template_comparision= u0_template_pilot_comparision_waveform,
                                                        plot_flag = True)

#%% u1 Initial Read
plt.close('all')
print('\nUser 1 Analysis --------')
u1_file = '../Oscope_Recordings/5_5_2026/6dB/Random/NOMA_Test_1MHz_40mV_3ms__u1a0_1.csv'


u1_original_recording = utils.Single_Channel_Recorded_Samples_Original(file_name_I_channel= u1_file,
                                                                    plot_flag = False)

u1_vals = u1_original_recording.ch1_volts
u1_sps = int(np.round(u1_original_recording.fs/mod_rate))

if balanced_flag == True:
    u1_decoder_sps = int(u1_sps/2)
else:
    u1_decoder_sps = u1_sps

#Create a template waveform of the pilot_sequence   
u1_template_pilot_comparision_waveform = np.repeat(comparision_pilot_sequence, u1_decoder_sps)


#%% u1 Filtered and Aligned

#Pass through a band-pass filter
u1_Filtered = utils.Filtered_Values(input_vals = u1_vals, 
                                           fs = u1_original_recording.fs, 
                                           f_lp = f_lp,
                                           f_hp = f_hp,
                                           sps = u1_sps,
                                           plot_flag = False,
                                           speed_plot_flag=True)


#Baseline-correction
u1_Adjusted = utils.Baseline_Adjustment(input_vals=u1_Filtered.vals_filtered)


#Detect if the user has been flipped. Will be used for the input into the user matched filter
start_search_i = int(159.3e3)
end_search_i = int(start_search_i + 10e3)
u1_flipped_flag =  utils.flipped_flag(input_vals = u1_Adjusted.baseline_adjusted_seq[start_search_i:end_search_i], 
                   template_comparision = u1_template_pilot_comparision_waveform , 
                   sps = u1_decoder_sps)

if(u1_flipped_flag == True):
    u1_framed_matched_input = -1*u1_Adjusted.baseline_adjusted_seq[start_search_i:]
else:
    u1_framed_matched_input = u1_Adjusted.baseline_adjusted_seq[start_search_i:]


u1_Framed_Matched = utils.Framed_Data_Matched_Filtering(input_vals = u1_framed_matched_input, 
                                                        input_time = u1_original_recording.ch1_time_secs[start_search_i:], 
                                                        x_truth_seq = mu.u1_pilot_longer_v1, 
                                                        fs = u1_original_recording.fs, 
                                                        mod_rate = mod_rate,
                                                        sps = u1_decoder_sps,
                                                        template_comparision= u1_template_pilot_comparision_waveform,
                                                        plot_flag = True)
#%% SI Initial Read
plt.close('all')
print('\n Super-imposed Analysis --------')
SI_file = '../Oscope_Recordings/5_5_2026/6dB/Random/NOMA_Test_1MHz_40mV_3ms__SIa0__2_users_only_1.csv'


SI_original_recording = utils.Single_Channel_Recorded_Samples_Original(file_name_I_channel= SI_file,
                                                                    plot_flag = False)

SI_vals = SI_original_recording.ch1_volts
SI_sps = int(np.round(SI_original_recording.fs/mod_rate))

if balanced_flag == True:
    SI_decoder_sps = int(SI_sps/2)
else:
    SI_decoder_sps = SI_sps
    
#Create a template waveform of the pilot_sequence   
SI_template_pilot_comparision_waveform = np.repeat(comparision_pilot_sequence, SI_decoder_sps)

#%% SI Filter and Frame
plt.close('all')
start_search_i = int(350.1e3)
# start_search_i = int(306.3e3)
end_search_i = int(start_search_i + 5e3)



SI_Filtered = utils.Filtered_Values(input_vals = SI_vals, 
                                            fs = SI_original_recording.fs, 
                                            f_lp = f_lp,
                                            f_hp = f_hp,
                                            sps = SI_sps,
                                            plot_flag = True,
                                            speed_plot_flag = True)

SI_Adjusted = utils.Baseline_Adjustment(input_vals=SI_Filtered.vals_filtered)

SI_flipped_flag = utils.flipped_flag(input_vals = SI_Adjusted.baseline_adjusted_seq[start_search_i:end_search_i], 
                   template_comparision = SI_template_pilot_comparision_waveform, 
                   sps = SI_decoder_sps)


if(SI_flipped_flag == True):
    SI_framed_matched_input = -1*SI_Adjusted.baseline_adjusted_seq[start_search_i:]
else:
    SI_framed_matched_input = SI_Adjusted.baseline_adjusted_seq[start_search_i:]
    
    

SI_Framed_Matched = utils.Framed_Data_Matched_Filtering(input_vals = SI_framed_matched_input, 
                                                        input_time = SI_original_recording.ch1_time_secs[start_search_i : end_search_i], 
                                                        x_truth_seq = comparision_pilot_sequence, 
                                                        fs = SI_original_recording.fs, 
                                                        mod_rate = mod_rate,
                                                        sps = SI_decoder_sps,
                                                        end_frame_sample_idx= int(5e3),
                                                        template_comparision= SI_template_pilot_comparision_waveform,
                                                        plot_flag = True,
                                                        corr_sens= 10)

if(SI_flipped_flag == True):
    SI_Framed_Matched.aligned_input_vals = -1*SI_Framed_Matched.aligned_input_vals
    
#%%
SI_Pilots_Analysis = utils.SIC_Pilots_Checker(SI_input_vals = -1 * SI_Adjusted.baseline_adjusted_seq, 
                                              sps = SI_decoder_sps, 
                                              num_users=2,
                                              thresh_y_vals_arr= [-0.01,0.02],
                                              template_comparision= SI_template_pilot_comparision_waveform,
                                              u0_guess_attempts_arr = [0.0625], 
                                              start_align_idx_arr = [int(160.9e3), int(488e3)], 
                                              samples_search_amount = int(5e3), 
                                              detect_bound_index_shift = 0,
                                              plot_flag = False, 
                                              speed_plot_flag = True)

#%%
SI_Decoder = utils.SIC_Decoder(SI_input_vals = SI_Adjusted.baseline_adjusted_seq, 
                               sps = SI_decoder_sps, 
                               num_users = 2, 
                               users_channel_coeffs_arr = [-0.0545, .02],
                               thresh_y_vals_arr= [-0.01, 0.06],
                               template_comparision= SI_template_pilot_comparision_waveform,
                               start_align_idx_arr = [int(160.8e3), int(488.5e3)], 
                               samples_search_amount = int(5e3), 
                               detect_bound_index_shift_arr = [0, -2],
                               plot_flag = True, 
                               speed_plot_flag = True)


#%%
#%% BER Calculation
#BER Calculation
u0_random_truth = mu.u0_rand_message
u1_random_truth = mu.u1_rand_message
u0_decoded = SI_Decoder.users_decoded[0]
u1_decoded = SI_Decoder.users_decoded[1]

pilot_skip_amount = len(comparision_pilot_sequence_orig)
message_length = len(u0_random_truth)


u0_BER, u0_comparision = utils.calc_BER(truth_arr = u0_random_truth, 
                        decoded_arr = u0_decoded, 
                        skip_amount = pilot_skip_amount) #minus one weird



print('u0 BER %: ' + str(np.round(u0_BER,3) * 100))

u1_BER, u1_comparision = utils.calc_BER(truth_arr = u1_random_truth, 
                        decoded_arr = u1_decoded, 
                        skip_amount = pilot_skip_amount)


print('u1 BER %: ' + str(np.round(u1_BER,3) * 100))