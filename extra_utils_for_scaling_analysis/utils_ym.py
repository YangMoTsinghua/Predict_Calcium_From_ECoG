"""
Some misc. utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
from rastermap import Rastermap
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from tqdm import tqdm
from scipy.signal import savgol_filter
from PopulationCoding.predict import canonical_cov
import logging

DEFAULT_N_JOBS = 8

def plot_neurons_behavior_ym(
    neurons, ECoG_fp, normalized_pupil_area, t, clim=[-0.3, 1.5], zspacing=10
):
    fig = plt.figure(figsize=(8, 6))
    
    # 
    ax = plt.subplot(10, 1, (1, 5))
    ax.imshow(
        neurons.T, cmap="viridis", aspect="auto", clim=clim, interpolation="bicubic"
    )
    ax.set_xticks([])
    ax.set_ylabel("Neuron #")

    # 
    ax = plt.subplot(10, 1, (6, 9))
    time_indices = np.linspace(0, len(t) - 1, ECoG_fp.shape[0])
    for i in range(ECoG_fp.shape[1]):
        ax.plot(time_indices,ECoG_fp[:, i] - zspacing * i, color="k")
    ax.set_xlim([0, len(t) - 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("ECoG")

    # 
    ax = plt.subplot(10, 1, 10)
    ax.plot(t, normalized_pupil_area, color="k")
    ax.set_xlim([t[0], t[-1]])
    ax.set_yticks([])
    ax.set_ylabel("Normalized\npupil area")
    ax.set_xlabel("Time (sec)")

    return fig


def bandstop_filter(data, order=3, fs=1000, stopbands=[[2, 3], [4, 6], [49, 51]]):
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    filtered_data = data.copy()

    for freq_stop in stopbands:
        b, a = signal.butter(order, [f / (fs / 2) for f in freq_stop], btype='bandstop')
        filtered_data = signal.filtfilt(b, a, filtered_data, axis=1)

    if data.shape[0] == 1:
        filtered_data = np.squeeze(filtered_data, axis=0)

    return filtered_data

def preprocess_calcium_data(res_path, load_path, threshold=50):
    denoised_data = np.load(os.path.join(res_path, "neuron_denoised_records_whole_brain.npy"))
    position_x = pd.read_csv(os.path.join(load_path, "valid_neuron_x.csv"))
    position_y = pd.read_csv(os.path.join(load_path, "valid_neuron_y.csv"))

    denoised_data[denoised_data > threshold] = threshold

    neurons = zscore(denoised_data).T
    centers = np.hstack((position_x, position_y)).T

    return neurons, centers


def preprocess_ecog_data(load_path, kbd1, evt06):
    fp_32chs = pd.read_csv(os.path.join(load_path, "ele_signal_burst_supp_fp01-fp32.csv")).values.T
    mean_across_channels = np.mean(fp_32chs, axis=0)
    fp_32chs_filtered = np.zeros_like(fp_32chs)

    for ch in range(fp_32chs.shape[0]):
        signal_without_mean = fp_32chs[ch] - mean_across_channels
        fp_32chs_filtered[ch] = bandstop_filter(signal_without_mean)

    # 
    second_start_Ele = kbd1 - 120  # s
    second_stop_Ele = second_start_Ele + 900  # s
    fhz_Ele = 1000  # Hz
    
    t_idx_Ele = np.arange(int(fhz_Ele * second_start_Ele), int(fhz_Ele * second_stop_Ele))
    fp_32chs_example = zscore(fp_32chs_filtered.T[t_idx_Ele, :32])

    return fp_32chs_example


def perform_rastermap_analysis(neurons, kbd1, evt06, perform_rastermap):
    second_start_Ca = kbd1 - evt06 - 120
    second_stop_Ca = second_start_Ca + 900
    fhz_Ca = 10
    t_idx_Ca = np.arange(int(fhz_Ca * second_start_Ca), int(fhz_Ca * second_stop_Ca))
    neurons_example = zscore(neurons[t_idx_Ca, :])
    neurons_example_sorted = None
    if perform_rastermap:
        model = Rastermap(n_PCs=32, n_clusters=8).fit(neurons_example.T)
        neurons_example_sorted = neurons_example[:, model.isort]
    return neurons_example, neurons_example_sorted
 

def downsample_signal(fp_32chs_example, factor=100):
    fp_32chs_downsampled = fp_32chs_example.reshape(-1, factor, fp_32chs_example.shape[1]).mean(axis=1)
    assert fp_32chs_downsampled.shape[0] == 9000
    return fp_32chs_downsampled

def extract_features(channel_data):
    mean = np.mean(channel_data)
    std_dev = np.std(channel_data)
    variance = np.var(channel_data)
    max_val = np.max(channel_data)
    min_val = np.min(channel_data)
    median = np.median(channel_data)
    ptp = np.ptp(channel_data)
    iqr = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)
    energy = np.sum(channel_data ** 2) / len(channel_data)
    skewness = skew(channel_data)
    kurt = kurtosis(channel_data)
    rms = np.sqrt(np.mean(channel_data**2))
    waveform_factor = rms / np.mean(np.abs(channel_data))
    crest_factor = max_val / rms
    zero_crossings = np.where(np.diff(np.signbit(channel_data)))[0].size

    freqs, psd = welch(channel_data, fs=1000, nperseg=len(channel_data))  
    dominant_freq = freqs[np.argmax(psd)]
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    spectral_entropy = -np.sum((psd/np.sum(psd)) * np.log2(psd/np.sum(psd)))
    bandwidth = np.sqrt(np.sum(psd * (freqs - spectral_centroid)**2) / np.sum(psd))

    features = [
        mean, std_dev, variance, max_val, min_val, median,
        ptp, iqr, energy, skewness, kurt, waveform_factor, crest_factor,
        zero_crossings, dominant_freq, spectral_centroid, spectral_entropy, bandwidth
    ]
    
    return features

def extract_all_features(fp_32chs_example, num_windows, num_channels):
    features_per_channel = 18
    output_matrix = np.zeros((num_windows, num_channels * features_per_channel))

    for window_idx in tqdm(range(num_windows)):
        start_idx = window_idx * 100
        end_idx = start_idx + 100
        window_data = fp_32chs_example[start_idx:end_idx, :]

        feature_vector = []
        for channel_idx in range(num_channels):
            channel_data = window_data[:, channel_idx]
            features = extract_features(channel_data)
            feature_vector.extend(features)

        output_matrix[window_idx, :] = feature_vector

    return output_matrix

def zscore_and_visualize_ecog_features(ECoG_Extracted_Features, output_directory, savefigs_1):

    ECoG_Extracted_Features = zscore(zscore(ECoG_Extracted_Features.T).T)

    if savefigs_1:
        model_ECoG = Rastermap(n_PCs=32, n_clusters=8).fit(ECoG_Extracted_Features.T)
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        clim1 = [-0.3, 100]
        clim2 = [-0.3, 3]

        axs[0].imshow(
            ECoG_Extracted_Features[:, model_ECoG.isort].T, cmap="viridis", aspect="auto", clim=clim1, interpolation="bicubic"
        )
        axs[0].set_xticks([])
        axs[0].set_ylabel("Neuron #")
        axs[0].set_title('Original TrainX')

        axs[1].imshow(
            ECoG_Extracted_Features[:, model_ECoG.isort].T, cmap="viridis", aspect="auto", clim=clim2, interpolation="bicubic"
        )
        axs[1].set_xticks([])
        axs[1].set_ylabel("Neuron #")
        axs[1].set_title('Original TrainX')

        output_path_tif = os.path.join(output_directory, "fig6", "ECoG_Features.tif")
        output_path_png = os.path.join(output_directory, "fig6", "ECoG_Features.png")
        
        
        fig.savefig(output_path_tif, dpi=600, format='tif')
        fig.savefig(output_path_png, dpi=600, format='png')
    
        plt.close(fig)


def get_list_shape(lst):
    if isinstance(lst, list):
        return (len(lst),) + get_list_shape(lst[0])
    else:
        return ()
    

def plot_svc_time_series(neurons_example, ex_u, ex_v, ex_ntrain, ex_ntest, t_start, t_end, res_path, filename, n_nneur=-1):
    
    svc_indices = [0, 7, 15, 63, 255, 511]  # SVC 1, 8, 16, 64, 256, 512
    
    plt.figure(figsize=(12, 18))

    for i, svc_index in enumerate(svc_indices):
        
        neurons_subset_train = neurons_example[t_start*10:t_end*10, ex_ntrain[n_nneur]]
        svc_time_series_train = np.dot(neurons_subset_train, ex_u[n_nneur][:, svc_index])

        
        neurons_subset_test = neurons_example[t_start*10:t_end*10, ex_ntest[n_nneur]]
        svc_time_series_test = np.dot(neurons_subset_test, ex_v[n_nneur][:, svc_index])

        
        correlation = np.corrcoef(svc_time_series_train, svc_time_series_test)[0, 1]

        
        plt.subplot(len(svc_indices), 1, i + 1)
        time_axis = np.arange(svc_time_series_train.shape[0])/10  

        plt.plot(time_axis, svc_time_series_train, label=f'Training Set SVC {svc_index + 1}', color='blue')
        plt.plot(time_axis, svc_time_series_test, label=f'Testing Set SVC {svc_index + 1}', color='orange')

       
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        plt.xlabel('Time')
        plt.ylabel('SVC Value')
        plt.title(f'SVC {svc_index + 1} Time Series')
        plt.legend()

    plt.tight_layout()
    
    
    save_dir = os.path.join(res_path, "SVCA set1 and set2")
    os.makedirs(save_dir, exist_ok=True)

   
    output_path_png = os.path.join(save_dir, f"1_8_16_64_256_512_{filename}.png",)
    plt.savefig(output_path_png, format='png')
    plt.close()

    
def compare_and_plot(model, trainX, testX, u_sub, v_sub, vp_train, vp_test, ll, kk, res_path, base_path, centers):

    dictpath = os.path.join(res_path, "predicted_neurons_versus_trainX_testX", base_path)
    os.makedirs(dictpath, exist_ok=True)
    plot_filename_png = f"neurons_l{ll}_k{kk}.png"
    plot_filepath_png = os.path.join(dictpath, plot_filename_png)
    plot_filename_tif = f"neurons_l{ll}_k{kk}.tif"
    plot_filepath_tif = os.path.join(dictpath, plot_filename_tif)


    trainX_approximation = vp_train.T @ u_sub.T
    testX_approximation = vp_test.T @ v_sub.T
    
    trainX_approximation = np.real(trainX_approximation)
    testX_approximation = np.real(testX_approximation)
    trainX_approximation = zscore(zscore(trainX_approximation.T).T)
    testX_approximation = zscore(zscore(testX_approximation.T).T)
    
    real = np.hstack((trainX, testX))
    predicted = np.hstack((trainX_approximation, testX_approximation))
    
    plot_neuron_correlation(real, predicted, centers, res_path, f"{base_path}l{ll}k{kk}")

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    clim = [-0.3, 1.5]

    axs[0].imshow(
        real[:, model.isort].T, cmap="viridis", aspect="auto", clim=clim, interpolation="bicubic"
    )
    axs[0].set_xticks([])
    axs[0].set_ylabel("Neuron #")
    axs[0].set_title('Original')

    axs[1].imshow(
        predicted[:, model.isort].T, cmap="viridis", aspect="auto", clim=clim, interpolation="bicubic"
    )
    axs[1].set_xticks([])
    axs[1].set_ylabel("Neuron #")
    axs[1].set_title('Predicted')
    
    for ax in axs.flat:
        ax.set(xlabel='Time', ylabel='Neurons')

    plt.tight_layout()
    plt.savefig(plot_filepath_png)
    plt.savefig(plot_filepath_tif)
    plt.close(fig)
    print(f"Plot saved at {dictpath}")


import pickle

def plot_quantitative_statistics(correlations, res_path, suffix):
    sorted_indices = np.argsort(correlations)[::-1]  
    top_1000_indices = sorted_indices[:1000]
    
    top_1000_correlations = [correlations[i] for i in top_1000_indices]
    all_average_correlation = np.mean(correlations)
    top_1000_average_correlation = np.mean(top_1000_correlations)

    all_std_error = np.std(correlations) / np.sqrt(len(correlations))* 1.96
    top_1000_std_error = np.std(top_1000_correlations) / np.sqrt(len(top_1000_correlations)) * 1.96

    # Prepare data to save
    data_to_save = {
        'correlations':correlations,
        'sorted_indices': sorted_indices,
        'top_1000_indices': top_1000_indices,
        'top_1000_correlations': top_1000_correlations,
        'all_average_correlation': all_average_correlation,
        'top_1000_average_correlation': top_1000_average_correlation,
        'all_std_error': all_std_error,
        'top_1000_std_error': top_1000_std_error
    }

    # Save variables to a file using pickle
    save_dir = os.path.join(res_path, "neuron_correlation_statistics")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"variables_{suffix}.pkl"), 'wb') as f:
        pickle.dump(data_to_save, f)

    # Plotting
    plt.figure(figsize=(8, 6))
    labels = ['All Neurons', 'Top 1000 Neurons']
    averages = [all_average_correlation, top_1000_average_correlation]
    errors = [all_std_error, top_1000_std_error]

    plt.bar(labels, averages, yerr=errors, color=['blue', 'green'], alpha=0.7, capsize=5)
    plt.ylabel('Average Pearson Correlation')
    plt.title('Comparison of Average Correlation with Error Bars')

    plt.savefig(os.path.join(save_dir, f"neuron_correlation_statistics_{suffix}.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"neuron_correlation_statistics_{suffix}.pdf"))
    plt.close()



def plot_neuron_correlation(real_activity, predicted_activity, centers, res_path, suffix):
    correlations = []
    for i in range(real_activity.shape[1]):
        if np.std(real_activity[:, i]) == 0 or np.std(predicted_activity[:, i]) == 0:
            correlations.append(0)
        else:
            corr = np.corrcoef(real_activity[:, i], predicted_activity[:, i])[0, 1]
            correlations.append(corr)
    correlations = np.nan_to_num(correlations, nan=0) 
    correlations = np.maximum(correlations, 0)

    s = max(2, 100 / np.sqrt(centers.shape[1]))  
    np.save(os.path.join(res_path, f'correlations_{suffix}.npy'), {
            'correlations': correlations,
        })  

    np.save(os.path.join(res_path, f'real_predicted_activity_centers_{suffix}.npy'), {
            'real_activity': real_activity,
            'predicted_activity': predicted_activity,
            'centers': centers,
        })

    # Plotting scatter plots
    for vmax_value in [0.6, 1]:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            centers[0, :], 
            centers[1, :], 
            c=correlations,
            cmap='Reds', 
            s=10, 
            alpha=correlations,
            edgecolors='none',
            vmin=0, 
            vmax=vmax_value
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label('Pearson Correlation')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Neuron Activity Correlation ({suffix})')

        save_dir = os.path.join(res_path, "neuron_correlation_maps")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"neuron_correlation_{suffix}_{vmax_value}.png"), dpi=300)
        plt.savefig(os.path.join(save_dir, f"neuron_correlation_{suffix}_{vmax_value}.pdf"))
        plt.close()   

    # Calculate distance from center
    center_point = np.mean(centers, axis=1)
    distances = np.linalg.norm(centers.T - center_point, axis=1)

    # Define a threshold to separate central and peripheral neurons
    threshold_distance = np.percentile(distances, 50)  # Median distance as separation

    central_correlations = correlations[distances <= threshold_distance]
    peripheral_correlations = correlations[distances > threshold_distance]

    # Plot average correlation
    plt.figure()
    plt.bar(['Central', 'Peripheral'], [np.mean(central_correlations), np.mean(peripheral_correlations)])
    plt.ylabel('Average Pearson Correlation')
    plt.title('Average Correlation: Central vs Peripheral Neurons')
    plt.savefig(os.path.join(save_dir, f"average_correlation_central_vs_peripheral_{suffix}.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"average_correlation_central_vs_peripheral_{suffix}.pdf"))
    plt.close()

    # Plot correlation vs distance
    plt.figure()
    plt.scatter(distances, correlations, c=correlations, cmap='Reds')
    plt.colorbar(label='Pearson Correlation')
    plt.xlabel('Distance from Center')
    plt.ylabel('Pearson Correlation')
    plt.title('Correlation vs Distance from Center')
    plt.savefig(os.path.join(save_dir, f"correlation_vs_distance_{suffix}.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"correlation_vs_distance_{suffix}.pdf"))
    plt.close()
    plot_quantitative_statistics(correlations, res_path, suffix)


# def plot_neuron_correlation(real_activity, predicted_activity, centers, res_path, suffix):
#     correlations = []
#     for i in range(real_activity.shape[1]):
#         if np.std(real_activity[:, i]) == 0 or np.std(predicted_activity[:, i]) == 0:
#             correlations.append(0)
#         else:
#             corr = np.corrcoef(real_activity[:, i], predicted_activity[:, i])[0, 1]
#             correlations.append(corr)
#     correlations = np.nan_to_num(correlations, nan=0) 
#     correlations = np.maximum(correlations, 0)
    
#     s = max(2, 100 / np.sqrt(centers.shape[1]))  
#     np.save(os.path.join(res_path, f'correlations_{suffix}.npy'), {
#             'correlations': correlations,
#         })  

#     np.save(os.path.join(res_path, f'real_predicted_activity_centers_{suffix}.npy'), {
#             'real_activity': real_activity,
#             'predicted_activity': predicted_activity,
#             'centers': centers,
#         })

#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(
#         centers[0, :], 
#         centers[1, :], 
#         c=correlations,
#         cmap='Reds', 
#         s=10, 
#         alpha=correlations,
#         edgecolors='none',
#         vmin=0, 
#         vmax=0.6
#     )

#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Pearson Correlation')
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.title(f'Neuron Activity Correlation ({suffix})')
    
#     save_dir = os.path.join(res_path, "neuron_correlation_maps")
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, f"neuron_correlation_{suffix}_0.6.png"), dpi=300)
#     plt.savefig(os.path.join(save_dir, f"neuron_correlation_{suffix}_0.6.pdf"))
#     plt.close()   
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(
#         centers[0, :], 
#         centers[1, :], 
#         c=correlations,
#         cmap='Reds', 
#         s=10, 
#         alpha=correlations,
#         edgecolors='none',
#         vmin=0, 
#         vmax=1
#     )

#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Pearson Correlation')
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.title(f'Neuron Activity Correlation ({suffix})')
    
#     save_dir = os.path.join(res_path, "neuron_correlation_maps")
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, f"neuron_correlation_{suffix}_1.png"), dpi=300)
#     plt.savefig(os.path.join(save_dir, f"neuron_correlation_{suffix}_1.pdf"))
#     plt.close()   
#     plot_quantitative_statistics(correlations, res_path, suffix)
    


def reduced_rank_regressions_ym(ranks, ECoG_Extracted_Features, projs1, projs2, lams, res_path,
                             ex_itrain, ex_itest, ex_u, ex_v, nsvc_predict, trainX, testX, centers):
    
    nrank1 = ranks[ranks <= ECoG_Extracted_Features.shape[1]]
    print("nrank1:", nrank1)

    n_nneurs = 0
    t_start_plot = 50
    t_end_plot = 400

    for l in tqdm(range(len(lams))):  # try various regularization
        atrain, btrain, _, _ = canonical_cov(
            projs1[ex_itrain[n_nneurs], :], ECoG_Extracted_Features[ex_itrain[n_nneurs], :], lams[l], npc=max(nrank1)
        )
        atest, btest, _, _ = canonical_cov(
            projs2[ex_itrain[n_nneurs], :], ECoG_Extracted_Features[ex_itrain[n_nneurs], :], lams[l], npc=max(nrank1)
        )
    
        for k in tqdm(range(len(nrank1))):  # try various ranks
            vp_train = (
                atrain[:, :nrank1[k]] @ btrain[:, :nrank1[k]].T @ ECoG_Extracted_Features.T
            )
            vp_test = atest[:, :nrank1[k]] @ btest[:, :nrank1[k]].T @ ECoG_Extracted_Features.T

            window_length = 51  
            polyorder = 11       
            vp_train_filtered = savgol_filter(vp_train.T, window_length=window_length, polyorder=polyorder, axis=0).T
            vp_test_filtered = savgol_filter(vp_test.T, window_length=window_length, polyorder=polyorder, axis=0).T
            vp_train = vp_train_filtered
            vp_test = vp_test_filtered

            print("vp_train:", vp_train.shape)
            print("vp_test:", vp_test.shape)
            print("projs1:", projs1.shape)
            print("projs2:", projs2.shape)

            plot_components = [0, 1, 2, 3, 4, 5, 6, 7, 15, 31, 63]


            def save_plots(proj, vp, filepath_suffix, filename_suffix):
                fig, axes = plt.subplots(len(plot_components), 1, figsize=(15, 25))
                for i, vec_idx in enumerate(plot_components):
                    ax = axes[i]
                    ax.plot(proj[t_start_plot*10:t_end_plot*10, vec_idx], color='b', alpha=0.5, label='real')
                    ax.plot(vp.T[t_start_plot*10:t_end_plot*10, vec_idx], color='r', alpha=0.5, label='predicted')
                    ax.set_title(f'SVC {vec_idx + 1}')
                    ax.legend()
                    ax.set_xlabel('time')
                    ax.set_ylabel('SVCs')
                plt.tight_layout()
                filename_png = f"SVCs_l{l}_k{k}_nneur{n_nneurs}_{filename_suffix}.png"
                filename_pdf = f"SVCs_l{l}_k{k}_nneur{n_nneurs}_{filename_suffix}.pdf"
                directory_path = os.path.join(res_path, "predicted_SVCs", filepath_suffix)
                os.makedirs(directory_path, exist_ok=True)
                filepath_png = os.path.join(directory_path, filename_png)
                filepath_pdf = os.path.join(directory_path, filename_pdf)        
                plt.savefig(filepath_png)
                plt.savefig(filepath_pdf, format='pdf')
                fig.clf()
                
            def center_data(data, axis=0):
                mean = np.mean(data, axis=axis, keepdims=True)
                centered_data = data - mean
                return centered_data
            vp_train_filtered=center_data(vp_train_filtered,axis=1)
            vp_test_filtered =center_data(vp_test_filtered, axis=1)    

            # Save plots using the defined helper function
            save_plots(projs1, vp_train_filtered, "Train_And_Test", "projs_1")
            save_plots(projs1[ex_itrain[n_nneurs]], vp_train_filtered[:, ex_itrain[n_nneurs]], "Train_Only" , "projs_1")
            save_plots(projs1[ex_itest[n_nneurs]], vp_train_filtered[:, ex_itest[n_nneurs]], "Test_Only", "projs_1")
            save_plots(projs2, vp_test_filtered,  "Train_And_Test", "projs_2")
            save_plots(projs2[ex_itrain[n_nneurs]], vp_test_filtered[:, ex_itrain[n_nneurs]], "Train_Only" , "projs_2")
            save_plots(projs2[ex_itest[n_nneurs]], vp_test_filtered[:, ex_itest[n_nneurs]], "Test_Only", "projs_2")

            u_sub = ex_u[n_nneurs][:, :nsvc_predict]
            v_sub = ex_v[n_nneurs][:, :nsvc_predict]

            model = Rastermap(n_PCs=32, n_clusters=8).fit(np.hstack((trainX, testX)).T)

            if (l==len(lams)-1) and (k==len(nrank1)-1):
                np.save(os.path.join(res_path, 'vp_train_test__projs1_2.npy'), {
                'vp_train': vp_train,
                'vp_test': vp_test,
                'projs1': projs1,
                'projs2':projs2
                 })
                
                plot_burst=1
                KBD1=191.4
                burst_start=[241,	244.2,	247.3,	252.2,	255.5,	258.8,	265,	270,	279.1,	280,	286.9,	301.9,	311.8,	317.5,	323.4,	334.1,	338.7,	350.6,	368.5,	376.6]
                supp_start=[243.5,	246.1,	248.8,	254,	257.8,	260,	270,	275.4,	280,	283.4,	289.8,	304,	312.5,	318.5,	324.4,	334.5,	339.5,	351.4,	369.5,	377.7]
                burst_start = np.array(burst_start)
                supp_start = np.array(supp_start)
                burst_start=burst_start-KBD1+120
                supp_start=supp_start-KBD1+120
                sampling_rate = 10
                def calculate_intervals(start_times, end_times):
                    return [
                        (int(start * sampling_rate), int(end * sampling_rate))
                        for start, end in zip(start_times, end_times)
                    ]

                if plot_burst == 1:
                    # Burst time intervals
                    burst_intervals = calculate_intervals(burst_start, supp_start)
                    selected_indices = np.concatenate([np.arange(start, end) for start, end in burst_intervals])
                else:
                    # Supp time intervals, excluding the last-to-first transition
                    supp_intervals = calculate_intervals(supp_start[:-1], burst_start[1:])
                    selected_indices = np.concatenate([np.arange(start, end) for start, end in supp_intervals])

                # Function to slice data using indices
                def get_sliced_data(data, indices):
                    return data[indices, :]

                # Plotting logic
                compare_and_plot(
                    model,
                    get_sliced_data(trainX, selected_indices),
                    get_sliced_data(testX, selected_indices),
                    u_sub,
                    v_sub,
                    vp_train[:, selected_indices],
                    vp_test[:, selected_indices],
                    l,
                    k,
                    res_path,
                    "Train_And_Test",
                    centers
                )

                # Filter the indices for test and train specifically within selected_indices
                selected_test_indices = ex_itest[n_nneurs][np.isin(ex_itest[n_nneurs], selected_indices)]
                selected_train_indices = ex_itrain[n_nneurs][np.isin(ex_itrain[n_nneurs], selected_indices)]

                compare_and_plot(
                    model,
                    trainX[selected_test_indices, :],
                    testX[selected_test_indices, :],
                    u_sub,
                    v_sub,
                    vp_train[:, selected_test_indices],
                    vp_test[:, selected_test_indices],
                    l,
                    k,
                    res_path,
                    "Test_Only",
                    centers
                )

                compare_and_plot(
                    model,
                    trainX[selected_train_indices, :],
                    testX[selected_train_indices, :],
                    u_sub,
                    v_sub,
                    vp_train[:, selected_train_indices],
                    vp_test[:, selected_train_indices],
                    l,
                    k,
                    res_path,
                    "Train_Only",
                    centers
                )

                break

                plot_ane=1
                if plot_ane:  
                    
                    compare_and_plot(model, trainX[ex_itest[n_nneurs][(ex_itest[n_nneurs] >= 500) & (ex_itest[n_nneurs] <= 4000)], :],
                                        testX[ex_itest[n_nneurs][(ex_itest[n_nneurs] >= 500) & (ex_itest[n_nneurs] <= 4000)], :],
                                        u_sub, v_sub,
                                        vp_train[:, ex_itest[n_nneurs][(ex_itest[n_nneurs] >= 500) & (ex_itest[n_nneurs] <= 4000)]],
                                        vp_test[:, ex_itest[n_nneurs][(ex_itest[n_nneurs] >= 500) & (ex_itest[n_nneurs] <= 4000)]],
                                        l, k, res_path, "Test_Only", centers)
                    break
                    compare_and_plot(model, trainX[500:4000, :], testX[500:4000, :], u_sub, v_sub, vp_train[:, 500:4000], vp_test[:, 500:4000], l, k, res_path, "Train_And_Test", centers)
                    compare_and_plot(model, trainX[ex_itrain[n_nneurs][(ex_itrain[n_nneurs] >= 500) & (ex_itrain[n_nneurs] <= 4000)], :],
                                        testX[ex_itrain[n_nneurs][(ex_itrain[n_nneurs] >= 500) & (ex_itrain[n_nneurs] <= 4000)], :],
                                        u_sub, v_sub,
                                        vp_train[:, ex_itrain[n_nneurs][(ex_itrain[n_nneurs] >= 500) & (ex_itrain[n_nneurs] <= 4000)]],
                                        vp_test[:, ex_itrain[n_nneurs][(ex_itrain[n_nneurs] >= 500) & (ex_itrain[n_nneurs] <= 4000)]],
                                        l, k, res_path, "Train_Only", centers)   
                    
                else:

                    new_start1 = 0
                    new_end1 = 500
                    new_start2 = 4000
                    new_end2 = 9000
                    combined_slice = np.r_[new_start1:new_end1, new_start2:new_end2]
                    compare_and_plot(model, 
                                    trainX[combined_slice, :], 
                                    testX[combined_slice, :], 
                                    u_sub, v_sub, 
                                    vp_train[:, combined_slice],
                                    vp_test[:, combined_slice],
                                    l, k, res_path, 
                                    "Train_And_Test", centers)

                    test_condition = ((ex_itest[n_nneurs] >= new_start1) & (ex_itest[n_nneurs] < new_end1)) | \
                                    ((ex_itest[n_nneurs] >= new_start2) & (ex_itest[n_nneurs] < new_end2))
                    selected_test_indices = ex_itest[n_nneurs][test_condition]
                    compare_and_plot(model, 
                                    trainX[selected_test_indices, :],
                                    testX[selected_test_indices, :],
                                    u_sub, v_sub,
                                    vp_train[:, selected_test_indices],
                                    vp_test[:, selected_test_indices],
                                    l, k, res_path, 
                                    "Test_Only", centers)
                    break
                    train_condition = ((ex_itrain[n_nneurs] >= new_start1) & (ex_itrain[n_nneurs] < new_end1)) | \
                                    ((ex_itrain[n_nneurs] >= new_start2) & (ex_itrain[n_nneurs] < new_end2))
                    selected_train_indices = ex_itrain[n_nneurs][train_condition]
                    compare_and_plot(model, 
                                    trainX[selected_train_indices, :],
                                    testX[selected_train_indices, :],
                                    u_sub, v_sub,
                                    vp_train[:, selected_train_indices],
                                    vp_test[:, selected_train_indices],
                                    l, k, res_path,
                                    "Train_Only", centers) 


def center_data(data, axis=0):
                mean = np.mean(data, axis=axis, keepdims=True)
                centered_data = data - mean
                return centered_data


def reduced_rank_regressions_ym_2(ranks, ECoG_Extracted_Features, projs1, projs2, lams, res_path,
                                ex_itrain, ex_itest, ex_u, ex_v, nsvc_predict, trainX, testX, centers):
    
    nrank1 = ranks[ranks <= ECoG_Extracted_Features.shape[1]]
    print("nrank1:", nrank1)

    n_nneurs = 0
    t_start_plot = 50
    t_end_plot = 400

    for l in tqdm(range(len(lams))):
        vp_combined_train = np.zeros((projs1.shape[1], ECoG_Extracted_Features.shape[0]))
        vp_combined_test = np.zeros_like(vp_combined_train)

        # cross1: use ex_itrain as training set, use ex_itest as test set
        atrain_p1, btrain_p1, _, _ = canonical_cov(
            projs1[ex_itrain[n_nneurs], :], ECoG_Extracted_Features[ex_itrain[n_nneurs], :], lams[l], npc=max(nrank1)
        )
        atest_p1, btest_p1, _, _ = canonical_cov(
            projs2[ex_itrain[n_nneurs], :], ECoG_Extracted_Features[ex_itrain[n_nneurs], :], lams[l], npc=max(nrank1)
        )
        # cross2: use ex_itest as training set, use ex_itrain as test set
        atrain_p2, btrain_p2, _, _ = canonical_cov(
            projs1[ex_itest[n_nneurs], :], ECoG_Extracted_Features[ex_itest[n_nneurs], :], lams[l], npc=max(nrank1)
        )
        atest_p2, btest_p2, _, _ = canonical_cov(
            projs2[ex_itest[n_nneurs], :], ECoG_Extracted_Features[ex_itest[n_nneurs], :], lams[l], npc=max(nrank1)
        )

        for k in tqdm(range(len(nrank1))):  # try various ranks
            vp_train_p1 = (
                atrain_p1[:, :nrank1[k]] @ btrain_p1[:, :nrank1[k]].T @ ECoG_Extracted_Features[ex_itest[n_nneurs]].T
            )
            vp_test_p1 = atest_p1[:, :nrank1[k]] @ btest_p1[:, :nrank1[k]].T @ ECoG_Extracted_Features[ex_itest[n_nneurs]].T
            vp_train_p1=center_data(vp_train_p1,axis=1)
            vp_test_p1 =center_data(vp_test_p1, axis=1)    
            vp_combined_train[:, ex_itest[n_nneurs]] = zscore(vp_train_p1.T).T
            vp_combined_test[:, ex_itest[n_nneurs]] = zscore(vp_test_p1.T).T
            
            vp_train_p2 = (
                atrain_p2[:, :nrank1[k]] @ btrain_p2[:, :nrank1[k]].T @ ECoG_Extracted_Features[ex_itrain[n_nneurs]].T
            )
            vp_test_p2 = atest_p2[:, :nrank1[k]] @ btest_p2[:, :nrank1[k]].T @ ECoG_Extracted_Features[ex_itrain[n_nneurs]].T
            vp_train_p2=center_data(vp_train_p2,axis=1)
            vp_test_p2 =center_data(vp_test_p2, axis=1)   
            vp_combined_train[:, ex_itrain[n_nneurs]] = zscore(vp_train_p2.T).T
            vp_combined_test[:, ex_itrain[n_nneurs]] = zscore(vp_test_p2.T).T

            # filter
            window_length = 51  
            polyorder = 11
            vp_combined_train_filtered = savgol_filter(vp_combined_train.T, window_length, polyorder, axis=0).T
            vp_combined_test_filtered = savgol_filter(vp_combined_test.T, window_length, polyorder, axis=0).T

            # plot and save
            plot_components = [0, 1, 2, 3, 4, 5, 6, 7, 15, 31, 63]
            def save_plots(proj, vp, filepath_suffix, filename_suffix):
                fig, axes = plt.subplots(len(plot_components), 1, figsize=(15, 25))
                for i, vec_idx in enumerate(plot_components):
                    ax = axes[i]
                    ax.plot(proj[t_start_plot*10:t_end_plot*10, vec_idx], color='b', alpha=0.5, label='real')
                    ax.plot(vp.T[t_start_plot*10:t_end_plot*10, vec_idx], color='r', alpha=0.5, label='predicted')
                    ax.set_title(f'SVC {vec_idx + 1}')
                    ax.legend()
                    ax.set_xlabel('time')
                    ax.set_ylabel('SVCs')
                plt.tight_layout()
                filename_png = f"SVCs_l{l}_k{k}_nneur{n_nneurs}_{filename_suffix}.png"
                filename_pdf = f"SVCs_l{l}_k{k}_nneur{n_nneurs}_{filename_suffix}.pdf"
                directory_path = os.path.join(res_path, "predicted_SVCs", filepath_suffix)
                os.makedirs(directory_path, exist_ok=True)
                filepath_png = os.path.join(directory_path, filename_png)
                filepath_pdf = os.path.join(directory_path, filename_pdf)        
                plt.savefig(filepath_png)
                plt.savefig(filepath_pdf, format='pdf')
                fig.clf()
        
            save_plots(projs1, vp_combined_train_filtered, "2-fold cross validation", "projs_1")
            save_plots(projs2, vp_combined_test_filtered, "2-fold cross validation", "projs_2")

            u_sub = ex_u[n_nneurs][:, :nsvc_predict]
            v_sub = ex_v[n_nneurs][:, :nsvc_predict]
        
            model = Rastermap(n_PCs=32, n_clusters=8).fit(np.hstack((trainX, testX)).T)

            # test_indices = ex_itest[n_nneurs][(ex_itest[n_nneurs] >= 500) & (ex_itest[n_nneurs] <= 4000)]
            compare_and_plot(model, trainX[500:4000], testX[500:4000], u_sub, v_sub,
                        vp_combined_train_filtered[:, 500:4000], vp_combined_test_filtered[:, 500:4000],
                        l, k, res_path, "2-fold cross validation")
