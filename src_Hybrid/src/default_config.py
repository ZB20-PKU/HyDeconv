import argparse


def default_config_Hybrid_SIM():
    parser = argparse.ArgumentParser(description='Hybrid-SIM reconstruction ... ')
    ##
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda')
    parser.add_argument('--SIM_raw_data_path', type=str, default='', help='SIM raw data file path')
    ##  BF-SIM parameters    
    parser.add_argument('--SIM_BF_flag', type=bool, default=True)
    parser.add_argument('--SIM_BF_offset', type=float, default=100, help='Camera offset value')    
    parser.add_argument('--SIM_BF_depth_infocus', type=int, default=4)
    parser.add_argument('--SIM_BF_depth_outfocus', type=int, default=50)
    parser.add_argument('--SIM_BF_PSF_path', type=str, default='None')
    parser.add_argument('--SIM_Raw_mask_sigma', type=float, default=1, help='sigmoid windows function params')
    parser.add_argument('--SIM_Raw_pixel_size', type=float, default=43.33e-9)
    parser.add_argument('--SIM_excitation_NA', type=float, default=1.4, help='excitation numerical aperture')
    parser.add_argument('--SIM_emission_NA', type=float, default=1.4, help='emission numerical aperture')
    parser.add_argument('--SIM_excitation_wavelength', type=float, default=488e-9)    
    parser.add_argument('--SIM_emission_wavelength', type=float, default=525e-9)    
    parser.add_argument('--SIM_Pattern_average_number', type=int, default=50)
    parser.add_argument('--SIM_Pattern_vector_search_ratio', type=float, default=0.05)
    parser.add_argument('--SIM_Pattern_orientation_number', type=int, default=3, help='number of structure illuminaion orientations')
    parser.add_argument('--SIM_Pattern_phase_number', type=int, default=3, help='number of structure illuminaion phases')
    parser.add_argument('--SIM_Pattern_phase_interval', type=list, default=[1, 1, 1])
    parser.add_argument('--SIM_Pattern_phase_EMD', type=int, default=2)
    parser.add_argument('--SIM_Pattern_module_depth_EMD', type=int, default=0)
    parser.add_argument('--SIM_Pattern_estimation_overlap_ratio', type=float, default=0.2)    
    parser.add_argument('--SIM_Recon_wiener_parameter', type=float, default=0.2, help='wiener_sigma = 1/SNR')
    parser.add_argument('--SIM_Recon_OTF_path', type=str, default='./src_Hybrid/src_Optics/SIM_Recon_OTF/SIM_Recon_OTF_WL525_NA1.4_PSXY43.tif')
    parser.add_argument('--SIM_Pattern_reuse_flag', type=bool, default=False)
    ##  Hessian denoising parameters
    parser.add_argument('--Hessian_iteration_number', type=int, default=100)
    parser.add_argument('--Hessian_fidelity', type=float, default=100)
    parser.add_argument('--Hessian_Z_continuity', type=float, default=2)
    parser.add_argument('--Hessian_Z_rolling_window_size', type=int, default=100)
    ##  TDV denoising parameters    
    parser.add_argument('--TDV_mode', type=str, default='inference')
    parser.add_argument('--TDV_step_size', type=float, default=None, help='TDV step size')
    parser.add_argument('--TDV_model_path', type=str, default='./src_Hybrid/src_model/TDV_150XSIM_Actin.pth')
    parser.add_argument('--TDV_fidelity', type=float, default=100)
    parser.add_argument('--TDV_offset', type=float, default=0.1)
    parser.add_argument('--TDV_iteration_number', type=int, default=30)
    parser.add_argument('--TDV_weight', type=float, default=1)
    ##  Sparse deconvolution parameters
    parser.add_argument('--Sparse_NA', type=float, default=3)
    parser.add_argument('--Sparse_iteration_number', type=int, default=20)
    parser.add_argument('--Sparse_offset', type=float, default=0.3)    
    ##
    args = parser.parse_args()
    return args


def default_config_TDV_SIM_training():
    parser = argparse.ArgumentParser(description='TDV-SIM training ... ')
    ##
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda')
    parser.add_argument('--SIM_raw_data_folder_path', type=str, default='', help='SIM raw data folder path')
    parser.add_argument('--SIM_raw_data_file_path', type=str, default='', help='SIM raw data file path')
    ##  BF-SIM parameters    
    parser.add_argument('--SIM_BF_flag', type=bool, default=True)
    parser.add_argument('--SIM_BF_offset', type=float, default=100, help='Camera offset value')    
    parser.add_argument('--SIM_BF_depth_infocus', type=int, default=4)
    parser.add_argument('--SIM_BF_depth_outfocus', type=int, default=50)
    parser.add_argument('--SIM_BF_PSF_path', type=str, default='None')
    parser.add_argument('--SIM_Raw_mask_sigma', type=float, default=1, help='sigmoid windows function params')
    parser.add_argument('--SIM_Raw_pixel_size', type=float, default=65e-9)
    parser.add_argument('--SIM_excitation_NA', type=float, default=1.4, help='excitation numerical aperture')
    parser.add_argument('--SIM_emission_NA', type=float, default=1.4, help='emission numerical aperture')
    parser.add_argument('--SIM_excitation_wavelength', type=float, default=488e-9)    
    parser.add_argument('--SIM_emission_wavelength', type=float, default=525e-9)    
    parser.add_argument('--SIM_Pattern_average_number', type=int, default=50)
    parser.add_argument('--SIM_Pattern_vector_search_ratio', type=float, default=0.05)
    parser.add_argument('--SIM_Pattern_orientation_number', type=int, default=3, help='number of structure illuminaion orientations')
    parser.add_argument('--SIM_Pattern_phase_number', type=int, default=3, help='number of structure illuminaion phases')
    parser.add_argument('--SIM_Pattern_phase_interval', type=list, default=[1, 1, 1])
    parser.add_argument('--SIM_Pattern_phase_EMD', type=int, default=2)
    parser.add_argument('--SIM_Pattern_module_depth_EMD', type=int, default=0)
    parser.add_argument('--SIM_Pattern_estimation_overlap_ratio', type=float, default=0.2)    
    parser.add_argument('--SIM_Recon_wiener_parameter', type=float, default=0.2, help='wiener_sigma = 1/SNR')
    parser.add_argument('--SIM_Recon_OTF_path', type=str, default='./src_Hybrid/src_Optics/SIM_Recon_OTF/SIM_Recon_OTF_WL525_NA1.4_PSXY65.tif')
    parser.add_argument('--SIM_Pattern_reuse_flag', type=bool, default=False)
    ##  Dataset parameters
    parser.add_argument('--Dataset_Z_average_number', type=int, default=20)
    parser.add_argument('--Dataset_Z_skip_number', type=int, default=5)
    parser.add_argument('--Dataset_XY_Poisson_noise_level', type=float, default=50)
    parser.add_argument('--Dataset_XY_Gaussian_noise_level', type=float, default=50)
    parser.add_argument('--Dataset_XY_margin_size', type=int, default=16)
    parser.add_argument('--Dataset_XY_block_size', type=int, default=256)
    parser.add_argument('--Dataset_XY_block_interval', type=int, default=128)
    parser.add_argument('--Dataset_XY_overexposure_ratio', type=float, default=0.5, help='The proportion of pixels with Max-normalization gray value > 0.9')
    parser.add_argument('--Dataset_XY_minimal_heterogeneity', type=float, default=0.6, help='Minimal standard deviation')
    parser.add_argument('--Dataset_XY_minimal_contrast', type=float, default=0.3, help='Minimal range')
    ##  TDV-DNN training parameters    
    parser.add_argument('--Train_batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--Train_epoch_number', type=int, default=50, help='Epoch number')
    parser.add_argument('--Train_learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--Train_learning_rate_decay_ratio', type=float, default=0.9, help='Learning rate decay ratio')
    parser.add_argument('--Train_learning_rate_minimum', type=float, default=1e-6, help='Learning rate minimum')
    parser.add_argument('--Train_abnormal_factor', type=float, default=3, help='Skip threshold (mean + factor*std) for abnormal samples')
    parser.add_argument('--Train_SSIM_MSE_ratio', type=float, default=0.2, help='SSIM/MSE ratio for loss function')
    parser.add_argument('--Train_Max_reinitialization_number', type=int, default=5, help='Max reinitialization number')
    parser.add_argument('--Train_sample', type=str, default='Actin', help='Sample name')
    parser.add_argument('--Train_system', type=str, default='100XSIM', help='System name')
    parser.add_argument('--TDV_mode', type=str, default='train', help='DNN mode')
    ##
    args = parser.parse_args()
    return args


def default_config_Hybrid_FM():
    parser = argparse.ArgumentParser(description='Hybrid-FM reconstruction ... ')
    ##
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda')
    parser.add_argument('--Raw_data_path', type=str, default='', help='FM raw data file path')
    ##  Hessian denoising parameters
    parser.add_argument('--Hessian_iteration_number', type=int, default=100)
    parser.add_argument('--Hessian_fidelity', type=float, default=150)
    parser.add_argument('--Hessian_Z_continuity', type=float, default=0)
    parser.add_argument('--Hessian_Z_rolling_window_size', type=int, default=100)
    ##  TDV denoising parameters    
    parser.add_argument('--TDV_mode', type=str, default='inference')
    parser.add_argument('--TDV_step_size', type=float, default=None, help='TDV step size')
    parser.add_argument('--TDV_model_path', type=str, default='./src_Hybrid/src_model/TDV_STED_Mito.pth')
    parser.add_argument('--TDV_fidelity', type=float, default=100)
    parser.add_argument('--TDV_offset', type=float, default=0.1)
    parser.add_argument('--TDV_iteration_number', type=int, default=30)
    parser.add_argument('--TDV_weight', type=float, default=1)
    ##  Sparse deconvolution parameters
    parser.add_argument('--Rolling_ball_radius', type=float, default=30)
    parser.add_argument('--Rolling_ball_paraboloid_flag', type=bool, default=True)
    parser.add_argument('--Emission_wavelength', type=float, default=608e-9)   
    parser.add_argument('--Raw_pixel_size', type=float, default=15e-9)   
    parser.add_argument('--Sparse_NA', type=float, default=6)
    parser.add_argument('--Sparse_iteration_number', type=int, default=20)
    parser.add_argument('--Sparse_offset', type=float, default=0.05)    
    ##
    args = parser.parse_args()
    return args


def default_config_TDV_FM_training():
    parser = argparse.ArgumentParser(description='TDV-FM training ... ')
    ##
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda')
     ##  FM parameters    
    parser.add_argument('--FM_raw_data_folder_path', type=str, default='', help='FM raw data folder path')
    parser.add_argument('--FM_raw_data_file_path', type=str, default='', help='FM raw data file path')   
    parser.add_argument('--FM_Raw_pixel_size', type=float, default=65e-9)
    parser.add_argument('--FM_emission_NA', type=float, default=1.4, help='emission numerical aperture')  
    parser.add_argument('--FM_emission_wavelength', type=float, default=525e-9)    
    parser.add_argument('--FM_wiener_parameter', type=float, default=0.2, help='wiener_sigma = 1/SNR')
    ##  Dataset parameters
    parser.add_argument('--Dataset_Z_average_number', type=int, default=20)
    parser.add_argument('--Dataset_Z_skip_number', type=int, default=5)
    parser.add_argument('--Dataset_XY_Poisson_noise_level', type=float, default=300)
    parser.add_argument('--Dataset_XY_Gaussian_noise_level', type=float, default=300)
    parser.add_argument('--Dataset_XY_margin_size', type=int, default=16)
    parser.add_argument('--Dataset_XY_block_size', type=int, default=256)
    parser.add_argument('--Dataset_XY_block_interval', type=int, default=128)
    parser.add_argument('--Dataset_XY_overexposure_ratio', type=float, default=0.5, help='The proportion of pixels with Max-normalization gray value > 0.9')
    parser.add_argument('--Dataset_XY_minimal_heterogeneity', type=float, default=0.6, help='Minimal standard deviation')
    parser.add_argument('--Dataset_XY_minimal_contrast', type=float, default=0.3, help='Minimal range')
    ##  TDV-DNN training parameters    
    parser.add_argument('--Train_batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--Train_epoch_number', type=int, default=50, help='Epoch number')
    parser.add_argument('--Train_learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--Train_learning_rate_decay_ratio', type=float, default=0.9, help='Learning rate decay ratio')
    parser.add_argument('--Train_learning_rate_minimum', type=float, default=1e-6, help='Learning rate minimum')
    parser.add_argument('--Train_abnormal_factor', type=float, default=3, help='Skip threshold (mean + factor*std) for abnormal samples')
    parser.add_argument('--Train_SSIM_MSE_ratio', type=float, default=0.2, help='SSIM/MSE ratio for loss function')
    parser.add_argument('--Train_Max_reinitialization_number', type=int, default=5, help='Max reinitialization number')
    parser.add_argument('--Train_sample', type=str, default='Actin', help='Sample name')
    parser.add_argument('--Train_system', type=str, default='FM', help='System name')
    parser.add_argument('--TDV_mode', type=str, default='train', help='DNN mode')
    ##
    args = parser.parse_args()
    return args


def default_config_Hybrid_WFM():
    parser = argparse.ArgumentParser(description='Hybrid-WFM reconstruction ... ')
    ##
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda')
    parser.add_argument('--WFM_raw_data_path', type=str, default='', help='WFM raw data file path')
    ##  BF-SIM parameters    
    parser.add_argument('--WFM_BF_flag', type=bool, default=True)
    parser.add_argument('--WFM_BF_offset', type=float, default=100, help='Camera offset value')    
    parser.add_argument('--WFM_BF_depth_infocus', type=int, default=4)
    parser.add_argument('--WFM_BF_depth_outfocus', type=int, default=50)
    parser.add_argument('--WFM_BF_PSF_path', type=str, default='None')
    parser.add_argument('--FM_Raw_pixel_size', type=float, default=65e-9)    
    parser.add_argument('--FM_emission_wavelength', type=float, default=525e-9)   
    parser.add_argument('--FM_emission_NA', type=float, default=1.4)
    ##  Hessian denoising parameters
    parser.add_argument('--Hessian_iteration_number', type=int, default=100)
    parser.add_argument('--Hessian_fidelity', type=float, default=100)
    parser.add_argument('--Hessian_Z_continuity', type=float, default=0)
    parser.add_argument('--Hessian_Z_rolling_window_size', type=int, default=100)
    ##  TDV denoising parameters    
    parser.add_argument('--TDV_mode', type=str, default='inference')
    parser.add_argument('--TDV_step_size', type=float, default=None, help='TDV step size')
    parser.add_argument('--TDV_model_path', type=str, default='./src_Hybrid/src_model/TDV_WFM_Peroxisome.pth')
    parser.add_argument('--TDV_fidelity', type=float, default=100)
    parser.add_argument('--TDV_offset', type=float, default=0.1)
    parser.add_argument('--TDV_iteration_number', type=int, default=30)
    parser.add_argument('--TDV_weight', type=float, default=1)    ##  Sparse deconvolution parameters
    
    parser.add_argument('--Sparse_NA', type=float, default=1)
    parser.add_argument('--Sparse_iteration_number', type=int, default=20)
    parser.add_argument('--Sparse_offset', type=float, default=0.1)    
    ##
    args = parser.parse_args()
    return args


def default_config_TDV_WFM_training():
    parser = argparse.ArgumentParser(description='TDV-WFM training ... ')
    ##
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda')
     ##  FM parameters    
    parser.add_argument('--FM_raw_data_folder_path', type=str, default='', help='FM raw data folder path')
    parser.add_argument('--FM_raw_data_file_path', type=str, default='', help='FM raw data file path')   
    parser.add_argument('--FM_Raw_pixel_size', type=float, default=65e-9)
    parser.add_argument('--FM_emission_NA', type=float, default=1.4, help='emission numerical aperture')  
    parser.add_argument('--FM_emission_wavelength', type=float, default=525e-9)    
    parser.add_argument('--FM_wiener_parameter', type=float, default=0.2, help='wiener_sigma = 1/SNR')
    ##  WFM parameters
    parser.add_argument('--WFM_BF_flag', type=bool, default=True)
    parser.add_argument('--WFM_BF_offset', type=float, default=100, help='Camera offset value')    
    parser.add_argument('--WFM_BF_depth_infocus', type=int, default=4)
    parser.add_argument('--WFM_BF_depth_outfocus', type=int, default=50)
    parser.add_argument('--WFM_BF_PSF_path', type=str, default='./src_Hybrid/src_Optics/BF_PSF/BF_PSF_WL525_PS65.tif')
    ##  Dataset parameters
    parser.add_argument('--Dataset_Z_average_number', type=int, default=20)
    parser.add_argument('--Dataset_Z_skip_number', type=int, default=5)
    parser.add_argument('--Dataset_XY_Poisson_noise_level', type=float, default=80)
    parser.add_argument('--Dataset_XY_Gaussian_noise_level', type=float, default=80)
    parser.add_argument('--Dataset_XY_margin_size', type=int, default=16)
    parser.add_argument('--Dataset_XY_block_size', type=int, default=256)
    parser.add_argument('--Dataset_XY_block_interval', type=int, default=128)
    parser.add_argument('--Dataset_XY_overexposure_ratio', type=float, default=0.5, help='The proportion of pixels with Max-normalization gray value > 0.9')
    parser.add_argument('--Dataset_XY_minimal_heterogeneity', type=float, default=0.6, help='Minimal standard deviation')
    parser.add_argument('--Dataset_XY_minimal_contrast', type=float, default=0.3, help='Minimal range')
    ##  TDV-DNN training parameters    
    parser.add_argument('--Train_batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--Train_epoch_number', type=int, default=50, help='Epoch number')
    parser.add_argument('--Train_learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--Train_learning_rate_decay_ratio', type=float, default=0.9, help='Learning rate decay ratio')
    parser.add_argument('--Train_learning_rate_minimum', type=float, default=1e-6, help='Learning rate minimum')
    parser.add_argument('--Train_abnormal_factor', type=float, default=3, help='Skip threshold (mean + factor*std) for abnormal samples')
    parser.add_argument('--Train_SSIM_MSE_ratio', type=float, default=0.2, help='SSIM/MSE ratio for loss function')
    parser.add_argument('--Train_Max_reinitialization_number', type=int, default=5, help='Max reinitialization number')
    parser.add_argument('--Train_sample', type=str, default='Actin', help='Sample name')
    parser.add_argument('--Train_system', type=str, default='FM', help='System name')
    parser.add_argument('--TDV_mode', type=str, default='train', help='DNN mode')
    ##
    args = parser.parse_args()
    return args