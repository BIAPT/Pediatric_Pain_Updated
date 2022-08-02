%{
    Liz Teel 05/08/2021

    Setting up the general parameters neede for the pain projects analysis.
    The code is called during the step_1_generatefeatures file.

    Modified from:

    Danielle Nadin 2020-04-27
    Setup experimental variables for analyzing tDCS scalp data. 
    Modified from Yacine's motif analysis augmented code. 
%}

% General Experiment Variables
settings = load_settings();
raw_data_path = settings.raw_data_path;
output_path = settings.output_path;

spectro_params = struct();
spectro_params.tapers = [2 3];
spectro_params.Fs = 300;
spectro_params.fpass = [1 50];
spectro_params.trialave = 1;

% Structure for the Features of All Analyses Run
feature_params = struct();

% General Features (used for all analyses which require them)
feature_params.general_param.win_size = 10;  
feature_params.general_param.step_size = 10; 

%Spectrogram Features
feature_params.spr_param.window_size = 10;
feature_params.spr_param.time_bandwith_product = 2;
feature_params.spr_param.number_tapers = 3;
feature_params.spr_param.spectrum_window_size = 3; % in seconds
feature_params.spr_param.step_size = 10; % in seconds

% wPLI/dPLI Variables
feature_params.pli_param.number_surrogate = 20; % Number of surrogate wPLI to create
feature_params.pli_param.p_value = 0.05; % the p value to make our test on 

% Permutation Entropy
feature_params.pe_param.embedding_dimension = 5;
feature_params.pe_param.time_lag = 4;

% The other parameters are recording dependant and will be dynamically generated
