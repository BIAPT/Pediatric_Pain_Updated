%{
    Liz Teel 2021-05-13
    
    Calculating the graph theory outcomes of interest for the project
    Using thresholds calculated in step 2 

    * Warning: This experiment use the setup_experiments.m script to 
    load variables. Therefore if you are trying to edit this code and you
    don't know what a variable means, take a look at the setup_experiments.m
    script. It contains all the parameters for a project.
%}

%% Seting up the variables
clear;
setup_painexperiments; % name of your set-up experiments file

addpath(genpath('/Users/elizabethteel/Desktop/Matlab_BIAPTFunctions')); %adding the NeuroAlgo library so matlab can find the necessary functions called

% Create the input/output directories
in_path = '/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/EEG Features/wPLI/';
output_path = '/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/EEG Features/';

% Create the output directories for all the analyses
hubs_output_path = mkdir_if_not_exist(output_path,'Binary Graph');
csv_output_path = mkdir_if_not_exist(hubs_output_path,'CSV');

%Create CSV files to output errors for checking later
OUT_FILE = strcat(csv_output_path,filesep, "binarygraph_features_%s.csv");
OUT_LOG = strcat(output_path, "errors_%s.csv"); %log of indiviudals who were skipped to troubleshoot

% Adding in the channel locations for the DSI-24
data = load('full_headset_location.mat');
max_location = data.max_location; %channel locations for max number channels (i.e. no missing channels)
max_regions = ["frontal", "frontal", "frontal", "frontal", "frontal", "frontal", "frontal", "temporal", "temporal", "temporal", "temporal", "central", "central", "central", "parietal", "parietal", "parietal", "occipital", "occipital"]; %channel regions for max number channels (i.e. no missing channels)

header = ["id", "group", "state"]; %name of headers that will be used in CSV file
bandpass_names = {'delta','theta', 'alpha', 'beta'}; %names of frequency bands
bandpass_freqs = {[1 4], [4 8], [8 13], [13 30]}; %ranges for each frequency band

states = {'Baseline', 'Cold'};

modename = "wPLI";

file_id = fopen(OUT_LOG,'w');
fclose(file_id);


%% Calculate Features

%store current number of participant processed
num_participant = 1;

directories = dir(in_path);
for p = 9
%for p = 16:length(directories)

    folder = directories(p);
    participant = folder.name; %get the name of the participant being analyzed
    disp(participant); %display participant's name in command window

    % participant variable init
    p_num = str2num(extractAfter(participant,"E"));
    is_healthy = contains(participant, 'HE');
    participant_path = strcat(in_path,filesep,participant);
        
    % Iterate over all the files within each participant's folder
    features_allstates = [];
    %for s = 2
    for s = 1:length(states) %when testing/de-bugging, only using one condition
        
        filename = states{s};
        
        if strcmp(filename,'Baseline')
            statename='Baseline'; %creating a character name to display in command window
            statenumber = 1; %creating a numeric value for export to CSV file
        end         
        
        if strcmp(filename,'Cold')
            statename='Cold';
            statenumber = 3;
        end        
        
        disp(strcat("State: ", statename));
        
        features_allfreq = [];
        features_singlefreq = [];
            %for b_i = 1
            for b_i = 1:length(bandpass_names) %loop through all frequency bands          

                bandname = bandpass_names(b_i); %determine frequency band
                disp(bandname); %display frequency band name in display window
                
                %Import wPLI matrix data
                data_path = strcat(in_path,filesep,participant,filesep,statename,filesep,modename,bandname,'_structure.mat');
                
                %creating output directories to export figures/results from each analysis
                hubs_participant_output_path = mkdir_if_not_exist(hubs_output_path,strcat(participant,filesep,statename));

                %% Load Data
                try
                    data = load(data_path);
                    wpli_matrix = data.result_wpli.data.avg_wpli;
                    timeresolved_matrix = data.result_wpli.data.wpli;
                    channels_location = data.result_wpli.metadata.channels_location;

                    catch
                        %Skip loop if file missing
                        disp(strcat("Problem with file: ", participant));
                    continue
                end          

                %% Calculating graph theory outcomes

                %Initialize null networks
                number_null_network = 100;
                [num_window, number_channels, ~] = size(timeresolved_matrix);

                null_networks = zeros(number_null_network, number_channels, number_channels);
                
                
                %initiate matrix to store calculations during each loop
                des_participant = zeros(num_window,1);
                des_group = zeros(num_window,1);
                des_state = zeros(num_window,1);
                des_num_window = zeros(num_window,1);
                
                %initiate matrix to store calculations during each loop
                wpli_geff_norm = zeros(num_window,1);
                wpli_pathlength_norm = zeros(num_window,1); 
                wpli_cluster_norm = zeros(num_window,1);
                wpli_bsw = zeros(num_window,1); 
                wpli_mod = zeros(num_window,1); 
                wpli_frontal_strength = zeros(num_window,1); 
                wpli_temporal_strength = zeros(num_window,1); 
                wpli_central_strength = zeros(num_window,1); 
                wpli_parietal_strength = zeros(num_window,1); 
                wpli_occipital_strength = zeros(num_window,1); 
                
                for s_i = 1:num_window %loop through all windows in the wPLI matrix
                    
                    % this participant has no functional connectivity at
                    % this conditon, skipping loop to avoid error
                    if contains(participant, 'HE005') && statenumber == 3 && b_i == 1 && s_i == 1
             
                        continue
                    end
                    
                    % this participant has no functional connectivity at
                    % this conditon, skipping loop to avoid error
                    if contains(participant, 'HE006') && statenumber == 1 && b_i == 1 && s_i == 3
             
                        continue
                    end
                        
                    disp(strcat(modename," Window #",string(s_i)));
                    wpli_segment = squeeze(timeresolved_matrix(s_i,:,:));
                    
                    % Filter the channels location to match the filtered motifs
                    %[wpli_2dorder, r_labels, r_regions, r_location] = reorder_channels(wpli_segment, channels_location,'biapt_dsi24.csv');
                    
                    %Set threshold based on participant's minimally spanning map
                    thresholdtable = readtable(strcat('/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Thresholds/thresholds_',participant,'.csv'));
                    current_threshold = thresholdtable{1,b_i};
                    
                    % Binarize the network
                    t_network = threshold_matrix(wpli_segment, current_threshold); %the threshold is here graph_param.threshold(p,t)
                    b_network = binarize_matrix(t_network);

                    % Iterate and populate the null network matrix for all binary matrices
                    parfor i = 1:number_null_network 
                        disp(strcat("Random network #",string(i)));
                        [null_matrix,~] = randmio_und(b_network,10);
                        null_networks(i,:,:) = null_matrix; 
                    end

                    des_participant(s_i) = p_num;
                    des_group(s_i) = is_healthy;
                    des_state(s_i) = statenumber;
                    des_num_window(s_i) = s_i;

                    %calculating global efficiency and path length
                    [~, norm_geff, ~, norm_lambda] = binary_global_efficiency(b_network,null_networks);
                    wpli_geff_norm(s_i) = norm_geff;
                    wpli_pathlength_norm(s_i) = norm_lambda;

                    %calculating clustering coefficient 
                    [~, norm_cluster] = undirected_binary_clustering_coefficient(b_network,null_networks);
                    wpli_cluster_norm(s_i) = norm_cluster;

                    %calculating small-worldness
                    [b_small_worldness] = undirected_binary_small_worldness(b_network,null_networks);
                    wpli_bsw(s_i) = b_small_worldness;

                    [~,mod] = community_louvain(b_network,1); % find community, modularity
                    wpli_mod(s_i) = mod;

                    % calculate node strength (not degree b/c that is problematic with fully connected graph;
                    node_strength = strengths_und(b_network);
                    pad_strength = pad_result(node_strength, channels_location, max_location); %pad missing channels with NaN
                    
                    %frontal ROI average
                    frontal_array = pad_strength(1:7);
                    frontal_strength = nanmean(frontal_array);
                    wpli_frontal_strength(s_i) = frontal_strength;
                    
                    %Temporal ROI average
                    temporal_array = pad_strength(8:11);
                    temporal_strength = nanmean(temporal_array);
                    wpli_temporal_strength(s_i) = temporal_strength;
                    
                    %Central ROI average
                    central_array = pad_strength(12:14);
                    central_strength = nanmean(central_array);
                    wpli_central_strength(s_i) = central_strength;
                    
                    %Parietal ROI average
                    parietal_array = pad_strength(15:17);
                    parietal_strength = nanmean(parietal_array);
                    wpli_parietal_strength(s_i) = parietal_strength;
                    
                    %Occipital ROI average
                    occipital_array = pad_strength(18:19);
                    occipital_strength = nanmean(occipital_array);
                    wpli_occipital_strength(s_i) = occipital_strength;
                        
                end
                    
                %create structure and store all dPLI graph theory outcomes
                result_wpligraph.geff_norm = wpli_geff_norm; %global efficiency
                result_wpligraph.pathlength_norm = wpli_pathlength_norm; %path length
                result_wpligraph.cluster_norm = wpli_cluster_norm; %clustering coefficient
                result_wpligraph.bsw = wpli_bsw;  %small  worldness
                result_wpligraph.mod = wpli_mod; %modularity (note: doesn't need to be normalized against random networks)
                result_wpligraph.frontal_strength = wpli_frontal_strength; % average frontal degree
                result_wpligraph.temporal_strength = wpli_temporal_strength; % average temporal degree
                result_wpligraph.central_strength = wpli_central_strength; % average central degree
                result_wpligraph.parietal_strength = wpli_parietal_strength; % average parietal degree
                result_wpligraph.occipital_strength = wpli_occipital_strength; % average frontal degree

                %save dPLI graph theory structure to participant folder
                wpligraph_struct_filename = strcat(hubs_participant_output_path,filesep,'BinaryGraph_', modename, '_', bandname,'_structure.mat');
                save(wpligraph_struct_filename, 'result_wpligraph');

                features_singlefreq = horzcat(wpli_pathlength_norm, wpli_cluster_norm, wpli_bsw, wpli_mod, wpli_frontal_strength, wpli_temporal_strength, wpli_central_strength, wpli_parietal_strength, wpli_occipital_strength); % concatenate variables in a single frequency band
                features_allfreq = horzcat(features_allfreq, features_singlefreq); %concatenate variables for all frequency bands together  
                
            end
            
            features_descriptives = horzcat(des_participant, des_group, des_state, des_num_window, features_allfreq); % concatenate the descriptive info with the graph theory outcomes
            features_allstates = vertcat(features_allstates, features_descriptives);
    end
    
    column_names = {'id', 'group', 'state', 'window_number', 'delta_pathlength', 'delta_cluster', 'delta_bsw', 'delta_mod', 'delta_frontal_strength', 'delta_temporal_strength', 'delta_central_strength', 'delta_parietal_strength', 'delta_occipital_strength', 'theta_pathlength', 'theta_cluster', 'theta_bsw', 'theta_mod', 'theta_frontal_strength', 'theta_temporal_strength', 'theta_central_strength', 'theta_parietal_strength', 'theta_occipital_strength', 'alpha_pathlength', 'alpha_cluster', 'alpha_bsw', 'alpha_mod', 'alpha_frontal_strength', 'alpha_temporal_strength', 'alpha_central_strength', 'alpha_parietal_strength', 'alpha_occipital_strength','beta_pathlength', 'beta_cluster', 'beta_bsw', 'beta_mod', 'beta_frontal_strength', 'beta_temporal_strength', 'beta_central_strength', 'beta_parietal_strength', 'beta_occipital_strength'};
    T = array2table(features_allstates, 'VariableNames', column_names);
    
    graph_output = strcat(csv_output_path,filesep,'binarygraph_features_',participant,'.csv');
    writetable(T,graph_output);
end

%% Functions Needed to Run Code

function [pad_vector] = pad_result(vector, location, max_location)
% PAD_RESULT : will pad the result with the channels it has missing
% This is used to have a normalized power that has the same number of
% channels for all values. Will put NaN where a channel is missing.

    [num_window,~] = size(vector);
    pad_vector = zeros(num_window, length(max_location));
    for w = 1:num_window
        for l = 1:length(max_location)
            label = max_location(l).labels;

            % The channel may not be in the same order as location
            index = get_label_index(label, location);

            if (index == 0)
                pad_vector(w,l) = NaN; 
            else
                pad_vector(w,l) = vector(w, index);
            end
        end
    end
end


% Function to check if a label is present in a given location
function [label_index] = get_label_index(label, location)
    label_index = 0;
    for i = 1:length(location)
       if(strcmp(label,location(i).labels))
          label_index = i;
          return
       end
    end
end

function write_feature_vector(file_id, max_location, bandpass_name, feature_type)
    for c = 1:length(max_location)
        channel_label = max_location(c).labels;
        feature_label = sprintf("%s_%s_%s",channel_label, bandpass_name, feature_type);
        fprintf(file_id,'%s,', lower(feature_label)); 
    end
end