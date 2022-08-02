%{
    Liz Teel 2021-02-17
    

    Code to extract the EEG features for the machine learning analysis.

    Spectral, functional connectivity, & permutation entropy features
    included here. Graph theory features are in Step 3, as they require
    an additional step (threshold sweep) before calculating. 
  
    Adapted from Yacine Mahdid July 10
        -generate_features.m
        -EEG_pain_detection -> Milestones -> 3_first_draft_paper
   
%}

%% Seting up the variables
clear;
setup_painexperiments; % name of your set-up experiments file

addpath(genpath('/Users/elizabethteel/Desktop/Matlab_BIAPTFunctions')); %adding the NeuroAlgo library so matlab can find the necessary functions called

% Create the input/output directories
in_path = '/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/eeg_pain_data';
output_path = '/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results';

% Create the output directories for all the analyses
power_output_path = mkdir_if_not_exist(output_path,'Power');
wpli_output_path = mkdir_if_not_exist(output_path,'wPLI');
dpli_output_path = mkdir_if_not_exist(output_path,'dPLI');
pe_output_path = mkdir_if_not_exist(output_path,'PE');
csv_output_path = mkdir_if_not_exist(output_path,'CSV');

%Create CSV files to output errors for checking later
OUT_FILE = strcat(csv_output_path,filesep, "features_%s.csv");
OUT_LOG = strcat(output_path, "errors_%s.csv"); %log of indiviudals who were skipped to troubleshoot

% Adding in the channel locations for the DSI-24
data = load('full_headset_location.mat');
max_location = data.max_location; %channel locations for max number channels (i.e. no missing channels)
max_regions = ["frontal", "frontal", "frontal", "frontal", "frontal", "frontal", "frontal", "temporal", "temporal", "temporal", "temporal", "central", "central", "central", "parietal", "parietal", "parietal", "occipital", "occipital"]; %channel regions for max number channels (i.e. no missing channels)

header = ["id", "group", "state"]; %name of headers that will be used in CSV file
bandpass_names = {'delta','theta', 'alpha', 'beta'}; %names of frequency bands
bandpass_freqs = {[1 4], [4 8], [8 13], [13 30]}; %ranges for each frequency band

p_id = {'HE007'};  % list of participants to analyze
states = {'_nopain.set', '_cold.set', '_hot2.set'};  % list of conditions to analyze

file_id = fopen(OUT_LOG,'w');
fclose(file_id);

%% Calculate Features

%store current number of participant processed
num_participant = 1;

for p = 1:length(p_id)
    
    participant = p_id{p};
    disp(participant);

    % participant variable init
    p_num = str2num(extractAfter(participant,"E"));
    is_healthy = contains(participant, 'HE');
    participant_path = strcat(in_path,filesep,participant);
    
    out_file_participant = sprintf(OUT_FILE,participant);
    write_header(out_file_participant, header, bandpass_names, max_location)
    
    % Iterate over all the files within each participant's folder
    for s = 1:length(states) %when testing/de-bugging, only using one condition
        
        filename = states{s};
        
        if strcmp(filename,'_nopain.set')
            statename='Baseline'; %creating a character name to display in command window
            statenumber = 1; %creating a numeric value for export to CSV file
        end    
        
        if strcmp(filename,'_nopain_edited.set')
            statename='Baseline'; 
            statenumber = 1; 
        end 
        
        if strcmp(filename,'_hot1.set')
            statename='Hot 1';
            statenumber = 2;
        end  
        
        if strcmp(filename,'_hot1_edited.set')
            statename='Hot 1';
            statenumber = 2;
        end   
        
        if strcmp(filename,'_cold.set')
            statename='Cold';
            statenumber = 3;
        end  
        
        if strcmp(filename,'_cold_edited.set')
            statename='Cold';
            statenumber = 3;
        end  
        
        if strcmp(filename,'_hot2.set')
            statename='Hot 2';
            statenumber = 4;
        end   
        
        if strcmp(filename,'_hot2_edited.set')
            statename='Hot 2';
            statenumber = 4;
        end   
        
        if strcmp(filename,'_covas.set')
            statename='COVAS Baseline';
            statenumber = 5;
        end   
        
        if strcmp(filename,'_covas_edited.set')
            statename='COVAS Baseline';
            statenumber = 5;
        end   
        
        disp(strcat("State: ", statename));
        
        %creating output directories to export figures/results from each analysis
        power_participant_output_path = mkdir_if_not_exist(power_output_path,strcat(participant,filesep,statename));
        wpli_participant_output_path = mkdir_if_not_exist(wpli_output_path,strcat(participant,filesep,statename));
        dpli_participant_output_path = mkdir_if_not_exist(dpli_output_path,strcat(participant,filesep,statename));
        pe_participant_output_path = mkdir_if_not_exist(pe_output_path,strcat(participant,filesep,statename));
        
        fullname = strcat(participant,filename);
     
            try
            	%Load file if available
                recording = load_set(fullname,participant_path);
                
            catch
                %Skip loop if file missing
                disp(sprintf("Skipping Participant Because File is Missing"));
                % Output info for skipped loops in log to check for errors
                disp(strcat("Problem with file: ", participant))
                file_id = fopen(OUT_LOG,'a'); 
                fprintf(file_id, strcat("Problem Loading File: ", participant,"\n"));
                fclose(file_id);
                continue
            end
        
            %% Calculate Features
            
            %Unpack parameters needed to caclulate features
            %these parameters are house in the "setup_painexperiement" code
            win_size = feature_params.general_param.win_size;
            step_size = feature_params.general_param.step_size;

            %Power Params
            time_bandwith_product = feature_params.spr_param.time_bandwith_product;
            number_tapers = feature_params.spr_param.number_tapers;

            %wPLI & dPLI Params
            number_surrogate = feature_params.pli_param.number_surrogate; % Number of surrogate wPLI to create
            p_value = feature_params.pli_param.p_value; % the p value to make our test on

            %Permutation Entropy Params
            embedding_dimension = feature_params.pe_param.embedding_dimension;
            time_lag = feature_params.pe_param.time_lag;

            features = [];
            for b_i = 1:length(bandpass_freqs)
            %for b_i = 3 %when testing/de-bugging, only using one frequency band
                bandpass = bandpass_freqs{b_i}; %pull out the frequency range to filter by
                name = bandpass_names{b_i}; %pull out the name of the frequency band for the dataset
                fprintf("Calculating Feature at %s\n",name);            
                
                %% Power per channels
                power_struct = na_topographic_distribution(recording, win_size, step_size, bandpass); %calculate power per channel
                powers = power_struct.data.power; %pull out the power matrix from the power structure
                power_location = power_struct.metadata.channels_location; %pull out the location file needed to identify missing channels
                pad_powers = pad_result(powers, power_location, max_location); %pad missing channels with NaN
                
                %save power structure
                power_struct_filename = strcat(power_participant_output_path,filesep, 'Power', name,'_structure.mat');
                save(power_struct_filename, 'power_struct');
                                             
                %% Peak Frequency
                result_sp = na_spectral_power(recording, win_size, time_bandwith_product, number_tapers, bandpass, step_size);  %calculate peak frequency
                peak_frequency = result_sp.data.peak_frequency'; %pull out peak frequency variable from power structure  
                
                %save peak frequency structure
                peakpower_struct_filename = strcat(power_participant_output_path,filesep, 'PeakPower', name,'_structure.mat');
                save(peakpower_struct_filename, 'result_sp');
                        
                %% wPLI
                result_wpli = na_wpli_dsi24(recording, bandpass, win_size, step_size, number_surrogate, p_value); %calculate wPLI
                wpli_location = result_wpli.metadata.channels_location;
                wpli_channels = struct2cell(result_wpli.metadata.channels_location); %get channel location to pad missing channels
                wpli_timeresolved_matrix = result_wpli.data.wpli; %pull out the time-resolved 3D matrix for wPLI
                avg_wpli = mean(result_wpli.data.wpli,3); %create single value for each channel at each window (average across row for data reduction purposes
                pad_avg_wpli = pad_result(avg_wpli, wpli_location, max_location); %pad missing channels with NaN          
                
                %save wPLI struct, wPLI matrix, and channel location to participant folder
                wpli_struct_filename = strcat(wpli_participant_output_path,filesep, 'wPLI', name,'_structure.mat');
                save(wpli_struct_filename, 'result_wpli');
                wpli_matrix_filename = strcat(wpli_participant_output_path,filesep, 'wPLI', name,'_matrix.mat');
                save(wpli_matrix_filename, 'wpli_timeresolved_matrix');
                wpli_channel_filename = strcat(wpli_participant_output_path,filesep, 'wPLI', name,'_channels.mat');
                save(wpli_channel_filename,'wpli_channels');
                       
                %% dPLI
                result_dpli = na_dpli_dsi24(recording, bandpass, win_size, step_size, number_surrogate, p_value); %calculate dPLI
                dpli_location = result_dpli.metadata.channels_location;
                dpli_channels = struct2cell(result_dpli.metadata.channels_location); %get channel location to pad missing channels
                dpli_timeresolved_matrix = result_dpli.data.dpli; %pull out the time-resolved 3D matrix for wPLI
                avg_dpli = mean(result_dpli.data.dpli,3); %create single value for each channel at each window (average across row for data reduction purposes
                pad_avg_dpli = pad_result(avg_dpli, dpli_location, max_location); %pad missing channels with NaN
                        
                %save dPLI structure, matrix, and channel location to participant folder
                dpli_struct_filename = strcat(dpli_participant_output_path,filesep, 'dPLI', name,'_structure.mat');
                save(dpli_struct_filename, 'result_dpli');
                dpli_matrix_filename = strcat(dpli_participant_output_path,filesep, 'dPLI', name,'_matrix.mat');
                save(dpli_matrix_filename, 'dpli_timeresolved_matrix');
                dpli_channel_filename = strcat(dpli_participant_output_path,filesep, 'dPLI', name,'_channels.mat');
                save(dpli_channel_filename,'dpli_channels');
                
                %% PE
                pe_struct = na_permutation_entropy(recording, bandpass, win_size , step_size, embedding_dimension, time_lag); %calculate permutation entropy
                pe_location = pe_struct.metadata.channels_location; %get channel location to pad missing channels
                pe = pe_struct.data.permutation_entropy; %pull out pe values
                pad_pe = pad_result(pe, pe_location, max_location);%pad missing channels with NaN
                norm_pe = pe_struct.data.normalized_permutation_entropy; %pull out normalized pe values
                pad_norm_pe = pad_result(norm_pe, pe_location, max_location); %pad missing channels with NaN
                
                %save PE structure
                pe_struct_filename = strcat(pe_participant_output_path,filesep, 'PE', name,'_structure.mat');
                save(pe_struct_filename, 'pe_struct');
                
                %concetnate all features
                features = horzcat(features, pad_powers, peak_frequency, pad_avg_wpli, pad_avg_dpli, pad_pe, pad_norm_pe);
                
            end
            
            
       %Write the features to file
       [num_window, ~] = size(features);
       for w = 1:num_window
           row = features(w,:);
           dlmwrite(out_file_participant, [p_num, is_healthy, statenumber, row], '-append');
       end


    end
    
end

%% Functions Needed to Run Code

function write_header(OUT_FILE, header, bandpass_names, max_location)
    %% Create data set
    % Overwrite the file
    
    delete(OUT_FILE);

    % Write header to the features file
    file_id = fopen(OUT_FILE,'w');
    for i = 1:length(header)
        fprintf(file_id,'%s,', header(i));
    end

    % Write the rest of the header for the channel-wise power
    for b_i = 1:length(bandpass_names)
        bandpass_name = bandpass_names{b_i};
        
        write_feature_vector(file_id, max_location, bandpass_name, "power") %power         

        %Peak Frequency
        feature_label = sprintf("peak_freq_%s",bandpass_name);
        fprintf(file_id, '%s,',lower(feature_label));

        write_feature_vector(file_id, max_location, bandpass_name, "wpli") %wPLI 
        write_feature_vector(file_id, max_location, bandpass_name, "dpli") %dPLI 
        write_feature_vector(file_id, max_location, bandpass_name, "pe") %PE
        write_feature_vector(file_id, max_location, bandpass_name, "norm_pe") %normalized PE
        
    end

    fprintf(file_id,"\n");
    fclose(file_id);
end

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
