% Liz Teel 05-08-2021
% sweep through pain data to find the minimally connected map
% sweeping through baseline files only, but seperate thresholds will be generated for each frequency band

% modifed from: Yacine Mahdid 2019-12-12 and Danielle Nadin 2020-02-25 

%% Set-up Variables 

clear;
setup_painexperiments; % see this file to edit the experiments

addpath(genpath('/Users/elizabethteel/Desktop/Matlab_BIAPTFunctions')); %adding the NeuroAlgo library so matlab can find the necessary functions called

% Create the input/output directories
in_path = '/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results';
output_path = '/Users/elizabethteel/Desktop/Research/Projects/Musculoskeletal Pain/Results/Thresholds';

%Create CSV files to output errors for checking later
OUT_FILE = strcat(output_path, filesep, "thresholds_%s.csv");
OUT_LOG = strcat(output_path, filesep, "thresholderrors_%s.csv"); %log of indiviudals who were skipped to troubleshoot

mode = ["wPLI"]; %names of functional connectivity analyses to assess
bandpass_names = ["delta", "theta", "alpha", "beta"]; %names of frequency bands

p_id = {'ME046'};  % list of participants to analyze
states = {'_nopain.set'};  %list of conditions to analyze


%% Create Loops to Analyze Baseline Files for All Participants at Each Frequency Band

num_participant = 1;

for m = 1:length(mode)
    
    modename = mode{m};
    display(modename);
    
    %mode specific input  path
    mode_path = strcat(in_path,filesep,mode);
    
    for p = 1:length(p_id)

        participant = p_id{p};
        disp(participant);

        % participant variable init
        p_num = str2num(extractAfter(participant,"E"));
        is_healthy = contains(participant, 'HE');
        
        %setting up input path for data depending on which baseline condition was used
        if strcmp(states, '_nopain.set')
            participant_path = strcat(mode_path,filesep,participant,filesep,'Baseline');
        elseif strcmp(states, '_nopain_edited.set')
            participant_path = strcat(mode_path,filesep,participant,filesep,'Baseline');
        elseif strcmp(states, '_covas.set')
            participant_path = strcat(mode_path,filesep,participant,filesep,'COVAS Baseline');
        elseif strcmp(states, '_covas_edited.set')
            participant_path = strcat(mode_path,filesep,participant,filesep,'COVAS Baseline');
        end
        
        out_file_participant = sprintf(OUT_FILE,participant);

        result_threshold = struct();        
        result_threshold.threshold = zeros(1,length(bandpass_names));

        for b_i = 1:length(bandpass_names)

            bandname = bandpass_names(b_i);
            disp(bandname);

            %Import pli data
            %Only want data from CogTest1- in filename and not a loop
            pli_input_path = strcat(participant_path,filesep,modename,bandname,'_structure.mat');

            %% Load Data
            try
                data = load(pli_input_path);
                if strcmp(modename, 'dPLI')
                    pli_matrix = data.result_dpli.data.avg_dpli;
                    channels_location = data.result_dpli.metadata.channels_location;
                elseif strcmp(modename, 'wPLI')
                    pli_matrix = data.result_wpli.data.avg_wpli;
                    channels_location = data.result_wpli.metadata.channels_location;
                elseif strcmp(mode, 'AEC')
                    pli_matrix = data.result_aec.aec;
                    [hight, width, len] = size(pli_matrix);
                    temp = zeros(hight, width);
                    for i=1:hight
                        for j=1:width
                            for k=1:len
                                temp(i,j) = temp(i,j) + pli_matrix(i,j,k);
                            end
                        end
                    end
                    temp = temp/len;
                    pli_matrix=temp;
                    channels_location = data.result_aec.labels;
                 end

            catch
                % Output info for skipped loops in log to check for errors
                disp(strcat("Problem with file: ", participant))
                file_id = fopen(OUT_LOG,'a'); 
                fprintf(file_id, strcat("Problem Loading File: ", participant,modename));
                fclose(file_id);
                continue
            end

            %% Perform Threshold Sweep

            %setting parameters for values to sweep the data at, more connected to less connected
            sweep_param.range = 0.90:-0.01:0.01;

            %loop through thresholds
            for j = 1:length(sweep_param.range) 
                current_threshold = sweep_param.range(j);
                disp(strcat("Doing the threshold : ", string(current_threshold)));

            % Thresholding and binarization using the current threshold
                t_network = threshold_matrix(pli_matrix, current_threshold);
                b_network = binarize_matrix(t_network);

            % check if the binary network is disconnected
            % Here our binary network (b_network) is a weight matrix but also an
            % adjacency matrix.
                distance = distance_bin(b_network);

            % Here we check if there is one node that is disconnected
                if(sum(isinf(distance(:))))
                    disp(strcat("Final threshold: ", string(sweep_param.range(j-1))));
                    break; 
                end
                
                final_threshold = current_threshold;
                result_threshold.threshold(1,b_i) = final_threshold;

            end

        end

    header = {'delta_threshold', 'theta_threshold', 'alpha_threshold', 'beta_threshold'}; %name of headers that will be used in CSV file
    row = result_threshold.threshold;

    table = array2table(row,'VariableNames', header);

    writetable(table, out_file_participant);

    end
end
