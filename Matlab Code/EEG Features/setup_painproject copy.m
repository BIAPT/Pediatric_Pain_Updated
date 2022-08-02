%{
    Liz Teel 2021-02-17

    Modifed from Yacine and Danielle's code

    This script is run to make sure that the project uses the right
    code by downloading the right libraries and adding to path the
    right folders.
%}

% Setting up the paths
[folder, name, ext] = fileparts(mfilename('fullpath'));
library_path = strcat(folder,filesep,'library');
utils_path = strcat(folder,filesep,'utils');
temp_path = strcat(folder,filesep,'temp.zip');

%Add that folder plus all subfolders to the path.
disp("Adding resource to path.")
addpath(genpath(library_path));
addpath(genpath(utils_path));
