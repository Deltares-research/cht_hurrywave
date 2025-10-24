% Make Hurrywave model
clear
close all
clc
addpath('p:\11204750-hurrywave\02_modelling\SWIVT_cases\f998dcs13\02_scripts\')
addpath('c:\OET\matlab\')

try

    oetsettings;

catch

    disp('FAILED TO LOAD OPENEARTHTOOLS');

end

fname = 'DCSM-feb2023-v1_c';
destout = 'p:\11204750-hurrywave\02_modelling\SWIVT_cases\f998dcs13\01_data\matlab_generated\';


zs = 0;

trefs = datenum(2013,12,03,0,0,0);
tstops = datenum(2013,12,08,0,0,0);
%tstops = datenum(2023,1,30,0,0,0);


% % --- bc
[spec_w_1, s1] = find_spec_characteristics_from_SWIVT_2D('p:\11204750-hurrywave\01_data\00_SWIVT\sessions\session001\SWAN4131AB\f998dcs13001_000\f998dcs13001\model_io\f998dcs13001.bnd');

points = [s1.lon' s1.lat'];

% create bnd file with a dummy coordinate
fid = fopen([destout, 'hurrywave.bnd'],'wt');
fprintf(fid,'%.2f\t%.2f\n', points');
fclose(fid);

time_sec= (s1.time' - s1.time(1))*24*3600;

create_bnd(destout, time_sec, spec_w_1.hs, spec_w_1.Tps, spec_w_1.meanDir, spec_w_1.dspr, spec_w_1.hs*0+zs);
