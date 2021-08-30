clear all
close all
clc
origin = fopen('DTM_MAP_02_N00E006S03E009SC.img','r');
data = fread(origin,[12288,12288],'int16','l');
size(data)
colormap('jet');
imagesc(data);colorbar;

