
clear all
close all
clc
pix = 64;
max_angle = 5;
is_mixangle = true;
is_noise = true;
is_boulder = false;
if is_noise
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem(noisy)_ver2";
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem(noisy)"; % 9/5
    if is_mixangle;
        folder_name = string(pix)+"pix_(0-"+string(max_angle)+"deg)_dem(noisy)"; % 9/13
    end
else
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem_ver2";
end
mkdir(folder_name);
mkdir(folder_name,'image');
mkdir(folder_name,'label');
mkdir(folder_name,'model');
% 7680:1:16640
for i=0:1:7680
   if is_mixangle;
       angle = round((max_angle)*rand(1)) % スロープ角(1~5)
   end
   f = fractal_terrain_generation(i,0,pix,angle,folder_name,is_noise,is_boulder);
%   f = base_DEM(i,2,pix,angle,folder_name,is_noise);

   % 引数　i:イテレーション, mode, pix:解像度, angle:斜度, folder_name:セーブディレクトリ,
   % is_noise:パーリンノイズ付加の有無
   % モード0:LOGデータ保存、モード1は動画生成、モード2：3次元プロット
end

function f = count(i)
    f = i^2;
    iter_data =  zeros(1,1); 
    filenum = string(i)
    filename = "crater/data_"+filenum
    save(filename,'iter_data')
    
end