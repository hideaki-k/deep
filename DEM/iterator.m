
clear all
close all
clc
pix = 64;
max_angle = 5;
angle = 0;
is_mixangle = true;
is_noise = true;
is_boulder = false;

addpath(' C:\Users\aki\Documents\GitHub\deep\DEM\terrain_generation');
%10/25 NC研　評価用データセット
% 斜度、ノイズ、クレータ数、クレータ半径を指定可能
is_evaluate = true;
evaluate_angle = 0;

if is_evaluate %評価用
    folder_name = string(pix)+"pix_("+string(evaluate_angle)+"deg)_dem(noisy)_evaluate";


elseif is_noise
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem(noisy)_ver2";
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem(noisy)"; % 9/5
    if is_mixangle; %アングル混在
        folder_name = string(pix)+"pix_(0-"+string(max_angle)+"deg)_dem(noisy)"; % 9/13
    else %アングル固定
        folder_name = string(pix)+"pix_(0-"+string(angle)+"deg)_dem(noisy)";
    end
else
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem_ver2";
end
mkdir(folder_name)
mkdir(folder_name,'image')
mkdir(folder_name,'label');
mkdir(folder_name,'model');
% 7680:1:16640
for i=512:1:640
   if is_evaluate
       evaluate_terrain_generation(i,2,pix,evaluate_angle,folder_name,is_noise,is_boulder);
    
    
   elseif is_mixangle
       angle = round((max_angle)*rand(1)) % スロープ角(1~5)
       f = fractal_terrain_generation(i,0,pix,angle,folder_name,is_noise,is_boulder);
       
   else
       f = fractal_terrain_generation(i,0,pix,angle,folder_name,is_noise,is_boulder);
   end
  
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