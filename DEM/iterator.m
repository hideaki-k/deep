
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
addpath(' C:\Users\hp731\Documents\GitHub\deep\DEM\terrain_generation');
%10/25 NCç ã??è©ä¾¡ç¨ã?ã¼ã¿ã»ã?ã?
% æåº¦ããã¤ãºãã¯ã¬ã¼ã¿æ°ãã¯ã¬ã¼ã¿åå¾?ãæå®å¯è½
is_evaluate = true;
evaluate_angle = 0;

% 0deg,5deg¯¶n`ÉXÎpñÂ
is_double_terrain = true;

if is_evaluate %è©ä¾¡ç¨
    folder_name = string(pix)+"pix_("+string(evaluate_angle)+"deg)_dem(noisy)_evaluate_1124";


elseif is_noise
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem(noisy)_ver2";
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem(noisy)"; % 9/5
    if is_mixangle; %ã¢ã³ã°ã«æ··å¨
        folder_name = string(pix)+"pix_(0-"+string(max_angle)+"deg)_dem(noisy)"; % 9/13
    else %ã¢ã³ã°ã«åºå®?
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
for i=0:1:128
   if is_double_terrain
       evaluate_angle=5
        double_terrain_generation(i,0,pix,evaluate_angle,folder_name,is_noise,is_boulder);
    
   elseif is_evaluate
       evaluate_terrain_generation(i,0,pix,evaluate_angle,folder_name,is_noise,is_boulder);
    
    
   elseif is_mixangle
       angle = round((max_angle)*rand(1)) % ã¹ã­ã¼ãè§?(1~5)
       f = fractal_terrain_generation(i,0,pix,angle,folder_name,is_noise,is_boulder);
       
   else
       f = fractal_terrain_generation(i,0,pix,angle,folder_name,is_noise,is_boulder);
   end
  
%   f = base_DEM(i,2,pix,angle,folder_name,is_noise);

   % å¼æ°ã?i:ã¤ã?ã¬ã¼ã·ã§ã³, mode, pix:è§£ååº¦, angle:æåº¦, folder_name:ã»ã¼ããã£ã¬ã¯ããª,
   % is_noise:ãã?¼ãªã³ãã¤ãºä»å ã®æç¡
   % ã¢ã¼ã?0:LOGã?ã¼ã¿ä¿å­ã?ã¢ã¼ã?1ã¯åç»çæ?ã?ã¢ã¼ã?2?¼?3æ¬¡å?ãã­ã?ã?
end

function f = count(i)
    f = i^2;
    iter_data =  zeros(1,1); 
    filenum = string(i)
    filename = "crater/data_"+filenum
    save(filename,'iter_data')
    
end