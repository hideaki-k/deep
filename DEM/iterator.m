
clear all
close all
clc
pix = 128;
angle = 0;
folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem";
mkdir(folder_name);
mkdir(folder_name,'image');
mkdir(folder_name,'label');
mkdir(folder_name,'model');
for i=0:1:12800
  % f = put_crater(i,0);
   % f = make_scan_crater(i,0);
  % f = make_scan_bolder_perlin(i,2);
  % f =  make_crater(i,1)
   f = base_DEM(i,0,pix,angle,folder_name)
  % f = put_boulder_perlin(i,0)

   %モード0:LOGデータ保存、モード1は動画生成、モード2：3次元プロット
end

function f = count(i)
    f = i^2;
    iter_data =  zeros(1,1); 
    filenum = string(i)
    filename = "crater/data_"+filenum
    save(filename,'iter_data')
    
end