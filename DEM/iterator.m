for i=1:1:12800
  % f = put_crater(i,2);
   %f =  make_crater(i,2)
   %f =  make_crater(i,2)
   f = base_DEM(i,0)
   %モード0:LOGデータ保存、モード1は動画生成、モード2：3次元プロット
end

function f = count(i)
    f = i^2;
    iter_data =  zeros(1,1); 
    filenum = string(i)
    filename = "crater/data_"+filenum
    save(filename,'iter_data')
    
end