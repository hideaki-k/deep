for i=1:1:1
   %f =  make_bolder_perlin(i,0)
   f =  make_crater(i,2)
   %モード0:なし、モード1は動画生成、モード2：3次元プロット
end

function f = count(i)
    f = i^2;
    iter_data =  zeros(1,1); 
    filenum = string(i)
    filename = "crater/data_"+filenum
    save(filename,'iter_data')
    
end