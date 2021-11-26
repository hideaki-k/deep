function f = double_terrain_generation(k,mode,pix,angle,folder_name,is_noise,is_boulder)
    f = mode;

%% 斜面 & フラクタルの付�?
    size_factor = pix;
    time_scale = 20;
    base_0 = zeros(size_factor,size_factor);
    base = zeros(size_factor,size_factor);
    label_data = zeros(size_factor,size_factor);
    time_data_0 = zeros(size_factor,size_factor,time_scale);
    time_data_5 = zeros(size_factor,size_factor,time_scale);
    
    direct = round(rand(1),0);
    up_down= round(rand(1),0);
    
    if up_down == 1
        up_down = 1;
    else 
        up_down = -1;
    end

    for i=1:1:size_factor
        for j=1:1:size_factor
            if direct==1
                base(:,j) = (up_down)*j*tan(deg2rad(angle));
            else
                base(i,:) = (up_down)*i*tan(deg2rad(angle));   
            end
        end
    end

   %% クレータの個数 & 座標を決�?
   crater_num = round(1 + (1 + 6)*rand(1)); % クレータ個数(1~6)
   center_x_list = zeros(crater_num,1); %�?クレータ中�?座標_x
   center_y_list = zeros(crater_num,1); % クレータ中�?座標_y
   
   alpha = [];
   R = zeros(crater_num,1);

   for crater = 1:1:crater_num
       R(crater) = 3 + (-3 + 15)*rand(1); %クレータ半�?(3~10)
       x_cord = 3 + (3 + 64)*rand(1);
       y_cord = 3 + (3 + 64)*rand(1);
       center_x_list(crater) = round(x_cord);
       center_y_list(crater) = round(y_cord);
   end
   %% ボル�?ーの個数 & 座標を決�?

    boulder_num = round(5*rand(1)); % ボル�?ー個数(0~5)
    boulder_center_x_list =  zeros(boulder_num,1); % ボル�?ー中�?座標x
    boulder_center_y_list = zeros(boulder_num,1); % ボル�?ー中�?座標y
    boulder_xziku_list = zeros(boulder_num,1); % ボル�?ーx軸長�?
    boulder_yziku_list = zeros(boulder_num,1); % ボル�?ーy軸長�?
    boulder_zziku_list = zeros(boulder_num,1); % ボル�?ーz軸長�?
   
   if is_boulder
   for boulder = 1:1:boulder_num
       % 座�?
       x_cord = 3 + (3 + 64)*rand(1);
       y_cord = 3 + (3 + 64)*rand(1);
       boulder_center_x_list(boulder) = round(x_cord);
       boulder_center_y_list(boulder) = round(y_cord);
       % 大きさ
       xr = abs(5*rand(1));
       yr = abs(5*rand(1));
       zr = 3+abs(1*rand(1));
       boulder_xziku_list(boulder) = round(xr);
       boulder_yziku_list(boulder) = round(yr);
       boulder_zziku_list(boulder) = round(zr);
   end

   end
    DEM_0 =  put_hazard(base_0,is_noise, center_x_list, center_y_list, R, is_boulder, boulder_center_x_list, boulder_center_y_list, boulder_xziku_list, boulder_yziku_list, boulder_zziku_list);
    DEM_5 = put_hazard(base,is_noise, center_x_list, center_y_list, R, is_boulder, boulder_center_x_list, boulder_center_y_list, boulder_xziku_list, boulder_yziku_list, boulder_zziku_list);
   
    true_DEM_0 = DEM_0;    
    DEM_0 = round(DEM_0,0);
    true_DEM_5 = DEM_5;    
    DEM_5 = round(DEM_5,0);
   %% 三次�?プロ�?�?
    if mode==2
        figure(1)
        s = surf(DEM);
        s.EdgeColor = 'none';
        xlabel('X');
        ylabel('Y');
        zlim([-20 20])
        colormap gray
        colorbar
        view(3)
        savefig('model.fig')
       
%         figure(2)
%         s = surf(label_data);
%         s.EdgeColor = 'none';
%         xlabel('X');
%         ylabel('Y');
%         zlim([-10 24])
%         colormap turbo
%         colorbar
%         view(3)
%         savefig('label.fig')
        
    end
    time = 0;
    lidar_data = zeros(size_factor,size_factor);

    max_elevation_0 = max(DEM_0(:));
    min_elevation = max_elevation_0-20;
    for i = max_elevation_0:-1:min_elevation

        time = time+1;
        lidar_data(DEM_0==i)=1;
        time_data_0(:,:,time) = lidar_data;
        if mode == 1
            imagesc(lidar_data);
            colorbar;
            hold on;
            pause(0.1);
            frame = lidar_data;
            writeVideo(v,frame);
        end
        lidar_data = zeros(size_factor,size_factor);
    end
    
    
    time = 0;
    lidar_data = zeros(size_factor,size_factor);

    max_elevation_5 = max(DEM_5(:));
    min_elevation = max_elevation_5-20;
    for i = max_elevation_5:-1:min_elevation

        time = time+1;
        lidar_data(DEM_5==i)=1;
        time_data_5(:,:,time) = lidar_data;
        if mode == 1
            imagesc(lidar_data);
            colorbar;
            hold on;
            pause(0.1);
            frame = lidar_data;
            writeVideo(v,frame);
        end
        lidar_data = zeros(size_factor,size_factor);
    end


    %% 教師�?ータとして保�?
    if mode == 0
        % 0deg
        filenum = string(k);
        % true_DEM.mat hazard_label評価用
        filename = folder_name+"/model/real_model_0_"+filenum;
        save(filename,'true_DEM_0');
        
        % observed_DEM.mat hazard_label評価用 8/29追�?
        filename = folder_name+"/model/observed_model_0_"+filenum;
        save(filename,'DEM_0');
        
        % model_png 見た目で評価?��丸め�?�なしとあり?�?
        kyorigazou = mat2gray(true_DEM_0);
        filename = folder_name+"/model/model_true_0_"+filenum+'.png';
        imwrite(kyorigazou,filename);
        kyorigazou = mat2gray(DEM_0);
        filename = folder_name+"/model/model_0_"+filenum+'.png';
        imwrite(kyorigazou,filename);
        
        % label_data.mat crater中�?を塗りつぶした
        filename = folder_name+"/label/label_"+filenum;
        save(filename,'label_data');
     
        % time_data lidar再現�?ータ
        filename = folder_name+"/image/image_0_"+filenum
        save(filename,'time_data_0');
        
        % 5deg
        filenum = string(k);
        % true_DEM.mat hazard_label評価用
        filename = folder_name+"/model/real_model_5_"+filenum;
        save(filename,'true_DEM_5');
        
        % observed_DEM.mat hazard_label評価用 8/29追�?
        filename = folder_name+"/model/observed_model_5_"+filenum;
        save(filename,'DEM_5');
        
        % model_png 見た目で評価?��丸め�?�なしとあり?�?
        kyorigazou = mat2gray(true_DEM_5);
        filename = folder_name+"/model/model_true_5_"+filenum+'.png';
        imwrite(kyorigazou,filename);
        kyorigazou = mat2gray(DEM_5);
        filename = folder_name+"/model/model_5_"+filenum+'.png';
        imwrite(kyorigazou,filename);
        
        % label_data.mat crater中�?を塗りつぶした
        filename = folder_name+"/label/label_"+filenum;
        save(filename,'label_data');
     
        % time_data lidar再現�?ータ
        filename = folder_name+"/image/image_5_"+filenum
        save(filename,'time_data_5');
    end