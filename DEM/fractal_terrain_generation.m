function f = fractal_terrain_generation(k,mode,pix,angle,folder_name,is_noise)
    f = mode;
%% generate background fractal terrain
    x = [0:0.1:64];
    y = [0:0.1:64];
%% Generate the random terrain with Perlin noise
    if is_noise == true
        % Initial calculation
        [X, Y] = meshgrid(x,y);
        f = @(t) myinterpolation(t);
        H = perlin_2d(f, 1, X, Y);
        size(H);
    end
%% 斜面 & フラクタルの付与
    size_factor = pix;
    time_scale = 20;
    base = zeros(size_factor,size_factor);
    label_data = zeros(size_factor,size_factor);
    time_data = zeros(size_factor,size_factor,time_scale);
    
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
    % フラクタル化
     noise_min = 0;
     noise_max = 0.8;
     noise_val = (noise_max-noise_min).*rand(1)+noise_min;
    for i=1:1:size_factor
        for j=1:1:size_factor
            base(i,j) = base(i,j) + noise_val*H(i,j);
        end
    end
   %% クレータの個数 & 座標を決定
   center_x_list = []; %　クレータ中心座標_x
   center_y_list = []; % クレータ中心座標_y
   crater_num = 1 + (1 + 10)*rand(1) %クレータ個数(1~10)
   alpha = [];
   R = [];
   for crater = 1:1:crater_num
        %% set parameter
        R(end+1) = 3 + (-3 + 15)*rand(1) %クレータ半径(3~10)


   end

   for crater = 1:1:crater_num
       x_cord = 3 + (3 + 64)*rand(1);
       y_cord = 3 + (3 + 64)*rand(1);
       center_x_list(end+1) = round(x_cord);
       center_y_list(end+1) = round(y_cord);
   end
   
    [DEM,label_data] = put_crater(base, center_x_list, center_y_list, R);
    
    if mode ~= 2     
    DEM = round(DEM,0);
    end
   %% 三次元プロット
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
       
        figure(2)
        s = surf(label_data);
        s.EdgeColor = 'none';
        xlabel('X');
        ylabel('Y');
        zlim([-10 24])
        colormap turbo
        colorbar
        view(3)
        savefig('label.fig')
        
    end
    time = 0;
    lidar_data = zeros(size_factor,size_factor);

    max_elevation = max(DEM(:));
    min_elevation = max_elevation-10;
    for i = max_elevation:-1:min_elevation
        i;
        time = time+1;
        lidar_data(DEM==i)=1;
        time_data(:,:,time) = lidar_data;
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
    if mode == 1
        close(v);
    end

    %% 教師データとして保存
    if mode == 0

        filenum = string(k);
        kyorigazou = mat2gray(DEM);
        filename = folder_name+"/model/model_"+filenum+'.png';
        imwrite(kyorigazou,filename);
        
        filename = folder_name+"/label/label_"+filenum;
        save(filename,'label_data');
     
        filename = folder_name+"/image/image_"+filenum
        save(filename,'time_data');
    end