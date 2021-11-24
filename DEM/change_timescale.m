% mode  0:保存 1:ビデオ保存 2:三次元プロット 
addpath 'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0deg)_dem(noisy)_evaluate\model'
folder_name = 'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0deg)_dem(noisy)_evaluate';
mode = 0;
size_factor = 64;
num_time = 70
% real_modelファイルの読み込み
for i=0:1:16640
    i
    % file読み込み
    file_path = append('real_model_',string(i),'.mat');
    DEM = load(file_path,'true_DEM')
    DEM = DEM.true_DEM;
        
    % 丸目
    DEM = round(DEM,1);
    
    % time_data 用意
    time_scale = 70;
    time_data = zeros(size_factor,size_factor,time_scale);
    if mode==1
        v = VideoWriter('50_image.avi')
        open(v)
    end
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
        max_elevation = max(DEM(:))
        max_elevation = min(DEM(:))
    end

    time = 0;
    lidar_data = zeros(size_factor,size_factor);
    max_elevation = max(DEM(:))
    min_elevation = max(DEM(:))-7;
    max_elevation-min_elevation
    for h = max_elevation:-0.1:min_elevation
         time = time+1;
        lidar_data(DEM==h)=1;
        time_data(:,:,time) = lidar_data;
        if mode == 1
            %imagesc(lidar_data);
            %colorbar;
            hold on;
            pause(0.2);
            frame = lidar_data;
            writeVideo(v,frame);
        end
        lidar_data = zeros(size_factor,size_factor);
    end
    if mode == 1
        close(v)
    end
    %% 教師データとして保存
    if mode == 0
        filenum = string(i);
        filename = folder_name+"/image/image(t-70)_"+filenum;
        save(filename,'time_data');
    end
  
end