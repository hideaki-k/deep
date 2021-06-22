function f=base_DEM(k,mode)
    f = mode;
    %% init parameter
    size_factor = 64;
    time_scale = 20;
    %angle = rand(1)*10;
    angle = 16;
    model = zeros(size_factor,size_factor);
    label_data = zeros(size_factor,size_factor);
    direct = round(rand(1),0);
    up_down= round(rand(1),0);
    if up_down == 1
        up_down = 1;
    else up_down == 0
        up_down = -1;
    end
    if direct==1
        for x=1:1:size_factor
            for y=1:1:size_factor
                model(:,y) = (up_down)*y*tan(deg2rad(angle)); 
            end
        end
    end
    if direct==0
        for x=1:1:size_factor
            for y=1:1:size_factor
                model(x,:) = (up_down)*x*tan(deg2rad(angle)); 
            end
        end
    end

    %% set parameter
    R = 10 + abs(randn(1)); %クレータ半径
    % H_r = 150 + abs(5*randn())
    H_ro = 0.036*(2*R)^1.014;
    H_r = H_ro;
    H_c = 0.196*(2*R)^1.010 - H_ro ;
    W_r = 0.257*(2*R)^1.011;
    % RANGE
    alpha = (H_c+H_r)*R/(H_c+H_ro); %クレータ内縁
    beta = R+(1-(H_c+H_r)/(H_c+H_ro))*W_r ;%クレータ外縁
    dist_to_zero =20; %標高ゼロまでの距離
    A = -3*R^3 + 2*R^2*beta + 2*R*beta^2 + 2*beta^3;
    % クレータ１の中心座標
    x_center = size_factor/2+20*(-1 + (1+1)*rand(1));
    y_center = size_factor/2+20*(-1 + (1+1)*rand(1));

    %% 生成したクレータ2が近接した場合に削除する処理
    cnt = 0;
    while 1
        x_center_2 = size_factor/2-20*(-1 + (1+1)*rand(1));
        y_center_2 = size_factor/2-20*(-1 + (1+1)*rand(1));
        if sqrt(abs(x_center-x_center_2)^2) + sqrt(abs(y_center-y_center_2)^2) > 3.5*R
            break
        elseif cnt >=100
            break
        end
        cnt=cnt+1
    end
    %% クレータ付与
    for i =  1:1:size_factor
        for j = 1:1:size_factor

            % y = wgn(1,1,-3);
             r = sqrt(abs(i-x_center)^2 + abs(j-y_center)^2);
             r_ = sqrt(abs(i-x_center_2)^2 + abs(j-y_center_2)^2);


             if r <= alpha || r_ <= alpha
                 label_data(i,j) = 1;
                 if r <= alpha
                    h = (H_c+H_ro)*(r^2/R^2)-H_c;

                 elseif r_ <= alpha
                    h = (H_c+H_ro)*(r_^2/R^2)-H_c;
                 end

    %              elseif r <= R || r_ <= R
    %                  disp("R")
    %                  r
    %                  if r <= R
    %                     h = ((H_c + H_ro)^2/(H_r - H_ro) - H_c)*((r/R) - 1)^2 + H_r;
    %                  elseif r_ <= R
    %                     h = ((H_c + H_ro)^2/(H_r - H_ro) - H_c)*((r_/R) - 1)^2 + H_r;
    %                  end
    %              elseif r < beta || r_ < beta
    %                  if r <= beta
    %                     h =  (H_r*(R+W_r)^3*A)/(W_r*beta^4*(R-beta)^2*(3*R^2+3*R*W_r+W_r^2)) * (r-R)^2*(r-beta*(1+(beta^3-R^3)/A))+H_r;
    %                  elseif r_ <= beta
    %                     h =  (H_r*(R+W_r)^3*A)/(W_r*beta^4*(R-beta)^2*(3*R^2+3*R*W_r+W_r^2)) * (r_-R)^2*(r_-beta*(1+(beta^3-R^3)/A))+H_r;
    %                  end

             elseif r <= dist_to_zero || r_ <= dist_to_zero
                 if r <= dist_to_zero
                     h =  H_r*(R+W_r)^3/((R+W_r)^3-R^3)*(r/R)^(-3) - (H_r*R^3)/((R+W_r)^3-R^3);
                 elseif r_ <= dist_to_zero
                     h =  H_r*(R+W_r)^3/((R+W_r)^3-R^3)*(r_/R)^(-3) - (H_r*R^3)/((R+W_r)^3-R^3);
                 end

             else    
               % h = H_r*(R+W_r)^3/((R+W_r)^3-R^3)*(r/R)^(-3) - (H_r*R^3)/((R+W_r)^3-R^3);
                %h = h + H_r*(R+W_r)^3/((R+W_r)^3-R^3)*(r_/R)^(-3) - (H_r*R^3)/((R+W_r)^3-R^3);
                h = -1;
             end

             model(i,j) = model(i,j)+h;
        end
    end
    model;
    model = round(model,0);
    model;
    
    %% 三次元プロット
    if mode==2
        figure(1)
        s = surface(model);
        s.EdgeColor = 'none';
        xlabel('X');
        ylabel('Y');
        zlim([-10 24])
        colorbar
        view(3)
        savefig('model')
        figure(2)
        s = surface(label_data);
        s.EdgeColor = 'none';
        xlabel('X');
        ylabel('Y');
        zlim([-10 24])
        colorbar
        view(3)
        savefig('label')
    end
    %% 動画 v open

    if mode == 1
        v = VideoWriter('SLOPE_two_craters_image.avi')
        open(v)
    end
    time = 0;
    lidar_data = zeros(size_factor,size_factor);
    time_data = zeros(size_factor,size_factor);
    max_elevation = max(model(:));
    min_elevation = max_elevation-time_scale;
    for i = max_elevation:-1:min_elevation
        time = time+1;
        lidar_data(model==i)=1;
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
        folder_name = string(size_factor)+"pix_("+string(angle)+"deg)_craters";
        mkdir(folder_name);
        mkdir(folder_name,'image');
        mkdir(folder_name,'label');
        filenum = string(k);
        filename = folder_name+"/label/label_"+filenum;
        save(filename,'label_data');
     
        filename = folder_name+"/image/image_"+filenum
        save(filename,'time_data');
    end
end