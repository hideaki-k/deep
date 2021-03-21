size_factor = 500
center = size_factor/2
model = zeros(size_factor,size_factor);
for i =  1:1:size_factor
    for j = 1:1:size_factor
        f = abs(i-center)^2 + abs(j-center)^2;
        model(i,j) = f;
    end
end
model



v = VideoWriter('peaks_1.avi');
open(v);
lidar_data = zeros(size_factor,size_factor);
for i =  1:1:size_factor*10
  
    lidar_data(model==i) = 1;
    
    if rem(i,10) == 0
        imagesc(lidar_data);
        colorbar;
        hold on;
        pause(0.1)
        frame = lidar_data;
        writeVideo(v,frame);
        lidar_data =  zeros(size_factor,size_factor);
    end
    

    
end
close(v);
imagesc(lidar_data);
colorbar;

imagesc(model);
colorbar;
% f1 = figure;
% f2 = figure;
% figure(f1);
% imagesc(data);colorbar;
% figure(f2);
% imagesc(trim);colorbar;


s = surface(model);
s.EdgeColor = 'none';
%colorbar
view(3)