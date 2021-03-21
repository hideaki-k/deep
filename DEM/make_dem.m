size_factor = 500
center = size_factor/2
model = zeros(size_factor,size_factor);
R = 100
H_r = 5
H_c = 10

W_r = 200
for i =  1:1:size_factor
    for j = 1:1:size_factor
        

        r = sqrt(abs(i-center)^2 + abs(j-center)^2);
        if r <= R
            h = (H_c+H_r)*r^2/R^2;
        else
            h = H_r*(R+W_r)^3/((R+W_r)^3-R^3)*(r/R)^(-3) - (H_r*R^3)/((R+W_r)^3-R^3);
        end
        model(i,j) = h;
    end
end
model


% 
% v = VideoWriter('peaks_1.avi');
% open(v);
% lidar_data = zeros(size_factor,size_factor);
% for i =  1:1:size_factor*10
%   
%     lidar_data(model==i) = 1;
%     
%     if rem(i,10) == 0
%         imagesc(lidar_data);
%         colorbar;
%         hold on;
%         pause(0.1)
%         frame = lidar_data;
%         writeVideo(v,frame);
%         lidar_data =  zeros(size_factor,size_factor);
%     end
%     
% 
%     
% end
% close(v);
% imagesc(lidar_data);
% colorbar;

% imagesc(model);
% colorbar;
% f1 = figure;
% f2 = figure;
% figure(f1);
% imagesc(data);colorbar;
% figure(f2);
% imagesc(trim);colorbar;


s = surface(model);
s.EdgeColor = 'none';
zlim([0 50])
%colorbar
view(3)