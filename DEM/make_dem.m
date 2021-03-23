size_factor = 500
center = size_factor/2
model = zeros(size_factor,size_factor);


H_r = 3
H_ro = 0.036*(2*R)^1.014
H_r = H_ro
H_c = 0.196*(2*R)^1.010 - H_ro
W_r = 0.257*(2*R)^1.011
% RANGE
R = 100
alpha = (H_c+H_r)*R/(H_c+H_ro)
beta = R+(1-(H_c+H_r)/(H_c+H_ro))*W_r

A = -3*R^3 + 2*R^2*beta + 2*R*beta^2 + 2*beta^3

for i =  1:1:size_factor
    for j = 1:1:size_factor
        

        r = sqrt(abs(i-center)^2 + abs(j-center)^2);
         if r <= alpha
             h = (H_c+H_ro)*(r^2/R^2)-H_c;
         elseif r <= R
             h = ((H_c + H_ro)^2/(H_r - H_ro) - H_c)*((r/R) - 1)^2 + H_r;
         elseif r < beta
             h =  (H_r*(R+W_r)^3*A)/(W_r*beta^4*(R-beta)^2*(3*R^2+3*R*W_r+W_r^2)) * (r-R)^2*(r-beta*(1+(beta^3-R^3)/A))+H_r;
        else    
            h = H_r*(R+W_r)^3/((R+W_r)^3-R^3)*(r/R)^(-3) - (H_r*R^3)/((R+W_r)^3-R^3);
        end
        model(i,j) = h;
    end
end
model;


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
zlim([-50 50])
%colorbar
view(3)