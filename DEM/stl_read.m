b = figure(2);
model = stlread('tri.stl')
a = model.ConnectivityList  
% model.Points
a(1,3)
img = zeros(100,100);
for i = 1:1:9
    for j = 1:1:9
       i,j
       a(i,j);
    end
end
imshow(img)

% model = pcread('circle.ply');
% pcshow(model);
% model(:)
% %triplot(data)
% % surf(data);
% % colormap(jet);
% lidar_data = zeros(50,50);
% 
% 
% % imagesc(data);colorbar;
% % a = colormap('jet');
% view(3)
% %trim = data(1:500,1:500);
% for i =  15:-1:0
%     disp(i)
%     lidar_data(model==i) = 1;
%    
%     imagesc(lidar_data);
%     colorbar;
%     hold on;
%     pause(0.01)
%     frame = lidar_data;
%     %writeVideo(v,frame);
%     lidar_data =  zeros(50,50);
%     
% end


