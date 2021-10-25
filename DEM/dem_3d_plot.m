clear
close all
filename = 'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(5deg)_dem(noisy)\model\real_model_16611.mat'
DEM = load(filename,'true_DEM');
DEM = DEM.true_DEM;

figure(1)
s = surf(DEM);
s.EdgeColor = 'none';
xlabel('X');
ylabel('Y');
zlim([-20 20])
colormap turbo
colorbar
view(3)
