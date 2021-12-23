clear
close all
filename = 'C:\Users\aki\OneDrive - 東京理科大学\研究\64pix_(0-5deg)_dem(noisy)\model\real_model_396.mat'
DEM = load(filename,'true_DEM');
DEM = DEM.true_DEM;

figure(1)
s = surf(DEM);
s.EdgeColor = 'none';
xlabel('X[m]');
ylabel('Y[m]');
zlabel('Altitude[m]');
zlim([-20 20])
colormap turbo
c = colorbar;
c.Label.String = 'm';
view(3)
