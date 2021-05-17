%(0,-0.5,0,6,3.25,3.25) %(xc,yc,zc) を中心座標、(xr,yr,zr) を半軸の長さ

%    (X-XC)^2     (Y-YC)^2     (Z-ZC)^2
%    --------  +  --------  +  --------  =  1
%      XR^2         YR^2         ZR^2
%% initialize parameter
xc = 100;
yc = 100;
zc = 0;
xr = 80;
yr = 50;
zr = 30;

model = zeros(200,200);
%% Set the range of area to generate the terrain
x = [0:0.1:20];
y = [0:0.1:20];
%% Generate the random terrain with Perlin noise
% Initial calculation
[X, Y] = meshgrid(x,y);
f = @(t) myinterpolation(t);
H = perlin_2d(f, 6, X, Y);
size(H) % 301   301
%% generate elipsoid
for i =  1:1:200
    for j = 1:1:200
       if j > yc
           if j < yc + yr*sqrt(1-(i-xc)^2/xr^2)
               
                z = genrate_ellipsoid(i,j,xc, yc, zc, xr, yr, zr);
                model(i,j) = z +H(i,j);
           end
       else
          if j > yc - yr*sqrt(1-(i-xc)^2/xr^2)
                z = genrate_ellipsoid(i,j,xc, yc, zc, xr, yr, zr);
                
                model(i,j) = z+H(i,j) ;
          end
       end
    end
end
%%
s = surface(model);
s.EdgeColor = 'none';
axis equal
zlim([0 30])
xlim([0 200])
ylim([0 200])
%colorbar
view(3)
%%
function z =genrate_ellipsoid(x ,y, xc, yc, zc, xr, yr, zr)
%x,y
noise = wgn(1,1,-3);
%y = y+noise
z =  zc + zr*sqrt(1-((x-xc)^2/xr^2)-((y-yc)^2/yr^2));

end