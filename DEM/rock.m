%(0,-0.5,0,6,3.25,3.25) %(xc,yc,zc) を中心座標、(xr,yr,zr) を半軸の長さ

%    (X-XC)^2     (Y-YC)^2     (Z-ZC)^2
%    --------  +  --------  +  --------  =  1
%      XR^2         YR^2         ZR^2
size_factor = 5
xc = 0;
yc = 0;
zc = 0;
xr = 10;
yr = 5;
zr = 10;
model = zeros(2*xr,2*yr);
for i =  1:1:xr
    for j = 1:1:yr
       if j < yr*sqrt(1-i^2/xr^2)
            z = genrate_ellipsoid(i,j,xc, yc, zc, 10, 5, 10);
        	model(i,j) = z;
       end
    end
end
s = surface(model);
s.EdgeColor = 'none';
zlim([0 30])
%colorbar
view(3)
function z =genrate_ellipsoid(x ,y, xc, yc, zc, xr, yr, zr)
x,y
z =  zc + zr*sqrt(1-((x-xc)^2/xr^2)-((y-yc)^2/yr^2))

end