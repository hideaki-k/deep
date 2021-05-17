%(0,-0.5,0,6,3.25,3.25) %(xc,yc,zc) を中心座標、(xr,yr,zr) を半軸の長さ

%    (X-XC)^2     (Y-YC)^2     (Z-ZC)^2
%    --------  +  --------  +  --------  =  1
%      XR^2         YR^2         ZR^2

xc = 100;
yc = 100;
zc = 0;
xr = 80;
yr = 30;
zr = 10;
model = zeros(200,200);
for i =  1:1:200
    for j = 1:1:200
       if j > yc
           if j < yc + yr*sqrt(1-(i-xc)^2/xr^2)
                z = genrate_ellipsoid(i,j,xc, yc, zc, xr, yr, zr);
                
                model(i,j) = z ;
           end
       else
          if j > yc - yr*sqrt(1-(i-xc)^2/xr^2)
                z = genrate_ellipsoid(i,j,xc, yc, zc, xr, yr, zr);
                
                model(i,j) = z ;
          end
       end
    end
end
s = surface(model);
s.EdgeColor = 'none';
zlim([0 30])
xlim([0 200])
ylim([0 200])
%colorbar
view(3)
function z =genrate_ellipsoid(x ,y, xc, yc, zc, xr, yr, zr)
x,y
z =  zc + zr*sqrt(1-((x-xc)^2/xr^2)-((y-yc)^2/yr^2));

end