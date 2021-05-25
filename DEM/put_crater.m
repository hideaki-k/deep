%% init parameter
size_factor = 512;
num_crater = 2;


model = zeros(size_factor,size_factor);


R = 50 + abs(5*randn(1)) %クレータ半径
% H_r = 150 + abs(5*randn())
H_ro = 0.036*(2*R)^1.014;
H_r = H_ro;
H_c = 0.196*(2*R)^1.010 - H_ro ;
W_r = 0.257*(2*R)^1.011;

% RANGE
alpha = (H_c+H_r)*R/(H_c+H_ro) %クレータ内縁
beta = R+(1-(H_c+H_r)/(H_c+H_ro))*W_r %クレータ外縁
dist_to_zero =200; %標高ゼロまでの距離
A = -3*R^3 + 2*R^2*beta + 2*R*beta^2 + 2*beta^3

x_center = size_factor/2+100*randn(1)
y_center = size_factor/2+100*randn(1)

%% 生成したクレータが近接した場合に削除する処理
while 1
    x_center_2 = 500*rand(1)
    y_center_2 = 500*rand(1)
    if sqrt(abs(x_center-x_center_2)^2) + sqrt(abs(y_center-y_center_2)^2) > R
        break
    end
end

%%
    for i =  1:1:size_factor
        for j = 1:1:size_factor

            % y = wgn(1,1,-3);
             r = sqrt(abs(i-x_center)^2 + abs(j-y_center)^2);
             r_ = sqrt(abs(i-x_center_2)^2 + abs(j-y_center_2)^2);
             
            
             if r <= alpha || r_ <= alpha
            
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
                h = 0;
             end
           
             model(i,j) = h;
        end
    end
model;
model = round(model,1);
model;
%%
figure(1)
s = surface(model);
s.EdgeColor = 'none';
zlim([-100 100])
colorbar
view(3)

%%
% figure(2)
% plot3(model)
% xlabel('X(m)')
% ylabel('Y(m)')
% zlabel('Z(m)')
% axis equal
%%