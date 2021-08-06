function [DEM,label_data] = put_crater(base, center_of_x, center_of_y,R)
    

    size_factor = size(base)
    DEM = zeros(size_factor);
    label_data = zeros(size_factor);


    
   %% 障害物付与
    for i =  1:1:size_factor(1)
        for j = 1:1:size_factor(2)
               for ind = 1:1:length(center_of_x)
                    x = center_of_x(ind);
                    y = center_of_y(ind);
                    r = sqrt(abs(i-x)^2 + abs(j-y)^2);
                    h = base(i,j);
                    
                    H_ro = 0.036*(2*R(ind))^1.014;
                    H_r = H_ro;
                    H_c = 0.196*(2*R(ind))^1.010 - H_ro ;
                    W_r = 0.257*(2*R(ind))^1.011;
                    alpha = (H_c+H_r)*R(ind)/(H_c+H_ro); %クレータ内縁
                  %%  
                    if r <= alpha
                        h = (H_c+H_ro)*(r^2/R(ind)^2)-H_c;
                        label_data(i,j) = 1;
                    else
                        h = H_r*(R(ind)+W_r)^3/((R(ind)+W_r)^3-R(ind)^3)*(r/R(ind))^(-3) - (H_r*R(ind)^3)/((R(ind)+W_r)^3-R(ind)^3);
                    end
                   base(i,j) = base(i,j) + h;
               end
               DEM(i,j) = base(i,j);

        end
    end





end
