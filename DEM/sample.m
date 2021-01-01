A = randi(10,10);
A

imagesc(A);
colorbar;
A(A==1)=255

imagesc(A);
colorbar;
emp = zeros(10,10);
emp(2,2)=255;

% imagesc(emp);
% colorbar;