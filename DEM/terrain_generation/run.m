%% Initialize
clear
close

%% Set the range of area to generate the terrain
x = [0:0.1:30];
y = [0:0.1:30];

%% Generate the random terrain with Perlin noise
% Initial calculation
[X, Y] = meshgrid(x,y);
f = @(t) myinterpolation(t);
H = perlin_2d(f, 1, X, Y);
size(H) % 301   301

% Fractal Perlin noise
% n = 5;
% for i = 1:n
%     xtmp = 2^i*x;
%     ytmp = 2^i*y;
%     [Xtmp, Ytmp] = meshgrid(xtmp, ytmp)
%     Htmp = perlin_2d(f, 10*0.5^i, Xtmp, Ytmp);
%     H = H + Htmp;
% end
%% Generate daen
% H(1,1)
% model = zeros(300,300)
% for i = 1:1:300
%     for j =1:1:300
%  
%         model(i,j) = i+j+H(i,j);
%     end
% end
% s = surface(model);
% s.EdgeColor = 'none';
% zlim([0 30])
% xlim([0 200])
% ylim([0 200])
% %colorbar
% view(3)
%% Display the result
h = surf(X, Y, 0.8*H);
h.EdgeColor = 'none';
axis([0 3 0 3 -15 15])
axis square