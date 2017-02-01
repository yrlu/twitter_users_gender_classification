function [] = Draw_Eigen_Vecs_For_Cov (pixels, h)
smean = mean(pixels,1);
R = cov(pixels);
[E D] = svd(R);
figure(h);
x = [smean(1) smean(1)+sqrt(D(1,1)).*E(1,1)];
y = [smean(2) smean(2)+sqrt(D(1,1)).*E(2,1)];
plot(x,y,'k','Linewidth',1.5);

x = [smean(1) smean(1)+sqrt(D(2,2)).*E(1,2)];
y = [smean(2) smean(2)+sqrt(D(2,2)).*E(2,2)];
plot(x,y,'k','Linewidth',1.5);
end