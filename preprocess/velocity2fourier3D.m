function y = velocity2fourier3D(x)
[dim,m,n,p]=size(x);
fourier= fft2(reshape(x(1,:,:),m,n,p));
vx=fftshift(fourier);
fourier= fft2(reshape(x(2,:,:),m,n,p));
vy=fftshift(fourier);
fourier= fft2(reshape(x(3,:,:),m,n,p));
vz=fftshift(fourier);
y=zeros(dim,m,n);
for i=1:m
for j=1:n
    for k=1:p
    y(1,i,j,k)= vx(i,j,k);
    y(2,i,j,k) = vy(i,j,k);
    y(3,i,j,k) = vz(i,j,k);
    end
end
end
end