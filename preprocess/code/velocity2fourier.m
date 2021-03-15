function y = velocity2fourier(x)
[dim,m,n]=size(x);
fourier= fft2(reshape(x(1,:,:),m,n));
vx=fftshift(fourier);
fourier= fft2(reshape(x(2,:,:),m,n));
vy=fftshift(fourier);
fourier= fft2(reshape(x(3,:,:),m,n));
vz=fftshift(fourier);
y=zeros(dim,m,n);
for i=1:m
for j=1:n
    y(1,i,j)= vx(i,j);
    y(2,i,j) = vy(i,j);
    y(3,i,j) = vz(i,j);
end
end

end