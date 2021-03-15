function y = image2fourier3D(x)
fourier= fftn(x);
y=fftshift(fourier);
end