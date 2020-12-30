function y = image2fourier(x)
fourier= fft2(x);
y=fftshift(fourier);
end