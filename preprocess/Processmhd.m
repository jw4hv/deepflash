clear all;
%%%%%%%%%%%% Source, target and velocity fields directories%%%%%%%%%%%%%%%% 
FilesV=dir('./velocity/*.mhd');
FilesS=dir('./source/*.mhd');
FilesT=dir('./target/*.mhd');

%%%%%%%%%%%% Folds of frequency files%%%%%%%%%%%%%%%% 
mkdir Rnet/velo_fourier_real_x
mkdir Rnet/velo_fourier_real_y
mkdir Rnet/velo_fourier_real_z

mkdir Inet/velo_fourier_imag_x
mkdir Inet/velo_fourier_imag_y
mkdir Inet/velo_fourier_imag_z

mkdir Rnet/src_fourier_real
mkdir Rnet/tar_fourier_real

mkdir Inet/src_fourier_imag
mkdir Inet/tar_fourier_imag

%%% Write out frequencies and truncated low-dimensional velocity fields %%%
trunc_dim = 16;
for k=1:size(FilesV)
    FileNames=FilesV(k).name;
    Filefolder=FilesV(k).folder;
    path=strcat( Filefolder,'/',FileNames) ;
    alldataV(k,:,:,:,:)=loadMETA(path);
    [h1,comdim,hth,wth,lth]=size(alldataV(1,:,:,:,:));
    %%%%%%%%%%%%%%%%%%%%%%%%%% Extracted Low-frequencies of 3D velocity fields %%%%%%%%%%%%%%%%%%%%%%%%%%
    if (lth ~= 1)
    alldataVf(k,:,:,:,:)= velocity2fourier3D(reshape(alldataV(k,:,:,:,:),comdim,hth,wth,lth));
    alldataVf(k,:,:,:,:) = alldataVf(k,:,:,:,:)/(hth*wth*lth);
    allLowVf (k,:,:,:,:) = TruncatedLowF3D(alldataVf(k,:,:,:,:),trunc_dim,trunc_dim,trunc_dim);
    fname_vel_R_X = sprintf('./Rnet/velo_fourier_real_x/%dvelocityRFX.mhd', k);
    writeMETA(reshape(real(allLowVf(k,1,:,:,:)), trunc_dim+1, trunc_dim+1, trunc_dim+1),fname_vel_R_X);
    fname_vel_R_Y = sprintf('./Rnet/velo_fourier_real_y/%dvelocityRFY.mhd', k);
    writeMETA(reshape(real(allLowVf(k,2,:,:,:)), trunc_dim+1, trunc_dim+1, trunc_dim+1),fname_vel_R_Y);
    fname_vel_R_Z = sprintf('./Rnet/velo_fourier_real_z/%dvelocityRFZ.mhd', k);
    writeMETA(reshape(real(allLowVf(k,3,:,:,:)), trunc_dim+1, trunc_dim+1, trunc_dim+1),fname_vel_R_Z);

    fname_vel_R_X = sprintf('./Inet/velo_fourier_imag_x/%dvelocityIFX.mhd', k);
    writeMETA(reshape(imag(allLowVf(k,1,:,:,:)), trunc_dim+1, trunc_dim+1, trunc_dim+1),fname_vel_R_X);
    fname_vel_R_Y = sprintf('./Inet/velo_fourier_imag_y/%dvelocityIFY.mhd', k);
    writeMETA(reshape(imag(allLowVf(k,2,:,:,:)), trunc_dim+1, trunc_dim+1, trunc_dim+1),fname_vel_R_Y);
    fname_vel_R_Z = sprintf('./Inet/velo_fourier_imag_z/%dvelocityIFZ.mhd', k);
    writeMETA(reshape(imag(allLowVf(k,3,:,:,:)), trunc_dim+1, trunc_dim+1, trunc_dim+1),fname_vel_R_Z);

    %%%%%%%%%%%%%%%%%%%%%%%%%% Extracted Low-frequencies of 2D velocity fields %%%%%%%%%%%%%%%%%%%%%%%%%%
    else
    alldataVf(k,:,:,:)= velocity2fourier(reshape(alldataV(k,:,:,:,:),comdim,hth,wth));
    alldataVf(k,:,:,:) = alldataVf(k,:,:,:)/(hth*wth*lth);
    allLowVf (k,:,:,:) = TruncatedLowF(alldataVf(k,:,:,:),trunc_dim,trunc_dim,1);

    fname_vel_R_X = sprintf('./Rnet/velo_fourier_real_x/%dvelocityRFX.mhd', k);
    writeMETA(reshape(real(allLowVf(k,1,:,:)), trunc_dim+1, trunc_dim+1),fname_vel_R_X);
    fname_vel_R_Y = sprintf('./Rnet/velo_fourier_real_y/%dvelocityRFY.mhd', k);
    writeMETA(reshape(real(allLowVf(k,2,:,:)), trunc_dim+1, trunc_dim+1),fname_vel_R_Y);
    fname_vel_R_Z = sprintf('./Rnet/velo_fourier_real_z/%dvelocityRFZ.mhd', k);
    writeMETA(reshape(real(allLowVf(k,3,:,:)), trunc_dim+1, trunc_dim+1),fname_vel_R_Z);

    fname_vel_R_X = sprintf('./Inet/velo_fourier_imag_x/%dvelocityIFX.mhd', k);
    writeMETA(reshape(imag(allLowVf(k,1,:,:)), trunc_dim+1, trunc_dim+1),fname_vel_R_X);
    fname_vel_R_Y = sprintf('./Inet/velo_fourier_imag_y/%dvelocityIFY.mhd', k);
    writeMETA(reshape(imag(allLowVf(k,2,:,:)), trunc_dim+1, trunc_dim+1),fname_vel_R_Y);
    fname_vel_R_Z = sprintf('./Inet/velo_fourier_imag_z/%dvelocityIFZ.mhd', k);
    writeMETA(reshape(imag(allLowVf(k,3,:,:)), trunc_dim+1, trunc_dim+1),fname_vel_R_Z);
    end


    FileNames=FilesS(k).name;
    Filefolder=FilesS(k).folder;
    path=strcat( Filefolder,'/',FileNames) ;
    alldataS(k,:,:,:)=loadMETA(path);
    [h1,hth,wth,lth]=size(alldataS(1,:,:,:));
    if (lth ~= 1)
    alldataSf(k,:,:,:)= image2fourier3D(reshape(alldataS(k,:,:,:,:),hth,wth,lth));
    alldataSf(k,:,:,:) = alldataSf(k,:,:,:)/ (hth*wth*lth);
    fname_src_R = sprintf('./Rnet/src_fourier_real/%dsrcRF.mhd', k);
    writeMETA(reshape(real(alldataSf(k,:,:,:)),hth, wth, lth),fname_src_R);  
    fname_src_I = sprintf('./Inet/src_fourier_imag/%dsrcIF.mhd', k);
    writeMETA(reshape(imag(alldataSf(k,:,:,:)),hth, wth, lth),fname_src_I);
    % alldataSfcon(k,:,:)= reshape(getconj(reshape(alldataSf(k,:,:),hth,wth)),1,floor(hth/2)+1,wth);
    else
    alldataSf(k,:,:)= image2fourier(reshape(alldataS(k,:,:,:),hth,wth));
    alldataSf(k,:,:) = alldataSf(k,:,:)/ (hth*wth);
    fname_src_R = sprintf('./src_fourier_real/%dsrcRF.mhd', k);
    writeMETA(reshape(real(alldataSf(k,:,:,:)),hth, wth ),fname_src_R);  
    fname_src_I = sprintf('./src_fourier_imag/%dsrcIF.mhd', k);
    writeMETA(reshape(imag(alldataSf(k,:,:,:)),hth, wth ),fname_src_I);  
    end
    
    FileNames=FilesT(k).name;
    Filefolder=FilesT(k).folder;
    path=strcat( Filefolder,'/',FileNames) ;
    alldataT(k,:,:,:)=loadMETA(path);
    [h1,hth,wth,lth]=size(alldataS(1,:,:,:,:));
    if (lth ~= 1)
    alldataTf(k,:,:,:)= image2fourier(reshape(alldataT(k,:,:,:),hth,wth,lth));
    alldataTf(k,:,:,:) = alldataTf(k,:,:,:)/ (hth*wth*lth);
    fname_tar_R = sprintf('./Rnet/tar_fourier_real/%dtarRF.mhd', k);
    writeMETA(reshape(real(alldataTf(k,:,:,:)), hth, wth, lth),fname_tar_R);
    fname_tar_I = sprintf('./Inet/tar_fourier_imag/%dtarIF.mhd', k);
    writeMETA(reshape(imag(alldataTf(k,:,:,:)), hth, wth, lth),fname_tar_I);
    % alldataTfcon(k,:,:)= reshape(getconj(reshape(alldataSf(k,:,:),hth,wth)),1,floor(hth/2)+1,wth);  
    else
    alldataTf(k,:,:)= image2fourier(reshape(alldataT(k,:,:,:),hth,wth));
    alldataTf(k,:,:) = alldataTf(k,:,:)/ (hth*wth);
    fname_tar_R = sprintf('./Rnet/tar_fourier_real/%dtarRF.mhd', k);
    writeMETA(reshape(real(alldataTf(k,:,:)), hth, wth),fname_tar_R);
    fname_tar_I = sprintf('./Inet/tar_fourier_imag/%dtarIF.mhd', k);
    writeMETA(reshape(imag(alldataTf(k,:,:)), hth, wth),fname_tar_I);
    end
end
