a=loadMETA('./mri_1.mhd');
b=imresize3(a,[160,192,224]);
B = imrotate3(b,180,[0 0 1]);
C=imresize3(B,[160,192,224]);
writeMETA(C,'./mri1.mhd');
