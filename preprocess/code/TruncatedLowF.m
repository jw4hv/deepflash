 %%%%%input x (k, 3, 128, 128)
function y = TruncatedLowF (x, truncX, truncY, truncZ)
%x = alldataVf(2,:,:,:);
[n,cop,h,w,d]=size(x);
x=reshape(x,cop,h,w);
%%%%%%%%%%%%%%%%%%%Cut high frequency %%%%%%%%%%%%%%%% 
y = zeros(cop, truncX+1, truncY+1, truncZ);
for id = 1:cop
for i = ((h/2)- (truncX/2)+ 1): ((h/2)- (truncX/2)+ truncX+1)
    for j =((w/2)- (truncY/2) +1 ): ((w/2)- (truncY/2) + truncY+1 )
        %for k =((d/2)- (truncZ/2)): ((h/2) + (truncZ/2))
         y (id, i-(h/2-truncX/2), j-(h/2-truncX/2)) = x (id,i,j);
        %end
    end
end
end

