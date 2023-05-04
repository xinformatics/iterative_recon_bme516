% here we create a phantom and k-space data (ground truth)
P = phantom('Modified Shepp-Logan',256);
figure(1); imshow(abs(P).^0.5)
kSpaceData = fftshift(fft2(fftshift(P)));
figure(2); imagesc(abs(kSpaceData).^0.3);colormap("gray"); axis equal off


% here we simulate a case that part of the k-space data are corrupted
kSpaceData_corrupted = kSpaceData;
kSpaceData_corrupted(:,180) = kSpaceData(:,180)*10;
P_corrupted = ifftshift(ifft2(ifftshift(kSpaceData_corrupted)));
figure(3); imshow(abs(P_corrupted).^0.5)
figure(4); imagesc(abs(kSpaceData_corrupted).^0.3);colormap("gray"); axis equal off

% Assuming that we know that line#180 of the k-space data are corrupted 
% but don't know the correct value. 
% We will use an iterative approach to produce an image with reduced
% artifact

% Step 1:
corruped_image = P_corrupted;
corruped_image_fixed = corruped_image;
corruped_image_fixed(:,[1:32,end-31:end])=0;
figure(5); imshow(abs(corruped_image_fixed).^0.5);
corrupted_kdata_fixed = fftshift(fft2(fftshift(corruped_image_fixed)));
figure(6); imagesc(abs(corrupted_kdata_fixed).^0.3);colormap("gray"); axis equal off
% Step 2:
corrupted_kdata_fixed_again = kSpaceData_corrupted;
corrupted_kdata_fixed_again(:,180) = corrupted_kdata_fixed(:,180);
corruped_image_fixed_again = ifftshift(ifft2(ifftshift(corrupted_kdata_fixed_again)));
figure(7); imshow(abs(corruped_image_fixed_again).^0.5);
figure(8); imagesc(abs(corrupted_kdata_fixed_again).^0.3);colormap("gray"); axis equal off

% check:
figure(100);
plot(1:256,abs(kSpaceData_corrupted(:,180)),"g", 1:256, abs(corrupted_kdata_fixed_again(:,180)),"r", 1:256,abs(kSpaceData(:,180)),"k")
% since the k-data are improved, the above listed procedure could be
% repeated!
% Step 1a:
corruped_image = corruped_image_fixed_again;% note that we assign new values to corrupted image
corruped_image_fixed = corruped_image;
corruped_image_fixed(:,[1:32,end-31:end])=0;
figure(9); imshow(abs(corruped_image_fixed).^0.5);
corrupted_kdata_fixed = fftshift(fft2(fftshift(corruped_image_fixed)));
figure(10); imagesc(abs(corrupted_kdata_fixed).^0.3);colormap("gray"); axis equal off
% Step 2a:
corrupted_kdata_fixed_again = kSpaceData_corrupted;
corrupted_kdata_fixed_again(:,180) = corrupted_kdata_fixed(:,180);
corruped_image_fixed_again = ifftshift(ifft2(ifftshift(corrupted_kdata_fixed_again)));
figure(11); imshow(abs(corruped_image_fixed_again).^0.5);
figure(12); imagesc(abs(corrupted_kdata_fixed_again).^0.3);colormap("gray"); axis equal off

% check:
figure(101);
plot(1:256,abs(kSpaceData_corrupted(:,180)),"g", 1:256, abs(corrupted_kdata_fixed_again(:,180)),"r", 1:256,abs(kSpaceData(:,180)),"k")
% since the k-data are getting closer to the ground truth, we should write
% a for-loop to repeat the above-listed procedure again, for many more
% times!! (say 200 times)

for cnt = 1:200
    % Step 1a:
    corruped_image = corruped_image_fixed_again;% note that we assign new values to corrupted image
    corruped_image_fixed = corruped_image;
    corruped_image_fixed(:,[1:32,end-31:end])=0;
    corrupted_kdata_fixed = fftshift(fft2(fftshift(corruped_image_fixed)));
    % Step 2a:
    corrupted_kdata_fixed_again = kSpaceData_corrupted;
    corrupted_kdata_fixed_again(:,180) = corrupted_kdata_fixed(:,180);
    corruped_image_fixed_again = ifftshift(ifft2(ifftshift(corrupted_kdata_fixed_again)));
end
% check:
figure(102);
plot(1:256,abs(kSpaceData_corrupted(:,180)),"g", 1:256, abs(corrupted_kdata_fixed_again(:,180)),"r", 1:256,abs(kSpaceData(:,180)),"k")
% looks good -- the k-data are getting very close to the ground truth

figure(13); imshow(abs(corruped_image_fixed_again).^0.5);
figure(14); imagesc(abs(corrupted_kdata_fixed_again).^0.3);colormap("gray"); axis equal off







