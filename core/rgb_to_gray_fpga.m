function gray_img = rgb_to_gray_fpga(rgb_img)
% RGB_TO_GRAY_FPGA RGB를 Grayscale로 변환 (FPGA rgb2gray.cpp 알고리즘)
%
% Y = (R + 2*G + B) / 4

rgb_img = double(rgb_img);

R = rgb_img(:,:,1);
G = rgb_img(:,:,2);
B = rgb_img(:,:,3);

gray_img = (R + 2*G + B) / 4;

gray_img = round(gray_img);
gray_img = max(0, min(255, gray_img));
gray_img = uint8(gray_img);
end
