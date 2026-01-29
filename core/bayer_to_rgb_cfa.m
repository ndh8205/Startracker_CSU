function rgb_img = bayer_to_rgb_cfa(bayer_img)
% BAYER_TO_RGB_CFA Bayer 패턴을 RGB로 변환 (FPGA cfa.cpp 알고리즘 재현)
%
% FPGA cfa.cpp의 3x3 윈도우 기반 bilinear 보간과 동일

bayer_img = double(bayer_img);
[rows, cols] = size(bayer_img);

% 패딩 (3x3 윈도우를 위해)
padded = padarray(bayer_img, [1 1], 'replicate');

% 출력 이미지
R = zeros(rows, cols);
G = zeros(rows, cols);
B = zeros(rows, cols);

for row = 1:rows
    for col = 1:cols
        pr = row + 1;
        pc = col + 1;
        w = padded(pr-1:pr+1, pc-1:pc+1);

        r_idx = mod(row-1, 2);
        c_idx = mod(col-1, 2);
        pos_idx = r_idx * 2 + c_idx;

        switch pos_idx
            case 0  % R 위치
                R(row,col) = w(2,2);
                G(row,col) = (w(1,2) + w(2,1) + w(2,3) + w(3,2)) / 4;
                B(row,col) = (w(1,1) + w(1,3) + w(3,1) + w(3,3)) / 4;
            case 1  % Gr 위치
                R(row,col) = (w(2,1) + w(2,3)) / 2;
                G(row,col) = w(2,2);
                B(row,col) = (w(1,2) + w(3,2)) / 2;
            case 2  % Gb 위치
                R(row,col) = (w(1,2) + w(3,2)) / 2;
                G(row,col) = w(2,2);
                B(row,col) = (w(2,1) + w(2,3)) / 2;
            case 3  % B 위치
                R(row,col) = (w(1,1) + w(1,3) + w(3,1) + w(3,3)) / 4;
                G(row,col) = (w(1,2) + w(2,1) + w(2,3) + w(3,2)) / 4;
                B(row,col) = w(2,2);
        end
    end
end

R = max(0, min(255, R));
G = max(0, min(255, G));
B = max(0, min(255, B));

rgb_img = uint8(cat(3, R, G, B));
end
