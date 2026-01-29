function [gray_img, method_info] = bayer_to_gray_direct(bayer_img, method)
% BAYER_TO_GRAY_DIRECT Bayer 패턴에서 직접 Grayscale로 변환
%
% 방법:
%   'raw'      - Bayer 값 그대로 사용 (해상도 유지)
%   'binning'  - 2x2 바이닝 (R+G+G+B)/4 (해상도 1/2)
%   'green'    - Green 채널만 추출 (해상도 유지)
%   'weighted' - 2x2 가중 평균 (해상도 유지)

if nargin < 2
    method = 'raw';
end

bayer_img = double(bayer_img);
[rows, cols] = size(bayer_img);

method_info.method = method;
method_info.input_size = [rows, cols];

switch lower(method)
    case 'raw'
        % RAW 값 그대로 사용
        gray_img = uint8(bayer_img);
        method_info.output_size = [rows, cols];
        method_info.description = 'RAW 값 그대로';

    case 'binning'
        % 2x2 바이닝
        out_rows = floor(rows / 2);
        out_cols = floor(cols / 2);
        gray_img = zeros(out_rows, out_cols);

        for r = 1:out_rows
            for c = 1:out_cols
                br = (r-1)*2 + 1;
                bc = (c-1)*2 + 1;
                R  = bayer_img(br, bc);
                Gr = bayer_img(br, bc+1);
                Gb = bayer_img(br+1, bc);
                B  = bayer_img(br+1, bc+1);
                gray_img(r, c) = (R + Gr + Gb + B) / 4;
            end
        end

        gray_img = uint8(round(gray_img));
        method_info.output_size = [out_rows, out_cols];
        method_info.description = '2x2 바이닝';

    case 'green'
        % Green 채널만 추출 + 보간
        gray_img = zeros(rows, cols);
        green_mask = false(rows, cols);

        for r = 1:rows
            for c = 1:cols
                r_idx = mod(r-1, 2);
                c_idx = mod(c-1, 2);
                if (r_idx == 0 && c_idx == 1) || (r_idx == 1 && c_idx == 0)
                    green_mask(r, c) = true;
                end
            end
        end

        gray_img(green_mask) = bayer_img(green_mask);

        padded = padarray(bayer_img, [1 1], 'replicate');
        for r = 1:rows
            for c = 1:cols
                if ~green_mask(r, c)
                    pr = r + 1;
                    pc = c + 1;
                    gray_img(r, c) = (padded(pr-1,pc) + padded(pr+1,pc) + ...
                                     padded(pr,pc-1) + padded(pr,pc+1)) / 4;
                end
            end
        end

        gray_img = uint8(round(gray_img));
        method_info.output_size = [rows, cols];
        method_info.description = 'Green 채널 추출';

    case 'weighted'
        % 2x2 가중 평균
        gray_img = zeros(rows, cols);
        padded = padarray(bayer_img, [1 1], 'replicate');

        for r = 1:rows
            for c = 1:cols
                pr = r + 1;
                pc = c + 1;
                r_idx = mod(r-1, 2);
                c_idx = mod(c-1, 2);
                pos_idx = r_idx * 2 + c_idx;

                switch pos_idx
                    case 0
                        gray_img(r,c) = (padded(pr,pc) + ...
                            (padded(pr-1,pc) + padded(pr,pc+1) + ...
                             padded(pr+1,pc) + padded(pr,pc-1))/4 * 2 + ...
                            padded(pr+1,pc+1)) / 4;
                    case 1
                        gray_img(r,c) = ((padded(pr,pc-1) + padded(pr,pc+1))/2 + ...
                            padded(pr,pc) * 2 + ...
                            (padded(pr-1,pc) + padded(pr+1,pc))/2) / 4;
                    case 2
                        gray_img(r,c) = ((padded(pr-1,pc) + padded(pr+1,pc))/2 + ...
                            padded(pr,pc) * 2 + ...
                            (padded(pr,pc-1) + padded(pr,pc+1))/2) / 4;
                    case 3
                        gray_img(r,c) = (padded(pr-1,pc-1) + ...
                            (padded(pr-1,pc) + padded(pr,pc+1) + ...
                             padded(pr+1,pc) + padded(pr,pc-1))/4 * 2 + ...
                            padded(pr,pc)) / 4;
                end
            end
        end

        gray_img = uint8(round(gray_img));
        method_info.output_size = [rows, cols];
        method_info.description = '2x2 가중 평균';

    otherwise
        error('Unknown method: %s', method);
end
end
