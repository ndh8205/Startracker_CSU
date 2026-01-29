function snr = calculate_peak_snr(img)
% CALCULATE_PEAK_SNR Peak SNR 계산
%
% Peak SNR = 20 * log10(peak_signal / noise_std)
% 별 영역의 최대값과 배경 노이즈의 표준편차 비율

img = double(img);

% 배경 추정 (중앙값)
bg = median(img(:));

% 배경 노이즈 표준편차 (배경 영역만)
bg_mask = img < (bg + 10);
if sum(bg_mask(:)) > 100
    noise_std = std(img(bg_mask));
else
    noise_std = std(img(:));
end

% Peak 신호 (배경 대비)
peak_signal = max(img(:)) - bg;

if noise_std < 0.1
    snr = 100;
else
    snr = 20 * log10(peak_signal / noise_std);
end
end
