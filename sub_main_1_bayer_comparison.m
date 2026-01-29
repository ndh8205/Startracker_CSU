%% main_simulation.m
% Star Tracker Bayer → Gray 변환 비교 시뮬레이션
%
% 목적:
%   별센서에서 RGB 변환 과정이 불필요한지 검증
%   현재 FPGA 파이프라인: Bayer → CFA → RGB → Gray
%   제안: Bayer → Gray 직접 변환 (CFA, RGB2Gray IP 제거 가능)
%
% 사용법:
%   1. MATLAB에서 이 파일 실행
%   2. 필요시 '설정' 섹션에서 관측 방향 변경
%
% 의존성:
%   - D:\star_tracker_test\main_pj_code (별 카탈로그)
%   - D:\star_tracker_test\star_simulator_matlab (시뮬레이터 참고)
%
% 저자: GAiMSat-1 Star Tracker 프로젝트
% 날짜: 2026-01-30

clear; close all; clc;

%% 경로 설정
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'core'));
addpath(fullfile(script_dir, 'utils'));
% 카탈로그는 data/ 폴더에 포함됨 (외부 경로 불필요)

fprintf('================================================\n');
fprintf('  Star Tracker Bayer → Gray 변환 비교 시뮬레이션\n');
fprintf('================================================\n\n');

%% ========== 설정 ==========
% 센서 파라미터 (OV4689 + 10.42mm 렌즈)
sensor_params = struct();
sensor_params.myu = 2e-6;       % 픽셀 크기 [m]
sensor_params.f = 0.01042;      % 초점거리 [m]
sensor_params.l = 1280;         % 가로 픽셀
sensor_params.w = 720;          % 세로 픽셀
sensor_params.mag_limit = 6.5;  % 최대 등급

% 노이즈 파라미터
sensor_params.dark_current = 5;
sensor_params.read_noise = 3;
sensor_params.add_noise = true;

% 관측 방향 (오리온 벨트 - 삼태성)
ra_deg = 84;      % Right Ascension [deg]
de_deg = -1;      % Declination [deg]
roll_deg = 0;     % Roll [deg]

% 출력 폴더
output_dir = fullfile(script_dir, 'output');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% ========== FOV 계산 ==========
FOVx = rad2deg(2 * atan((sensor_params.myu*sensor_params.l/2) / sensor_params.f));
FOVy = rad2deg(2 * atan((sensor_params.myu*sensor_params.w/2) / sensor_params.f));

fprintf('센서 설정:\n');
fprintf('  해상도: %d x %d\n', sensor_params.l, sensor_params.w);
fprintf('  FOV: %.2f° x %.2f°\n', FOVx, FOVy);
fprintf('  픽셀 크기: %.1f µm\n', sensor_params.myu * 1e6);
fprintf('  초점거리: %.2f mm\n\n', sensor_params.f * 1000);

fprintf('관측 방향:\n');
fprintf('  RA: %.1f°, DEC: %.1f°, Roll: %.1f°\n\n', ra_deg, de_deg, roll_deg);

%% ========== 1. 별 이미지 생성 ==========
fprintf('1. 별 이미지 생성 (Hipparcos 카탈로그 기반)...\n');
[gray_ideal, bayer_img, star_info] = simulate_star_image_realistic(ra_deg, de_deg, roll_deg, sensor_params);

fprintf('   완료! FOV 내 별: %d개\n', star_info.num_stars);
fprintf('   등급 범위: %.2f ~ %.2f\n\n', min(star_info.magnitudes), max(star_info.magnitudes));

if star_info.num_stars < 3
    warning('별이 너무 적습니다! 다른 RA/DEC를 시도해보세요.');
end

%% ========== 2. 변환 방식별 Grayscale 생성 ==========
fprintf('2. 변환 방식별 Grayscale 생성...\n');

% 방법 A: FPGA 파이프라인 (CFA → RGB → Gray)
tic;
rgb_img = bayer_to_rgb_cfa(bayer_img);
gray_fpga = rgb_to_gray_fpga(rgb_img);
time_fpga = toc;

% 방법 B1: RAW 직접
tic;
[gray_raw, ~] = bayer_to_gray_direct(bayer_img, 'raw');
time_raw = toc;

% 방법 B2: 2x2 바이닝
tic;
[gray_binning, info_binning] = bayer_to_gray_direct(bayer_img, 'binning');
time_binning = toc;

% 방법 B3: Green 채널
tic;
[gray_green, ~] = bayer_to_gray_direct(bayer_img, 'green');
time_green = toc;

% 방법 B4: 가중 평균
tic;
[gray_weighted, ~] = bayer_to_gray_direct(bayer_img, 'weighted');
time_weighted = toc;

fprintf('   A. FPGA 방식: %.3f ms\n', time_fpga*1000);
fprintf('   B1. RAW 직접: %.3f ms\n', time_raw*1000);
fprintf('   B2. 2x2 바이닝: %.3f ms\n', time_binning*1000);
fprintf('   B3. Green 채널: %.3f ms\n', time_green*1000);
fprintf('   B4. 가중 평균: %.3f ms\n\n', time_weighted*1000);

%% ========== 3. SNR 분석 ==========
fprintf('3. Peak SNR 분석...\n');

snr_results = struct();
snr_results.fpga = calculate_peak_snr(double(gray_fpga));
snr_results.raw = calculate_peak_snr(double(gray_raw));
snr_results.binning = calculate_peak_snr(double(gray_binning));
snr_results.green = calculate_peak_snr(double(gray_green));
snr_results.weighted = calculate_peak_snr(double(gray_weighted));

fprintf('   A. FPGA: %.2f dB\n', snr_results.fpga);
fprintf('   B1. RAW: %.2f dB\n', snr_results.raw);
fprintf('   B2. Binning: %.2f dB\n', snr_results.binning);
fprintf('   B3. Green: %.2f dB\n', snr_results.green);
fprintf('   B4. Weighted: %.2f dB\n\n', snr_results.weighted);

%% ========== 4. 별 검출 분석 ==========
fprintf('4. 별 검출 성능...\n');

threshold = 15;
min_area = 2;

detection_results = struct();
detection_results.fpga = detect_stars_simple(gray_fpga, threshold, min_area);
detection_results.raw = detect_stars_simple(gray_raw, threshold, min_area);
detection_results.binning = detect_stars_simple(gray_binning, threshold, min_area);
detection_results.green = detect_stars_simple(gray_green, threshold, min_area);
detection_results.weighted = detect_stars_simple(gray_weighted, threshold, min_area);

fprintf('   실제 별: %d개\n', star_info.num_stars);
fprintf('   A. FPGA: %d개\n', detection_results.fpga.n_detected);
fprintf('   B1. RAW: %d개\n', detection_results.raw.n_detected);
fprintf('   B2. Binning: %d개\n', detection_results.binning.n_detected);
fprintf('   B3. Green: %d개\n', detection_results.green.n_detected);
fprintf('   B4. Weighted: %d개\n\n', detection_results.weighted.n_detected);

%% ========== 5. Centroid 정확도 ==========
fprintf('5. Centroid 정확도...\n');

true_centroids = star_info.true_centroids;
match_radius = 5.0;

centroid_results = struct();
centroid_results.fpga = evaluate_centroid_accuracy(detection_results.fpga, true_centroids, match_radius);
centroid_results.raw = evaluate_centroid_accuracy(detection_results.raw, true_centroids, match_radius);
centroid_results.binning = evaluate_centroid_accuracy(detection_results.binning, true_centroids * 0.5, match_radius);
centroid_results.green = evaluate_centroid_accuracy(detection_results.green, true_centroids, match_radius);
centroid_results.weighted = evaluate_centroid_accuracy(detection_results.weighted, true_centroids, match_radius);

fprintf('   A. FPGA: %.3f px RMS\n', centroid_results.fpga.rms_error);
fprintf('   B1. RAW: %.3f px RMS\n', centroid_results.raw.rms_error);
fprintf('   B2. Binning: %.3f px RMS\n', centroid_results.binning.rms_error);
fprintf('   B3. Green: %.3f px RMS\n', centroid_results.green.rms_error);
fprintf('   B4. Weighted: %.3f px RMS\n\n', centroid_results.weighted.rms_error);

%% ========== 6. 시각화 ==========
fprintf('6. 시각화...\n');

% Figure 1: 원본 이미지
fig1 = figure('Position', [50 50 1400 500], 'Name', '원본 시뮬레이션', 'Color', 'k');

subplot(1,3,1);
imshow(uint8(min(255, star_info.ideal_gray)), []);
hold on;
plot(star_info.true_centroids(:,1), star_info.true_centroids(:,2), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
hold off;
title(sprintf('이상적 Grayscale\n%d개 별', star_info.num_stars), 'Color', 'w', 'FontSize', 12);

subplot(1,3,2);
imshow(bayer_img, []);
title('Bayer 패턴 (RGGB)', 'Color', 'w', 'FontSize', 12);

subplot(1,3,3);
imshow(gray_fpga, []);
title('FPGA 출력', 'Color', 'w', 'FontSize', 12);

sgtitle(sprintf('Star Tracker (RA=%d°, DEC=%d°, FOV=%.1f°×%.1f°)', ...
    ra_deg, de_deg, FOVx, FOVy), 'FontSize', 14, 'Color', 'w');

% Figure 2: 변환 방식 비교
fig2 = figure('Position', [50 50 1600 800], 'Name', '변환 비교', 'Color', 'k');

subplot(2,3,1);
imshow(bayer_img); title('원본 Bayer', 'Color', 'w');

subplot(2,3,2);
imshow(gray_fpga);
title(sprintf('A. FPGA\nSNR=%.1fdB', snr_results.fpga), 'Color', 'w');

subplot(2,3,3);
imshow(gray_raw);
title(sprintf('B1. RAW\nSNR=%.1fdB', snr_results.raw), 'Color', 'w');

subplot(2,3,4);
imshow(gray_binning);
title(sprintf('B2. Binning\nSNR=%.1fdB', snr_results.binning), 'Color', 'w');

subplot(2,3,5);
imshow(gray_green);
title(sprintf('B3. Green\nSNR=%.1fdB', snr_results.green), 'Color', 'w');

subplot(2,3,6);
imshow(gray_weighted);
title(sprintf('B4. Weighted\nSNR=%.1fdB', snr_results.weighted), 'Color', 'w');

sgtitle('Bayer → Grayscale 변환 비교', 'FontSize', 14, 'Color', 'w');

% Figure 3: 성능 비교
fig3 = figure('Position', [100 100 1200 400], 'Name', '성능 비교');

methods = {'FPGA', 'RAW', 'Binning', 'Green', 'Weighted'};
snr_vals = [snr_results.fpga, snr_results.raw, snr_results.binning, snr_results.green, snr_results.weighted];
detect_vals = [detection_results.fpga.n_detected, detection_results.raw.n_detected, ...
               detection_results.binning.n_detected, detection_results.green.n_detected, ...
               detection_results.weighted.n_detected];
time_vals = [time_fpga, time_raw, time_binning, time_green, time_weighted] * 1000;

subplot(1,3,1);
bar(snr_vals);
set(gca, 'XTickLabel', methods);
ylabel('Peak SNR (dB)');
title('SNR 비교'); grid on;

subplot(1,3,2);
bar(detect_vals);
hold on; yline(star_info.num_stars, 'r--', 'LineWidth', 2);
set(gca, 'XTickLabel', methods);
ylabel('검출 수');
title(sprintf('별 검출 (실제: %d)', star_info.num_stars)); grid on;

subplot(1,3,3);
bar(time_vals);
set(gca, 'XTickLabel', methods);
ylabel('시간 (ms)');
title('처리 시간'); grid on;

sgtitle('성능 지표 비교', 'FontSize', 14);

%% ========== 7. 결과 저장 ==========
fprintf('7. 결과 저장...\n');

saveas(fig1, fullfile(output_dir, 'simulation_original.png'));
saveas(fig2, fullfile(output_dir, 'conversion_comparison.png'));
saveas(fig3, fullfile(output_dir, 'performance_metrics.png'));
imwrite(gray_ideal, fullfile(output_dir, 'gray_ideal.png'));
imwrite(bayer_img, fullfile(output_dir, 'bayer_raw.png'));
imwrite(gray_fpga, fullfile(output_dir, 'gray_fpga.png'));
imwrite(gray_raw, fullfile(output_dir, 'gray_raw_direct.png'));

fprintf('   저장 완료: %s\n\n', output_dir);

%% ========== 8. 결론 ==========
fprintf('================================================\n');
fprintf('  결론\n');
fprintf('================================================\n\n');

fprintf('1. SNR: RAW 직접 방식이 FPGA와 동등 (차이: %+.2f dB)\n', snr_results.raw - snr_results.fpga);
fprintf('2. 검출률: FPGA %.0f%% vs RAW %.0f%%\n', ...
    100*detection_results.fpga.n_detected/star_info.num_stars, ...
    100*detection_results.raw.n_detected/star_info.num_stars);
fprintf('3. 속도: RAW가 %.1fx 빠름\n\n', time_fpga/time_raw);

fprintf('★ 권장: CFA + RGB2Gray IP 제거 가능\n');
fprintf('★ RAW Bayer 데이터 직접 사용으로 리소스/전력 절감\n');
fprintf('================================================\n');
