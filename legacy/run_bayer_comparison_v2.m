% ============================================================
% ⚠ 폐기된 파일 (DEPRECATED) - v2
% ============================================================
% 이 파일은 더 이상 사용되지 않습니다.
%
% [대체 파일]
%   sub_main_1_bayer_comparison.m (최신 버전)
%
% [버전 이력]
%   v1 (run_bayer_comparison.m):
%     - 무작위 별 이미지 기반 (simulate_bayer_star_image 사용)
%     - 물리 모델 없음, 재현성 부족
%   v2 (이 파일):
%     - Hipparcos 별 카탈로그 기반으로 개선 (simulate_star_image_realistic 사용)
%     - 실제 별 등급/좌표 사용으로 물리적 정확도 향상
%     - 외부 경로 하드코딩 여전히 존재
%   최신 (sub_main_1_bayer_comparison.m):
%     - v2의 로직을 리팩토링하여 독립 실행형으로 변환
%     - 외부 경로 제거, data/ 폴더에 카탈로그 포함
%     - main_simulation.m과 연계하여 동일 파라미터 사용 보장
%
% [v2에서 v_최신으로의 주요 변경]
%   - 외부 addpath 제거 → data/ 폴더의 자체 카탈로그 사용
%   - 관측 방향 변경 (RA=90°,DEC=20° → RA=84°,DEC=-1°, 오리온 벨트)
%   - 센서 파라미터를 main_simulation.m과 공유
%   - 출력 폴더에 결과 이미지 자동 저장
%
% [참고] 이 파일의 핵심 로직은 sub_main_1_bayer_comparison.m에 통합되었습니다.
% ============================================================

%% run_bayer_comparison_v2.m
% Bayer → RGB → Gray vs Bayer → Gray 직접 변환 비교 시뮬레이션
% 실제 별 카탈로그 기반 (star_simulator 방식)
%
% 목적:
%   별센서에서 RGB 변환 과정이 불필요한지 검증

clear; close all; clc;

%% 경로 설정
% ※ 외부 경로 하드코딩 — 최신 버전에서는 제거됨
%   이 경로들은 특정 PC에서만 동작하므로 이식성이 없음
%   최신 코드는 data/ 폴더에 카탈로그를 포함하여 독립 실행 가능
addpath(genpath('D:\star_tracker_test\main_pj_code'));
addpath(genpath('D:\star_tracker_test\star_simulator_matlab'));

fprintf('========================================\n');
fprintf('  Bayer → Gray 변환 비교 (v2 - 실제 카탈로그)\n');
fprintf('========================================\n\n');

%% 센서 파라미터 (OV4689 기반)
sensor_params = struct();
sensor_params.myu = 2e-6;       % 픽셀 크기 [m]
sensor_params.f = 0.01042;      % 초점거리 [m]
sensor_params.l = 1280;         % 가로 픽셀
sensor_params.w = 720;          % 세로 픽셀
sensor_params.mag_limit = 6.0;  % 최대 등급

% 노이즈 파라미터
sensor_params.dark_current = 5;
sensor_params.read_noise = 3;
sensor_params.add_noise = true;

% FOV 계산
FOVx = rad2deg(2 * atan((sensor_params.myu*sensor_params.l/2) / sensor_params.f));
FOVy = rad2deg(2 * atan((sensor_params.myu*sensor_params.w/2) / sensor_params.f));

fprintf('센서 설정:\n');
fprintf('  해상도: %d x %d\n', sensor_params.l, sensor_params.w);
fprintf('  FOV: %.2f° x %.2f°\n', FOVx, FOVy);
fprintf('  픽셀 크기: %.1f µm\n', sensor_params.myu * 1e6);
fprintf('  초점거리: %.2f mm\n', sensor_params.f * 1000);
fprintf('\n');

%% 1. 시뮬레이션 위치 설정 (별이 많은 영역)
% --- 관측 방향 ---
% v2에서는 RA=90°, DEC=20° (오리온 근처, 비교적 별이 많은 영역)
% 최신 버전에서는 RA=84°, DEC=-1° (오리온 벨트, 삼태성)으로 변경
%   → 밝은 별 3개가 집중되어 별센서 테스트에 더 적합
% 은하수 근처 또는 별이 밀집한 영역 선택
ra_deg = 90;    % Right Ascension [deg] - Orion 근처
de_deg = 20;    % Declination [deg]
roll_deg = 0;   % Roll [deg]

fprintf('관측 방향:\n');
fprintf('  RA: %.1f°, DEC: %.1f°, Roll: %.1f°\n\n', ra_deg, de_deg, roll_deg);

%% 2. 별 이미지 생성 (실제 카탈로그 기반)
fprintf('1. 별 이미지 생성 (Hipparcos 카탈로그 기반)...\n');
[gray_ideal, bayer_img, star_info] = simulate_star_image_realistic(ra_deg, de_deg, roll_deg, sensor_params);

fprintf('   생성 완료!\n');
fprintf('   FOV 내 별: %d개\n', star_info.num_stars);
fprintf('   등급 범위: %.2f ~ %.2f\n', min(star_info.magnitudes), max(star_info.magnitudes));
fprintf('   이미지 최대값: %d, 배경: %d\n\n', max(bayer_img(:)), median(bayer_img(:)));

if star_info.num_stars < 3
    warning('별이 너무 적습니다! 다른 RA/DEC를 시도해보세요.');
end

%% 3. 변환 방식별 Grayscale 이미지 생성
fprintf('2. 변환 방식별 Grayscale 생성...\n');

% 방법 A: 현재 FPGA 파이프라인 (CFA → RGB → Gray)
tic;
rgb_img = bayer_to_rgb_cfa(bayer_img);
gray_fpga = rgb_to_gray_fpga(rgb_img);
time_fpga = toc;
fprintf('   A. FPGA 방식 (CFA→RGB→Gray): %.3f ms\n', time_fpga*1000);

% 방법 B1: Bayer 직접 - RAW 값 그대로
tic;
[gray_raw, ~] = bayer_to_gray_direct(bayer_img, 'raw');
time_raw = toc;
fprintf('   B1. RAW 직접: %.3f ms\n', time_raw*1000);

% 방법 B2: Bayer 직접 - 2x2 바이닝
tic;
[gray_binning, info_binning] = bayer_to_gray_direct(bayer_img, 'binning');
time_binning = toc;
fprintf('   B2. 2x2 바이닝: %.3f ms (해상도 %dx%d)\n', ...
    time_binning*1000, info_binning.output_size(2), info_binning.output_size(1));

% 방법 B3: Bayer 직접 - Green 채널
tic;
[gray_green, ~] = bayer_to_gray_direct(bayer_img, 'green');
time_green = toc;
fprintf('   B3. Green 채널: %.3f ms\n', time_green*1000);

% 방법 B4: Bayer 직접 - 가중 평균
tic;
[gray_weighted, ~] = bayer_to_gray_direct(bayer_img, 'weighted');
time_weighted = toc;
fprintf('   B4. 가중 평균: %.3f ms\n\n', time_weighted*1000);

%% 4. SNR 분석
fprintf('3. Peak SNR 분석...\n');

snr_results = struct();
snr_results.fpga = calculate_peak_snr(double(gray_fpga));
snr_results.raw = calculate_peak_snr(double(gray_raw));
snr_results.binning = calculate_peak_snr(double(gray_binning));
snr_results.green = calculate_peak_snr(double(gray_green));
snr_results.weighted = calculate_peak_snr(double(gray_weighted));

fprintf('   A. FPGA 방식:  %.2f dB\n', snr_results.fpga);
fprintf('   B1. RAW 직접:  %.2f dB\n', snr_results.raw);
fprintf('   B2. 2x2 바이닝: %.2f dB\n', snr_results.binning);
fprintf('   B3. Green 채널: %.2f dB\n', snr_results.green);
fprintf('   B4. 가중 평균:  %.2f dB\n\n', snr_results.weighted);

%% 5. 별 검출 성능 분석
fprintf('4. 별 검출 성능 분석...\n');

threshold = 15;
min_area = 2;

detection_results = struct();
detection_results.fpga = detect_stars_simple(gray_fpga, threshold, min_area);
detection_results.raw = detect_stars_simple(gray_raw, threshold, min_area);
detection_results.binning = detect_stars_simple(gray_binning, threshold, min_area);
detection_results.green = detect_stars_simple(gray_green, threshold, min_area);
detection_results.weighted = detect_stars_simple(gray_weighted, threshold, min_area);

fprintf('   검출된 별 (실제: %d개):\n', star_info.num_stars);
fprintf('   A. FPGA 방식:  %d개\n', detection_results.fpga.n_detected);
fprintf('   B1. RAW 직접:  %d개\n', detection_results.raw.n_detected);
fprintf('   B2. 2x2 바이닝: %d개\n', detection_results.binning.n_detected);
fprintf('   B3. Green 채널: %d개\n', detection_results.green.n_detected);
fprintf('   B4. 가중 평균:  %d개\n\n', detection_results.weighted.n_detected);

%% 6. Centroid 정확도 분석
fprintf('5. Centroid 정확도 분석...\n');

true_centroids = star_info.true_centroids;
match_radius = 5.0;

centroid_results = struct();
centroid_results.fpga = evaluate_centroid_accuracy(detection_results.fpga, true_centroids, match_radius);
centroid_results.raw = evaluate_centroid_accuracy(detection_results.raw, true_centroids, match_radius);
% ★ 바이닝 해상도 스케일링:
%   2×2 바이닝 출력은 원본의 절반 해상도 (1280×720 → 640×360)
%   ground truth 좌표도 0.5배 스케일링 필요
%   다른 방법은 원본 해상도 유지 → 스케일링 불필요
centroid_results.binning = evaluate_centroid_accuracy(detection_results.binning, true_centroids * 0.5, match_radius);
centroid_results.green = evaluate_centroid_accuracy(detection_results.green, true_centroids, match_radius);
centroid_results.weighted = evaluate_centroid_accuracy(detection_results.weighted, true_centroids, match_radius);

fprintf('   Centroid RMS 오차:\n');
fprintf('   A. FPGA 방식:  %.3f px (매칭: %d개)\n', centroid_results.fpga.rms_error, centroid_results.fpga.n_matched);
fprintf('   B1. RAW 직접:  %.3f px (매칭: %d개)\n', centroid_results.raw.rms_error, centroid_results.raw.n_matched);
fprintf('   B2. 2x2 바이닝: %.3f px (매칭: %d개)\n', centroid_results.binning.rms_error, centroid_results.binning.n_matched);
fprintf('   B3. Green 채널: %.3f px (매칭: %d개)\n', centroid_results.green.rms_error, centroid_results.green.n_matched);
fprintf('   B4. 가중 평균:  %.3f px (매칭: %d개)\n\n', centroid_results.weighted.rms_error, centroid_results.weighted.n_matched);

%% 7. 원본 시뮬레이션 이미지 (별도 Figure)
fprintf('6. 원본 시뮬레이션 이미지...\n');

figure('Position', [50 50 1400 500], 'Name', '원본 시뮬레이션 이미지', 'Color', 'k');

% 이상적인 Grayscale (노이즈 없음)
subplot(1,3,1);
imshow(uint8(star_info.ideal_gray), []);
hold on;
plot(star_info.true_centroids(:,1), star_info.true_centroids(:,2), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
for i = 1:size(star_info.true_centroids, 1)
    text(star_info.true_centroids(i,1)+15, star_info.true_centroids(i,2), ...
        sprintf('%.1f', star_info.magnitudes(i)), 'Color', 'y', 'FontSize', 9);
end
hold off;
title(sprintf('이상적 Grayscale (노이즈 없음)\n%d개 별, 등급 %.1f~%.1f', ...
    star_info.num_stars, min(star_info.magnitudes), max(star_info.magnitudes)), 'Color', 'w', 'FontSize', 12);

% Bayer 패턴 이미지 (노이즈 포함)
subplot(1,3,2);
imshow(bayer_img, []);
hold on;
plot(star_info.true_centroids(:,1), star_info.true_centroids(:,2), 'g+', 'MarkerSize', 10, 'LineWidth', 1.5);
hold off;
title(sprintf('Bayer 패턴 (RGGB + 노이즈)\n최대값: %d, 배경: %d', max(bayer_img(:)), median(bayer_img(:))), 'Color', 'w', 'FontSize', 12);

% Bayer 패턴 확대 (첫 번째 별 주변)
subplot(1,3,3);
if ~isempty(star_info.true_centroids)
    cx = round(star_info.true_centroids(1,1));
    cy = round(star_info.true_centroids(1,2));
    win = 30;
    x1 = max(1, cx-win); x2 = min(sensor_params.l, cx+win);
    y1 = max(1, cy-win); y2 = min(sensor_params.w, cy+win);

    crop_img = bayer_img(y1:y2, x1:x2);
    imshow(crop_img, []);
    title(sprintf('Bayer 확대 (별 #1 주변)\n등급: %.2f', star_info.magnitudes(1)), 'Color', 'w', 'FontSize', 12);

    % Bayer 패턴 격자 표시
    hold on;
    for r = 1:2:size(crop_img,1)
        for c = 1:2:size(crop_img,2)
            rectangle('Position', [c-0.5, r-0.5, 2, 2], 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 0.5);
        end
    end
    hold off;
end

sgtitle(sprintf('Star Tracker 시뮬레이션 (RA=%.0f°, DEC=%.0f°, FOV=%.1f°×%.1f°)', ...
    ra_deg, de_deg, FOVx, FOVy), 'FontSize', 14, 'Color', 'w');

%% 8. 결과 시각화
fprintf('7. 변환 방식 비교 시각화...\n');

figure('Position', [50 50 1800 900], 'Name', 'Bayer 변환 비교 (실제 카탈로그)', 'Color', 'k');

% 원본 Bayer
subplot(2,3,1);
imshow(bayer_img);
hold on;
plot(star_info.true_centroids(:,1), star_info.true_centroids(:,2), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
hold off;
title(sprintf('원본 Bayer (별 %d개)', star_info.num_stars), 'Color', 'w');

% FPGA 방식
subplot(2,3,2);
imshow(gray_fpga);
hold on;
if detection_results.fpga.n_detected > 0
    plot(detection_results.fpga.centroids(:,1), detection_results.fpga.centroids(:,2), 'g+', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;
title(sprintf('A. FPGA (CFA→RGB→Gray)\nSNR=%.1fdB, 검출=%d', snr_results.fpga, detection_results.fpga.n_detected), 'Color', 'w');

% RAW 직접
subplot(2,3,3);
imshow(gray_raw);
hold on;
if detection_results.raw.n_detected > 0
    plot(detection_results.raw.centroids(:,1), detection_results.raw.centroids(:,2), 'g+', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;
title(sprintf('B1. RAW 직접\nSNR=%.1fdB, 검출=%d', snr_results.raw, detection_results.raw.n_detected), 'Color', 'w');

% 2x2 바이닝
subplot(2,3,4);
imshow(gray_binning);
hold on;
if detection_results.binning.n_detected > 0
    plot(detection_results.binning.centroids(:,1), detection_results.binning.centroids(:,2), 'g+', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;
title(sprintf('B2. 2x2 바이닝 (640x360)\nSNR=%.1fdB, 검출=%d', snr_results.binning, detection_results.binning.n_detected), 'Color', 'w');

% Green 채널
subplot(2,3,5);
imshow(gray_green);
hold on;
if detection_results.green.n_detected > 0
    plot(detection_results.green.centroids(:,1), detection_results.green.centroids(:,2), 'g+', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;
title(sprintf('B3. Green 채널\nSNR=%.1fdB, 검출=%d', snr_results.green, detection_results.green.n_detected), 'Color', 'w');

% 가중 평균
subplot(2,3,6);
imshow(gray_weighted);
hold on;
if detection_results.weighted.n_detected > 0
    plot(detection_results.weighted.centroids(:,1), detection_results.weighted.centroids(:,2), 'g+', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;
title(sprintf('B4. 가중 평균\nSNR=%.1fdB, 검출=%d', snr_results.weighted, detection_results.weighted.n_detected), 'Color', 'w');

sgtitle(sprintf('Bayer → Grayscale 비교 (RA=%.0f°, DEC=%.0f°)', ra_deg, de_deg), 'FontSize', 14, 'Color', 'w');

%% 8. 성능 비교 막대 그래프
figure('Position', [100 100 1200 400], 'Name', '성능 비교');

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
title('SNR 비교');
grid on;

subplot(1,3,2);
bar(detect_vals);
hold on;
yline(star_info.num_stars, 'r--', 'LineWidth', 2);
set(gca, 'XTickLabel', methods);
ylabel('검출된 별 개수');
title(sprintf('별 검출 (실제: %d개)', star_info.num_stars));
grid on;

subplot(1,3,3);
bar(time_vals);
set(gca, 'XTickLabel', methods);
ylabel('처리 시간 (ms)');
title('처리 시간');
grid on;

sgtitle('성능 지표 비교', 'FontSize', 14);

%% 9. 결론 출력
fprintf('\n========================================\n');
fprintf('  분석 결론\n');
fprintf('========================================\n\n');

fprintf('1. Peak SNR:\n');
fprintf('   FPGA:     %.2f dB\n', snr_results.fpga);
fprintf('   RAW:      %.2f dB (차이: %+.2f dB)\n', snr_results.raw, snr_results.raw - snr_results.fpga);
fprintf('   Binning:  %.2f dB (차이: %+.2f dB)\n\n', snr_results.binning, snr_results.binning - snr_results.fpga);

fprintf('2. 별 검출률:\n');
fprintf('   FPGA:     %d/%d (%.1f%%)\n', detection_results.fpga.n_detected, star_info.num_stars, 100*detection_results.fpga.n_detected/star_info.num_stars);
fprintf('   RAW:      %d/%d (%.1f%%)\n', detection_results.raw.n_detected, star_info.num_stars, 100*detection_results.raw.n_detected/star_info.num_stars);
fprintf('   Binning:  %d/%d (%.1f%%)\n\n', detection_results.binning.n_detected, star_info.num_stars, 100*detection_results.binning.n_detected/star_info.num_stars);

fprintf('3. 처리 시간:\n');
fprintf('   FPGA:     %.1f ms\n', time_fpga*1000);
fprintf('   RAW:      %.1f ms (%.1fx 빠름)\n', time_raw*1000, time_fpga/time_raw);
fprintf('   Binning:  %.1f ms (%.1fx 빠름)\n\n', time_binning*1000, time_fpga/time_binning);

fprintf('4. 권장 사항:\n');
fprintf('   ★ RAW 직접 방식이 별센서에 적합\n');
fprintf('   ★ CFA + RGB2Gray IP 제거 가능\n');
fprintf('   ★ 바이닝: SNR 향상 필요시 고려\n');

fprintf('\n========================================\n');
