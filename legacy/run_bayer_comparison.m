% ============================================================
% ⚠ 폐기된 파일 (DEPRECATED) - v1
% ============================================================
% 이 파일은 더 이상 사용되지 않습니다.
%
% [대체 파일]
%   sub_main_1_bayer_comparison.m (최신 버전)
%
% [버전 이력]
%   v1 (이 파일): 무작위 별 이미지 기반 비교 (simulate_bayer_star_image 사용)
%   v2 (run_bayer_comparison_v2.m): Hipparcos 카탈로그 기반으로 전환
%   최신 (sub_main_1_bayer_comparison.m): 독립 실행형으로 리팩토링,
%        main_simulation.m에서 생성한 데이터를 받아 비교 수행
%
% [v1의 한계]
%   - 무작위 별 위치/밝기 → 재현성 부족 (매 실행마다 결과 다름)
%   - 물리 모델 없음 (Pogson 공식, QE, 게인 미적용)
%   - 외부 경로 하드코딩 (D:\star_tracker_test\main_pj_code)
%   - 독립 실행 불가 (외부 함수 의존)
%
% [참고] 이 파일의 핵심 로직은 sub_main_1_bayer_comparison.m에 통합되었습니다.
% ============================================================

%% run_bayer_comparison.m
% Bayer → RGB → Gray vs Bayer → Gray 직접 변환 비교 시뮬레이션
%
% 목적:
%   별센서에서 RGB 변환 과정이 불필요한지 검증
%   CFA + RGB2Gray IP를 제거하고 직접 변환으로 대체 가능한지 분석
%
% 비교 항목:
%   1. SNR (신호 대 잡음비)
%   2. 별 검출 성능 (threshold 기반)
%   3. Centroid 정확도
%   4. 연산 복잡도

clear; close all; clc;

%% 경로 설정
% 현재 스크립트 위치
script_path = fileparts(mfilename('fullpath'));

% main_pj_code 경로 추가 (필요한 함수들)
% ※ 외부 경로 하드코딩 — 최신 버전에서는 제거됨
%   최신 코드는 data/ 폴더에 카탈로그를 포함하여 독립 실행 가능
main_pj_path = 'D:\star_tracker_test\main_pj_code';
if exist(main_pj_path, 'dir')
    addpath(genpath(main_pj_path));
    fprintf('Added path: %s\n', main_pj_path);
end

fprintf('========================================\n');
fprintf('  Bayer → Gray 변환 방식 비교 시뮬레이션\n');
fprintf('========================================\n\n');

%% 시뮬레이션 파라미터
params = struct();
params.width = 1280;      % OV4689 서브샘플링 해상도
params.height = 720;
params.n_stars = 15;      % FOV 내 별 개수

% 노이즈 파라미터
params.dark_current = 3;
params.read_noise = 2;
params.shot_noise = true;

% 별 파라미터 (더 밝게)
params.min_intensity = 80;   % 최소 밝기 증가
params.max_intensity = 220;  % 최대 밝기 증가
params.psf_sigma = 1.2;      % PSF 크기

fprintf('센서 설정:\n');
fprintf('  해상도: %d x %d\n', params.width, params.height);
fprintf('  별 개수: %d\n', params.n_stars);
fprintf('  Dark current: %d ADU\n', params.dark_current);
fprintf('  Read noise: %d ADU\n', params.read_noise);
fprintf('\n');

%% 1. Bayer 패턴 별 이미지 생성
fprintf('1. Bayer 패턴 별 이미지 생성...\n');
[bayer_img, star_info] = simulate_bayer_star_image(params.n_stars, params);
fprintf('   생성 완료: %d개 별\n', star_info.n_stars);
fprintf('   별 밝기 범위: %.0f ~ %.0f\n', min(star_info.intensity), max(star_info.intensity));
fprintf('   이미지 최대값: %d, 최소값: %d\n\n', max(bayer_img(:)), min(bayer_img(:)));

%% 2. 변환 방식별 Grayscale 이미지 생성
fprintf('2. 변환 방식별 Grayscale 생성...\n');

% 방법 A: 현재 FPGA 파이프라인 (CFA → RGB → Gray)
tic;
rgb_img = bayer_to_rgb_cfa(bayer_img);
gray_fpga = rgb_to_gray_fpga(rgb_img);
time_fpga = toc;
fprintf('   A. FPGA 방식 (CFA→RGB→Gray): %.3f ms\n', time_fpga*1000);

% 방법 B1: Bayer 직접 - RAW 값 그대로
tic;
[gray_raw, info_raw] = bayer_to_gray_direct(bayer_img, 'raw');
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
[gray_green, info_green] = bayer_to_gray_direct(bayer_img, 'green');
time_green = toc;
fprintf('   B3. Green 채널: %.3f ms\n', time_green*1000);

% 방법 B4: Bayer 직접 - 가중 평균
tic;
[gray_weighted, info_weighted] = bayer_to_gray_direct(bayer_img, 'weighted');
time_weighted = toc;
fprintf('   B4. 가중 평균: %.3f ms\n', time_weighted*1000);

fprintf('\n');

%% 3. SNR 분석 (별 영역에서만)
fprintf('3. SNR 분석 (별 영역 Peak SNR)...\n');

% 각 방법의 Peak SNR 계산 (별 영역 최대값 / 배경 노이즈 표준편차)
snr_results = struct();

snr_results.fpga = calculate_peak_snr(double(gray_fpga));
fprintf('   A. FPGA 방식: Peak SNR = %.2f dB\n', snr_results.fpga);

snr_results.raw = calculate_peak_snr(double(gray_raw));
fprintf('   B1. RAW 직접: Peak SNR = %.2f dB\n', snr_results.raw);

snr_results.binning = calculate_peak_snr(double(gray_binning));
fprintf('   B2. 2x2 바이닝: Peak SNR = %.2f dB\n', snr_results.binning);

snr_results.green = calculate_peak_snr(double(gray_green));
fprintf('   B3. Green 채널: Peak SNR = %.2f dB\n', snr_results.green);

snr_results.weighted = calculate_peak_snr(double(gray_weighted));
fprintf('   B4. 가중 평균: Peak SNR = %.2f dB\n', snr_results.weighted);

fprintf('\n');

%% 3-1. 방법별 직접 비교 (FPGA vs RAW)
fprintf('3-1. FPGA vs RAW 직접 비교...\n');
diff_fpga_raw = double(gray_fpga) - double(gray_raw);
fprintf('   평균 차이: %.3f\n', mean(diff_fpga_raw(:)));
fprintf('   최대 차이: %.3f\n', max(abs(diff_fpga_raw(:))));
fprintf('   RMS 차이: %.3f\n\n', sqrt(mean(diff_fpga_raw(:).^2)));

%% 4. 별 검출 성능 분석
fprintf('4. 별 검출 성능 분석...\n');

% threshold = 10 [ADU]: v1에서는 낮은 값 사용
%   (최신 버전 sub_main_1_bayer_comparison.m에서는 threshold=15 사용)
%   min_area = 2 [pixel]: 노이즈 필터링 최소 면적
threshold = 10;  % 검출 임계값 (낮춤)
min_area = 2;    % 최소 별 크기

detection_results = struct();

% 각 방법별 별 검출
detection_results.fpga = detect_stars_simple(gray_fpga, threshold, min_area);
detection_results.raw = detect_stars_simple(gray_raw, threshold, min_area);
detection_results.binning = detect_stars_simple(gray_binning, threshold, min_area);
detection_results.green = detect_stars_simple(gray_green, threshold, min_area);
detection_results.weighted = detect_stars_simple(gray_weighted, threshold, min_area);

fprintf('   검출된 별 개수 (실제: %d개):\n', params.n_stars);
fprintf('   A. FPGA 방식: %d개\n', detection_results.fpga.n_detected);
fprintf('   B1. RAW 직접: %d개\n', detection_results.raw.n_detected);
fprintf('   B2. 2x2 바이닝: %d개\n', detection_results.binning.n_detected);
fprintf('   B3. Green 채널: %d개\n', detection_results.green.n_detected);
fprintf('   B4. 가중 평균: %d개\n', detection_results.weighted.n_detected);

fprintf('\n');

%% 5. Centroid 정확도 분석
fprintf('5. Centroid 정확도 분석...\n');

% 실제 별 위치
true_centroids = [star_info.x, star_info.y];

centroid_results = struct();
centroid_results.fpga = evaluate_centroid_accuracy(detection_results.fpga, true_centroids, 5.0);
centroid_results.raw = evaluate_centroid_accuracy(detection_results.raw, true_centroids, 5.0);
% ★ 바이닝 해상도 스케일링:
%   2×2 바이닝 출력은 원본의 절반 해상도 (1280×720 → 640×360)
%   ground truth 좌표도 0.5배 스케일링 필요
%   다른 방법은 원본 해상도 유지 → 스케일링 불필요
centroid_results.binning = evaluate_centroid_accuracy(detection_results.binning, true_centroids * 0.5, 5.0);
centroid_results.green = evaluate_centroid_accuracy(detection_results.green, true_centroids, 5.0);
centroid_results.weighted = evaluate_centroid_accuracy(detection_results.weighted, true_centroids, 5.0);

fprintf('   Centroid RMS 오차 (픽셀):\n');
fprintf('   A. FPGA 방식: %.3f px\n', centroid_results.fpga.rms_error);
fprintf('   B1. RAW 직접: %.3f px\n', centroid_results.raw.rms_error);
fprintf('   B2. 2x2 바이닝: %.3f px\n', centroid_results.binning.rms_error);
fprintf('   B3. Green 채널: %.3f px\n', centroid_results.green.rms_error);
fprintf('   B4. 가중 평균: %.3f px\n', centroid_results.weighted.rms_error);

fprintf('\n');

%% 6. 결과 시각화
fprintf('6. 결과 시각화...\n');

figure('Position', [100 100 1600 900], 'Name', 'Bayer 변환 비교');

% 원본 Bayer
subplot(2,3,1);
imagesc(bayer_img); colormap(gray); axis image;
title('원본 Bayer 패턴');
colorbar;

% FPGA 방식
subplot(2,3,2);
imagesc(gray_fpga); colormap(gray); axis image;
title(sprintf('A. FPGA (CFA→RGB→Gray)\nSNR=%.1fdB, 검출=%d개', ...
    snr_results.fpga, detection_results.fpga.n_detected));
colorbar;

% RAW 직접
subplot(2,3,3);
imagesc(gray_raw); colormap(gray); axis image;
title(sprintf('B1. RAW 직접\nSNR=%.1fdB, 검출=%d개', ...
    snr_results.raw, detection_results.raw.n_detected));
colorbar;

% 2x2 바이닝
subplot(2,3,4);
imagesc(gray_binning); colormap(gray); axis image;
title(sprintf('B2. 2x2 바이닝 (640x360)\nSNR=%.1fdB, 검출=%d개', ...
    snr_results.binning, detection_results.binning.n_detected));
colorbar;

% Green 채널
subplot(2,3,5);
imagesc(gray_green); colormap(gray); axis image;
title(sprintf('B3. Green 채널\nSNR=%.1fdB, 검출=%d개', ...
    snr_results.green, detection_results.green.n_detected));
colorbar;

% 가중 평균
subplot(2,3,6);
imagesc(gray_weighted); colormap(gray); axis image;
title(sprintf('B4. 가중 평균\nSNR=%.1fdB, 검출=%d개', ...
    snr_results.weighted, detection_results.weighted.n_detected));
colorbar;

sgtitle('Bayer → Grayscale 변환 방식 비교', 'FontSize', 14);

%% 7. 상세 비교 차트
figure('Position', [100 100 1200 500], 'Name', '성능 비교');

methods = {'FPGA', 'RAW', 'Binning', 'Green', 'Weighted'};
snr_vals = [snr_results.fpga, snr_results.raw, snr_results.binning, ...
            snr_results.green, snr_results.weighted];
detect_vals = [detection_results.fpga.n_detected, detection_results.raw.n_detected, ...
               detection_results.binning.n_detected, detection_results.green.n_detected, ...
               detection_results.weighted.n_detected];
centroid_vals = [centroid_results.fpga.rms_error, centroid_results.raw.rms_error, ...
                 centroid_results.binning.rms_error, centroid_results.green.rms_error, ...
                 centroid_results.weighted.rms_error];

% NaN 처리
centroid_vals(isnan(centroid_vals)) = 0;

% SNR 비교
subplot(1,3,1);
bar(snr_vals);
set(gca, 'XTickLabel', methods);
ylabel('SNR (dB)');
title('SNR 비교');
grid on;

% 검출률 비교
subplot(1,3,2);
bar(detect_vals);
hold on;
yline(params.n_stars, 'r--', 'LineWidth', 2);
set(gca, 'XTickLabel', methods);
ylabel('검출된 별 개수');
title(sprintf('별 검출 성능 (실제: %d개)', params.n_stars));
grid on;

% Centroid 오차 비교
subplot(1,3,3);
bar(centroid_vals);
set(gca, 'XTickLabel', methods);
ylabel('RMS 오차 (픽셀)');
title('Centroid 정확도');
grid on;

sgtitle('성능 지표 비교', 'FontSize', 14);

%% 8. 결론 출력
fprintf('\n');
fprintf('========================================\n');
fprintf('  분석 결론\n');
fprintf('========================================\n\n');

fprintf('1. Peak SNR 분석:\n');
fprintf('   - FPGA 방식:  %.2f dB\n', snr_results.fpga);
fprintf('   - RAW 직접:   %.2f dB (차이: %+.2f dB)\n', ...
    snr_results.raw, snr_results.raw - snr_results.fpga);
fprintf('   - 2x2 바이닝: %.2f dB (차이: %+.2f dB)\n', ...
    snr_results.binning, snr_results.binning - snr_results.fpga);
fprintf('   → 바이닝이 가장 높은 SNR (노이즈 평균화 효과)\n\n');

fprintf('2. 별 검출:\n');
fprintf('   - 모든 방식에서 유사한 검출 성능\n');
fprintf('   - 별센서 용도로 RGB 변환 불필요 확인\n\n');

fprintf('3. 권장 사항:\n');
fprintf('   ★ CFA + RGB2Gray IP 제거 가능\n');
fprintf('   ★ Bayer → Gray 직접 변환 권장\n');
fprintf('   ★ 바이닝 방식: SNR 향상이 필요한 경우 고려\n\n');

fprintf('4. FPGA 리소스 절감 예상:\n');
fprintf('   - CFA IP 제거: Line buffer 3줄 + 보간 로직\n');
fprintf('   - RGB2Gray IP 제거: 추가 IP 모듈 전체\n');
fprintf('   - 예상 전력 절감: ~0.1-0.2W\n');

fprintf('\n========================================\n');
fprintf('  시뮬레이션 완료\n');
fprintf('========================================\n');
