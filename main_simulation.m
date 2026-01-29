%% main_simulation.m
% Star Tracker 별 이미지 시뮬레이션
%
% 목적:
%   Hipparcos 카탈로그 기반 물리적으로 정확한 별 이미지 생성
%   - 실제 별 위치/등급 사용
%   - Pogson 공식 기반 플럭스 계산
%   - 광학 시스템 PSF 적용
%   - 센서 노이즈 모델링 (샷 노이즈, 읽기 노이즈, 다크 전류)
%
% 사용법:
%   1. MATLAB에서 이 파일 실행
%   2. '설정' 섹션에서 관측 방향 변경 가능
%
% 의존성:
%   - 별 카탈로그 파일 필요 (경로 설정 필요)
%   - Image Processing Toolbox
%   - Statistics Toolbox (poissrnd)
%
% 관련 서브 시뮬레이션:
%   - sub_main_1_bayer_comparison.m : Bayer → Gray 변환 방식 비교
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
fprintf('  Star Tracker 별 이미지 시뮬레이션\n');
fprintf('================================================\n\n');

%% ========== 센서 설정 ==========
% OV4689 + 10.42mm 렌즈 기반
sensor_params = struct();
sensor_params.myu = 2e-6;       % 픽셀 크기 [m]
sensor_params.f = 0.01042;      % 초점거리 [m]
sensor_params.l = 1280;         % 가로 픽셀
sensor_params.w = 720;          % 세로 픽셀
sensor_params.mag_limit = 6.5;  % 최대 등급 (육안 가시 한계)

%% ========== 노이즈 설정 ==========
sensor_params.dark_current = 5;   % 다크 전류 [ADU]
sensor_params.read_noise = 3;     % 읽기 노이즈 [ADU]
sensor_params.add_noise = true;   % 노이즈 활성화

%% ========== 관측 방향 설정 ==========
% 예시: 오리온 벨트 (삼태성)
ra_deg = 84;      % Right Ascension [deg] (적경)
de_deg = -1;      % Declination [deg] (적위)
roll_deg = 0;     % Roll [deg] (회전)

% 다른 관측 대상 예시:
% - 북극성 부근: ra_deg = 37.95, de_deg = 89.26
% - 큰곰자리 북두칠성: ra_deg = 165, de_deg = 55
% - 전갈자리 안타레스: ra_deg = 247.35, de_deg = -26.43

%% ========== 출력 폴더 ==========
output_dir = fullfile(script_dir, 'output');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% ========== FOV 계산 ==========
FOVx = rad2deg(2 * atan((sensor_params.myu*sensor_params.l/2) / sensor_params.f));
FOVy = rad2deg(2 * atan((sensor_params.myu*sensor_params.w/2) / sensor_params.f));

fprintf('센서 파라미터:\n');
fprintf('  해상도: %d x %d 픽셀\n', sensor_params.l, sensor_params.w);
fprintf('  픽셀 크기: %.1f µm\n', sensor_params.myu * 1e6);
fprintf('  초점거리: %.2f mm\n', sensor_params.f * 1000);
fprintf('  FOV: %.2f° x %.2f°\n\n', FOVx, FOVy);

fprintf('관측 방향:\n');
fprintf('  RA: %.2f° (적경)\n', ra_deg);
fprintf('  DEC: %.2f° (적위)\n', de_deg);
fprintf('  Roll: %.2f°\n\n', roll_deg);

%% ========== 1. 별 이미지 생성 ==========
fprintf('1. 별 이미지 생성 (Hipparcos 카탈로그 기반)...\n');
tic;
[gray_ideal, bayer_img, star_info] = simulate_star_image_realistic(ra_deg, de_deg, roll_deg, sensor_params);
gen_time = toc;

fprintf('   생성 시간: %.2f 초\n', gen_time);
fprintf('   FOV 내 별: %d개\n', star_info.num_stars);

if star_info.num_stars > 0
    fprintf('   등급 범위: %.2f ~ %.2f\n', min(star_info.magnitudes), max(star_info.magnitudes));
end
fprintf('\n');

if star_info.num_stars < 3
    warning('별이 너무 적습니다! 다른 RA/DEC를 시도해보세요.');
end

%% ========== 2. 물리 모델 정보 출력 ==========
fprintf('2. 물리 모델:\n');
fprintf('   PSF (Point Spread Function):\n');
fprintf('     - 모델: 2D Gaussian\n');
fprintf('     - sigma = 1.2 픽셀 (광학 시스템에 의해 결정)\n');
fprintf('     - 6-sigma 윈도우 사용\n\n');

fprintf('   Pogson 공식 (등급-플럭스 관계):\n');
fprintf('     - total_flux = ref_flux * 10^(-0.4 * (mag - ref_mag))\n');
fprintf('     - 기준: 6등급 = 1000 ADU\n');
fprintf('     - 등급 5 차이 = 100배 플럭스 차이\n\n');

fprintf('   노이즈 모델:\n');
fprintf('     - 샷 노이즈: Poisson 분포\n');
fprintf('     - 읽기 노이즈: %.1f ADU (Gaussian)\n', sensor_params.read_noise);
fprintf('     - 다크 전류: %.1f ADU\n\n', sensor_params.dark_current);

%% ========== 3. 시각화 ==========
fprintf('3. 시각화...\n');

% Figure 1: 이상적 Grayscale + 별 위치 표시
fig1 = figure('Position', [50 50 1400 500], 'Name', '별 이미지 시뮬레이션', 'Color', 'k');

subplot(1,3,1);
imshow(uint8(min(255, star_info.ideal_gray)), []);
title('이상적 Grayscale (노이즈 없음)', 'Color', 'w', 'FontSize', 12);

subplot(1,3,2);
imshow(uint8(min(255, star_info.ideal_gray)), []);
hold on;
if star_info.num_stars > 0
    plot(star_info.true_centroids(:,1), star_info.true_centroids(:,2), ...
        'ro', 'MarkerSize', 12, 'LineWidth', 2);

    % 밝은 별에 등급 레이블 표시
    for i = 1:min(10, star_info.num_stars)
        [~, idx] = sort(star_info.magnitudes);
        bright_idx = idx(i);
        text(star_info.true_centroids(bright_idx,1)+10, ...
             star_info.true_centroids(bright_idx,2), ...
             sprintf('%.1f', star_info.magnitudes(bright_idx)), ...
             'Color', 'yellow', 'FontSize', 9);
    end
end
hold off;
title(sprintf('별 위치 표시 (%d개)', star_info.num_stars), 'Color', 'w', 'FontSize', 12);

subplot(1,3,3);
imshow(bayer_img, []);
title('Bayer 패턴 (RGGB) + 노이즈', 'Color', 'w', 'FontSize', 12);

sgtitle(sprintf('Star Tracker 시뮬레이션 (RA=%.1f°, DEC=%.1f°, FOV=%.1f°×%.1f°)', ...
    ra_deg, de_deg, FOVx, FOVy), 'FontSize', 14, 'Color', 'w');

% Figure 2: 등급별 별 분포
if star_info.num_stars > 0
    fig2 = figure('Position', [100 100 600 400], 'Name', '등급 분포');
    histogram(star_info.magnitudes, 'BinWidth', 0.5, 'FaceColor', [0.3 0.5 0.8]);
    xlabel('등급 (Magnitude)');
    ylabel('별 개수');
    title(sprintf('FOV 내 별 등급 분포 (총 %d개)', star_info.num_stars));
    grid on;

    % 등급별 플럭스 참고선
    yyaxis right;
    mags = 0:0.5:7;
    ref_flux = 1000; ref_mag = 6.0;
    fluxes = ref_flux * 10.^(-0.4 * (mags - ref_mag));
    plot(mags, fluxes, 'r-', 'LineWidth', 2);
    ylabel('상대 플럭스 (ADU)');
    legend('별 개수', 'Pogson 플럭스', 'Location', 'northeast');
end

%% ========== 4. 별 정보 테이블 ==========
fprintf('4. 밝은 별 정보 (상위 10개):\n');
fprintf('   %-4s  %-8s  %-8s  %-6s  %-6s\n', '순위', 'X (px)', 'Y (px)', '등급', '플럭스');
fprintf('   %-4s  %-8s  %-8s  %-6s  %-6s\n', '----', '------', '------', '----', '------');

if star_info.num_stars > 0
    [sorted_mag, idx] = sort(star_info.magnitudes);
    ref_flux = 1000; ref_mag = 6.0;

    for i = 1:min(10, star_info.num_stars)
        j = idx(i);
        flux = ref_flux * 10^(-0.4 * (star_info.magnitudes(j) - ref_mag));
        fprintf('   %-4d  %-8.1f  %-8.1f  %-6.2f  %-6.0f\n', ...
            i, star_info.true_centroids(j,1), star_info.true_centroids(j,2), ...
            star_info.magnitudes(j), flux);
    end
end
fprintf('\n');

%% ========== 5. 결과 저장 ==========
fprintf('5. 결과 저장...\n');

% 이미지 저장
imwrite(uint8(min(255, star_info.ideal_gray)), fullfile(output_dir, 'star_image_ideal.png'));
imwrite(bayer_img, fullfile(output_dir, 'star_image_bayer.png'));
saveas(fig1, fullfile(output_dir, 'star_simulation_result.png'));

if star_info.num_stars > 0
    saveas(fig2, fullfile(output_dir, 'magnitude_distribution.png'));
end

% 별 정보 MAT 파일 저장
save(fullfile(output_dir, 'star_info.mat'), 'star_info', 'sensor_params', 'ra_deg', 'de_deg', 'roll_deg');

fprintf('   저장 완료: %s\n\n', output_dir);

%% ========== 6. 요약 ==========
fprintf('================================================\n');
fprintf('  시뮬레이션 완료\n');
fprintf('================================================\n\n');

fprintf('생성된 이미지:\n');
fprintf('  - star_image_ideal.png : 이상적 Grayscale\n');
fprintf('  - star_image_bayer.png : Bayer 패턴 (노이즈 포함)\n');
fprintf('  - star_info.mat : 별 정보 데이터\n\n');

fprintf('추가 분석:\n');
fprintf('  - Bayer→Gray 변환 비교: sub_main_1_bayer_comparison.m 실행\n');
fprintf('================================================\n');
