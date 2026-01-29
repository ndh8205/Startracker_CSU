function [gray_img, bayer_img, star_info] = simulate_star_image_realistic(ra_deg, de_deg, roll_deg, sensor_params)
% SIMULATE_STAR_IMAGE_REALISTIC 실제 별 카탈로그 기반 별 이미지 시뮬레이션
%
% 입력:
%   ra_deg   - Right Ascension (degrees)
%   de_deg   - Declination (degrees)
%   roll_deg - Roll angle (degrees)
%   sensor_params - 센서 파라미터 구조체
%
% 출력:
%   gray_img  - Grayscale 이미지 (이상적)
%   bayer_img - Bayer 패턴 이미지 (RGGB)
%   star_info - 별 정보 구조체
%
% 기반: star_simulator_main.m (Brian Catraguna's Python code 변환)

%% 기본 파라미터
if nargin < 4
    sensor_params = struct();
end

% OV4689 기반 센서 파라미터
sensor_params = set_default(sensor_params, 'myu', 2e-6);      % 픽셀 크기 [m]
sensor_params = set_default(sensor_params, 'f', 0.01042);     % 초점거리 [m]
sensor_params = set_default(sensor_params, 'l', 1280);        % 가로 픽셀
sensor_params = set_default(sensor_params, 'w', 720);         % 세로 픽셀
sensor_params = set_default(sensor_params, 'mag_limit', 6.5); % 최대 등급

% 노이즈 파라미터
sensor_params = set_default(sensor_params, 'dark_current', 5);
sensor_params = set_default(sensor_params, 'read_noise', 3);
sensor_params = set_default(sensor_params, 'add_noise', true);

% Bayer 채널 감도
sensor_params = set_default(sensor_params, 'sensitivity_R', 1.0);
sensor_params = set_default(sensor_params, 'sensitivity_G', 1.0);
sensor_params = set_default(sensor_params, 'sensitivity_B', 0.9);

%% Convert to radians
ra = deg2rad(ra_deg);
de = deg2rad(de_deg);
roll = deg2rad(roll_deg);

%% Calculate FOV
myu = sensor_params.myu;
f = sensor_params.f;
l = sensor_params.l;
w = sensor_params.w;

FOVy = rad2deg(2 * atan((myu*w/2) / f));
FOVx = rad2deg(2 * atan((myu*l/2) / f));

star_info.FOVx = FOVx;
star_info.FOVy = FOVy;
star_info.ra_deg = ra_deg;
star_info.de_deg = de_deg;
star_info.roll_deg = roll_deg;

%% Create Rotation Matrix
M = create_rotation_matrix(ra, de, roll);

%% Load Star Catalogue (local data folder)
script_dir = fileparts(mfilename('fullpath'));
base_dir = fileparts(script_dir);  % bayer_comparison 폴더
data_dir = fullfile(base_dir, 'data');

catalog_path = fullfile(data_dir, 'star_catalog_kvector.mat');
csv_path = fullfile(data_dir, 'Hipparcos_Below_6.0.csv');

if exist(catalog_path, 'file')
    data = load(catalog_path);
    catalog_data = data.catalog_data;

    % 카탈로그에서 별 정보 추출
    ra_stars = catalog_data.star_catalog.RA;
    de_stars = catalog_data.star_catalog.DEC;
    magnitudes = catalog_data.star_catalog.Magnitude;
elseif exist(csv_path, 'file')
    % 대체: CSV 파일 사용
    opts = detectImportOptions(csv_path);
    opts.VariableNamingRule = 'preserve';
    star_catalogue = readtable(csv_path, opts);
    ra_stars = star_catalogue.RA;
    de_stars = star_catalogue.DE;
    magnitudes = star_catalogue.Magnitude;
else
    error('별 카탈로그를 찾을 수 없습니다. data/ 폴더에 star_catalog_kvector.mat 또는 Hipparcos_Below_6.0.csv 파일이 필요합니다.');
end

%% Search for stars within FOV
R = sqrt(deg2rad(FOVx)^2 + deg2rad(FOVy)^2) / 2;
alpha_start = ra - R/cos(de);
alpha_end = ra + R/cos(de);
delta_start = de - R;
delta_end = de + R;

% Find stars in range
star_within_ra = (alpha_start <= ra_stars) & (ra_stars <= alpha_end);
star_within_de = (delta_start <= de_stars) & (de_stars <= delta_end);
stars_in_fov = star_within_ra & star_within_de;

% 등급 제한
mag_ok = magnitudes <= sensor_params.mag_limit;
stars_in_fov = stars_in_fov & mag_ok;

% Filter stars
ra_i = ra_stars(stars_in_fov);
de_i = de_stars(stars_in_fov);
mag_i = magnitudes(stars_in_fov);

%% Convert to sensor coordinates
star_sensor_coords = zeros(length(ra_i), 3);
M_transpose = M';

for i = 1:length(ra_i)
    dir_vector = [cos(ra_i(i))*cos(de_i(i));
                  sin(ra_i(i))*cos(de_i(i));
                  sin(de_i(i))];
    star_sensor_coords(i, :) = (M_transpose * dir_vector)';
end

%% Convert to image coordinates (star_simulator 방식 - z 체크 없음)
star_loc = zeros(length(ra_i), 2);
for i = 1:length(ra_i)
    x = f * (star_sensor_coords(i,1) / star_sensor_coords(i,3));
    y = f * (star_sensor_coords(i,2) / star_sensor_coords(i,3));
    star_loc(i, :) = [x, y];
end

pixel_per_length = 1 / myu;

%% Convert to pixel coordinates
pixel_coords = [];
filtered_mag = [];
true_centroids = [];

for i = 1:size(star_loc, 1)
    x1 = star_loc(i, 1);
    y1 = star_loc(i, 2);

    x1pixel = pixel_per_length * x1;
    y1pixel = pixel_per_length * y1;

    % Check if within bounds
    if abs(x1pixel) > l/2 || abs(y1pixel) > w/2
        continue;
    end

    pixel_coords = [pixel_coords; x1pixel, y1pixel];
    filtered_mag = [filtered_mag; mag_i(i)];

    % 실제 이미지 좌표 (centroid)
    true_x = l/2 + x1pixel;
    true_y = w/2 - y1pixel;
    true_centroids = [true_centroids; true_x, true_y];
end

star_info.num_stars = length(filtered_mag);
star_info.pixel_coords = pixel_coords;
star_info.magnitudes = filtered_mag;
star_info.true_centroids = true_centroids;

%% Create Grayscale image (ideal)
gray_img = zeros(w, l);

for i = 1:length(filtered_mag)
    x = l/2 + pixel_coords(i,1);
    y = w/2 - pixel_coords(i,2);

    % 등급 기반 별 특성 (물리적으로 정확한 모델)
    mag = filtered_mag(i);

    % === 물리 모델 ===
    % 1. PSF sigma: 광학 시스템에 의해 결정 (상수)
    %    OV4689 + 10.42mm 렌즈 기준, 약 1.2 pixel sigma
    sigma = 1.2;  % 모든 별 동일

    % 2. 총 플럭스 (Pogson 공식)
    %    등급 5 차이 = 100배 플럭스 차이
    %    기준: 6등급별 = 총 플럭스 1000 (ADU)
    ref_mag = 6.0;
    ref_flux = 1000;  % 6등급 별의 총 광자 수 (ADU)
    total_flux = ref_flux * 10^(-0.4 * (mag - ref_mag));

    % 3. PSF 적용하여 그리기
    %    Gaussian PSF의 적분값 = total_flux
    gray_img = draw_star_psf(gray_img, x, y, sigma, total_flux);
end

star_info.ideal_gray = gray_img;

%% Add noise to grayscale (optional)
if sensor_params.add_noise
    % Dark current
    gray_noisy = gray_img + sensor_params.dark_current;

    % Shot noise (Poisson)
    gray_noisy = poissrnd(max(0, gray_noisy));

    % Read noise (Gaussian)
    gray_noisy = gray_noisy + sensor_params.read_noise * randn(size(gray_noisy));

    % Clamp
    gray_noisy = max(0, min(255, gray_noisy));
    gray_img = uint8(round(gray_noisy));
else
    gray_img = uint8(round(gray_img));
end

%% Create Bayer pattern image (RGGB)
bayer_img = zeros(w, l);

% 먼저 노이즈 없는 ideal 이미지로 Bayer 생성
ideal_gray = star_info.ideal_gray;

% Bayer 채널 감도 적용
for row = 1:w
    for col = 1:l
        r_idx = mod(row-1, 2);
        c_idx = mod(col-1, 2);

        if r_idx == 0 && c_idx == 0
            sensitivity = sensor_params.sensitivity_R;  % R
        elseif r_idx == 1 && c_idx == 1
            sensitivity = sensor_params.sensitivity_B;  % B
        else
            sensitivity = sensor_params.sensitivity_G;  % G
        end

        bayer_img(row, col) = ideal_gray(row, col) * sensitivity;
    end
end

% Add noise
if sensor_params.add_noise
    bayer_img = bayer_img + sensor_params.dark_current;
    bayer_img = poissrnd(max(0, bayer_img));
    bayer_img = bayer_img + sensor_params.read_noise * randn(size(bayer_img));
end

bayer_img = max(0, min(255, bayer_img));
bayer_img = uint8(round(bayer_img));

star_info.sensor_params = sensor_params;

end

%% Helper Functions
function params = set_default(params, field, value)
    if ~isfield(params, field)
        params.(field) = value;
    end
end

function M = create_rotation_matrix(ra, de, roll)
    ra_exp = ra - (pi/2);
    de_exp = de + (pi/2);

    M1 = [cos(ra_exp), -sin(ra_exp), 0;
          sin(ra_exp),  cos(ra_exp), 0;
          0,            0,           1];

    M2 = [1,  0,            0;
          0,  cos(de_exp), -sin(de_exp);
          0,  sin(de_exp),  cos(de_exp)];

    M3 = [cos(roll), -sin(roll), 0;
          sin(roll),  cos(roll), 0;
          0,          0,         1];

    M = M1 * M2 * M3;
end

function img = draw_star_psf(img, cx, cy, sigma, total_flux)
    % 물리적으로 정확한 PSF 렌더링
    % total_flux: PSF 적분값 (총 광자 수, ADU)
    % sigma: PSF 표준편차 (픽셀)
    %
    % 2D Gaussian: I(x,y) = A * exp(-(x^2+y^2)/(2*sigma^2))
    % 적분값: integral = 2 * pi * sigma^2 * A
    % 따라서: A = total_flux / (2 * pi * sigma^2)

    [h, w] = size(img);

    % 피크 값 계산 (정규화)
    peak_amplitude = total_flux / (2 * pi * sigma^2);

    % 윈도우 크기 (6-sigma 범위)
    win = ceil(6 * sigma);

    x_start = max(1, round(cx) - win);
    x_end = min(w, round(cx) + win);
    y_start = max(1, round(cy) - win);
    y_end = min(h, round(cy) + win);

    for y = y_start:y_end
        for x = x_start:x_end
            % 서브픽셀 거리
            dx = x - cx;
            dy = y - cy;
            dist_sq = dx^2 + dy^2;

            % Gaussian PSF
            val = peak_amplitude * exp(-dist_sq / (2 * sigma^2));

            % 누적 (겹치는 별 처리)
            img(y, x) = img(y, x) + val;
        end
    end
end
