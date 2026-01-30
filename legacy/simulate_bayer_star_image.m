function [bayer_img, star_info] = simulate_bayer_star_image(n_stars, params)
% ============================================================
% ⚠ 폐기된 파일 (DEPRECATED)
% ============================================================
% 이 파일은 더 이상 사용되지 않습니다.
%
% [대체 파일]
%   core/simulate_star_image_realistic.m
%
% [폐기 사유]
%   이 함수는 별 위치와 밝기를 무작위(random)로 생성합니다.
%   최신 버전은 Hipparcos 별 카탈로그를 사용하여 실제 천구 좌표 기반으로
%   물리적으로 정확한 별 이미지를 생성합니다.
%
% [주요 차이점]
%   이 파일 (레거시)                   │ 최신 파일
%   ─────────────────────────────────────┼───────────────────────────
%   별 위치: 무작위 (rand)             │ Hipparcos 카탈로그 좌표
%   별 밝기: 균등 분포 (50~250 ADU)    │ Pogson 공식 (실제 등급)
%   입력: 별 개수 (n_stars)            │ RA, DEC, Roll (관측 방향)
%   노이즈: dark_current 상수값 [ADU]  │ dark_current_rate [e-/px/s]
%   센서 모델: 없음 (직접 ADU)         │ QE, 게인, 노출시간 적용
%
% [참고] run_bayer_comparison.m (v1)에서 호출됩니다.
%        최신 비교는 sub_main_1_bayer_comparison.m을 사용하세요.
% ============================================================
%
% SIMULATE_BAYER_STAR_IMAGE 별 이미지를 Bayer 패턴으로 시뮬레이션
%
% 입력:
%   n_stars - 생성할 별 개수
%   params  - 센서/시뮬레이션 파라미터 구조체
%
% 출력:
%   bayer_img - Bayer 패턴 이미지 (RGGB)
%   star_info - 별 정보 구조체 (위치, 밝기, 실제 centroid)
%
% Bayer 패턴 (RGGB):
%   R  G  R  G  ...
%   G  B  G  B  ...

%% 기본 파라미터
if nargin < 2
    params = struct();
end

% 센서 파라미터 (OV4689 기반)
params = set_default(params, 'width', 1280);
params = set_default(params, 'height', 720);
params = set_default(params, 'bit_depth', 8);
params = set_default(params, 'max_val', 255);

% 별 파라미터
params = set_default(params, 'min_intensity', 50);
params = set_default(params, 'max_intensity', 250);
params = set_default(params, 'psf_sigma', 1.2);
params = set_default(params, 'margin', 20);

% 노이즈 파라미터
params = set_default(params, 'dark_current', 5);
params = set_default(params, 'read_noise', 3);
params = set_default(params, 'shot_noise', true);

% Bayer 채널 감도 (상대값)
params = set_default(params, 'sensitivity_R', 1.0);
params = set_default(params, 'sensitivity_G', 1.0);
params = set_default(params, 'sensitivity_B', 0.9);

%% 별 위치 및 밝기 생성
rng('shuffle');

star_info.n_stars = n_stars;
star_info.x = params.margin + rand(n_stars, 1) * (params.width - 2*params.margin);
star_info.y = params.margin + rand(n_stars, 1) * (params.height - 2*params.margin);
star_info.intensity = params.min_intensity + ...
    rand(n_stars, 1) * (params.max_intensity - params.min_intensity);
star_info.psf_sigma = params.psf_sigma * ones(n_stars, 1);

%% 이상적인 Grayscale 이미지 생성 (PSF 적용)
ideal_img = zeros(params.height, params.width);

% PSF 윈도우 크기 (6-sigma 범위)
psf_radius = ceil(6 * params.psf_sigma);
[px, py] = meshgrid(-psf_radius:psf_radius, -psf_radius:psf_radius);

for i = 1:n_stars
    cx = star_info.x(i);
    cy = star_info.y(i);

    % PSF (2D Gaussian)
    sigma = star_info.psf_sigma(i);
    psf = exp(-((px - (cx - round(cx))).^2 + (py - (cy - round(cy))).^2) / (2*sigma^2));
    % PSF 정규화:
    %   psf / sum(psf(:)) → 적분값 = 1 (에너지 보존)
    %   × intensity → 적분값 = intensity [ADU]
    %   이 방식은 에너지 보존형 정규화 (energy-conserving normalization)
    %   최신 코드(draw_star_psf)에서는 peak_amplitude = flux/(2πσ²) 방식 사용
    psf = psf / sum(psf(:)) * star_info.intensity(i);

    % 이미지에 추가
    x_start = round(cx) - psf_radius;
    x_end = round(cx) + psf_radius;
    y_start = round(cy) - psf_radius;
    y_end = round(cy) + psf_radius;

    % 경계 체크
    if x_start >= 1 && x_end <= params.width && ...
       y_start >= 1 && y_end <= params.height
        ideal_img(y_start:y_end, x_start:x_end) = ...
            ideal_img(y_start:y_end, x_start:x_end) + psf;
    end
end

%% Bayer 패턴 이미지 생성 (RGGB)
% sensitivity_map: 각 픽셀의 Bayer 채널 감도를 담은 2D 행렬
%   RGGB 패턴에 따라 R=1.0, G=1.0, B=0.9 값이 반복됨
%   element-wise 곱셈으로 한 번에 모든 픽셀에 감도 적용 (벡터화)
%   ※ 최신 코드에서도 동일한 RGGB 감도 매핑 사용

% 채널별 감도 매트릭스 생성
sensitivity_map = zeros(params.height, params.width);

for row = 1:params.height
    for col = 1:params.width
        r_idx = mod(row-1, 2);
        c_idx = mod(col-1, 2);

        if r_idx == 0 && c_idx == 0
            sensitivity_map(row, col) = params.sensitivity_R;
        elseif r_idx == 1 && c_idx == 1
            sensitivity_map(row, col) = params.sensitivity_B;
        else
            sensitivity_map(row, col) = params.sensitivity_G;
        end
    end
end

% 감도 적용
bayer_img = ideal_img .* sensitivity_map;

%% 노이즈 추가
% --- 노이즈 모델 ---
% 이 레거시 코드는 dark_current를 상수 ADU로 직접 더합니다.
% 최신 코드(simulate_star_image_realistic.m)에서는:
%   dark_electrons = dark_current_rate [e-/px/s] × exposure_time [s]
%   dark_adu = dark_electrons × analog_gain × digital_gain
% 로 물리적으로 정확한 노이즈 모델을 사용합니다.

% 다크 전류
bayer_img = bayer_img + params.dark_current;

% 샷 노이즈 (Poisson)
if params.shot_noise
    bayer_img = poissrnd(max(0, bayer_img));
end

% 읽기 노이즈 (Gaussian)
bayer_img = bayer_img + params.read_noise * randn(size(bayer_img));

%% 클램핑 및 양자화
bayer_img = round(bayer_img);
bayer_img = max(0, min(params.max_val, bayer_img));
bayer_img = uint8(bayer_img);

%% 추가 정보 저장
star_info.params = params;
star_info.ideal_img = ideal_img;
star_info.sensitivity_map = sensitivity_map;

end

function params = set_default(params, field, value)
    if ~isfield(params, field)
        params.(field) = value;
    end
end
