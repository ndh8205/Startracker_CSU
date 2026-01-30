%% sub_main_2_optimal_weights.m
% =========================================================================
% OV4689 우주 환경 Bayer→Gray 최적 가중치 연구
% =========================================================================
%
% [목적]
%   별 추적기(Star Tracker)에서 Bayer 패턴 센서 데이터를 그레이스케일로
%   변환할 때, 별 검출 SNR을 최대화하는 최적 채널 가중치를 도출합니다.
%
% [배경]
%   기존 FPGA 파이프라인: Bayer → CFA(RGB) → rgb2gray (R+2G+B)/4
%   이 가중치는 인간 시각 기반이며, 별 검출 최적화와 무관합니다.
%
%   본 연구는 물리적 근거(Planck 흑체복사 + 센서 QE + 노이즈 모델)에
%   기반하여 최적 가중치를 수학적으로 도출합니다.
%
% [핵심 이론: Inverse Variance Weighting]
%   N개의 독립적 측정값 x_i (신호 S_i, 노이즈 σ_i)를 조합할 때,
%   SNR을 최대화하는 가중치:
%
%     w_i = S_i / σ_i²     (정규화: Σw_i = 1)
%
%   증명: 조합 신호 S_comb = Σ w_i S_i, 노이즈 σ_comb = √(Σ w_i² σ_i²)
%         SNR = S_comb/σ_comb 를 Lagrange 승수법으로 최대화
%
% [실행 방법]
%   MATLAB에서: sub_main_2_optimal_weights
%
% [출력]
%   - Figure 1: 흑체복사 스펙트럼 + 센서 채널 응답
%   - Figure 2: 별 온도별 최적 가중치
%   - Figure 3: 노이즈 레짐별 가중치 변화
%   - Figure 4: 기존 방법 vs 최적 가중치 SNR 비교
%   - output/optimal_weights_result.mat: 결과 데이터
%
% [관련 파일]
%   - core/bayer_to_gray_direct.m : 직접 변환 함수 ('optimal' 모드)
%   - sub_main_1_bayer_comparison.m : 기존 변환 방식 비교
%   - main_simulation.m : 별 이미지 생성
%
% [참고문헌]
%   - Planck's law: B(λ,T) = 2hc²/λ⁵ × 1/(exp(hc/λkT)-1)
%   - Liebe (2002), "Accuracy Performance of Star Trackers"
%   - OV4689 Product Brief (OmniVision)
%
% =========================================================================

clear; clc; close all;
fprintf('=== OV4689 우주 환경 Bayer→Gray 최적 가중치 연구 ===\n\n');

%% ========================================================================
%  Section 1: 물리 상수 및 센서 파라미터 정의
%  ========================================================================

% --- 물리 상수 ---
h = 6.626e-34;       % Planck 상수 [J·s]
c = 2.998e8;         % 광속 [m/s]
k_B = 1.381e-23;     % Boltzmann 상수 [J/K]

% --- OV4689 센서 파라미터 (시뮬레이션과 동일) ---
sensor.pixel_size = 2e-6;         % 픽셀 크기 [m]
sensor.focal_length = 10.42e-3;   % 초점거리 [m]
sensor.exposure_time = 0.022;     % 노출 시간 [s] (22ms, 레지스터 0xC350)
sensor.analog_gain = 16;          % 아날로그 게인 [배]
sensor.digital_gain = 1.0;        % 디지털 게인 [배]
sensor.quantum_eff = 0.5;         % 양자 효율 (전체 평균)
sensor.dark_current = 0.1;        % 다크 전류 [e-/px/s] (지상)
sensor.dark_current_space = 0.01; % 다크 전류 [e-/px/s] (우주, -20°C 추정)
sensor.read_noise = 3.0;          % 읽기 노이즈 [e- RMS]
sensor.aperture_diam = 6.5e-3;    % 유효 구경 [m] (f/1.6)

% --- 파장 범위 ---
lambda = (350:1:900) * 1e-9;      % 350~900nm [m]
lambda_nm = lambda * 1e9;         % 표시용 [nm]

fprintf('센서: OV4689 (pixel=2µm, f=10.42mm, t_exp=22ms, gain=16x)\n');
fprintf('파장 범위: %d~%d nm (%d points)\n\n', ...
    min(lambda_nm), max(lambda_nm), length(lambda));

%% ========================================================================
%  Section 2: 흑체복사 스펙트럼 모델링 (Planck's Law)
%  ========================================================================
%
%  별은 흑체(blackbody)에 가까운 복사를 합니다.
%  Planck 법칙으로 별의 온도에 따른 파장별 광자 방출률을 계산합니다.
%
%  B_photon(λ, T) = B(λ, T) / (hc/λ)
%    = (2c/λ⁴) × 1/(exp(hc/λkT) - 1)   [photons/s/m²/sr/m]
%
%  광자 수 기준을 사용하는 이유:
%  센서의 QE는 에너지가 아닌 광자 수에 비례하기 때문입니다.
%  (1개의 광자 = 1개의 전자 생성, 에너지와 무관)

% 항성 스펙트럼 유형별 대표 온도
star_temps = [3000, 4000, 5000, 5800, 7500, 10000, 15000, 25000];
star_types = {'M형(적색)', 'K형', 'G후기', 'G2형(태양)', 'F형', 'A형(백색)', 'B후기', 'B형(청색)'};

% 주요 항법별 온도 (Star Tracker 카탈로그에서 자주 사용되는 별)
nav_star_temps = struct();
nav_star_temps.name = {'시리우스', '카노푸스', '아크투루스', '베가', '카펠라', '리겔', '프로키온', '베텔게우스', '아케르나르', '알타이르'};
nav_star_temps.temp = [9940, 7400, 4290, 9600, 5700, 12100, 6530, 3500, 15000, 7670];
nav_star_temps.mag  = [-1.46, -0.74, -0.05, 0.03, 0.08, 0.13, 0.34, 0.42, 0.46, 0.77];

% Planck 광자 스펙트럼 계산
fprintf('--- Section 2: 흑체복사 스펙트럼 ---\n');
B_photon = zeros(length(star_temps), length(lambda));
for i = 1:length(star_temps)
    T = star_temps(i);
    % 광자 수 기반 Planck 함수 [photons/s/m²/sr/m]
    B_photon(i,:) = (2*c ./ lambda.^4) ./ (exp(h*c ./ (lambda * k_B * T)) - 1);

    % Wien 변위 법칙으로 피크 파장 확인
    [~, idx] = max(B_photon(i,:));
    fprintf('  %s (%dK): 피크 %d nm\n', star_types{i}, T, round(lambda_nm(idx)));
end
fprintf('\n');

%% ========================================================================
%  Section 3: OV4689 채널별 스펙트럼 응답 모델링
%  ========================================================================
%
%  OV4689의 상세 스펙트럼 응답 곡선은 공개되지 않았습니다.
%  일반적인 CMOS Bayer 필터(BSI 2µm 기술)의 특성을 기반으로
%  가우시안 근사 모델을 사용합니다.
%
%  각 채널의 응답 = 색 필터 투과율 × 실리콘 QE 기저 곡선
%
%  [참고: OmniVision BSI-2 기술]
%  - Back-Side Illumination으로 QE가 향상됨
%  - 2µm 픽셀에서도 높은 감도 (특히 가시광)
%  - NIR 영역(>750nm) 감도는 제한적

fprintf('--- Section 3: 센서 채널 응답 모델 ---\n');

% --- 실리콘 기저 QE 곡선 ---
% BSI CMOS의 전형적인 QE 곡선 (에너지 흡수 + 간섭 효과)
% 피크: ~520nm, 장파장 쪽은 실리콘 밴드갭(1100nm)까지 감소
lambda_peak_si = 520e-9;
sigma_si = 200e-9;
QE_base = 0.55 * exp(-0.5 * ((lambda - lambda_peak_si) / sigma_si).^2);
% 장파장 보정: 실리콘 흡수 특성 (NIR 감소)
QE_base = QE_base .* (1 - 0.3 * max(0, (lambda - 600e-9)) / 300e-9);
% 단파장 보정: UV 영역 급격히 감소
QE_base = QE_base .* (1 ./ (1 + exp(-(lambda - 370e-9) / 15e-9)));

% --- Bayer 색 필터 모델 (가우시안 근사) ---
% 각 필터의 투과율 = 가우시안 피크 + 스커트 (광대역 누설)

% Red 필터
R_center = 620e-9;   % 중심 파장 [m]
R_sigma = 55e-9;     % 표준편차 (FWHM ≈ 130nm)
R_peak = 0.85;       % 피크 투과율
R_skirt = 0.05;      % 대역외 누설
filter_R = R_peak * exp(-0.5*((lambda - R_center)/R_sigma).^2) + R_skirt;
% Red는 장파장에서 추가 투과 (롱패스 특성)
filter_R = filter_R + 0.3 * (1 ./ (1 + exp(-(lambda - 650e-9) / 30e-9)));
filter_R = min(filter_R, 0.95);  % 최대 투과율 제한

% Green 필터
G_center = 530e-9;
G_sigma = 45e-9;     % FWHM ≈ 106nm
G_peak = 0.90;
G_skirt = 0.03;
filter_G = G_peak * exp(-0.5*((lambda - G_center)/G_sigma).^2) + G_skirt;

% Blue 필터
B_center = 460e-9;
B_sigma = 35e-9;     % FWHM ≈ 82nm (가장 좁음)
B_peak = 0.80;
B_skirt = 0.03;
filter_B = B_peak * exp(-0.5*((lambda - B_center)/B_sigma).^2) + B_skirt;

% --- 최종 채널별 QE (필터 × 기저 QE) ---
QE_R = QE_base .* filter_R;
QE_G = QE_base .* filter_G;
QE_B = QE_base .* filter_B;

% 피크값 출력
[peak_R, idx_R] = max(QE_R);
[peak_G, idx_G] = max(QE_G);
[peak_B, idx_B] = max(QE_B);
fprintf('  R 채널: 피크 QE = %.1f%% @ %d nm\n', peak_R*100, round(lambda_nm(idx_R)));
fprintf('  G 채널: 피크 QE = %.1f%% @ %d nm\n', peak_G*100, round(lambda_nm(idx_G)));
fprintf('  B 채널: 피크 QE = %.1f%% @ %d nm\n', peak_B*100, round(lambda_nm(idx_B)));

% 적분 QE (전체 파장 대역)
dlambda = lambda(2) - lambda(1);
int_R = sum(QE_R) * dlambda;
int_G = sum(QE_G) * dlambda;
int_B = sum(QE_B) * dlambda;
fprintf('  적분 QE: R=%.4f, G=%.4f, B=%.4f\n', int_R, int_G, int_B);
fprintf('  상대 비율: R:G:B = %.2f : %.2f : %.2f\n\n', ...
    int_R/int_G, 1.0, int_B/int_G);

%% ========================================================================
%  Section 4: 별 온도별 채널 신호량 계산
%  ========================================================================
%
%  각 별 온도에서 R/G/B 채널이 수집하는 광자 수를 계산합니다.
%
%  S_ch(T) = ∫ B_photon(λ,T) × QE_ch(λ) dλ × A_lens × Ω_pixel × t_exp
%
%  여기서:
%    B_photon: 광자 스펙트럼 [photons/s/m²/sr/m]
%    QE_ch:    채널별 양자효율 (필터 × 실리콘)
%    A_lens:   렌즈 집광 면적 [m²]
%    Ω_pixel:  1 픽셀의 입체각 [sr]
%    t_exp:    노출 시간 [s]

fprintf('--- Section 4: 채널별 신호 계산 ---\n');

% 렌즈 집광 면적
A_lens = pi * (sensor.aperture_diam/2)^2;  % [m²]

% 1 픽셀의 입체각
pixel_solid_angle = (sensor.pixel_size / sensor.focal_length)^2;  % [sr]

% 별 온도별 채널 신호 계산 (상대값)
S_R = zeros(1, length(star_temps));
S_G = zeros(1, length(star_temps));
S_B = zeros(1, length(star_temps));

for i = 1:length(star_temps)
    % 각 채널의 광자 수 적분 [상대값, 정규화 전]
    S_R(i) = sum(B_photon(i,:) .* QE_R) * dlambda;
    S_G(i) = sum(B_photon(i,:) .* QE_G) * dlambda;
    S_B(i) = sum(B_photon(i,:) .* QE_B) * dlambda;
end

% 정규화 (G 채널 5800K 기준)
S_total_5800 = S_R(4) + S_G(4) + S_B(4);
S_R_norm = S_R / S_total_5800;
S_G_norm = S_G / S_total_5800;
S_B_norm = S_B / S_total_5800;

fprintf('\n  별 온도별 채널 신호 비율 (정규화):\n');
fprintf('  %-14s  %6s  %6s  %6s  %8s\n', '별 유형', 'R', 'G', 'B', 'R:G:B');
fprintf('  %s\n', repmat('-', 1, 50));
for i = 1:length(star_temps)
    total = S_R_norm(i) + S_G_norm(i) + S_B_norm(i);
    r_frac = S_R_norm(i)/total;
    g_frac = S_G_norm(i)/total;
    b_frac = S_B_norm(i)/total;
    fprintf('  %-14s  %5.3f  %5.3f  %5.3f  %.2f:%.2f:%.2f\n', ...
        star_types{i}, r_frac, g_frac, b_frac, ...
        r_frac/g_frac, 1.0, b_frac/g_frac);
end
fprintf('\n');

%% ========================================================================
%  Section 5: 최적 가중치 도출 (Inverse Variance Weighting)
%  ========================================================================
%
%  [이론]
%  N개의 독립적 측정 x_i ~ N(S, σ_i²)을 가중합으로 조합:
%    x_comb = Σ w_i x_i,   Σ w_i = 1
%
%  조합된 SNR:
%    SNR_comb = (Σ w_i S_i) / √(Σ w_i² σ_i²)
%
%  SNR 최대화 조건 (Lagrange):
%    w_i = (S_i / σ_i²) / Σ(S_j / σ_j²)
%
%  [노이즈 모델]
%  각 픽셀의 노이즈 분산:
%    σ² = S_signal + S_dark + σ_read²
%
%  여기서:
%    S_signal: 별 신호 [e-] (Poisson)
%    S_dark:   다크 전류 [e-] (Poisson)
%    σ_read:   읽기 노이즈 [e-] (Gaussian, 게인 적용 전)

fprintf('--- Section 5: 최적 가중치 도출 ---\n\n');

% 6등급 별의 기준 광자 플럭스 (시뮬레이션 캘리브레이션)
ref_photon_flux = 96;   % [photons/s] for mag=6.0 (지상)
ref_photon_flux_space = ref_photon_flux / 0.5;  % 우주: 대기 손실 없음 (×2)

% 대상 등급 범위
magnitudes = 0:0.5:7;

% --- 지상 vs 우주 환경 비교 ---
environments = struct();
environments(1).name = '지상 (대기 있음)';
environments(1).ref_flux = ref_photon_flux;
environments(1).dark_current = sensor.dark_current;
environments(1).atm_factor = 0.5;

environments(2).name = '우주 (대기 없음, -20°C)';
environments(2).ref_flux = ref_photon_flux_space;
environments(2).dark_current = sensor.dark_current_space;
environments(2).atm_factor = 1.0;

% --- 별 온도별 + 등급별 + 환경별 최적 가중치 계산 ---
% 결과 저장 구조체
results = struct();

for env_idx = 1:2
    env = environments(env_idx);
    fprintf('=== 환경: %s ===\n\n', env.name);

    for t_idx = [1, 4, 6, 8]  % M형, G2형, A형, B형 (대표 4개)
        T = star_temps(t_idx);
        type_name = star_types{t_idx};

        fprintf('  [%s, %dK]\n', type_name, T);
        fprintf('  %-6s  %8s  %8s  %8s  | %6s  %6s  %6s  | %6s\n', ...
            '등급', 'S_R[e-]', 'S_G[e-]', 'S_B[e-]', 'w_R', 'w_G', 'w_B', 'SNR');
        fprintf('  %s\n', repmat('-', 1, 72));

        for m_idx = 1:length(magnitudes)
            mag = magnitudes(m_idx);

            % 별의 광자 플럭스
            photon_flux = env.ref_flux * 10^(-0.4 * (mag - 6.0));

            % 채널별 상대 신호 비율 계산
            S_total_T = S_R(t_idx) + S_G(t_idx) + S_B(t_idx);
            frac_R = S_R(t_idx) / S_total_T;
            frac_G = S_G(t_idx) / S_total_T;
            frac_B = S_B(t_idx) / S_total_T;

            % 채널별 전자 수 [e-]
            electrons = photon_flux * sensor.exposure_time * sensor.quantum_eff;
            e_R = electrons * frac_R;
            e_G = electrons * frac_G;
            e_B = electrons * frac_B;

            % 채널별 노이즈 [e-²]
            dark_e = env.dark_current * sensor.exposure_time;
            read_var = sensor.read_noise^2;

            var_R = e_R + dark_e + read_var;  % shot + dark + read
            var_G = e_G + dark_e + read_var;
            var_B = e_B + dark_e + read_var;

            % Inverse Variance Weighting
            w_R_raw = e_R / var_R;
            w_G_raw = e_G / var_G;
            w_B_raw = e_B / var_B;

            w_total = w_R_raw + w_G_raw + w_B_raw;
            w_R_opt = w_R_raw / w_total;
            w_G_opt = w_G_raw / w_total;
            w_B_opt = w_B_raw / w_total;

            % 조합 SNR
            S_comb = w_R_opt*e_R + w_G_opt*e_G + w_B_opt*e_B;
            var_comb = w_R_opt^2*var_R + w_G_opt^2*var_G + w_B_opt^2*var_B;
            SNR_opt = S_comb / sqrt(var_comb);

            % ADU 변환
            adu_R = e_R * sensor.analog_gain * sensor.digital_gain;
            adu_G = e_G * sensor.analog_gain * sensor.digital_gain;
            adu_B = e_B * sensor.analog_gain * sensor.digital_gain;

            fprintf('  %4.1f   %8.1f  %8.1f  %8.1f  | %5.3f  %5.3f  %5.3f  | %5.1f\n', ...
                mag, adu_R, adu_G, adu_B, w_R_opt, w_G_opt, w_B_opt, SNR_opt);

            % 결과 저장 (우주 환경, G2형일 때)
            if env_idx == 2 && t_idx == 4
                results(m_idx).mag = mag;
                results(m_idx).w_R = w_R_opt;
                results(m_idx).w_G = w_G_opt;
                results(m_idx).w_B = w_B_opt;
                results(m_idx).SNR = SNR_opt;
                results(m_idx).e_R = e_R;
                results(m_idx).e_G = e_G;
                results(m_idx).e_B = e_B;
            end
        end
        fprintf('\n');
    end
end

%% ========================================================================
%  Section 6: 노이즈 레짐 분석
%  ========================================================================
%
%  가중치가 노이즈 레짐에 따라 어떻게 변하는지 분석합니다.
%
%  1. Shot noise 지배 (밝은 별): σ² ≈ S → w ∝ S/S = 1 (균등)
%  2. Read noise 지배 (어두운 별): σ² ≈ σ_read² → w ∝ S (신호 비례)
%  3. 전환점: S_signal ≈ σ_read² = 9 e-

fprintf('--- Section 6: 노이즈 레짐 분석 ---\n\n');

% G2형 태양(5800K) 기준, 우주 환경
T_ref = 5800;
t_ref_idx = 4;
S_total_ref = S_R(t_ref_idx) + S_G(t_ref_idx) + S_B(t_ref_idx);
frac_R_ref = S_R(t_ref_idx) / S_total_ref;
frac_G_ref = S_G(t_ref_idx) / S_total_ref;
frac_B_ref = S_B(t_ref_idx) / S_total_ref;

% 등급 0~7까지 세밀하게
mag_fine = linspace(-1, 7, 200);
w_R_vs_mag = zeros(size(mag_fine));
w_G_vs_mag = zeros(size(mag_fine));
w_B_vs_mag = zeros(size(mag_fine));
SNR_vs_mag = zeros(size(mag_fine));
SNR_equal_vs_mag = zeros(size(mag_fine));
SNR_fpga_vs_mag = zeros(size(mag_fine));

for i = 1:length(mag_fine)
    mag = mag_fine(i);
    photon_flux = ref_photon_flux_space * 10^(-0.4 * (mag - 6.0));
    electrons = photon_flux * sensor.exposure_time * sensor.quantum_eff;

    e_R = electrons * frac_R_ref;
    e_G = electrons * frac_G_ref;
    e_B = electrons * frac_B_ref;

    dark_e = sensor.dark_current_space * sensor.exposure_time;
    read_var = sensor.read_noise^2;

    var_R = e_R + dark_e + read_var;
    var_G = e_G + dark_e + read_var;
    var_B = e_B + dark_e + read_var;

    % 최적 가중치
    w_R_raw = e_R / var_R;
    w_G_raw = e_G / var_G;
    w_B_raw = e_B / var_B;
    w_total = w_R_raw + w_G_raw + w_B_raw;

    w_R_vs_mag(i) = w_R_raw / w_total;
    w_G_vs_mag(i) = w_G_raw / w_total;
    w_B_vs_mag(i) = w_B_raw / w_total;

    % 최적 가중치 SNR
    S_opt = w_R_vs_mag(i)*e_R + w_G_vs_mag(i)*e_G + w_B_vs_mag(i)*e_B;
    var_opt = w_R_vs_mag(i)^2*var_R + w_G_vs_mag(i)^2*var_G + w_B_vs_mag(i)^2*var_B;
    SNR_vs_mag(i) = S_opt / sqrt(var_opt);

    % 균등 가중치(1/3) SNR
    w_eq = 1/3;
    S_eq = w_eq*(e_R + e_G + e_B);
    var_eq = w_eq^2*(var_R + var_G + var_B);
    SNR_equal_vs_mag(i) = S_eq / sqrt(var_eq);

    % FPGA 가중치 (0.25, 0.50, 0.25) SNR
    S_fpga = 0.25*e_R + 0.50*e_G + 0.25*e_B;
    var_fpga = 0.25^2*var_R + 0.50^2*var_G + 0.25^2*var_B;
    SNR_fpga_vs_mag(i) = S_fpga / sqrt(var_fpga);
end

% 레짐 전환점 계산
% Shot noise = Read noise 일 때: S = σ_read² = 9 e-
transition_electrons = sensor.read_noise^2;  % 9 e-
transition_flux = transition_electrons / (sensor.exposure_time * sensor.quantum_eff);
transition_mag = 6.0 - 2.5 * log10(transition_flux / ref_photon_flux_space);

fprintf('  노이즈 레짐 전환점:\n');
fprintf('    Read noise = %d e- → σ² = %d e-²\n', sensor.read_noise, sensor.read_noise^2);
fprintf('    전환 신호: %.1f e- (= %.1f ADU)\n', transition_electrons, ...
    transition_electrons * sensor.analog_gain);
fprintf('    전환 등급: %.1f mag (우주)\n\n', transition_mag);

% 어두운 별(6등급)과 밝은 별(2등급) 비교
fprintf('  가중치 비교 (G2형, 우주):\n');
fprintf('  %-12s  %6s  %6s  %6s  | %s\n', '등급', 'w_R', 'w_G', 'w_B', '레짐');
fprintf('  %s\n', repmat('-', 1, 55));

check_mags = [0, 2, 4, 6];
for mag = check_mags
    idx = find(abs(mag_fine - mag) < 0.05, 1);
    if isempty(idx), continue; end

    photon_flux = ref_photon_flux_space * 10^(-0.4 * (mag - 6.0));
    electrons = photon_flux * sensor.exposure_time * sensor.quantum_eff;

    if electrons > transition_electrons * 3
        regime = 'Shot noise 지배';
    elseif electrons < transition_electrons / 3
        regime = 'Read noise 지배';
    else
        regime = '혼합';
    end

    fprintf('  %4.1f등급      %5.3f  %5.3f  %5.3f  | %s\n', ...
        mag, w_R_vs_mag(idx), w_G_vs_mag(idx), w_B_vs_mag(idx), regime);
end
fprintf('\n');

%% ========================================================================
%  Section 7: 실용적 최적 가중치 결정
%  ========================================================================
%
%  FPGA에서는 등급별로 가중치를 동적으로 변경할 수 없으므로,
%  전체 카탈로그에 걸쳐 가장 효과적인 "단일 고정 가중치"를 결정합니다.
%
%  방법: 등급 가중 평균 (어두운 별에 더 큰 가중치)
%  이유: 어두운 별이 더 많고, SNR이 더 중요

fprintf('--- Section 7: 실용적 최적 가중치 결정 ---\n\n');

% 별 수 분포 (Hipparcos 카탈로그 기반 추정)
% 등급별 별 수: N(m) ∝ 10^(0.6*m) (별 개수 법칙)
mag_weights = 10.^(0.6 * mag_fine);
mag_weights = mag_weights / sum(mag_weights);

% 등급 가중 평균 가중치
w_R_avg = sum(w_R_vs_mag .* mag_weights);
w_G_avg = sum(w_G_vs_mag .* mag_weights);
w_B_avg = sum(w_B_vs_mag .* mag_weights);

% 정규화
w_total_avg = w_R_avg + w_G_avg + w_B_avg;
w_R_final = w_R_avg / w_total_avg;
w_G_final = w_G_avg / w_total_avg;
w_B_final = w_B_avg / w_total_avg;

fprintf('  === 최적 가중치 (우주 환경, G2형 기준, 등급 가중 평균) ===\n\n');
fprintf('    w_R = %.4f  (Red)\n', w_R_final);
fprintf('    w_G = %.4f  (Green)\n', w_G_final);
fprintf('    w_B = %.4f  (Blue)\n\n', w_B_final);

% 별 온도별 최적 가중치도 계산
fprintf('  별 온도별 최적 가중치 (등급 가중 평균):\n');
fprintf('  %-14s  %6s  %6s  %6s\n', '별 유형', 'w_R', 'w_G', 'w_B');
fprintf('  %s\n', repmat('-', 1, 40));

optimal_by_temp = zeros(length(star_temps), 3);
for t_idx = 1:length(star_temps)
    T = star_temps(t_idx);
    S_total_T = S_R(t_idx) + S_G(t_idx) + S_B(t_idx);
    fr = S_R(t_idx) / S_total_T;
    fg = S_G(t_idx) / S_total_T;
    fb = S_B(t_idx) / S_total_T;

    w_r_arr = zeros(size(mag_fine));
    w_g_arr = zeros(size(mag_fine));
    w_b_arr = zeros(size(mag_fine));

    for i = 1:length(mag_fine)
        mag = mag_fine(i);
        pf = ref_photon_flux_space * 10^(-0.4 * (mag - 6.0));
        el = pf * sensor.exposure_time * sensor.quantum_eff;

        er = el * fr; eg = el * fg; eb = el * fb;
        dark_e = sensor.dark_current_space * sensor.exposure_time;
        rv = sensor.read_noise^2;

        vr = er + dark_e + rv;
        vg = eg + dark_e + rv;
        vb = eb + dark_e + rv;

        wr = er/vr; wg = eg/vg; wb = eb/vb;
        wt = wr + wg + wb;
        w_r_arr(i) = wr/wt;
        w_g_arr(i) = wg/wt;
        w_b_arr(i) = wb/wt;
    end

    wr_avg = sum(w_r_arr .* mag_weights);
    wg_avg = sum(w_g_arr .* mag_weights);
    wb_avg = sum(w_b_arr .* mag_weights);
    wt_avg = wr_avg + wg_avg + wb_avg;

    optimal_by_temp(t_idx, :) = [wr_avg/wt_avg, wg_avg/wt_avg, wb_avg/wt_avg];
    fprintf('  %-14s  %5.3f  %5.3f  %5.3f\n', star_types{t_idx}, ...
        optimal_by_temp(t_idx,1), optimal_by_temp(t_idx,2), optimal_by_temp(t_idx,3));
end
fprintf('\n');

% --- FPGA 구현용 정수 근사 ---
fprintf('  === FPGA 구현용 정수 근사 ===\n\n');

% 방법 1: /4 나눗셈 (2-bit 시프트)
% (a*R + b*G + c*B) / 4, a+b+c = 4 (정수)
a4 = round(w_R_final * 4);
b4 = round(w_G_final * 4);
c4 = round(w_B_final * 4);
% 합이 4가 되도록 조정
diff4 = 4 - (a4 + b4 + c4);
[~, max_ch] = max([w_R_final, w_G_final, w_B_final]);
if max_ch == 1, a4 = a4 + diff4;
elseif max_ch == 2, b4 = b4 + diff4;
else, c4 = c4 + diff4; end

fprintf('  /4 근사: (%d*R + %d*G + %d*B) / 4\n', a4, b4, c4);
fprintf('    실제 비율: %.3f : %.3f : %.3f\n', a4/4, b4/4, c4/4);
fprintf('    오차: R=%.1f%%, G=%.1f%%, B=%.1f%%\n', ...
    (a4/4 - w_R_final)/w_R_final*100, ...
    (b4/4 - w_G_final)/w_G_final*100, ...
    (c4/4 - w_B_final)/w_B_final*100);

% 방법 2: /8 나눗셈 (3-bit 시프트)
a8 = round(w_R_final * 8);
b8 = round(w_G_final * 8);
c8 = round(w_B_final * 8);
diff8 = 8 - (a8 + b8 + c8);
[~, max_ch8] = max([w_R_final, w_G_final, w_B_final]);
if max_ch8 == 1, a8 = a8 + diff8;
elseif max_ch8 == 2, b8 = b8 + diff8;
else, c8 = c8 + diff8; end

fprintf('\n  /8 근사: (%d*R + %d*G + %d*B) / 8\n', a8, b8, c8);
fprintf('    실제 비율: %.3f : %.3f : %.3f\n', a8/8, b8/8, c8/8);
fprintf('    오차: R=%.1f%%, G=%.1f%%, B=%.1f%%\n', ...
    (a8/8 - w_R_final)/w_R_final*100, ...
    (b8/8 - w_G_final)/w_G_final*100, ...
    (c8/8 - w_B_final)/w_B_final*100);

% 방법 3: /16 나눗셈 (4-bit 시프트)
a16 = round(w_R_final * 16);
b16 = round(w_G_final * 16);
c16 = round(w_B_final * 16);
diff16 = 16 - (a16 + b16 + c16);
[~, max_ch16] = max([w_R_final, w_G_final, w_B_final]);
if max_ch16 == 1, a16 = a16 + diff16;
elseif max_ch16 == 2, b16 = b16 + diff16;
else, c16 = c16 + diff16; end

fprintf('\n  /16 근사: (%d*R + %d*G + %d*B) / 16\n', a16, b16, c16);
fprintf('    실제 비율: %.3f : %.3f : %.3f\n', a16/16, b16/16, c16/16);
fprintf('    오차: R=%.1f%%, G=%.1f%%, B=%.1f%%\n\n', ...
    (a16/16 - w_R_final)/w_R_final*100, ...
    (b16/16 - w_G_final)/w_G_final*100, ...
    (c16/16 - w_B_final)/w_B_final*100);

%% ========================================================================
%  Section 8: 기존 방법과 SNR 비교
%  ========================================================================

fprintf('--- Section 8: 가중치별 SNR 비교 ---\n\n');

% 기존 방법 정의
methods = struct();
methods(1).name = '균등 (1/3)';
methods(1).w = [1/3, 1/3, 1/3];
methods(2).name = 'FPGA (R+2G+B)/4';
methods(2).w = [0.25, 0.50, 0.25];
methods(3).name = 'Green only';
methods(3).w = [0, 1, 0];
methods(4).name = '최적 (본 연구)';
methods(4).w = [w_R_final, w_G_final, w_B_final];
methods(5).name = sprintf('FPGA /8 (%d,%d,%d)', a8, b8, c8);
methods(5).w = [a8/8, b8/8, c8/8];

fprintf('  방법별 가중치:\n');
for i = 1:length(methods)
    fprintf('    %-20s: R=%.3f, G=%.3f, B=%.3f\n', ...
        methods(i).name, methods(i).w(1), methods(i).w(2), methods(i).w(3));
end
fprintf('\n');

% G2형 5800K, 우주 환경에서 등급별 SNR 비교
fprintf('  등급별 SNR 비교 (G2형, 우주):\n');
fprintf('  %-6s', '등급');
for i = 1:length(methods)
    fprintf('  %12s', methods(i).name);
end
fprintf('\n  %s\n', repmat('-', 1, 6 + 14*length(methods)));

SNR_all = zeros(length(methods), length(mag_fine));

for m = 1:length(mag_fine)
    mag = mag_fine(m);
    pf = ref_photon_flux_space * 10^(-0.4 * (mag - 6.0));
    el = pf * sensor.exposure_time * sensor.quantum_eff;

    er = el * frac_R_ref;
    eg = el * frac_G_ref;
    eb = el * frac_B_ref;

    dark_e = sensor.dark_current_space * sensor.exposure_time;
    rv = sensor.read_noise^2;

    vr = er + dark_e + rv;
    vg = eg + dark_e + rv;
    vb = eb + dark_e + rv;

    for i = 1:length(methods)
        w = methods(i).w;
        S_comb = w(1)*er + w(2)*eg + w(3)*eb;
        var_comb = w(1)^2*vr + w(2)^2*vg + w(3)^2*vb;
        SNR_all(i, m) = S_comb / sqrt(var_comb);
    end
end

% 주요 등급에서 출력
for mag = [0, 1, 2, 3, 4, 5, 6]
    idx = find(abs(mag_fine - mag) < 0.05, 1);
    if isempty(idx), continue; end
    fprintf('  %4.1f  ', mag);
    for i = 1:length(methods)
        fprintf('  %12.2f', SNR_all(i, idx));
    end
    fprintf('\n');
end
fprintf('\n');

% 최적 vs FPGA 개선율
idx_3 = find(abs(mag_fine - 3) < 0.05, 1);
idx_6 = find(abs(mag_fine - 6) < 0.05, 1);
if ~isempty(idx_3) && ~isempty(idx_6)
    improve_3 = (SNR_all(4, idx_3) / SNR_all(2, idx_3) - 1) * 100;
    improve_6 = (SNR_all(4, idx_6) / SNR_all(2, idx_6) - 1) * 100;
    fprintf('  최적 vs FPGA(R+2G+B)/4 개선율:\n');
    fprintf('    3등급: %+.1f%%\n', improve_3);
    fprintf('    6등급: %+.1f%%\n\n', improve_6);
end

%% ========================================================================
%  Section 9: 시각화
%  ========================================================================

fprintf('--- Section 9: 그래프 생성 ---\n');

% === Figure 1: 스펙트럼 + 센서 응답 ===
figure('Name', '흑체복사 스펙트럼 + OV4689 채널 응답', ...
    'Position', [50, 50, 1200, 500]);

subplot(1,2,1);
colors_temp = [0.8 0 0; 0.9 0.5 0; 0.6 0.6 0; 1 1 0; 0.5 0.8 0.5; 0.3 0.5 1; 0 0.2 0.8; 0.3 0 0.8];
for i = 1:length(star_temps)
    % 정규화 (피크=1)
    B_norm = B_photon(i,:) / max(B_photon(i,:));
    plot(lambda_nm, B_norm, 'Color', colors_temp(i,:), 'LineWidth', 1.2);
    hold on;
end
xlabel('파장 [nm]');
ylabel('정규화된 광자 스펙트럼');
title('별 온도별 흑체복사 스펙트럼');
legend(star_types, 'Location', 'eastoutside', 'FontSize', 7);
xlim([350 900]);
grid on;

subplot(1,2,2);
area(lambda_nm, QE_R*100, 'FaceColor', [1 0.5 0.5], 'FaceAlpha', 0.4, 'EdgeColor', 'r', 'LineWidth', 1.5);
hold on;
area(lambda_nm, QE_G*100, 'FaceColor', [0.5 1 0.5], 'FaceAlpha', 0.4, 'EdgeColor', [0 0.6 0], 'LineWidth', 1.5);
area(lambda_nm, QE_B*100, 'FaceColor', [0.5 0.5 1], 'FaceAlpha', 0.4, 'EdgeColor', 'b', 'LineWidth', 1.5);
plot(lambda_nm, QE_base*100, 'k--', 'LineWidth', 1, 'DisplayName', '실리콘 기저 QE');
xlabel('파장 [nm]');
ylabel('양자효율 QE [%]');
title('OV4689 채널별 스펙트럼 응답 (모델)');
legend({'Red', 'Green', 'Blue', '실리콘 기저'}, 'Location', 'northeast');
xlim([350 900]);
ylim([0 50]);
grid on;

% === Figure 2: 등급별 최적 가중치 ===
figure('Name', '등급별 최적 가중치 변화', ...
    'Position', [50, 100, 800, 500]);

plot(mag_fine, w_R_vs_mag, 'r-', 'LineWidth', 2); hold on;
plot(mag_fine, w_G_vs_mag, 'g-', 'LineWidth', 2);
plot(mag_fine, w_B_vs_mag, 'b-', 'LineWidth', 2);

% 기존 FPGA 가중치 표시
yline(0.25, 'r--', 'FPGA R=0.25', 'Alpha', 0.5, 'LineWidth', 1);
yline(0.50, 'g--', 'FPGA G=0.50', 'Alpha', 0.5, 'LineWidth', 1);
yline(0.25, 'b--', 'FPGA B=0.25', 'Alpha', 0.5, 'LineWidth', 1);

% 노이즈 레짐 전환점 표시
xline(transition_mag, 'k:', sprintf('전환점 %.1f mag', transition_mag), 'LineWidth', 1.5);

% 최종 가중치 표시
plot(mag_fine(end), w_R_final, 'rv', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
plot(mag_fine(end), w_G_final, 'g^', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
plot(mag_fine(end), w_B_final, 'bs', 'MarkerSize', 12, 'MarkerFaceColor', 'b');

xlabel('별 등급 [mag]');
ylabel('최적 가중치');
title(sprintf('최적 가중치 vs 등급 (G2형, 우주)\n최종: R=%.3f, G=%.3f, B=%.3f', ...
    w_R_final, w_G_final, w_B_final));
legend({'w_R (최적)', 'w_G (최적)', 'w_B (최적)'}, 'Location', 'east');
xlim([-1 7]);
ylim([0 0.7]);
grid on;

text(transition_mag+0.3, 0.65, 'Shot noise 지배 ←', 'FontSize', 9);
text(transition_mag+0.3, 0.60, '→ Read noise 지배', 'FontSize', 9);

% === Figure 3: SNR 비교 ===
figure('Name', '가중치별 SNR 비교', ...
    'Position', [50, 150, 900, 500]);

line_styles = {'k-', 'r--', 'g:', 'b-', 'm-.'};
for i = 1:length(methods)
    plot(mag_fine, 20*log10(max(SNR_all(i,:), 0.01)), line_styles{i}, 'LineWidth', 2);
    hold on;
end

% 검출 한계 (SNR = 3, 약 9.5 dB)
yline(20*log10(3), 'k--', 'SNR=3 (검출한계)', 'LineWidth', 1, 'Alpha', 0.7);

xlabel('별 등급 [mag]');
ylabel('SNR [dB]');
title('가중치별 SNR 비교 (G2형, 우주 환경)');
legend({methods.name}, 'Location', 'southwest');
xlim([-1 7]);
grid on;

% === Figure 4: 별 온도별 최적 가중치 ===
figure('Name', '별 온도별 최적 가중치', ...
    'Position', [50, 200, 800, 400]);

bar_data = optimal_by_temp;
b = bar(bar_data, 'grouped');
b(1).FaceColor = [1 0.4 0.4];
b(2).FaceColor = [0.4 0.8 0.4];
b(3).FaceColor = [0.4 0.4 1];

set(gca, 'XTickLabel', star_types);
xtickangle(30);
ylabel('최적 가중치');
title('별 온도별 최적 R:G:B 가중치 (우주 환경)');
legend({'R', 'G', 'B'}, 'Location', 'northeast');
grid on;
ylim([0 0.7]);

fprintf('  Figure 1~4 생성 완료\n\n');

%% ========================================================================
%  Section 10: 결과 저장
%  ========================================================================

fprintf('--- Section 10: 결과 저장 ---\n');

% 출력 폴더 확인
output_dir = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 결과 구조체
optimal_result = struct();
optimal_result.date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
optimal_result.environment = '우주 (대기 없음, -20°C)';
optimal_result.reference_star = 'G2형 (5800K, 태양 유사)';

% 최종 최적 가중치
optimal_result.w_R = w_R_final;
optimal_result.w_G = w_G_final;
optimal_result.w_B = w_B_final;

% FPGA 구현용
optimal_result.fpga_div4 = [a4, b4, c4];
optimal_result.fpga_div8 = [a8, b8, c8];
optimal_result.fpga_div16 = [a16, b16, c16];

% 별 온도별 가중치
optimal_result.star_temps = star_temps;
optimal_result.star_types = star_types;
optimal_result.weights_by_temp = optimal_by_temp;

% 센서 모델 파라미터
optimal_result.sensor = sensor;
optimal_result.QE_R = QE_R;
optimal_result.QE_G = QE_G;
optimal_result.QE_B = QE_B;
optimal_result.lambda_nm = lambda_nm;

% 기존 방법과의 비교
optimal_result.methods = methods;
optimal_result.SNR_comparison = SNR_all;
optimal_result.mag_range = mag_fine;

% 저장
save_path = fullfile(output_dir, 'optimal_weights_result.mat');
save(save_path, 'optimal_result');
fprintf('  결과 저장: %s\n', save_path);

% Figure 저장
saveas(1, fullfile(output_dir, 'spectrum_and_QE.png'));
saveas(2, fullfile(output_dir, 'optimal_weights_vs_magnitude.png'));
saveas(3, fullfile(output_dir, 'SNR_comparison.png'));
saveas(4, fullfile(output_dir, 'weights_by_temperature.png'));
fprintf('  그래프 저장: output/ 폴더\n');

%% ========================================================================
%  최종 요약
%  ========================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('  최종 결과 요약\n');
fprintf('================================================================\n\n');
fprintf('  [최적 가중치 (우주, 전 스펙트럼 평균)]\n');
fprintf('    R = %.4f\n', w_R_final);
fprintf('    G = %.4f\n', w_G_final);
fprintf('    B = %.4f\n\n', w_B_final);
fprintf('  [FPGA 구현 권장]\n');
fprintf('    (%d*R + %d*G + %d*B) >> 3    (/8 나눗셈)\n', a8, b8, c8);
fprintf('    또는\n');
fprintf('    (%d*R + %d*G + %d*B) >> 4    (/16 나눗셈, 더 정밀)\n\n', a16, b16, c16);
fprintf('  [기존 FPGA 대비 개선]\n');
if ~isempty(idx_6)
    fprintf('    6등급 별 SNR: %+.1f%% 개선\n', ...
        (SNR_all(4, idx_6) / SNR_all(2, idx_6) - 1) * 100);
end
fprintf('\n  [핵심 발견]\n');
fprintf('    1. Read noise 지배 영역(어두운 별)에서 가중치 최적화 효과 최대\n');
fprintf('    2. Shot noise 지배 영역(밝은 별)에서는 균등 가중과 차이 미미\n');
fprintf('    3. 별 온도(스펙트럼 유형)에 따른 최적 가중치 변화는 상대적으로 작음\n');
fprintf('    4. 고정 가중치 하나로 전체 카탈로그 커버 가능\n');
fprintf('================================================================\n');
