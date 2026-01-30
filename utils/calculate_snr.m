function snr = calculate_snr(measured, reference)
% CALCULATE_SNR 레퍼런스 기반 SNR(Signal-to-Noise Ratio, 신호 대 잡음비) 계산 [dB]
%
% === 기능 설명 ===
% 처리된 영상(measured)이 기준 영상(reference, ground truth)으로부터
% 얼마나 벗어났는지를 데시벨(dB) 단위로 정량화합니다.
% bayer_comparison에서 다양한 Bayer->Gray 변환 방법의 품질을
% "깨끗한" 레퍼런스 영상과 비교하여 평가하는 데 사용됩니다.
%
% === 수학적 정의 ===
%   SNR [dB] = 10 * log10(signal_power / noise_power)
%
%   여기서:
%     signal_power = mean(reference^2)
%       = 레퍼런스 영상의 평균 에너지 [ADU^2]
%     noise_power  = mean((measured - reference)^2)
%       = 측정 영상과 레퍼런스 영상 차이의 평균 제곱 오차(MSE) [ADU^2]
%
% === 10*log10 vs 20*log10 ===
% 10*log10을 사용하는 이유: signal_power와 noise_power가 이미 제곱된
% 값(파워/에너지 단위)이기 때문입니다. 진폭(amplitude) 비율이면
% 20*log10을 사용하지만, 여기서는 파워 비율이므로 10*log10이 올바릅니다.
%
% === SNR 해석 기준 ===
%   SNR > 30 dB  : 우수 (excellent) - 레퍼런스와 거의 동일
%   20~30 dB     : 양호 (good) - 육안으로 차이 미미
%   SNR < 20 dB  : 노이즈 많음 (noisy) - 눈에 띄는 열화
%   높을수록 measured 영상이 reference에 가까움을 의미합니다.
%
% === 주요 사용 사례 ===
% Bayer->Gray 변환 방법 비교 시:
%   reference = 원본 별 영상에서 직접 생성한 이상적인 그레이스케일
%   measured  = Bayer 패턴 -> Gray 변환을 거친 결과 영상
%   SNR 값으로 각 변환 방법의 정보 손실 정도를 비교합니다.
%
% --- 입력 ---
%   measured  : 처리된 영상 (2D 행렬) [ADU] - Bayer->Gray 변환 결과
%   reference : 기준 영상 (2D 행렬) [ADU] - 이상적인 그레이스케일 (ground truth)
%               * measured와 크기가 다를 수 있음 (binning 방법의 경우)
%
% --- 출력 ---
%   snr : 신호 대 잡음비 [dB]
%         * noise가 0에 가까우면 100 dB 반환 (사실상 무한대)
%
% --- 관련 함수 ---
%   calculate_peak_snr.m : 단일 영상 기반 Peak SNR 계산 (레퍼런스 불필요)
%
% 사용 예:
%   snr_value = calculate_snr(processed_gray, ideal_gray);

% =========================================================================
% === 영상 크기 맞춤 (Size Matching) ===
% =========================================================================
% binning 방법은 2x2 픽셀을 1개로 합치므로 출력 해상도가 절반
% (예: 1280x720 -> 640x360 [pixel])이 됩니다.
% 다른 방법들(직접 변환, 디모자이킹 등)은 원본 해상도를 유지하므로
% measured와 reference의 크기가 다를 수 있습니다.
% 정확한 비교를 위해 크기를 일치시켜야 합니다.
[h1, w1] = size(measured);
[h2, w2] = size(reference);

% --- 크기가 다를 경우 reference를 measured 크기로 리사이즈 ---
% measured(처리 결과)가 아닌 reference(기준)를 리사이즈합니다.
% 이유: 처리 결과(measured)를 그대로 보존하여 변환 방법의 실제 출력을
%       왜곡 없이 평가하기 위함입니다.
% imresize 기본 보간법: bilinear (쌍선형 보간)
%   → 리사이즈 시 최소한의 아티팩트만 발생합니다.
if h1 ~= h2 || w1 ~= w2
    reference = imresize(reference, [h1, w1]);
end

% =========================================================================
% === 데이터 타입 변환 (double 변환) ===
% =========================================================================
% uint8 -> double 변환: 부동소수점 연산을 위해 필수
% uint8 상태에서 제곱 시 오버플로우 발생:
%   예) uint8(200)^2 = 40000 > 255 → uint8로는 255에서 포화(saturation)
% double로 변환하면 정밀한 제곱/평균 계산이 가능합니다.
measured = double(measured);
reference = double(reference);

% =========================================================================
% === 신호 파워(Signal Power) 계산 ===
% =========================================================================
% signal_power: 레퍼런스 영상의 평균 에너지 [ADU^2]
% 이상적인 영상이 가진 "신호의 크기"를 나타냅니다.
% measured가 아닌 reference를 신호 기준으로 사용하는 것이 표준 관례입니다.
% (:) : 2D 행렬을 1D 벡터로 펼침 (vectorization)
% .^2  : 원소별(element-wise) 제곱
signal_power = mean(reference(:).^2);

% =========================================================================
% === 잡음 파워(Noise Power) 계산 ===
% =========================================================================
% noise_power: 측정 영상과 레퍼런스 간의 평균 제곱 오차 (MSE) [ADU^2]
% "잡음"은 처리 과정에서 유입된 모든 차이를 포함합니다:
%   - 컬러 보간(demosaicing) 오차
%   - 해상도 손실 (binning에 의한)
%   - 양자화(quantization) 오차
%   - 기타 처리 아티팩트
noise_power = mean((measured(:) - reference(:)).^2);

% =========================================================================
% === SNR 계산 및 제로 노이즈 보호 ===
% =========================================================================
% --- 매직 넘버 1e-10: 사실상 노이즈 제로 판별 임계값 ---
% noise_power가 0에 근접하면 log10(signal/0) = log10(무한대)가 되어
% 나눗셈 오류(division by zero)가 발생합니다.
% 이 경우 100 dB을 "사실상 무한대 SNR"로 반환합니다.
% (calculate_peak_snr.m과 동일한 규약 사용)
% 발생 가능한 경우: 영상을 자기 자신과 비교하거나,
%                    처리 과정이 무손실(lossless)인 경우
if noise_power < 1e-10
    snr = 100;
else
    % --- 데시벨 변환 [dB] ---
    % 10*log10: 파워(에너지) 비율의 데시벨 변환
    % 10배 (20배 아님): signal_power, noise_power가 이미 제곱 값(파워)이므로
    % 예시: signal_power=1000, noise_power=10
    %       → SNR = 10 * log10(1000/10) = 10 * log10(100) = 20 [dB]
    snr = 10 * log10(signal_power / noise_power);
end
end
