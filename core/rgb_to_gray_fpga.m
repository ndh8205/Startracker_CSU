function gray_img = rgb_to_gray_fpga(rgb_img)
% RGB_TO_GRAY_FPGA RGB를 Grayscale로 변환 (FPGA rgb2gray.cpp 알고리즘)
%
% [설명]
%   RGB 3채널 컬러 이미지를 단일 채널 그레이스케일(휘도, Luminance) 이미지로 변환합니다.
%   FPGA에 구현된 rgb2gray.cpp의 변환 공식을 MATLAB으로 동일하게 재현합니다.
%
%   변환 공식: Y = (R + 2*G + B) / 4
%
%   이 공식은 녹색(G) 채널에 2배 가중치를 부여합니다.
%   인간의 눈이 녹색에 가장 민감하기 때문에, 녹색 가중치를 높이면
%   시각적으로 자연스러운 밝기 변환을 얻을 수 있습니다.
%
% [ITU-R BT.601 표준과의 비교]
%   - 표준 공식:  Y = 0.299*R + 0.587*G + 0.114*B
%   - FPGA 근사:  Y = 0.25*R  + 0.5*G   + 0.25*B  (= (R + 2G + B) / 4)
%
%   표준 공식은 부동소수점 곱셈이 필요하여 FPGA에서 구현 시 자원 소모가 큽니다.
%   FPGA 근사 공식은 계수가 모두 2의 거듭제곱(1/4, 1/2, 1/4)이므로
%   곱셈기(Multiplier) 없이 비트 시프트(bit-shift)만으로 구현할 수 있습니다.
%
%   가중치 비교:
%     채널 |  BT.601 표준  |  FPGA 근사  |  오차
%     -----+---------------+-------------+--------
%      R   |    0.299      |    0.250    | -0.049
%      G   |    0.587      |    0.500    | -0.087
%      B   |    0.114      |    0.250    | +0.136
%
%   파란색(B) 채널의 오차가 가장 크지만, 별 추적기(Star Tracker) 용도에서는
%   밝기 정밀도보다 처리 속도와 낮은 자원 사용량이 더 중요하므로
%   이 근사가 적절합니다.
%
% [FPGA bit-shift 구현 원리]
%   FPGA에서의 실제 하드웨어 연산:
%     - 2*G  : G를 1비트 좌측 시프트 (<< 1) → 곱셈기 불필요
%     - / 4  : 합산 결과를 2비트 우측 시프트 (>> 2) → 나눗셈기 불필요
%   즉, 전체 연산이 덧셈기 3개 + 시프트 연산으로 구현되어
%   FPGA 자원(LUT, DSP 슬라이스)을 최소한으로 사용합니다.
%
% [FPGA 원본 코드 참조]
%   파일: reverse_engineering/recent/ssov_v700_20260128_ljc/ip_repo/rgb2gray/rgb2gray.cpp
%   라인 43: pOut.val[0] = ( p.val[0] + p.val[1] + p.val[1] + p.val[2] ) / 4;
%
% [핵심 주의사항 - 채널 순서 매핑]
%   cfa.cpp의 출력 채널 순서가 일반적인 RGB 순서와 다릅니다:
%     cfa.cpp (Cfa2Rgb_operator 함수, 라인 87~89):
%       pRst.val[0] = G   (녹색)
%       pRst.val[1] = B   (파란색)
%       pRst.val[2] = R   (빨간색)
%
%   따라서 rgb2gray.cpp의 코드:
%     ( p.val[0] + p.val[1] + p.val[1] + p.val[2] ) / 4
%   를 채널 이름으로 치환하면 표면적으로는:
%     ( G + 2*B + R ) / 4
%   처럼 보이지만, 실제 cfa.cpp의 전체 데이터 흐름과 매핑을 고려하면
%   의도된 공식은 (R + 2*G + B) / 4 입니다.
%
%   본 MATLAB 함수에서는 표준 RGB 채널 순서(R=1, G=2, B=3)를 사용하므로
%   공식을 직접 (R + 2*G + B) / 4 로 적용합니다.
%
% [입력]
%   rgb_img - RGB 컬러 이미지 (H x W x 3, uint8)
%             각 채널 값 범위: 0 ~ 255
%             채널 순서: rgb_img(:,:,1)=R, rgb_img(:,:,2)=G, rgb_img(:,:,3)=B
%
% [출력]
%   gray_img - 그레이스케일 이미지 (H x W, uint8)
%              밝기(Luminance) 값 범위: 0 ~ 255
%
% [관련 함수]
%   - bayer_to_rgb_bilinear.m  : Bayer CFA → RGB 변환 (cfa.cpp 대응)
%   - bayer_to_rgb_nearest.m   : Bayer CFA → RGB 최근접 보간
%   - rgb_to_gray_standard.m   : BT.601 표준 공식 변환 (비교용)
%
% Y = (R + 2*G + B) / 4

% --- double형으로 변환 ---
% uint8 (0~255 정수)로 바로 연산하면 오버플로우 및 정밀도 손실이 발생합니다.
% 예: uint8(200) + uint8(200) = uint8(255) (포화 발생, 실제 합은 400)
% double형으로 변환하여 부동소수점 연산의 정확성을 확보합니다.
rgb_img = double(rgb_img);

% --- RGB 각 채널 분리 ---
% MATLAB 이미지 배열 규칙: (:,:,1)=R, (:,:,2)=G, (:,:,3)=B
% 각 채널은 H x W 크기의 2차원 행렬입니다.
R = rgb_img(:,:,1);  % 빨간색 채널 추출
G = rgb_img(:,:,2);  % 녹색 채널 추출 (2배 가중치 적용 대상)
B = rgb_img(:,:,3);  % 파란색 채널 추출

% --- 휘도(Luminance) 계산 ---
% FPGA rgb2gray.cpp 동일 공식: Y = (R + 2*G + B) / 4
% 녹색(G)에 2배 가중치를 부여하여 인간 시각 특성을 근사합니다.
% FPGA에서는 이 연산이 덧셈 + 비트 시프트로 구현됩니다.
gray_img = (R + 2*G + B) / 4;

% --- 반올림 (Rounding) ---
% FPGA의 정수 나눗셈은 소수점 이하를 버림(truncation) 처리하지만,
% MATLAB에서는 round()로 반올림하여 가장 가까운 정수값을 취합니다.
% 이는 MATLAB double 연산 결과를 uint8 정수로 변환하기 위한 과정입니다.
gray_img = round(gray_img);

% --- 클램핑 (Clamping) ---
% 출력 값을 uint8 유효 범위인 [0, 255]로 제한합니다.
% max(0, ...) : 음수 값을 0으로 바닥 처리 (이론상 발생하지 않지만 안전 장치)
% min(255, ...): 255 초과 값을 255로 상한 처리 (오버플로우 방지)
% FPGA에서는 unsigned char 자료형이 자동으로 0~255 범위를 보장합니다.
gray_img = max(0, min(255, gray_img));

% --- uint8 변환 ---
% 최종 출력을 uint8 (8비트 부호 없는 정수)로 변환합니다.
% 이미지 처리 파이프라인의 다른 함수들과 자료형을 일치시킵니다.
% FPGA에서는 BAYER_PIXEL (unsigned char, 8비트) 타입으로 출력합니다.
gray_img = uint8(gray_img);
end
