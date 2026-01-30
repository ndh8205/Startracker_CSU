function result = detect_stars_simple(img, threshold, min_area)
% DETECT_STARS_SIMPLE 간단한 별 검출 (Threshold + Connected Component 분석)
%
% [설명]
%   Grayscale 이미지에서 배경보다 밝은 영역을 찾아 별을 검출합니다.
%   실제 별센서의 별 검출 알고리즘을 간소화한 버전입니다.
%
% [알고리즘 흐름]
%   1단계: 배경 추정 - 이미지 전체 중앙값(median) 계산
%   2단계: 이진화   - pixel > (배경 + threshold) → 1, 아니면 0
%   3단계: CCL      - 연결된 밝은 픽셀을 하나의 객체로 그룹화
%   4단계: 필터링   - 최소 면적 이상인 객체만 별로 판정
%   5단계: 추출     - 각 별의 중심 좌표와 밝기 추출
%
% [FPGA 별센서 알고리즘과의 비교]
%   실제 FPGA (system.sdk/ssov/src/algorithm/):
%     - CCL (Connected Component Labeling) 하드웨어 구현
%     - 라인 스캔 기반 (1줄씩 처리)
%     - Centroid: CoG (Center of Gravity) 방식
%   이 MATLAB 버전:
%     - MATLAB 내장 bwconncomp (동일 알고리즘, 소프트웨어 구현)
%     - regionprops로 속성 추출
%
% [입력]
%   img       - uint8 또는 double, [H x W] Grayscale 이미지
%   threshold - double, 검출 임계값 [ADU] (기본값: 20)
%               배경 + threshold 이상인 픽셀을 별 후보로 판정
%               권장 범위: 10~30 ADU (센서 노이즈 수준에 따라 조정)
%   min_area  - double, 최소 별 면적 [pixel] (기본값: 3)
%               이보다 작은 객체는 노이즈(핫픽셀)로 판정하여 제거
%               PSF sigma=1.2에서 별의 예상 면적: 약 5~15 pixel
%
% [출력]
%   result - struct:
%     .n_detected  : 검출된 별 개수 [정수]
%     .centroids   : [N x 2] 검출된 별 중심 좌표 [x, y] (서브픽셀 정밀도)
%                    좌표계: MATLAB 이미지 좌표 (1-based, 좌상 원점)
%     .intensities : [N x 1] 각 별의 최대 밝기 [ADU]
%
% [사용 예]
%   result = detect_stars_simple(gray_img, 15, 2);
%   fprintf('검출: %d개\n', result.n_detected);
%   plot(result.centroids(:,1), result.centroids(:,2), 'ro');

% === 기본값 설정 ===
if nargin < 2, threshold = 20; end
if nargin < 3, min_area = 3; end

% double 변환 (uint8 입력 시 연산 오류 방지)
img = double(img);

% --- 1단계: 배경 추정 ---
% 중앙값(median) 사용 이유:
%   - 별 픽셀은 전체의 0.1% 미만 (1280×720 이미지에서 별은 수십~수백 픽셀)
%   - 중앙값은 극단값(별)에 영향받지 않는 강건한(robust) 배경 추정치
%   - 우주 이미지에서 배경은 대부분 0에 가까움 (다크 전류 + 노이즈)
bg = median(img(:));

% --- 2단계: 이진화 (Thresholding) ---
% 배경 + threshold [ADU] 이상인 픽셀만 별 후보로 선택
% 예: 배경=3 ADU, threshold=15 → 18 ADU 이상인 픽셀이 별 후보
% 주의: threshold가 너무 낮으면 노이즈를 별로 오인 (False Positive ↑)
%       threshold가 너무 높으면 어두운 별을 놓침 (False Negative ↑)
binary = img > (bg + threshold);

% --- 3단계: Connected Component 분석 ---
% bwconncomp: 이진 이미지에서 인접한 '1' 픽셀을 하나의 객체로 묶음
%   - 8-연결 사용 (상하좌우 + 대각선 4방향, 총 8방향)
%   - 출력 cc: 각 연결 영역의 픽셀 인덱스 목록
cc = bwconncomp(binary);

% regionprops: 각 연결 영역의 속성(properties) 계산
%   - Centroid: 밝기 가중 중심 좌표 [x, y] (CoG: Center of Gravity)
%     → 서브픽셀 정밀도로 별 위치 추정 가능
%   - Area: 영역 면적 [pixel 수]
%   - MaxIntensity: 영역 내 최대 밝기 [ADU] (별의 피크)
%   - MeanIntensity: 영역 내 평균 밝기 [ADU]
% 주의: img를 두 번째 인자로 전달해야 밝기 가중 Centroid가 계산됨
stats = regionprops(cc, img, 'Centroid', 'Area', 'MaxIntensity', 'MeanIntensity');

% --- 4단계: 크기 필터링 ---
% 최소 면적(min_area) 이상인 객체만 별로 판정
% 1~2 픽셀 객체는 대부분 노이즈(핫픽셀, 코스믹 레이)이므로 제거
% PSF sigma=1.2 pixel에서 별의 예상 면적:
%   FWHM = 2.355 × sigma ≈ 2.8 pixel → 면적 ≈ π × (FWHM/2)² ≈ 6 pixel
valid_idx = find([stats.Area] >= min_area);

% --- 5단계: 결과 구성 ---
result.n_detected = length(valid_idx);
result.centroids = zeros(length(valid_idx), 2);
result.intensities = zeros(length(valid_idx), 1);

for i = 1:length(valid_idx)
    % Centroid: [x, y] 좌표 (MATLAB 좌표계: 1-based, 열=x, 행=y)
    result.centroids(i, :) = stats(valid_idx(i)).Centroid;
    % 최대 밝기: 별의 피크 ADU 값 (등급 추정에 활용 가능)
    result.intensities(i) = stats(valid_idx(i)).MaxIntensity;
end
end
