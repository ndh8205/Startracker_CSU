function result = evaluate_centroid_accuracy(detection, true_centroids, match_radius)
% EVALUATE_CENTROID_ACCURACY Centroid 정확도 평가
%
% [설명]
%   Nearest Neighbor(최근접 이웃) 매칭 방식으로 검출된 별 위치와 실제 별 위치를
%   비교하여 Centroid 정확도를 평가하는 함수입니다.
%   각 실제 별(ground truth)에 대해 검출 결과에서 가장 가까운 별을 찾고,
%   두 위치 간의 유클리드 거리를 오차(error)로 기록합니다.
%   최종적으로 모든 매칭된 별들의 RMS(Root Mean Square) 오차를 계산합니다.
%
% [알고리즘]
%   1. 각 실제 별 위치(true_centroids)에 대해 반복
%   2. 검출된 모든 별(detection.centroids)과의 유클리드 거리 계산
%   3. 가장 가까운 검출 별까지의 거리가 match_radius 이내이면 매칭 성공
%   4. 매칭 성공 시 해당 거리를 오차 배열에 추가
%   5. 모든 매칭 완료 후 RMS 오차 산출: sqrt(mean(errors^2))
%
% [주의사항]
%   - 1:1 매칭이 아닙니다. 하나의 검출 별이 여러 실제 별에 동시에 매칭될 수
%     있습니다. 밀집된 별 영역에서는 이 점에 유의해야 합니다.
%   - 바이닝(binning) 모드를 사용하는 경우, 이미지 해상도가 절반으로 줄어들므로
%     true_centroids에 0.5를 곱하여 스케일링한 후 전달해야 합니다.
%     예: evaluate_centroid_accuracy(det, true_centroids * 0.5, match_radius)
%   - match_radius가 너무 크면 잘못된 매칭(false match)이 발생할 수 있고,
%     너무 작으면 정상적인 매칭도 놓칠 수 있습니다.
%
% [성능 기준]
%   별센서(Star Tracker) 목표 정확도: 0.015도 = 약 54 arcsec
%   OV4689 센서 기준 환산:
%     - 픽셀 크기: 2um (마이크로미터)
%     - 초점 거리(focal length): f = 10.42mm
%     - 1 pixel 각도 = atan(2um / 10.42mm) = 약 0.011도
%     - 목표 0.015도 / 0.011도 = 약 1.4 pixel RMS
%   따라서 result.rms_error가 약 1.4 pixel 이하이면 목표 달성으로 판단합니다.
%
% [입력]
%   detection       - 별 검출 결과 구조체 (struct)
%     .n_detected   : 검출된 별의 개수 (정수, 0 이상)
%     .centroids    : 검출된 별의 중심 좌표 [N x 2] 행렬
%                     각 행은 [x, y] 또는 [col, row] 좌표 (단위: pixel)
%   true_centroids  - 실제 별의 중심 좌표 [M x 2] 행렬
%                     시뮬레이션에서 생성한 ground truth 위치
%                     각 행은 [x, y] 좌표 (단위: pixel)
%   match_radius    - 매칭 판정 반경 (단위: pixel, 기본값: 5.0)
%                     검출 별과 실제 별 간 거리가 이 값 미만이면 매칭 성공
%
% [출력]
%   result          - 평가 결과 구조체 (struct)
%     .n_matched    : 매칭 성공한 별의 개수 (정수)
%     .errors       : 각 매칭 별의 위치 오차 배열 [K x 1] (단위: pixel)
%                     K = n_matched, 각 값은 유클리드 거리
%     .rms_error    : RMS 오차 (단위: pixel)
%                     매칭된 별이 없으면 NaN 반환

% --- 기본 매개변수 설정 ---
% match_radius가 지정되지 않으면 기본값 5.0 pixel 사용
% 일반적인 별센서 시뮬레이션에서 5 pixel은 충분한 매칭 범위
if nargin < 3, match_radius = 5.0; end

% --- 결과 구조체 초기화 ---
% n_matched: 매칭 카운터를 0으로 초기화
% errors: 오차를 저장할 빈 배열 (매칭될 때마다 값 추가)
result.n_matched = 0;
result.errors = [];

% --- 빈 입력 처리 (예외 상황 방어) ---
% 검출된 별이 0개이거나 실제 별 좌표가 비어있으면 비교 자체가 불가능
% 이 경우 rms_error를 NaN으로 설정하고 즉시 반환
if detection.n_detected == 0 || isempty(true_centroids)
    result.rms_error = NaN;
    return;
end

% --- 각 실제 별에 대해 Nearest Neighbor 매칭 수행 ---
% true_centroids의 행(row) 수만큼 반복 = 실제 별의 개수만큼 반복
for i = 1:size(true_centroids, 1)
    % i번째 실제 별의 좌표 추출 [1 x 2] 벡터
    true_pos = true_centroids(i, :);

    % 가장 가까운 검출 별 찾기
    % --- 유클리드 거리 계산 ---
    % detection.centroids [N x 2] 에서 true_pos [1 x 2]를 빼면
    % MATLAB 브로드캐스팅에 의해 [N x 2] 차이 행렬 생성
    % .^2: 각 성분(dx, dy)을 제곱
    % sum(..., 2): 행 방향으로 합산 → dx^2 + dy^2
    % sqrt(): 제곱근 → 유클리드 거리 = sqrt(dx^2 + dy^2)
    % 결과: distances는 [N x 1] 벡터 (각 검출 별까지의 거리)
    distances = sqrt(sum((detection.centroids - true_pos).^2, 2));

    % 최소 거리와 해당 인덱스 추출
    % min_dist: 가장 가까운 검출 별까지의 거리 (pixel 단위)
    [min_dist, ~] = min(distances);

    % --- match_radius 판정 ---
    % 최소 거리가 match_radius 미만이면 매칭 성공으로 판정
    % match_radius 이상이면 해당 실제 별은 검출되지 않은 것으로 간주 (miss)
    if min_dist < match_radius
        % 매칭 성공: 카운터 증가 및 오차 기록
        result.n_matched = result.n_matched + 1;
        result.errors = [result.errors; min_dist];
    end
end

% --- RMS 오차 계산 ---
% 매칭된 별이 하나도 없는 경우 NaN 반환 (0으로 나누기 방지)
if isempty(result.errors)
    result.rms_error = NaN;
else
    % RMS (Root Mean Square) 오차 수식:
    %   RMS = sqrt( (1/K) * sum(e_i^2) )
    % 여기서 K는 매칭 개수, e_i는 각 매칭의 위치 오차 (pixel)
    % mean(errors.^2): 오차 제곱의 평균
    % sqrt(): 제곱근을 취해 원래 단위(pixel)로 복원
    % 이 값이 약 1.4 pixel 이하이면 별센서 목표 정확도(0.015도) 달성
    result.rms_error = sqrt(mean(result.errors.^2));
end
end
