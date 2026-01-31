function [gray_img, method_info] = bayer_to_gray_direct(bayer_img, method, weights)
% BAYER_TO_GRAY_DIRECT Bayer 패턴에서 직접 Grayscale로 변환
%
% [설명]
%   Bayer 패턴 이미지(RAW)를 중간에 RGB 컬러 이미지를 생성하지 않고
%   직접 그레이스케일(휘도) 이미지로 변환하는 함수입니다.
%
%   기존 FPGA 파이프라인은 Bayer → CFA(RGB) → rgb2gray(Gray) 의 2단계를
%   거치지만, 이 함수는 Bayer → Gray 변환을 1단계로 직접 수행합니다.
%   이를 통해 연산량 감소 및 파이프라인 단순화가 가능합니다.
%
% [FPGA 파이프라인 비교도]
%
%   === 기존 FPGA 파이프라인 (2단계) ===
%
%     Bayer RAW ──→ [CFA 디모자이킹] ──→ RGB 3채널 ──→ [rgb2gray] ──→ Gray
%     (H x W)       cfa.cpp              (H x W x 3)   rgb2gray.cpp   (H x W)
%                   3x3 윈도우 보간       메모리 3배     (R+2G+B)/4
%                   FPGA IP              AXI Stream     FPGA IP
%
%   === 직접 변환 (1단계, 이 함수) ===
%
%     Bayer RAW ──→ [Direct Gray 변환] ──→ Gray
%     (H x W)       이 함수               (H x W)
%                   CFA+rgb2gray 통합
%                   또는 단순화된 접근
%
%   직접 변환의 장점:
%     - RGB 중간 데이터 불필요 → 메모리 대역폭 1/3로 감소
%     - FPGA IP 2개 → 1개로 통합 가능 → 자원(LUT/DSP) 절감
%     - 별 추적기(Star Tracker)는 컬러 정보가 불필요하므로 합리적인 접근
%
% [4가지 변환 방법 비교]
%
%   방법      | 출력 크기      | 속도 | 품질 | 설명
%   ----------+----------------+------+------+------------------------------------
%   'raw'     | H x W (동일)   | 최고 | 최저 | Bayer 값 그대로 사용 (변환 없음)
%   'binning' | H/2 x W/2      | 높음 | 낮음 | 2x2 블록 평균, 해상도 1/2 감소
%   'green'   | H x W (동일)   | 보통 | 보통 | Green 채널만 추출 + 보간
%   'weighted'| H x W (동일)   | 낮음 | 최고 | CFA+rgb2gray 통합, 가중 평균
%
%   별 추적기 용도 권장:
%     - 별 검출(Threshold): 'raw' 또는 'binning' (속도 우선)
%     - Centroid 계산: 'green' 또는 'weighted' (정확도 우선)
%     - FPGA 구현: 'binning' (하드웨어 가장 단순) 또는 'weighted' (기존 IP 대체)
%
% [입력]
%   bayer_img - Bayer RGGB 패턴 이미지 [ADU] (rows x cols, uint8 또는 double)
%               OV4689 센서 RAW 출력에 해당하는 단일 채널 영상
%               각 픽셀 값 범위: 0~255 [ADU] (8비트 ADC 출력)
%               기본 해상도: 2688 x 1520 [pixel] (OV4689 풀프레임)
%
%   method   - 변환 방법 문자열 (선택, 기본값: 'raw')
%               'raw'      : Bayer 값 그대로 사용
%               'binning'  : 2x2 바이닝 (R+Gr+Gb+B)/4
%               'green'    : Green 채널 추출 + 십자 보간
%               'weighted' : 2x2 가중 평균 (R+2G+B)/4
%               'optimal'  : SNR 최대화 가중치 (sub_main_2_optimal_weights 참조)
%
%   weights  - (선택) 'optimal' 모드용 [w_R, w_G, w_B] 가중치 벡터
%               합이 1이어야 함. 미지정 시 기본값 사용.
%               sub_main_2_optimal_weights.m에서 도출된 값 사용 권장.
%
% [출력]
%   gray_img    - 그레이스케일 이미지 [ADU] (uint8)
%                  크기는 method에 따라 상이:
%                    'raw', 'green', 'weighted': rows x cols [pixel]
%                    'binning': floor(rows/2) x floor(cols/2) [pixel]
%
%   method_info - 변환 방법 메타데이터 (struct)
%                  .method      : 사용된 method 문자열
%                  .input_size  : 입력 크기 [rows, cols] [pixel]
%                  .output_size : 출력 크기 [rows, cols] [pixel]
%                  .description : 방법 설명 (한국어)
%
% [사용 예제]
%   % 기본 사용 (RAW 모드)
%   gray = bayer_to_gray_direct(bayer_raw);
%
%   % 바이닝 모드 (SNR 향상)
%   [gray, info] = bayer_to_gray_direct(bayer_raw, 'binning');
%   fprintf('출력 크기: %dx%d\n', info.output_size(1), info.output_size(2));
%
%   % 가중 평균 모드 (FPGA 파이프라인 대체)
%   gray_weighted = bayer_to_gray_direct(bayer_raw, 'weighted');
%
% [관련 함수]
%   - bayer_to_rgb_cfa.m    : Bayer → RGB CFA 디모자이킹 (FPGA cfa.cpp 재현)
%   - rgb_to_gray_fpga.m    : RGB → Gray 변환 (FPGA rgb2gray.cpp 재현)
%     → 이 두 함수를 순차 적용하면 기존 FPGA 파이프라인과 동일한 결과
%     → 본 함수의 'weighted' 모드는 이 두 단계를 하나로 통합한 것
%
% [FPGA 원본 소스 경로]
%   CFA 디모자이킹:
%     reverse_engineering/recent/ssov_v700_20260128_ljc/ip_repo/cfa/cfa.cpp
%   RGB→Gray 변환:
%     reverse_engineering/recent/ssov_v700_20260128_ljc/ip_repo/rgb2gray/rgb2gray.cpp

% === 입력 인수 처리 ===
% method 미지정 시 'raw'를 기본값으로 사용 (가장 빠른 방법)
if nargin < 2
    method = 'raw';
end

% weights 미지정 시 기본값 설정 ('optimal' 모드에서 사용)
% 기본값: sub_main_2_optimal_weights.m의 practical_weights에서 도출된 값
% 이 값은 OV4689 센서의 분광 응답과 흑체복사 스펙트럼 기반으로
% inverse variance weighting을 적용하여 SNR을 최대화하는 가중치입니다.
% 도출 과정: Section 7 (전체 카탈로그 등급 가중 평균)
if nargin < 3
    % sub_main_2_optimal_weights.m 실행 결과 (2026-01-31 도출)
    % 우주 환경, 전 스펙트럼 등급 가중 평균 (G2형 기준)
    % Inverse Variance Weighting: w_i ∝ S_i / σ_i²
    weights = [0.4544, 0.3345, 0.2111];  % [w_R, w_G, w_B], 합 = 1
end

% === 전처리: double 형변환 ===
% uint8 정수 연산 시 오버플로우 및 반올림 오차를 방지하기 위해
% 부동소수점(double)으로 변환 후 연산합니다.
% 예: uint8(200) + uint8(200) = uint8(255) (포화), double이면 400.0 (정확)
bayer_img = double(bayer_img);
[rows, cols] = size(bayer_img);

% === 메타데이터 초기화 ===
% 변환 방법, 입출력 크기 등의 정보를 구조체에 기록합니다.
method_info.method = method;
method_info.input_size = [rows, cols];

switch lower(method)
    % =====================================================================
    % === 방법 1: RAW (패스스루) ===
    % =====================================================================
    % Bayer 패턴 값을 그대로 그레이스케일로 사용합니다.
    % 어떠한 보간이나 변환도 수행하지 않으며, 가장 빠른 방법입니다.
    %
    % [원리]
    %   Bayer 센서의 각 픽셀은 R, Gr, Gb, B 중 하나의 값을 가집니다.
    %   이 값들은 결국 해당 색 필터를 통과한 빛의 세기(intensity)이므로,
    %   밝은 별은 어떤 색 채널이든 높은 값을 가집니다.
    %
    % [별 추적기에서의 적합성]
    %   별은 점광원(point source)이므로 색상 정보가 중요하지 않습니다.
    %   밝기 기반 별 검출(thresholding)에서는 Bayer 값 그대로도 충분합니다.
    %   → 별의 밝기 > 임계값 여부만 판단하면 되므로 색 보간 불필요
    %
    % [한계]
    %   자연 이미지에서는 RGGB 패턴에 의한 격자 아티팩트(grid artifact)가
    %   나타납니다. 그러나 점광원(별)에서는 이 효과가 무시할 수준입니다.
    %
    %   격자 아티팩트 예시 (확대 시):
    %     [R ] [Gr] [R ] [Gr]     밝기가 R, G, B에 따라 다르므로
    %     [Gb] [B ] [Gb] [B ]  →  체크보드 형태의 밝기 변동 발생
    %     [R ] [Gr] [R ] [Gr]     (별 추적에는 무시 가능)
    %     [Gb] [B ] [Gb] [B ]
    case 'raw'
        % RAW 값 그대로 사용
        gray_img = uint8(bayer_img);
        method_info.output_size = [rows, cols];
        method_info.description = 'RAW 값 그대로';

    % =====================================================================
    % === 방법 2: 2x2 바이닝 (Binning) ===
    % =====================================================================
    % 2x2 RGGB 블록 4개 픽셀의 산술 평균을 1개 출력 픽셀로 변환합니다.
    % 출력 해상도가 입력의 1/2 (가로, 세로 각각)로 줄어듭니다.
    %
    % [변환 원리 - ASCII 다이어그램]
    %
    %   입력 Bayer 2x2 블록:        출력 1 픽셀:
    %
    %   +----+----+
    %   | R  | Gr |                      R + Gr + Gb + B
    %   +----+----+  ──→  Gray_out = ─────────────────────
    %   | Gb | B  |                            4
    %   +----+----+
    %   (2x2 = 4 픽셀)              (1x1 = 1 픽셀)
    %
    %   구체적 수식:
    %     Gray(r,c) = (R + Gr + Gb + B) / 4  [ADU]
    %
    %   전체 해상도 변화:
    %     입력: 2688 x 1520 [pixel] → 출력: 1344 x 760 [pixel]
    %     (OV4689 기준)
    %
    % [SNR(신호 대 잡음비) 향상]
    %   4개 픽셀을 평균하면 잡음이 sqrt(4) = 2배 감소합니다.
    %   → SNR 약 2배 (6 dB) 향상
    %   → 어두운 별 검출에 유리 (별 추적기 핵심 이점)
    %
    %   수식: SNR_out = SNR_in * sqrt(N), N=4 (평균한 픽셀 수)
    %
    % [(R+Gr+Gb+B)/4 ≈ 휘도(Luminance)인 이유]
    %   Bayer RGGB 패턴에서 Green이 2개(Gr, Gb)이므로:
    %     (R + Gr + Gb + B) / 4 = (R + 2G + B) / 4 = 0.25R + 0.5G + 0.25B
    %   이는 FPGA rgb2gray.cpp의 Y = (R + 2G + B) / 4 공식과 동일합니다!
    %   → Green 50% 가중치는 인간 시각 민감도(BT.601: G=0.587)와 근사
    %   → 참고: rgb_to_gray_fpga.m 의 변환 공식
    %
    % [FPGA 구현 용이성]
    %   이 연산은 FPGA에서 매우 간단하게 구현 가능합니다:
    %     - 덧셈기 3개 (R+Gr, +Gb, +B)
    %     - 2비트 우측 시프트 (>> 2) → /4와 동일, 나눗셈기 불필요
    %   → CFA + rgb2gray 두 IP를 단순한 덧셈+시프트로 완전히 대체 가능
    case 'binning'
        % 2x2 바이닝
        % --- 1단계: 출력 크기 계산 ---
        % floor()를 사용하여 입력이 홀수 크기일 때 마지막 행/열을 버립니다.
        out_rows = floor(rows / 2);
        out_cols = floor(cols / 2);
        gray_img = zeros(out_rows, out_cols);

        % --- 2단계: 2x2 블록 순회 및 평균 계산 ---
        for r = 1:out_rows
            for c = 1:out_cols
                % 출력 좌표 (r,c) → 입력 Bayer 2x2 블록의 좌상단 좌표 (br,bc)
                br = (r-1)*2 + 1;
                bc = (c-1)*2 + 1;
                % 2x2 RGGB 블록에서 각 색상 추출 [ADU]
                %   (br, bc)   = R  위치 (짝수행, 짝수열)
                %   (br, bc+1) = Gr 위치 (짝수행, 홀수열)
                %   (br+1, bc) = Gb 위치 (홀수행, 짝수열)
                %   (br+1, bc+1) = B 위치 (홀수행, 홀수열)
                R  = bayer_img(br, bc);
                Gr = bayer_img(br, bc+1);
                Gb = bayer_img(br+1, bc);
                B  = bayer_img(br+1, bc+1);
                % 4픽셀 산술 평균 = (R + 2G + B) / 4 [ADU]
                gray_img(r, c) = (R + Gr + Gb + B) / 4;
            end
        end

        % --- 3단계: uint8 변환 ---
        % round()로 반올림 후 uint8로 변환 (0~255 범위 [ADU])
        gray_img = uint8(round(gray_img));
        method_info.output_size = [out_rows, out_cols];
        method_info.description = '2x2 바이닝';

    % =====================================================================
    % === 방법 3: Green 채널 추출 ===
    % =====================================================================
    % Bayer 패턴에서 Green(Gr, Gb) 픽셀만 선택적으로 추출하고,
    % R/B 위치에서는 인접 Green 픽셀들의 십자 평균으로 보간합니다.
    % 출력 해상도는 입력과 동일합니다 (H x W).
    %
    % [Green 채널 ≈ 휘도(Luminance)인 이유]
    %   인간의 눈은 녹색(Green)에 가장 민감합니다.
    %   ITU-R BT.601 표준: Y = 0.299*R + 0.587*G + 0.114*B
    %   → Green 가중치가 58.7%로 가장 크므로, Green만 사용해도
    %     휘도의 약 60%를 캡처할 수 있습니다.
    %   → 별 추적기에서는 밝기 정보만 필요하므로 충분한 근사
    %
    % [RGGB 패턴에서 Green 픽셀 위치]
    %
    %   열(col):  1     2     3     4     5     6   ...
    %   행(row):
    %     1     [ R  ][*Gr*][ R  ][*Gr*][ R  ][*Gr*] ...
    %     2     [*Gb*][ B  ][*Gb*][ B  ][*Gb*][ B  ] ...
    %     3     [ R  ][*Gr*][ R  ][*Gr*][ R  ][*Gr*] ...
    %     4     [*Gb*][ B  ][*Gb*][ B  ][*Gb*][ B  ] ...
    %     ...
    %
    %   *표시* = Green 픽셀 (전체의 50%, 체스보드 패턴)
    %   Green 마스크 조건: (짝수행 AND 홀수열) OR (홀수행 AND 짝수열)
    %   → XOR 논리와 동일: row_idx XOR col_idx = 1
    %
    % [보간 방법 - 십자 평균]
    %   R 또는 B 위치에서 Green 값이 없으므로, 상하좌우 4개 이웃의 평균으로 보간:
    %
    %          G_up
    %           |
    %   G_left--[R/B]--G_right    →  G = (G_up + G_down + G_left + G_right) / 4
    %           |
    %         G_down
    %
    %   이 보간은 bayer_to_rgb_cfa.m의 bilinear 보간과 동일한 방식입니다.
    %   (참고: cfa.cpp의 3x3 윈도우 기반 보간)
    case 'green'
        % Green 채널만 추출 + 보간
        gray_img = zeros(rows, cols);
        green_mask = false(rows, cols);

        % --- 1단계: Green 마스크 생성 ---
        % RGGB Bayer 패턴에서 Green 위치를 논리 마스크로 표시합니다.
        % Green 위치 조건:
        %   r_idx=0, c_idx=1 → Gr (짝수행, 홀수열)
        %   r_idx=1, c_idx=0 → Gb (홀수행, 짝수열)
        % 이는 체스보드(checkerboard) 패턴과 동일합니다.
        for r = 1:rows
            for c = 1:cols
                % 0-based 인덱스로 변환 (MATLAB은 1-based)
                r_idx = mod(r-1, 2);
                c_idx = mod(c-1, 2);
                % Gr 위치(짝수행,홀수열) 또는 Gb 위치(홀수행,짝수열)인 경우 Green
                % 논리적으로 r_idx XOR c_idx == 1 과 동일
                if (r_idx == 0 && c_idx == 1) || (r_idx == 1 && c_idx == 0)
                    green_mask(r, c) = true;
                end
            end
        end

        % --- 2단계: Green 위치에 원본 값 직접 할당 ---
        % Green 픽셀 위치에서는 보간 없이 Bayer 원본 값을 그대로 사용합니다.
        % 이 값이 해당 위치의 실제 Green 밝기입니다 [ADU].
        gray_img(green_mask) = bayer_img(green_mask);

        % --- 3단계: R/B 위치에서 Green 값 보간 (십자 평균) ---
        % 'replicate' 패딩: 경계 처리를 위해 상하좌우 1픽셀씩 확장
        % → bayer_to_rgb_cfa.m과 동일한 경계 처리 방식
        % → FPGA cfa.cpp의 line_buffer 경계 처리와 동등
        padded = padarray(bayer_img, [1 1], 'replicate');
        for r = 1:rows
            for c = 1:cols
                if ~green_mask(r, c)
                    % R 또는 B 위치 → 상하좌우 Green 이웃의 평균으로 보간
                    % padded 이미지에서의 인덱스 (+1 오프셋)
                    pr = r + 1;
                    pc = c + 1;
                    % 십자(+) 형태 4방향 평균 [ADU]:
                    %   위: padded(pr-1, pc)
                    %   아래: padded(pr+1, pc)
                    %   왼쪽: padded(pr, pc-1)
                    %   오른쪽: padded(pr, pc+1)
                    gray_img(r, c) = (padded(pr-1,pc) + padded(pr+1,pc) + ...
                                     padded(pr,pc-1) + padded(pr,pc+1)) / 4;
                end
            end
        end

        % --- 4단계: uint8 변환 ---
        gray_img = uint8(round(gray_img));
        method_info.output_size = [rows, cols];
        method_info.description = 'Green 채널 추출';

    % =====================================================================
    % === 방법 4: 2x2 가중 평균 (Weighted) ===
    % =====================================================================
    % CFA 디모자이킹과 rgb2gray 변환을 하나의 단계로 통합한 방법입니다.
    % 각 픽셀 위치에서 Y = (R + 2*G + B) / 4 공식을 로컬 이웃으로 계산합니다.
    %
    % [핵심 원리]
    %   기존 FPGA 파이프라인: Bayer → cfa.cpp(RGB) → rgb2gray.cpp(Gray)
    %   이 방법:             Bayer → 직접 Gray (cfa + rgb2gray 통합)
    %
    %   rgb_to_gray_fpga.m의 Y = (R + 2G + B) / 4 공식을 적용하되,
    %   RGB 채널을 먼저 복원하는 대신, Bayer 패턴의 이웃 픽셀에서
    %   직접 R, G, B 값을 추출/보간하여 Y를 한 번에 계산합니다.
    %
    % [pos_idx 계산]
    %   bayer_to_rgb_cfa.m과 동일한 방식으로 Bayer 패턴 위치를 판별합니다.
    %   pos_idx = r_idx * 2 + c_idx (2비트 값: 0~3)
    %     0 = R 위치  (짝수행, 짝수열)
    %     1 = Gr 위치 (짝수행, 홀수열)
    %     2 = Gb 위치 (홀수행, 짝수열)
    %     3 = B 위치  (홀수행, 홀수열)
    %
    %   참고: FPGA cfa.cpp의 ap_uint<2> pos_idx와 동일한 계산
    %         (bayer_to_rgb_cfa.m 참조)
    case 'weighted'
        % 2x2 가중 평균
        gray_img = zeros(rows, cols);
        % 3x3 윈도우 처리를 위한 경계 패딩 (상하좌우 1픽셀)
        % bayer_to_rgb_cfa.m과 동일한 'replicate' 패딩 방식
        padded = padarray(bayer_img, [1 1], 'replicate');

        for r = 1:rows
            for c = 1:cols
                % padded 이미지에서의 중심 좌표 (+1 오프셋)
                pr = r + 1;
                pc = c + 1;
                % Bayer 패턴 위치 판별 (bayer_to_rgb_cfa.m과 동일)
                r_idx = mod(r-1, 2);
                c_idx = mod(c-1, 2);
                pos_idx = r_idx * 2 + c_idx;

                switch pos_idx
                    % =============================================================
                    % case 0: R 위치 (짝수행, 짝수열)
                    % -------------------------------------------------------------
                    % 3x3 윈도우의 Bayer 배치:
                    %
                    %   +------+------+------+
                    %   | B    | Gb   | B    |
                    %   +------+------+------+
                    %   | Gr   |[R]   | Gr   |  ← 중심 = Red
                    %   +------+------+------+
                    %   | B    | Gb   | B    |
                    %   +------+------+------+
                    %
                    % 각 채널 추출:
                    %   R = center = padded(pr, pc)                       [ADU]
                    %   G = 십자 평균 = (상+우+하+좌) / 4                  [ADU]
                    %     = (Gb + Gr + Gb + Gr) / 4
                    %   B = 대각 평균 = padded(pr+1, pc+1)                [ADU]
                    %     (주: 코드에서는 우하단 1개만 사용 - 근사)
                    %
                    % Y = (R + 2*G_avg + B_avg) / 4                       [ADU]
                    %   = (center + cross_avg*2 + diagonal) / 4
                    % -------------------------------------------------------------
                    case 0
                        gray_img(r,c) = (padded(pr,pc) + ...
                            (padded(pr-1,pc) + padded(pr,pc+1) + ...
                             padded(pr+1,pc) + padded(pr,pc-1))/4 * 2 + ...
                            padded(pr+1,pc+1)) / 4;
                    % =============================================================
                    % case 1: Gr 위치 (짝수행, 홀수열)
                    % -------------------------------------------------------------
                    % 3x3 윈도우의 Bayer 배치:
                    %
                    %   +------+------+------+
                    %   | Gb   | B    | Gb   |
                    %   +------+------+------+
                    %   | R    |[Gr]  | R    |  ← 중심 = Green (R행)
                    %   +------+------+------+
                    %   | Gb   | B    | Gb   |
                    %   +------+------+------+
                    %
                    % 각 채널 추출:
                    %   R = 좌우 평균 = (R_left + R_right) / 2            [ADU]
                    %     = (padded(pr, pc-1) + padded(pr, pc+1)) / 2
                    %   G = center = padded(pr, pc) (Gr 값 직접 사용)     [ADU]
                    %   B = 상하 평균 = (B_up + B_down) / 2               [ADU]
                    %     = (padded(pr-1, pc) + padded(pr+1, pc)) / 2
                    %
                    % Y = (R_avg + 2*G + B_avg) / 4                       [ADU]
                    % -------------------------------------------------------------
                    case 1
                        gray_img(r,c) = ((padded(pr,pc-1) + padded(pr,pc+1))/2 + ...
                            padded(pr,pc) * 2 + ...
                            (padded(pr-1,pc) + padded(pr+1,pc))/2) / 4;
                    % =============================================================
                    % case 2: Gb 위치 (홀수행, 짝수열)
                    % -------------------------------------------------------------
                    % 3x3 윈도우의 Bayer 배치:
                    %
                    %   +------+------+------+
                    %   | Gr   | R    | Gr   |
                    %   +------+------+------+
                    %   | B    |[Gb]  | B    |  ← 중심 = Green (B행)
                    %   +------+------+------+
                    %   | Gr   | R    | Gr   |
                    %   +------+------+------+
                    %
                    % 각 채널 추출:
                    %   R = 상하 평균 = (R_up + R_down) / 2               [ADU]
                    %     = (padded(pr-1, pc) + padded(pr+1, pc)) / 2
                    %   G = center = padded(pr, pc) (Gb 값 직접 사용)     [ADU]
                    %   B = 좌우 평균 = (B_left + B_right) / 2            [ADU]
                    %     = (padded(pr, pc-1) + padded(pr, pc+1)) / 2
                    %
                    % Y = (R_avg + 2*G + B_avg) / 4                       [ADU]
                    %
                    % [Gr vs Gb 위치 비교]
                    %   case 1 (Gr): R=좌우, B=상하
                    %   case 2 (Gb): R=상하, B=좌우  ← R과 B의 방향이 반전
                    %   이는 RGGB 패턴에서 R행/B행의 색 배치가 다르기 때문
                    % -------------------------------------------------------------
                    case 2
                        gray_img(r,c) = ((padded(pr-1,pc) + padded(pr+1,pc))/2 + ...
                            padded(pr,pc) * 2 + ...
                            (padded(pr,pc-1) + padded(pr,pc+1))/2) / 4;
                    % =============================================================
                    % case 3: B 위치 (홀수행, 홀수열)
                    % -------------------------------------------------------------
                    % 3x3 윈도우의 Bayer 배치:
                    %
                    %   +------+------+------+
                    %   | R    | Gr   | R    |
                    %   +------+------+------+
                    %   | Gb   |[B]   | Gb   |  ← 중심 = Blue
                    %   +------+------+------+
                    %   | R    | Gr   | R    |
                    %   +------+------+------+
                    %
                    % 각 채널 추출:
                    %   R = 대각 = padded(pr-1, pc-1)                     [ADU]
                    %     (주: 좌상단 1개만 사용 - 근사)
                    %   G = 십자 평균 = (상+우+하+좌) / 4                  [ADU]
                    %     = (Gr + Gb + Gr + Gb) / 4
                    %   B = center = padded(pr, pc)                       [ADU]
                    %
                    % Y = (R_avg + 2*G_avg + B) / 4                       [ADU]
                    %
                    % [case 0(R) vs case 3(B) 대칭성]
                    %   case 0: R=중심, G=십자, B=대각  → (R + 2G + B) / 4
                    %   case 3: R=대각, G=십자, B=중심  → (R + 2G + B) / 4
                    %   → R과 B의 역할이 교환되지만 공식 구조는 동일
                    % -------------------------------------------------------------
                    case 3
                        gray_img(r,c) = (padded(pr-1,pc-1) + ...
                            (padded(pr-1,pc) + padded(pr,pc+1) + ...
                             padded(pr+1,pc) + padded(pr,pc-1))/4 * 2 + ...
                            padded(pr,pc)) / 4;
                end
            end
        end

        % --- uint8 변환 ---
        % 반올림 후 8비트 정수로 변환 [ADU] (0~255 범위)
        gray_img = uint8(round(gray_img));
        method_info.output_size = [rows, cols];
        method_info.description = '2x2 가중 평균';

    % =====================================================================
    % === 방법 5: 최적 가중 평균 (Optimal) ===
    % =====================================================================
    % 우주 별 추적기 용도로 SNR(신호 대 잡음비)을 최대화하는 가중치를
    % 사용하여 Bayer → Gray 변환을 수행합니다.
    %
    % [기존 'weighted'와의 차이]
    %   weighted: Y = (R + 2G + B) / 4   (인간 시각 기반, 고정)
    %   optimal:  Y = w_R*R + w_G*G + w_B*B  (SNR 최대화, 가변)
    %
    % [이론적 배경 - Inverse Variance Weighting]
    %   각 채널의 신호 S_i와 노이즈 σ_i가 다를 때,
    %   합산 신호의 SNR을 최대화하는 가중치:
    %     w_i ∝ S_i / σ_i²
    %
    %   Shot noise 지배 (밝은 별): σ² ≈ S → w_i ∝ 1 (균등)
    %   Read noise 지배 (어두운 별): σ² ≈ σ_read² → w_i ∝ S_i (신호 비례)
    %
    %   상세 도출 과정: sub_main_2_optimal_weights.m 참조
    %
    % [우주 환경 고려사항]
    %   - 대기 투과율: τ_atm = 1.0 (지상 0.5~0.7 대비)
    %   - 암전류: ~0.01 e⁻/px/s @ -20°C (지상 0.1 대비)
    %   - 별 스펙트럼: 흑체복사 (2,000K ~ 25,000K)
    %   - OV4689 분광 응답: R/G/B 채널별 감도 차이 반영
    %
    % [가중치 파라미터]
    %   weights = [w_R, w_G, w_B], 합 = 1
    %   기본값은 nargin < 3 처리에서 설정됨
    %   sub_main_2_optimal_weights.m 결과로 업데이트 권장
    case 'optimal'
        % 최적 가중 평균
        gray_img = zeros(rows, cols);
        % 3x3 윈도우 처리를 위한 경계 패딩 (상하좌우 1픽셀)
        padded = padarray(bayer_img, [1 1], 'replicate');

        % 가중치 추출 (가독성 향상)
        w_R = weights(1);
        w_G = weights(2);
        w_B = weights(3);

        for r = 1:rows
            for c = 1:cols
                % padded 이미지에서의 중심 좌표 (+1 오프셋)
                pr = r + 1;
                pc = c + 1;
                % Bayer 패턴 위치 판별 (bayer_to_rgb_cfa.m과 동일)
                r_idx = mod(r-1, 2);
                c_idx = mod(c-1, 2);
                pos_idx = r_idx * 2 + c_idx;

                switch pos_idx
                    % case 0: R 위치 (짝수행, 짝수열)
                    %   R = 중심, G = 십자 평균, B = 대각
                    case 0
                        R_val = padded(pr, pc);
                        G_val = (padded(pr-1,pc) + padded(pr,pc+1) + ...
                                 padded(pr+1,pc) + padded(pr,pc-1)) / 4;
                        B_val = (padded(pr-1,pc-1) + padded(pr-1,pc+1) + ...
                                 padded(pr+1,pc-1) + padded(pr+1,pc+1)) / 4;
                    % case 1: Gr 위치 (짝수행, 홀수열)
                    %   R = 좌우 평균, G = 중심, B = 상하 평균
                    case 1
                        R_val = (padded(pr, pc-1) + padded(pr, pc+1)) / 2;
                        G_val = padded(pr, pc);
                        B_val = (padded(pr-1, pc) + padded(pr+1, pc)) / 2;
                    % case 2: Gb 위치 (홀수행, 짝수열)
                    %   R = 상하 평균, G = 중심, B = 좌우 평균
                    case 2
                        R_val = (padded(pr-1, pc) + padded(pr+1, pc)) / 2;
                        G_val = padded(pr, pc);
                        B_val = (padded(pr, pc-1) + padded(pr, pc+1)) / 2;
                    % case 3: B 위치 (홀수행, 홀수열)
                    %   R = 대각 평균, G = 십자 평균, B = 중심
                    case 3
                        R_val = (padded(pr-1,pc-1) + padded(pr-1,pc+1) + ...
                                 padded(pr+1,pc-1) + padded(pr+1,pc+1)) / 4;
                        G_val = (padded(pr-1,pc) + padded(pr,pc+1) + ...
                                 padded(pr+1,pc) + padded(pr,pc-1)) / 4;
                        B_val = padded(pr, pc);
                end

                % 최적 가중 합산: Y = w_R*R + w_G*G + w_B*B [ADU]
                gray_img(r, c) = w_R * R_val + w_G * G_val + w_B * B_val;
            end
        end

        % --- uint8 변환 ---
        gray_img = uint8(round(gray_img));
        method_info.output_size = [rows, cols];
        method_info.description = sprintf('최적 가중 평균 (R=%.3f, G=%.3f, B=%.3f)', ...
            w_R, w_G, w_B);
        method_info.weights = weights;

    otherwise
        error('Unknown method: %s', method);
end
end
