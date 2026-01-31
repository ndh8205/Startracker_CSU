# Star Tracker 별 이미지 시뮬레이션

Hipparcos 카탈로그 기반 물리적으로 정확한 Star Tracker 이미지 시뮬레이터입니다.

---

## 목차

1. [개요](#개요)
2. [빠른 시작](#빠른-시작)
3. [디렉토리 구조](#디렉토리-구조)
4. [코드 구조 상세 분석](#코드-구조-상세-분석)
   - [전체 아키텍처](#전체-아키텍처)
   - [데이터 흐름도](#데이터-흐름도)
   - [함수별 상세 분석](#함수별-상세-분석)
5. [이론적 배경](#이론적-배경)
   - [천구 좌표계와 관측 방향](#1-천구-좌표계와-관측-방향)
   - [회전 행렬과 좌표 변환](#2-회전-행렬과-좌표-변환)
   - [핀홀 카메라 모델과 투영](#3-핀홀-카메라-모델과-투영)
   - [FOV 계산](#4-시야각fov-계산)
   - [Pogson 등급 시스템과 플럭스](#5-pogson-등급-시스템과-플럭스-계산)
   - [PSF 모델링](#6-psf-모델링-point-spread-function)
   - [센서 신호 모델](#7-센서-신호-모델)
   - [센서 노이즈 모델](#8-센서-노이즈-모델)
   - [Bayer 패턴과 CFA](#9-bayer-패턴과-color-filter-array)
   - [CFA 디모자이킹](#10-cfa-디모자이킹-demosaicing)
   - [Grayscale 변환 방법론](#11-grayscale-변환-방법론)
   - [SNR 이론](#12-snr-이론-signal-to-noise-ratio)
   - [별 검출 알고리즘](#13-별-검출-알고리즘)
   - [Centroid 추정 이론](#14-centroid-추정-이론)
6. [시뮬레이션 기능](#시뮬레이션-기능)
7. [센서 파라미터](#센서-파라미터)
8. [의존성](#의존성)
9. [참고 문헌](#참고-문헌)

---

## 개요

### 목적
실제 별 카탈로그와 물리 모델을 사용하여 Star Tracker 센서 이미지를 시뮬레이션합니다:
- **실제 별 위치/등급**: Hipparcos 카탈로그 기반
- **물리적 정확성**: Pogson 공식, PSF 모델, 센서 신호 체인
- **센서 노이즈**: 샷 노이즈, 읽기 노이즈, 다크 전류

### 활용
- Star Tracker 알고리즘 검증 (별 검출, Centroid, Star ID)
- 센서/광학 파라미터 최적화
- FPGA 영상처리 파이프라인 검증
- Bayer→Gray 직접 변환 vs FPGA 파이프라인 성능 비교

---

## 빠른 시작

### 1. Git 설치 (처음 사용자)

Git이 설치되어 있지 않다면 먼저 설치하세요:
- **Windows**: https://git-scm.com/download/win 에서 다운로드 후 설치 (모두 기본값으로 Next)
- **Mac**: 터미널에서 `git --version` 입력하면 자동 설치 안내

설치 확인:
```bash
git --version
# git version 2.xx.x 같은 출력이 나오면 성공
```

### 2. 저장소 다운로드 (Clone)

원하는 폴더에서 **명령 프롬프트**(Windows) 또는 **터미널**(Mac)을 열고:

```bash
# 원하는 폴더로 이동 (예: D드라이브)
cd D:\

# 저장소 복제 (URL은 실제 저장소 주소로 변경)
git clone https://github.com/YOUR_USERNAME/bayer_comparison.git

# 폴더로 이동
cd bayer_comparison
```

> **팁**: Windows에서 폴더 경로창에 `cmd` 입력하면 해당 위치에서 명령 프롬프트 열림

### 3. MATLAB에서 실행

```matlab
% MATLAB에서 이 폴더로 이동 후 실행
cd('D:\bayer_comparison')   % 실제 다운받은 경로로 변경

% 방법 1: 스크립트 실행 (이미지 생성 + Figure 출력)
main_simulation

% 방법 2: GUI 실행 (인터랙티브 파라미터 튜닝)
gui_star_simulator
```

모든 데이터 파일(별 카탈로그 등)은 `data/` 폴더에 포함되어 있어 별도 설정이 필요 없습니다.

### 4. 나중에 업데이트 받기

프로젝트가 업데이트되면 최신 버전을 받을 수 있습니다:

```bash
# bayer_comparison 폴더에서
cd D:\bayer_comparison

# 최신 버전 다운로드
git pull
```

### 5. 결과

- Figure 1: 별 이미지 (이상적 Gray, 별 위치 표시, Bayer 패턴)
- Figure 2: 등급 분포 히스토그램
- `output/` 폴더에 이미지 및 데이터 저장

---

## 디렉토리 구조

```
bayer_comparison/
├── README.md                   # 이 문서
├── main_simulation.m           # ★ 메인 실행 파일 (별 이미지 생성)
├── gui_star_simulator.m        # ★ 인터랙티브 GUI (파라미터 튜닝)
├── sub_main_1_bayer_comparison.m  # 서브: Bayer→Gray 변환 비교
├── sub_main_2_optimal_weights.m   # 서브: 최적 가중치 도출 연구
│
├── 250131_이미지파이프라인_비효율분석.md  # 분석: FPGA 파이프라인 비효율
├── 250131_최적가중치연구.md               # 분석: Bayer→Gray 최적 가중치
│
├── core/                       # 핵심 함수
│   ├── simulate_star_image_realistic.m   # ★ 별 이미지 생성 (Hipparcos)
│   ├── bayer_to_gray_direct.m            # 직접 변환 (5가지 방법)
│   ├── bayer_to_rgb_cfa.m                # CFA 디모자이킹
│   └── rgb_to_gray_fpga.m                # RGB→Gray (FPGA 방식)
│
├── utils/                      # 유틸리티 함수
│   ├── calculate_snr.m                   # SNR 계산
│   ├── calculate_peak_snr.m              # Peak SNR 계산
│   ├── detect_stars_simple.m             # 별 검출 (threshold)
│   └── evaluate_centroid_accuracy.m      # Centroid 정확도
│
├── data/                       # 데이터 파일
│   ├── star_catalog_kvector.mat          # ★ Hipparcos 별 카탈로그 (MAT)
│   └── Hipparcos_Below_6.0.csv           # Hipparcos 카탈로그 (CSV 백업)
│
├── legacy/                     # 구버전 (참고용)
│   ├── run_bayer_comparison.m            # v1 (랜덤 별)
│   ├── run_bayer_comparison_v2.m         # v2 (카탈로그)
│   └── simulate_bayer_star_image.m       # 랜덤 별 생성
│
├── refs/                       # 참고 문헌 PDF
│   └── *.pdf
│
└── output/                     # 출력 이미지/데이터
    └── *.png, *.mat
```

---

## 코드 구조 상세 분석

### 전체 아키텍처

시뮬레이터는 **모듈형 아키텍처**로 설계되어 있으며, 크게 3계층으로 나뉩니다:

```
┌──────────────────────────────────────────────────────────┐
│                    실행 계층 (Entry Points)                │
│  main_simulation.m          sub_main_1_bayer_comparison.m │
│  (별 이미지 생성)            (변환 방법 비교)              │
│  gui_star_simulator.m       sub_main_2_optimal_weights.m  │
│  (인터랙티브 GUI)           (최적 가중치 연구)             │
└──────────────┬──────────────────────┬────────────────────┘
               │                      │
┌──────────────▼──────────────────────▼────────────────────┐
│                    핵심 계층 (Core)                        │
│  simulate_star_image_realistic.m  ← 물리 엔진             │
│  bayer_to_rgb_cfa.m              ← CFA 디모자이킹          │
│  bayer_to_gray_direct.m          ← 직접 변환 (5가지)       │
│  rgb_to_gray_fpga.m              ← FPGA 그레이스케일       │
└──────────────┬──────────────────────┬────────────────────┘
               │                      │
┌──────────────▼──────────────────────▼────────────────────┐
│                    분석 계층 (Utils)                       │
│  calculate_snr.m                 ← 참조 기반 SNR           │
│  calculate_peak_snr.m            ← 피크 SNR               │
│  detect_stars_simple.m           ← 별 검출                │
│  evaluate_centroid_accuracy.m    ← Centroid 정확도         │
└──────────────────────────────────────────────────────────┘
```

### 데이터 흐름도

#### main_simulation.m 실행 흐름

```
[입력: RA, DEC, Roll, sensor_params]
        │
        ▼
┌─────────────────────────────────────┐
│ simulate_star_image_realistic()     │
│                                     │
│  1. Hipparcos 카탈로그 로드          │
│     └→ data/star_catalog_kvector.mat│
│                                     │
│  2. 관측 방향 → 회전 행렬 생성       │
│     └→ M = M₁(RA) × M₂(DEC) × M₃(Roll) │
│                                     │
│  3. 천구 좌표 → 센서 좌표 변환       │
│     └→ v_sensor = M^T × v_celestial │
│                                     │
│  4. FOV 내 별 필터링                 │
│     └→ RA/DEC 범위 + 등급 제한       │
│                                     │
│  5. 핀홀 투영 → 픽셀 좌표            │
│     └→ x_px = f × (X/Z) / μ         │
│                                     │
│  6. PSF 렌더링 (2D Gaussian)         │
│     └→ σ=1.2px, 6σ 윈도우           │
│                                     │
│  7. 플럭스 계산 (Pogson 공식)        │
│     └→ F = 96 × 10^(-0.4×(m-6))     │
│     └→ ADU = F × t_exp × QE × Gain  │
│                                     │
│  8. Bayer 패턴(RGGB) 생성            │
│     └→ 채널별 감도 적용              │
│                                     │
│  9. 노이즈 추가                      │
│     ├→ 다크 전류 (상수)              │
│     ├→ 샷 노이즈 (Poisson)          │
│     └→ 읽기 노이즈 (Gaussian)        │
│                                     │
│  10. ADC 클램핑 (8-bit: 0~255)       │
└────────────┬────────────────────────┘
             │
             ▼
    [출력: gray_ideal, bayer_img, star_info]
             │
             ├→ Figure 1: 센서 출력 (1:1 Bayer)
             ├→ Figure 2: 이상적 Grayscale (1:1)
             ├→ Figure 3: 분석 (Ideal + 별 위치 + Bayer)
             ├→ Figure 4: 등급 분포 히스토그램
             └→ output/ 폴더에 PNG + MAT 저장
```

#### sub_main_1_bayer_comparison.m 실행 흐름

```
[simulate_star_image_realistic()로부터 bayer_img, star_info 수신]
        │
        ├──────────────────────────────────────────┐
        │                                          │
        ▼                                          ▼
┌───────────────────┐              ┌───────────────────────────┐
│ 방법 A: FPGA 경로  │              │ 방법 B: 직접 변환 (4가지)  │
│                   │              │                           │
│ bayer_to_rgb_cfa()│              │ bayer_to_gray_direct()    │
│   ↓ 3×3 bilinear │              │   ├→ B1: 'raw' (그대로)    │
│ rgb_to_gray_fpga()│              │   ├→ B2: 'binning' (2×2)  │
│   ↓ (R+2G+B)/4   │              │   ├→ B3: 'green' (G채널)  │
│                   │              │   └→ B4: 'weighted' (가중) │
│ → gray_fpga       │              │                           │
└────────┬──────────┘              └────────────┬──────────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   5가지 방법 비교     │
              │                     │
              │ calculate_peak_snr()│  → SNR (dB)
              │ detect_stars_simple()│ → 검출 수
              │ evaluate_centroid_  │
              │   accuracy()        │  → RMS 오차 (px)
              │ tic/toc             │  → 처리 시간 (ms)
              └─────────┬───────────┘
                        │
                        ▼
              [Figure: 비교 차트 3개]
              [output/에 PNG 저장]
```

### 함수별 상세 분석

#### 1. `simulate_star_image_realistic.m` (핵심 물리 엔진)

| 항목 | 내용 |
|------|------|
| **위치** | `core/simulate_star_image_realistic.m` (~400줄) |
| **역할** | Hipparcos 카탈로그 기반 별 이미지 생성 |
| **입력** | `ra_deg`, `de_deg`, `roll_deg` (관측 방향), `sensor_params` (센서 설정) |
| **출력** | `gray_img` (Grayscale uint8), `bayer_img` (Bayer uint8), `star_info` (메타데이터 struct) |
| **비고** | `sensor_params.preloaded_catalog`로 사전 로드된 카탈로그 전달 가능 (GUI 성능 최적화) |

**내부 함수**:

| 함수 | 줄 | 역할 |
|------|-----|------|
| `set_default(params, field, value)` | 327-331 | 구조체 기본값 설정 헬퍼 |
| `create_rotation_matrix(ra, de, roll)` | 333-350 | 천구→센서 회전 행렬 생성 (3개 축 순차 회전) |
| `draw_star_psf(img, cx, cy, sigma, total_flux)` | 352-388 | 2D Gaussian PSF를 이미지에 렌더링 |

**`star_info` 출력 구조체 필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `FOVx`, `FOVy` | double | 시야각 (도) |
| `ra_deg`, `de_deg`, `roll_deg` | double | 관측 방향 |
| `num_stars` | int | FOV 내 별 수 |
| `pixel_coords` | N×2 double | 센서 좌표계 위치 (중심 원점) |
| `magnitudes` | N×1 double | 각 별의 등급 |
| `true_centroids` | N×2 double | 이미지 좌표계 위치 (좌상 원점) |
| `ideal_gray` | H×W double | 노이즈 없는 이상적 이미지 |
| `sensor_params` | struct | 사용된 센서 파라미터 |

**처리 순서 상세**:

```
1. 기본 파라미터 설정 (set_default로 누락 필드 채움)
2. 라디안 변환: ra, de, roll
3. FOV 계산: FOVx = 2×arctan(μ×l/2f), FOVy = 2×arctan(μ×w/2f)
4. 회전 행렬 M = M₁(RA-π/2) × M₂(DEC+π/2) × M₃(Roll) 생성
5. 별 카탈로그 로드 (MAT 우선, 없으면 CSV)
6. FOV 범위 내 별 필터링 (RA, DEC, 등급 조건)
7. 각 별에 대해:
   a. 천구 방향 벡터 → 센서 좌표 변환 (M^T × v)
   b. 핀홀 투영: x = f×(X/Z), y = f×(Y/Z)
   c. 미터 → 픽셀 변환: x_px = x/μ
   d. FOV 경계 검사
   e. 이미지 좌표 변환: (l/2 + x_px, w/2 - y_px)
8. 이상적 grayscale 이미지 생성:
   a. Pogson 공식으로 플럭스 계산
   b. 센서 모델 적용 (노출, QE, 게인)
   c. 2D Gaussian PSF 렌더링
9. 노이즈 추가 (ADU→전자 역변환 → 다크 전류 합산 → Poisson 샷 노이즈 → 전자→ADU → 읽기 노이즈)
10. Bayer 패턴 생성 (RGGB + 채널 감도, ideal_gray 기반)
11. Bayer에도 동일 노이즈 모델 적용 (전자 도메인 Poisson)
12. 8-bit ADC 클램핑
```

#### 2. `bayer_to_rgb_cfa.m` (CFA 디모자이킹)

| 항목 | 내용 |
|------|------|
| **위치** | `core/bayer_to_rgb_cfa.m` (54줄) |
| **역할** | Bayer RGGB → RGB 3채널 변환 (FPGA `cfa.cpp` 재현) |
| **입력** | `bayer_img` (H×W uint8, 단일 채널 Bayer) |
| **출력** | `rgb_img` (H×W×3 uint8, RGB) |
| **알고리즘** | 3×3 윈도우 기반 bilinear 보간 |

**RGGB 패턴에서의 보간 규칙**:

```
Bayer 2×2 블록:
  ┌────┬────┐
  │ R  │ Gr │   행 0 (짝수 행)
  ├────┼────┤
  │ Gb │ B  │   행 1 (홀수 행)
  └────┴────┘
   열0   열1

pos_idx = row_idx×2 + col_idx  (0, 1, 2, 3)
```

| pos_idx | 위치 | R 보간 | G 보간 | B 보간 |
|---------|------|--------|--------|--------|
| 0 | R 위치 | 자기 자신 | 상하좌우 평균 (4개) | 대각선 평균 (4개) |
| 1 | Gr 위치 | 좌우 평균 (2개) | 자기 자신 | 상하 평균 (2개) |
| 2 | Gb 위치 | 상하 평균 (2개) | 자기 자신 | 좌우 평균 (2개) |
| 3 | B 위치 | 대각선 평균 (4개) | 상하좌우 평균 (4개) | 자기 자신 |

```
3×3 윈도우 (w):
  ┌──────┬──────┬──────┐
  │w(1,1)│w(1,2)│w(1,3)│
  ├──────┼──────┼──────┤
  │w(2,1)│w(2,2)│w(2,3)│  ← 중심 = 현재 픽셀
  ├──────┼──────┼──────┤
  │w(3,1)│w(3,2)│w(3,3)│
  └──────┴──────┴──────┘

예: R 위치 (pos_idx=0)에서:
  R = w(2,2)                                        ← 자기 자신
  G = (w(1,2) + w(2,1) + w(2,3) + w(3,2)) / 4      ← 상하좌우
  B = (w(1,1) + w(1,3) + w(3,1) + w(3,3)) / 4      ← 대각선
```

#### 3. `rgb_to_gray_fpga.m` (FPGA 그레이스케일 변환)

| 항목 | 내용 |
|------|------|
| **위치** | `core/rgb_to_gray_fpga.m` (18줄) |
| **역할** | RGB → Grayscale (FPGA `rgb2gray.cpp` 재현) |
| **입력** | `rgb_img` (H×W×3 uint8) |
| **출력** | `gray_img` (H×W uint8) |
| **수식** | `Y = (R + 2×G + B) / 4` |

이 수식은 ITU-R BT.601 표준의 **근사치**입니다:
- 표준: `Y = 0.299R + 0.587G + 0.114B`
- FPGA 근사: `Y = 0.25R + 0.5G + 0.25B = (R + 2G + B) / 4`

FPGA에서는 나눗셈을 2비트 시프트(`>> 2`)로 구현하여 곱셈기 없이 처리 가능합니다.

#### 4. `bayer_to_gray_direct.m` (직접 변환 5가지)

| 항목 | 내용 |
|------|------|
| **위치** | `core/bayer_to_gray_direct.m` (~588줄) |
| **역할** | CFA 디모자이킹 없이 Bayer에서 직접 Grayscale 생성 |
| **입력** | `bayer_img` (H×W), `method` ('raw'/'binning'/'green'/'weighted'/'optimal'), `weights` (optional, [w_R, w_G, w_B]) |
| **출력** | `gray_img`, `method_info` (메타데이터) |

**방법별 알고리즘**:

| 방법 | 출력 해상도 | 수식 | FPGA 리소스 |
|------|-----------|------|-------------|
| **RAW** | H×W (원본) | `Y(r,c) = Bayer(r,c)` | 없음 (패스스루) |
| **Binning** | H/2 × W/2 | `Y = (R + Gr + Gb + B) / 4` | 덧셈 3회 + 시프트 |
| **Green** | H×W | Green 픽셀: 자기 자신, 나머지: 4방향 보간 | 덧셈 3회 + 시프트 |
| **Weighted** | H×W | 위치별 `(R + 2G + B) / 4` 변형 | bilinear 수준 |
| **Optimal** | H×W | `Y = w_R×R + w_G×G + w_B×B` (SNR 최대화) | 시프트+덧셈 |

**RAW 방법** (가장 단순):
```
입력 그대로 출력. R/G/B 픽셀의 감도 차이가 약간의 격자 패턴을 만들지만,
별센서에서는 Star Tracker의 관심 대상인 '밝은 점'의 검출/위치에는 영향 미미.
```

**Binning 방법**:
```
2×2 블록을 하나의 픽셀로 합산:
  ┌────┬────┐
  │ R  │ Gr │  →  Y = (R + Gr + Gb + B) / 4
  ├────┼────┤
  │ Gb │ B  │
  └────┴────┘

해상도가 1/4로 감소하지만, 4개 픽셀 합산으로 SNR이 √4 = 2배(6dB) 향상.
```

**Green 방법**:
```
Green 픽셀(전체의 50%)은 그대로 사용.
R/B 위치는 주변 Green 픽셀 4개의 평균으로 보간:

  Gr 위치: Y = Bayer(r, c)
  R/B 위치: Y = (상 + 하 + 좌 + 우) / 4

인간 시각이 Green에 가장 민감하므로 합리적인 근사.
```

**Weighted 방법**:
```
각 위치에서 주변 R, G, B 값을 추정하고 (R + 2G + B) / 4 적용.
FPGA 방식과 동일한 가중치이나, CFA 디모자이킹 단계를 건너뜀.
```

**Optimal 방법** (2026-01-31 추가):
```
각 위치에서 주변 R, G, B 값을 추정하고,
SNR 최대화 가중치를 적용:
  Y = 0.4544×R + 0.3345×G + 0.2111×B

도출 근거: Inverse Variance Weighting (w_i ∝ S_i / σ_i²)
  - OV4689 분광 응답 + 흑체복사 스펙트럼 기반
  - 우주 환경, 전 스펙트럼(3000K~25000K) 등급 가중 평균
  - 6등급 별 기준 기존 FPGA 대비 SNR +10.6% 개선

FPGA 정수 근사: Y = (8R + 5G + 3B) >> 4
  → 곱셈기 불필요 (시프트+덧셈만)

상세: 250131_최적가중치연구.md
코드: sub_main_2_optimal_weights.m
```

#### 5. `calculate_snr.m` (참조 기반 SNR)

| 항목 | 내용 |
|------|------|
| **위치** | `utils/calculate_snr.m` (26줄) |
| **수식** | `SNR = 10 × log₁₀(P_signal / P_noise)` |
| **P_signal** | `mean(reference²)` |
| **P_noise** | `mean((measured - reference)²)` |

참조 이미지(이상적)와 측정 이미지(노이즈 포함) 간의 차이를 dB로 표현합니다.

#### 6. `calculate_peak_snr.m` (피크 SNR)

| 항목 | 내용 |
|------|------|
| **위치** | `utils/calculate_peak_snr.m` (29줄) |
| **수식** | `Peak SNR = 20 × log₁₀(S_peak / σ_noise)` |
| **S_peak** | `max(img) - median(img)` (피크 신호 - 배경) |
| **σ_noise** | `std(배경 영역)` (배경 < median + 10인 영역) |

별 영상에 특화된 SNR. 가장 밝은 별의 피크값과 배경 노이즈의 비율을 측정합니다.

#### 7. `detect_stars_simple.m` (별 검출)

| 항목 | 내용 |
|------|------|
| **위치** | `utils/detect_stars_simple.m` (31줄) |
| **알고리즘** | Threshold + Connected Component Labeling (CCL) |
| **파라미터** | `threshold` (기본 20 ADU), `min_area` (기본 3 px) |

```
처리 순서:
1. 배경 추정: bg = median(전체 이미지)
2. 이진화: binary = (img > bg + threshold)
3. CCL: bwconncomp(binary) → 연결 영역 추출
4. 속성 계산: regionprops(..., 'Centroid', 'Area', 'MaxIntensity', 'MeanIntensity')
5. 면적 필터: Area >= min_area인 영역만 선택
6. 결과: 중심 좌표, 밝기, 검출 수
```

#### 8. `evaluate_centroid_accuracy.m` (Centroid 정확도)

| 항목 | 내용 |
|------|------|
| **위치** | `utils/evaluate_centroid_accuracy.m` (33줄) |
| **알고리즘** | 최근접 매칭 + RMS 오차 |
| **파라미터** | `match_radius` (기본 5.0 px) |

```
각 참 위치(true centroid)에 대해:
  1. 모든 검출 별까지 유클리드 거리 계산
  2. 최소 거리 < match_radius이면 매칭 성공
  3. 매칭 오차 수집
  4. RMS = √(mean(errors²))
```

---

## 이론적 배경

### 1. 천구 좌표계와 관측 방향

#### 적도 좌표계 (Equatorial Coordinate System)

별의 위치를 표현하는 데 사용되는 천문학적 좌표계입니다. 지구의 자전축을 기준으로 합니다.

```
                    천구 북극 (NCP)
                        │
                        │ 적위(DEC) = +90°
                        │
                        │    ★ 별
                        │   /
                        │  / DEC (적위)
                        │ /
   ──────────────────────○──────────── 천구 적도 (DEC = 0°)
                        │\
                        │ \ RA (적경)
                        │  \
                        │   → 춘분점 (RA = 0°)
                        │
                    천구 남극 (SCP)
```

- **RA (Right Ascension, 적경)**: 춘분점으로부터 동쪽 방향으로 측정한 각도 [0°, 360°]
  - 지구의 경도(longitude)에 해당
  - 춘분점: 태양이 천구 적도를 남→북으로 지나는 점 (3월 21일경)
- **DEC (Declination, 적위)**: 천구 적도로부터의 각거리 [-90°, +90°]
  - 지구의 위도(latitude)에 해당
  - +90° = 천구 북극 (North Celestial Pole)
- **Roll**: 광축 주위의 회전각 [0°, 360°]
  - 카메라가 같은 방향을 보더라도 회전에 따라 이미지가 다름

#### 방향 벡터 (Direction Vector)

RA, DEC를 3D 단위 벡터로 변환합니다 (구면 → 직교):

```
       ⎡ cos(RA) × cos(DEC) ⎤
v̂  =  ⎢ sin(RA) × cos(DEC) ⎥
       ⎣ sin(DEC)            ⎦
```

이 벡터는 지구 중심에서 별 방향을 가리키는 단위 벡터입니다.

**코드 대응** (`simulate_star_image_realistic.m:136-138`):
```matlab
dir_vector = [cos(ra_i(i))*cos(de_i(i));
              sin(ra_i(i))*cos(de_i(i));
              sin(de_i(i))];
```

### 2. 회전 행렬과 좌표 변환

#### 목적

천구 좌표계(ECI: Earth-Centered Inertial)에서 표현된 별의 방향 벡터를 센서 좌표계(Body Frame)로 변환해야 합니다. 이 변환은 위성의 자세(attitude)를 나타내는 회전 행렬 `M`으로 수행됩니다.

#### 3축 순차 회전 (Euler Angles)

회전 행렬은 3개의 기본 회전(RA, DEC, Roll)의 곱으로 구성됩니다:

```
M = M₁(α) × M₂(δ) × M₃(φ)
```

여기서:
- `α = RA - π/2` (적경 보정)
- `δ = DEC + π/2` (적위 보정)
- `φ = Roll` (회전각)

**M₁: Z축 회전 (RA 방향)**
```
        ⎡ cos(α)  -sin(α)   0 ⎤
M₁  =  ⎢ sin(α)   cos(α)   0 ⎥
        ⎣ 0        0        1 ⎦
```

**M₂: X축 회전 (DEC 방향)**
```
        ⎡ 1    0       0    ⎤
M₂  =  ⎢ 0   cos(δ) -sin(δ)⎥
        ⎣ 0   sin(δ)  cos(δ)⎦
```

**M₃: Z축 회전 (Roll)**
```
        ⎡ cos(φ)  -sin(φ)   0 ⎤
M₃  =  ⎢ sin(φ)   cos(φ)   0 ⎥
        ⎣ 0        0        1 ⎦
```

#### 좌표 변환

센서 좌표 = 전치 행렬 × 천구 방향 벡터:
```
v_sensor = M^T × v_celestial
```

**왜 전치(transpose)를 사용하는가?**

회전 행렬 M은 직교 행렬(orthogonal matrix)이므로 M^(-1) = M^T입니다. M이 센서→천구 변환이라면, M^T는 그 역변환인 천구→센서 변환이 됩니다.

**코드 대응** (`simulate_star_image_realistic.m:133-140`):
```matlab
M_transpose = M';
for i = 1:length(ra_i)
    dir_vector = [cos(ra_i(i))*cos(de_i(i));
                  sin(ra_i(i))*cos(de_i(i));
                  sin(de_i(i))];
    star_sensor_coords(i, :) = (M_transpose * dir_vector)';
end
```

### 3. 핀홀 카메라 모델과 투영

#### 핀홀 모델 (Pinhole Camera Model)

이상적인 렌즈 시스템을 단순화한 모델입니다. 모든 광선이 하나의 점(핀홀)을 통과한다고 가정합니다.

```
                별 (무한 원점)
                  \
                   \  입사 광선
                    \
  ──────────────────●──────────── 렌즈 평면 (핀홀)
                   /│
                  / │ f (초점거리)
                 /  │
                /   │
    ──────────★────────────────── 이미지 평면 (센서)
             (x_img, y_img)
```

#### 투영 수식

센서 좌표계에서 (X, Y, Z) 방향의 별이 이미지 평면에 맺히는 위치:

```
x_img = f × (X / Z)     [미터]
y_img = f × (Y / Z)     [미터]
```

- `f`: 초점거리 (10.42 mm = 0.01042 m)
- `Z`: 광축 방향 성분 (양수 = 카메라 앞쪽)
- `X, Y`: 광축에 수직한 성분

#### 미터 → 픽셀 변환

```
x_pixel = x_img / μ     [pixel]
y_pixel = y_img / μ     [pixel]
```

- `μ`: 픽셀 물리적 크기 (2 µm = 2×10⁻⁶ m)

#### 이미지 좌표계

센서 좌표 원점(이미지 중심)에서 이미지 좌표(좌상 원점)로 변환:

```
x_image = W/2 + x_pixel     (가로: 오른쪽이 +)
y_image = H/2 - y_pixel     (세로: 아래쪽이 +, 반전)
```

Y축 반전의 이유: 이미지 좌표계는 위→아래가 양의 방향이지만, 센서 물리 좌표는 아래→위가 양의 방향입니다.

**코드 대응** (`simulate_star_image_realistic.m:144-176`):
```matlab
% 핀홀 투영
x = f * (star_sensor_coords(i,1) / star_sensor_coords(i,3));
y = f * (star_sensor_coords(i,2) / star_sensor_coords(i,3));

% 미터→픽셀
pixel_per_length = 1 / myu;
x1pixel = pixel_per_length * x1;

% 이미지 좌표
true_x = l/2 + x1pixel;
true_y = w/2 - y1pixel;
```

### 4. 시야각(FOV) 계산

#### 정의

FOV(Field of View)는 카메라가 볼 수 있는 각도 범위입니다.

```
                    ┌────────────── 센서 가장자리
                   /│
                  / │
                 /  │ f
                /   │
               / θ/2│
  핀홀 ──────●──────┘
              \   │
               \  │ f
                \ │
                 \│
                  └────────────── 센서 반대쪽 가장자리
```

#### 수식

```
FOVx = 2 × arctan(μ × l / (2 × f))
FOVy = 2 × arctan(μ × w / (2 × f))
```

- `μ × l / 2` = 센서 가로 절반 물리 크기 [m]
- `f` = 초점거리 [m]

#### 본 시뮬레이션의 FOV

```
μ = 2×10⁻⁶ m, f = 0.01042 m
l = 1280 px, w = 720 px

센서 가로 절반 = 2e-6 × 1280/2 = 1.28e-3 m = 1.28 mm
센서 세로 절반 = 2e-6 × 720/2  = 7.20e-4 m = 0.72 mm

FOVx = 2 × arctan(1.28e-3 / 0.01042) = 2 × 7.01° = 14.02°
FOVy = 2 × arctan(7.20e-4 / 0.01042) = 2 × 3.95° =  7.91°
대각선 FOV = 2 × arctan(√(1.28² + 0.72²)×10⁻³ / 0.01042) ≈ 16.10°
```

**코드 대응** (`main_simulation.m:90-91`):
```matlab
FOVx = rad2deg(2 * atan((sensor_params.myu*sensor_params.l/2) / sensor_params.f));
FOVy = rad2deg(2 * atan((sensor_params.myu*sensor_params.w/2) / sensor_params.f));
```

### 5. Pogson 등급 시스템과 플럭스 계산

#### Pogson 등급 시스템

1856년 Norman Pogson이 정립한 천문학 밝기 체계입니다.

**핵심 정의**:
- 등급 5 차이 = 밝기 정확히 100배 차이
- 등급 1 차이 = 밝기 10^(0.4) ≈ 2.512배 차이
- 등급이 **작을수록** 밝음 (0등급 > 6등급)

**수학적 관계**:

```
m₁ - m₂ = -2.5 × log₁₀(F₁ / F₂)
```

등가 형태:
```
F₁ / F₂ = 10^(-0.4 × (m₁ - m₂))
```

| 등급 | 대표 천체 | 상대 밝기 (6등급=1) |
|------|-----------|-------------------|
| -1.5 | 시리우스 (가장 밝은 별) | ~1585× |
| 0.0 | 베가 (기준별) | ~251× |
| 1.0 | 스피카 | ~100× |
| 2.0 | 북극성 | ~39.8× |
| 3.0 | — | ~15.8× |
| 4.0 | — | ~6.3× |
| 5.0 | 육안 한계 (도시) | ~2.5× |
| 6.0 | 육안 한계 (암흑) | 1× (기준) |
| 6.5 | 시뮬레이션 한계 | ~0.63× |

#### 플럭스 계산 (시뮬레이션 구현)

```
photon_flux = ref_photon_flux × 10^(-0.4 × (mag - ref_mag))    [photons/s]
```

여기서:
- `ref_mag = 6.0` (기준 등급: 육안 한계)
- `ref_photon_flux = 96` [photons/s] (캘리브레이션 값)

#### 캘리브레이션 근거

`ref_photon_flux = 96`의 산출 과정:

```
1. V-band 제로점 (0등급):
   φ₀ = 8.96×10⁵ photons/s/cm² (Johnson V-band)

2. 6등급 별의 플럭스:
   φ₆ = φ₀ × 10^(-0.4×6) = 8.96×10⁵ × 3.98×10⁻³
      = 3567 photons/s/cm²

3. 렌즈 집광 면적:
   f/1.6, 유효 구경 6.5mm → A = π(0.325)² ≈ 0.33 cm²

4. 보정 계수:
   - 대기 투과율 (지상): × 0.5
   - 광학계 손실: × 0.8
   - CMOS fill factor: × 0.5
   (QE는 별도 적용하므로 제외)

5. 이론값:
   φ_eff = 3567 × 0.33 × 0.5 × 0.8 × 0.5 ≈ 235 photons/s

6. 실측 검증 (ori_900000.png, 400ms, gain=1x):
   - 시뮬레이션 ref=1000 → 피크 1110.7 ADU
   - 실측 피크 = 107 ADU
   - 보정비: 107/1110.7 = 0.0963
   - → ref = 1000 × 0.0963 ≈ 96

결론: ref_photon_flux = 96 (지상 관측 조건 캘리브레이션)
```

**코드 대응** (`simulate_star_image_realistic.m:219-237`):
```matlab
ref_mag = 6.0;
ref_photon_flux = 96;  % 캘리브레이션된 값
photon_flux = ref_photon_flux * 10^(-0.4 * (mag - ref_mag));
```

### 6. PSF 모델링 (Point Spread Function)

#### 정의

PSF는 점광원(별)이 광학 시스템을 통과한 후 이미지 평면에 맺히는 밝기 분포입니다. 이상적인 점이 아닌 퍼진 원반 형태로 맺히며, 이를 **Airy disk**이라 합니다.

#### 2D Gaussian 근사

실제 PSF(Airy pattern)를 Gaussian 함수로 근사합니다:

```
I(x, y) = A × exp(-(Δx² + Δy²) / (2σ²))
```

여기서:
- `A` = 피크 진폭
- `Δx, Δy` = 별 중심으로부터의 거리 [pixel]
- `σ` = 표준편차 [pixel] (PSF의 퍼짐 정도)

#### 정규화 (에너지 보존)

PSF의 전체 적분값이 별의 총 플럭스와 같아야 합니다:

```
∫∫ I(x,y) dx dy = total_flux

2D Gaussian 적분값 = 2π × σ² × A

따라서:
A = total_flux / (2π × σ²)
```

#### 시뮬레이션 파라미터

```
σ = 1.2 pixel (광학 시스템에 의해 결정, 상수)
윈도우 크기 = ±6σ = ±7.2 pixel (±ceil(7.2) = ±8 pixel)
6σ 범위에서 전체 에너지의 99.99%를 포함
```

σ = 1.2 pixel의 물리적 의미:
- 별빛이 렌즈를 통과하면서 회절(diffraction), 수차(aberration), 대기 흔들림(seeing)에 의해 퍼짐
- OV4689의 2µm 픽셀 + 10.42mm 렌즈 조합에서 σ ≈ 1.2px는 합리적인 값
- 실제 값은 광학 설계와 제조 공차에 따라 달라질 수 있음

```
PSF 프로파일 (σ=1.2):
     1.0 ┤████
         │ ████
     0.8 ┤  ████
         │    ████
     0.6 ┤      ████
         │        ████
     0.4 ┤          ██████
         │              ██████
     0.2 ┤                  ████████
         │                        ████████████
     0.0 ┤─────────────────────────────────────────
         -6σ  -4σ  -2σ   0   +2σ  +4σ  +6σ
```

**코드 대응** (`simulate_star_image_realistic.m:352-388`):
```matlab
function img = draw_star_psf(img, cx, cy, sigma, total_flux)
    peak_amplitude = total_flux / (2 * pi * sigma^2);
    win = ceil(6 * sigma);
    for y = y_start:y_end
        for x = x_start:x_end
            dx = x - cx;
            dy = y - cy;
            dist_sq = dx^2 + dy^2;
            val = peak_amplitude * exp(-dist_sq / (2 * sigma^2));
            img(y, x) = img(y, x) + val;  % 겹치는 별 누적
        end
    end
end
```

### 7. 센서 신호 모델

별빛이 최종 디지털 값(ADU)으로 변환되는 전체 체인입니다:

```
별빛 (photons/s)
    │
    ├× exposure_time (노출 시간) [s]
    │   → photons (총 광자 수)
    │
    ├× quantum_efficiency (양자 효율) [0~1]
    │   → electrons (전자 수)
    │   광자가 전자를 생성하는 효율. CMOS에서 약 50%.
    │
    ├× analog_gain (아날로그 게인) [배]
    │   → amplified electrons
    │   OV4689: 1x ~ 64x (기본 16x)
    │
    ├× digital_gain (디지털 게인) [배]
    │   → ADU (Analog-to-Digital Units)
    │   OV4689: 기본 1.0x
    │
    └→ 최종 ADU 값 (8-bit: 0~255)
```

수식:
```
ADU = photon_flux × t_exp × QE × A_gain × D_gain
```

구체적 예시 (6등급 별):
```
photon_flux = 96 photons/s
t_exp = 22 ms = 0.022 s
QE = 0.5
A_gain = 16x
D_gain = 1.0x

electrons = 96 × 0.022 × 0.5 = 1.056 e-
ADU = 1.056 × 16 × 1.0 = 16.9 ADU

→ 6등급 별은 약 17 ADU (배경 포함 전)
```

**코드 대응** (`simulate_star_image_realistic.m:237-244`):
```matlab
photon_flux = ref_photon_flux * 10^(-0.4 * (mag - ref_mag));
electrons = photon_flux * exposure_time * quantum_eff;
total_flux = electrons * analog_gain * digital_gain;
```

### 8. 센서 노이즈 모델

실제 이미지 센서에서 발생하는 3가지 주요 노이즈 소스를 모델링합니다.

#### 전체 노이즈 모델

```
물리적 신호 체인:
  광자 → 전자 (Poisson) → 게인 증폭 → 읽기 노이즈 (Gaussian)

수식:
  electrons = signal_e + dark_e                 ← 전자 단위
  noisy_e = Poisson(electrons)                  ← 샷 노이즈 (전자 단위)
  ADU = noisy_e × gain                          ← 게인 증폭
  최종 = ADU + Gaussian(0, σ_read × gain)       ← 읽기 노이즈 (ADU 단위)
```

> **주의**: Poisson 노이즈는 반드시 **전자(electron) 단위**에서 적용해야 합니다.
> ADU(게인 적용 후) 단위에서 적용하면 샷 노이즈가 √gain 만큼 과소평가됩니다.
> (예: gain=16일 때 약 4배 과소평가)

#### 8-1. 다크 전류 (Dark Current)

**물리 원인**: 열에너지에 의해 빛이 없어도 전자가 생성되는 현상.

```
dark_electrons = dark_current_rate × exposure_time
               = 0.1 [e-/px/s] × 0.022 [s]
               = 0.0022 e-/pixel

dark_ADU = dark_electrons × analog_gain × digital_gain
         = 0.0022 × 16 × 1.0
         = 0.035 ADU (매우 작음)
```

- 온도 의존: 10°C 상승마다 약 2배 증가
- 우주 환경 (-20°C): 지상 대비 약 1/100로 감소
- 시뮬레이션에서는 0.1 e-/px/s로 설정 (일반적인 CMOS 수준)

**코드 대응** (`simulate_star_image_realistic.m:411-415`):
```matlab
% ADU → 전자 역변환 (게인 제거)
total_gain = analog_gain * digital_gain;
signal_electrons = gray_img / total_gain;

% 다크 전류 (전자 단위로 추가)
dark_electrons = sensor_params.dark_current_rate * exposure_time;
total_electrons = signal_electrons + dark_electrons;
```

> 다크 전류는 전자 단위에서 신호에 합산된 후, 함께 Poisson 과정을 거칩니다.

#### 8-2. 샷 노이즈 (Shot Noise / Photon Noise)

**물리 원인**: 광자의 도착이 확률적(Poisson 과정)이기 때문에 발생. 양자역학적 근본 노이즈로, 제거 불가능.

```
신호 = N photons → 노이즈 = √N (Poisson 분포)
SNR = N / √N = √N  (더 밝을수록 SNR 높음)
```

예시:
| 신호 (ADU) | 노이즈 (ADU) | SNR |
|-----------|-------------|-----|
| 10 | √10 ≈ 3.2 | 3.2 |
| 100 | √100 = 10 | 10 |
| 1000 | √1000 ≈ 31.6 | 31.6 |

**코드 대응** (`simulate_star_image_realistic.m:417-424`):
```matlab
% 샷 노이즈 (Poisson) - 전자 단위에서 적용
noisy_electrons = poissrnd(max(0, total_electrons));

% 전자 → ADU (게인 재적용)
gray_noisy = noisy_electrons * total_gain;
```

`poissrnd(λ)`: 평균값 λ인 Poisson 분포에서 난수 생성. 이미지의 각 픽셀이 독립적인 Poisson 과정을 겪습니다.

> Poisson 노이즈의 분산은 평균과 같으므로 (`σ² = λ`), 전자 1.06개인 신호의
> 샷 노이즈는 √1.06 ≈ 1.03 e⁻. 이것이 게인(×16)으로 증폭되어 ADU에서는 ~16.5 ADU의 노이즈가 됩니다.

#### 8-3. 읽기 노이즈 (Read Noise)

**물리 원인**: ADC(아날로그-디지털 변환기)와 읽기 회로의 전자적 노이즈. 신호 크기와 무관한 고정 노이즈.

```
σ_read = 3 e- RMS (전자 단위)

게인 적용 후:
σ_read_ADU = σ_read × analog_gain × digital_gain
           = 3 × 16 × 1.0
           = 48 ADU RMS
```

분포: 정규 분포 N(0, σ_read_ADU)

**코드 대응** (`simulate_star_image_realistic.m:426-428`):
```matlab
% 읽기 노이즈 (Gaussian) - ADU 단위
read_noise_adu = sensor_params.read_noise * total_gain;
gray_noisy = gray_noisy + read_noise_adu * randn(size(gray_noisy));
```

#### 노이즈 예산 (Noise Budget)

```
전체 노이즈 (ADU) = √(σ²_shot_ADU + σ²_dark_ADU + σ²_read_ADU)

6등급 별 (피크 ~17 ADU, gain=16):
  signal_electrons = 17 / 16 ≈ 1.06 e⁻

  σ_shot = √1.06 e⁻ → ×16 = 16.4 ADU   ← 전자 노이즈가 게인으로 증폭됨
  σ_dark ≈ √0.002 × 16 ≈ 0.7 ADU (무시 가능)
  σ_read = 3 e⁻ × 16 = 48 ADU
  ───────────────────
  σ_total = √(269 + 0.5 + 2304) ≈ 50.7 ADU

→ 읽기 노이즈가 지배적이나, 샷 노이즈도 무시 못함
→ 기존 ADU 기반 Poisson 적용 시: σ_shot = √17 ≈ 4.1 ADU (4배 과소평가!)
```

> **수정 이력**: 기존 코드에서 `poissrnd(ADU)` → `poissrnd(electrons)`로 수정됨.
> ADU 기반 적용 시 샷 노이즈가 √gain 배 과소평가되는 버그가 있었음.

### 9. Bayer 패턴과 Color Filter Array

#### Bayer 패턴이란?

Bryce Bayer가 1976년 특허(US3,971,065)를 취득한 칼라 이미지 센서 기술입니다. 단일 센서로 컬러 이미지를 얻기 위해, 각 픽셀 위에 R/G/B 필터 중 하나를 배치합니다.

#### RGGB 패턴

```
전체 이미지에서 2×2 블록이 반복:

  열 0  열 1  열 2  열 3  열 4  열 5  ...
  ┌────┬────┬────┬────┬────┬────┐
행0│ R  │ Gr │ R  │ Gr │ R  │ Gr │  ← 짝수 행: R-G 반복
  ├────┼────┼────┼────┼────┼────┤
행1│ Gb │ B  │ Gb │ B  │ Gb │ B  │  ← 홀수 행: G-B 반복
  ├────┼────┼────┼────┼────┼────┤
행2│ R  │ Gr │ R  │ Gr │ R  │ Gr │
  ├────┼────┼────┼────┼────┼────┤
행3│ Gb │ B  │ Gb │ B  │ Gb │ B  │
  └────┴────┴────┴────┴────┴────┘

Green이 50% (인간 시각이 녹색에 가장 민감)
Red이 25%, Blue가 25%
```

#### 시뮬레이션에서의 Bayer 생성

```
Bayer(r,c) = Gray_ideal(r,c) × sensitivity(channel)

channel은 위치에 따라 결정:
  (짝수행, 짝수열) → R → sensitivity_R = 1.0
  (짝수행, 홀수열) → G → sensitivity_G = 1.0
  (홀수행, 짝수열) → G → sensitivity_G = 1.0
  (홀수행, 홀수열) → B → sensitivity_B = 0.9
```

B 채널의 감도가 0.9인 이유: 실제 CMOS 센서에서 Blue 필터의 양자 효율이 Red/Green보다 약간 낮습니다.

**코드 대응** (`simulate_star_image_realistic.m:289-304`):
```matlab
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
```

### 10. CFA 디모자이킹 (Demosaicing)

#### 문제

Bayer 이미지에서 각 픽셀은 R/G/B 중 하나의 값만 갖습니다. 완전한 RGB 이미지를 만들려면 없는 2개 채널의 값을 **보간(interpolation)**으로 추정해야 합니다.

#### Bilinear 보간 (시뮬레이션 구현)

가장 기본적인 디모자이킹 방법입니다. 이웃 픽셀의 평균으로 빈 값을 채웁니다.

**예: R 위치에서 G 값 추정**

```
  ┌────┬────┬────┐
  │ B  │ [G]│ B  │     [G]는 보간할 위치
  ├────┼────┼────┤
  │[G] │ R★ │[G] │     R★: 현재 R 픽셀
  ├────┼────┼────┤
  │ B  │ [G]│ B  │
  └────┴────┴────┘

G(R위치) = (G_상 + G_좌 + G_우 + G_하) / 4
```

**예: R 위치에서 B 값 추정**

```
  ┌────┬────┬────┐
  │[B] │ G  │[B] │     [B]는 보간할 위치
  ├────┼────┼────┤
  │ G  │ R★ │ G  │     대각선 4개의 평균
  ├────┼────┼────┤
  │[B] │ G  │[B] │
  └────┴────┴────┘

B(R위치) = (B_좌상 + B_우상 + B_좌하 + B_우하) / 4
```

#### FPGA 구현 대응

이 시뮬레이션의 `bayer_to_rgb_cfa.m`은 역분석 대상인 FPGA의 `cfa.cpp` HLS 코드와 동일한 알고리즘을 MATLAB으로 재현한 것입니다. FPGA에서는:
- 3×3 윈도우를 라인 버퍼(line buffer)로 구현
- 픽셀 클록마다 1픽셀씩 처리 (스트리밍 파이프라인)
- 정수 연산만 사용 (부동소수점 없음)

### 11. Grayscale 변환 방법론

#### 변환 비교의 목적

현재 FPGA 파이프라인:
```
Bayer → [CFA IP] → RGB → [RGB2Gray IP] → Gray → [별 검출]
         (비용: LUT + BRAM)   (비용: LUT)
```

제안 파이프라인:
```
Bayer → [직접 변환] → Gray → [별 검출]
         (비용: 거의 0)
```

**질문**: CFA + RGB2Gray IP를 제거하면 별 검출 성능이 저하되는가?

#### 방법 A: FPGA 경로 (기준)

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Bayer (H×W) │ ──→ │ CFA 디모자이킹│ ──→ │ (R+2G+B)/4   │ ──→ Gray (H×W)
│             │     │ (bilinear)   │     │ (FPGA 공식)   │
└─────────────┘     └──────────────┘     └──────────────┘
```

#### 방법 B1: RAW (패스스루)

```
┌─────────────┐
│ Bayer (H×W) │ ──→ Gray (H×W)   그대로 사용
└─────────────┘
```

논거:
- 별은 R/G/B 필터에 관계없이 밝게 보임 (백색광에 가까움)
- Bayer 격자 패턴이 약간 보이지만, threshold 기반 검출에는 영향 미미
- **연산량: 0** (가장 빠름, FPGA 리소스 불필요)

#### 방법 B2: 2×2 Binning

```
┌────┬────┐
│ R  │ Gr │ ──→ (R + Gr + Gb + B) / 4 = 1 pixel
├────┼────┤
│ Gb │ B  │
└────┴────┘
```

논거:
- 4개 픽셀 합산 → SNR √4 = 2배 향상
- 해상도 1/2 (1280×720 → 640×360)
- 별 위치 정밀도 약 2배 저하
- 하드웨어: 덧셈기 + 2-bit 시프터만 필요

#### 방법 B3: Green 채널 추출

```
Bayer에서 Green 픽셀만 사용 (50%), 나머지는 이웃 보간

RGGB에서 Green 위치:
  ┌────┬────┐
  │    │ Gr │
  ├────┼────┤
  │ Gb │    │
  └────┴────┘

Green이 아닌 위치는 주변 Green 4개의 평균
```

논거:
- Green 필터의 투과 대역이 가장 넓어 가장 많은 광자를 수집
- 인간 시각 모델(Luminance)에서 Green이 58.7% 차지
- 하드웨어: 비교적 단순한 보간 회로

#### 방법 B4: 가중 평균

```
각 위치에서 주변 R, G, B를 추정한 후:
Y = (R_est + 2×G_est + B_est) / 4

CFA 디모자이킹과 유사하지만, RGB를 각각 출력하지 않고
한 번에 Gray를 계산
```

논거:
- FPGA 경로와 동일한 가중치 사용
- CFA + RGB2Gray를 하나의 연산으로 통합
- 중간 RGB 이미지 불필요 → 메모리/대역폭 절감

#### 방법 B5: 최적 가중 평균 (Optimal)

```
각 위치에서 주변 R, G, B를 추정한 후:
Y = 0.4544×R + 0.3345×G + 0.2111×B

Inverse Variance Weighting (w_i ∝ S_i / σ_i²) 기반:
  - OV4689 분광 응답 특성 반영
  - 흑체복사 스펙트럼 (3,000K~25,000K) 기반 채널별 신호량 계산
  - Read noise 지배 영역(어두운 별)에서 효과 최대

FPGA 정수 근사: Y = (8R + 5G + 3B) >> 4
```

논거:
- 기존 FPGA `(R+2G+B)/4`는 인간 시각 기반 → 별 추적에 비최적
- R 채널이 가장 많은 광자를 수집 (넓은 응답 + 별빛 R 우세)
- 6등급 어두운 별에서 SNR +10.6% 개선
- 곱셈기 불필요, DSP 0개 추가

### 12. SNR 이론 (Signal-to-Noise Ratio)

#### 참조 기반 SNR (`calculate_snr.m`)

참조(이상적) 이미지와 비교하여 SNR을 측정합니다:

```
         P_signal       mean(reference²)
SNR = 10 × log₁₀(───────) = 10 × log₁₀(─────────────────────────)  [dB]
         P_noise        mean((measured - reference)²)
```

| SNR (dB) | 의미 | 노이즈 수준 |
|----------|------|------------|
| 0 | 신호 = 노이즈 | 매우 나쁨 |
| 10 | 신호 10배 | 나쁨 |
| 20 | 신호 100배 | 보통 |
| 30 | 신호 1000배 | 좋음 |
| 40 | 신호 10000배 | 매우 좋음 |

#### 피크 SNR (`calculate_peak_snr.m`)

별 영상에 특화된 SNR 정의입니다. 가장 밝은 별(피크)과 배경 노이즈의 비율:

```
                  S_peak
Peak SNR = 20 × log₁₀(─────)  [dB]
                  σ_noise
```

여기서:
- `S_peak = max(image) - median(image)`: 피크 신호 (배경 기준)
- `σ_noise = std(배경 영역)`: 배경 노이즈 표준편차
- 배경 영역 = `image < median + 10` 인 픽셀 (별이 아닌 영역)

20을 곱하는 이유: 전압 비(amplitude ratio)이므로 20log₁₀ 사용 (전력비는 10log₁₀).

#### 왜 Peak SNR을 사용하는가?

일반적인 이미지 품질 지표인 PSNR(Peak Signal-to-Noise Ratio)은 전체 이미지의 평균 품질을 측정합니다. 하지만 별센서 이미지에서는:
- 이미지의 99%+ 가 검정 배경 (신호 없음)
- 관심 영역은 소수의 밝은 점(별)뿐
- 별의 피크 밝기 vs 배경 노이즈의 비율이 검출 성능을 직접 결정

### 13. 별 검출 알고리즘

#### Threshold + Connected Component Labeling (CCL)

가장 기본적인 별 검출 방법입니다:

```
Step 1: 배경 추정
        bg = median(전체 이미지)
        → 이미지의 50% 이상이 배경이므로 중앙값이 배경을 잘 추정

Step 2: 이진화 (Thresholding)
        binary(r,c) = 1  if  image(r,c) > bg + threshold
                      0  otherwise
        → threshold = 15~20 ADU (기본 설정)

Step 3: Connected Component Labeling
        인접한 '1' 픽셀들을 하나의 그룹(component)으로 묶음

        예시:
        0 0 0 0 0 0 0      0 0 0 0 0 0 0
        0 0 1 1 0 0 0      0 0 ■ ■ 0 0 0     ← 별 1 (area=4)
        0 0 1 1 0 0 0  →   0 0 ■ ■ 0 0 0
        0 0 0 0 0 0 0      0 0 0 0 0 0 0
        0 0 0 0 0 1 0      0 0 0 0 0 ● 0     ← 별 2 (area=3)
        0 0 0 0 1 1 0      0 0 0 0 ● ● 0
        0 0 0 0 0 0 0      0 0 0 0 0 0 0

Step 4: 면적 필터링
        area < min_area인 그룹 제거 (노이즈 점 제거)
        → min_area = 2~3 pixel

Step 5: 속성 추출
        각 그룹에서: Centroid(무게중심), Area(면적),
                    MaxIntensity(최대 밝기), MeanIntensity(평균 밝기)
```

**코드 대응** (`detect_stars_simple.m:10-30`):
```matlab
bg = median(img(:));
binary = img > (bg + threshold);
cc = bwconncomp(binary);
stats = regionprops(cc, img, 'Centroid', 'Area', 'MaxIntensity', 'MeanIntensity');
valid_idx = find([stats.Area] >= min_area);
```

### 14. Centroid 추정 이론

#### Center of Gravity (CoG)

검출된 별 영역의 밝기 가중 중심을 계산합니다:

```
         Σᵢ xᵢ × I(xᵢ, yᵢ)
x_c = ───────────────────────
         Σᵢ I(xᵢ, yᵢ)

         Σᵢ yᵢ × I(xᵢ, yᵢ)
y_c = ───────────────────────
         Σᵢ I(xᵢ, yᵢ)
```

- `(xᵢ, yᵢ)`: 별 영역 내 각 픽셀의 좌표
- `I(xᵢ, yᵢ)`: 해당 픽셀의 밝기

이 시뮬레이션에서는 MATLAB의 `regionprops(..., 'Centroid')`가 이 계산을 수행합니다 (밝기 미가중 기하학적 중심).

#### Centroid 정확도 평가

**정확도 지표: RMS 오차**

```
         ┌─────────────────────┐
RMS  =   │ 1/N × Σ(dᵢ²)       │    [pixel]
         └─────────────────────┘

dᵢ = ||detected_pos - true_pos||₂   (유클리드 거리)
```

**매칭 알고리즘**:
```
각 참 별 위치에 대해:
  1. 모든 검출된 별까지 유클리드 거리 계산
  2. 최소 거리가 match_radius(기본 5 px) 이내이면 매칭
  3. 매칭된 쌍의 거리를 오차로 기록
  4. 매칭되지 않으면 미검출(missed detection)
```

**코드 대응** (`evaluate_centroid_accuracy.m:14-25`):
```matlab
for i = 1:size(true_centroids, 1)
    true_pos = true_centroids(i, :);
    distances = sqrt(sum((detection.centroids - true_pos).^2, 2));
    [min_dist, ~] = min(distances);
    if min_dist < match_radius
        result.n_matched = result.n_matched + 1;
        result.errors = [result.errors; min_dist];
    end
end
result.rms_error = sqrt(mean(result.errors.^2));
```

#### Centroid 오차의 이론적 한계

Liebe (2002)에 따르면, Centroid 오차의 이론적 하한은:

```
σ_centroid = σ_PSF / SNR_star    [pixel]
```

여기서:
- `σ_PSF`: PSF의 표준편차 (1.2 px)
- `SNR_star`: 별의 SNR (√(total_flux))

예: 3등급 별 (total_flux ≈ 5000 ADU):
```
σ_centroid = 1.2 / √5000 ≈ 0.017 pixel ≈ 0.034 µm
```

---

## 시뮬레이션 기능

### 메인: 별 이미지 생성 (`main_simulation.m`)

실제 별 카탈로그 기반 물리적으로 정확한 이미지 생성:
- 관측 방향 (RA, DEC, Roll) 설정
- 센서 파라미터 (해상도, 픽셀 크기, FOV)
- 노이즈 모델 (샷 노이즈, 읽기 노이즈, 다크 전류)

### GUI: 인터랙티브 시뮬레이터 (`gui_star_simulator.m`)

MATLAB `uifigure` 기반 인터랙티브 GUI:
- **5개 탭**: Sensor / Exposure / Noise / Observation / Processing
- **이미지 비교**: 이상적(노이즈 없는) Grayscale vs 변환된 Grayscale 나란히 표시
- **실시간 메트릭**: FOV, 별 수, SNR, 검출 수, Centroid RMS, 처리 시간
- **3단계 캐싱**: Stage1(시뮬) → Stage2(변환) → Stage3(검출), dirty flag 기반 필요 단계만 재실행
- **프리셋**: Orion Belt, Polaris, Big Dipper 등 관측 프리셋 / Space, Ground 등 씬 프리셋
- **카탈로그 사전 로드**: 477MB 카탈로그를 GUI 시작 시 한번만 로드

> **버그 수정 이력 (2026-01-31)**:
> - Ideal 이미지에 `star_info.ideal_gray` (노이즈 없는 PSF 렌더링) 사용하도록 수정.
>   기존에는 `simulate_star_image_realistic()`의 첫 번째 반환값(노이즈 포함)을 사용해
>   Ideal과 변환 결과가 동일하게 노이즈가 포함되는 문제가 있었음.
> - 노이즈 모델 수정: Poisson 노이즈를 ADU가 아닌 전자(electron) 단위에서 적용하도록 변경.
>   기존 코드는 `poissrnd(ADU)`로 샷 노이즈를 √gain 배 과소평가하고 있었음.

### 서브: Bayer→Gray 변환 비교 (`sub_main_1_bayer_comparison.m`)

FPGA 파이프라인 vs 직접 변환 성능 비교:

| 방법 | 설명 | 용도 |
|------|------|------|
| **A. FPGA** | CFA → RGB → Gray | 현재 FPGA 구현 |
| **B1. RAW** | Bayer 값 그대로 | 가장 단순 |
| **B2. Binning** | 2x2 평균 | 해상도↓, SNR↑ |
| **B3. Green** | G 채널만 사용 | 50% 데이터 사용 |
| **B4. Weighted** | R+2G+B/4 | 중간 복잡도 |
| **B5. Optimal** | 0.454R+0.335G+0.211B | SNR 최대화 |

**핵심 결론**: RAW 직접 변환이 FPGA 방식과 동등한 성능을 제공하며, CFA + RGB2Gray IP 제거로 FPGA 리소스/전력 절감 가능.

### 서브: 최적 가중치 연구 (`sub_main_2_optimal_weights.m`)

Inverse Variance Weighting 기반 Bayer→Gray 최적 가중치 도출:
- **결과**: R=0.4544, G=0.3345, B=0.2111 (6등급 별 기준 기존 FPGA 대비 SNR +10.6%)
- **FPGA 구현**: `(8R + 5G + 3B) >> 4` (곱셈기 불필요)
- **상세 문서**: `250131_최적가중치연구.md`

---

## 센서 파라미터

### OV4689 센서

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 해상도 | 1280 × 720 | FPGA 처리 해상도 |
| 픽셀 크기 | 2 µm | 센서 스펙 |
| 초점거리 | 10.42 mm | 렌즈 |
| FOV | 14° × 8° | 시야각 |

### 노출/게인 (OV4689 레지스터 기반)

| 파라미터 | 값 | 레지스터 | 설명 |
|----------|-----|---------|------|
| 노출 시간 | 22 ms | 0x3500-0x3502 (=0xC350) | 60Hz, VTS=2350 기준 |
| 아날로그 게인 | 16× | 0x3508-0x3509 | 범위: 1x ~ 64x |
| 디지털 게인 | 1.0× | 0x352a (=0x08) | 기본 1.0x |
| 양자 효율 | 50% | — | CMOS 기준 |

### 노이즈 모델

| 노이즈 | 모델 | 기본값 |
|--------|------|--------|
| 샷 노이즈 | Poisson 분포 | 신호 의존 |
| 읽기 노이즈 | Gaussian | 3 e- RMS |
| 다크 전류 | 상수 | 0.1 e-/px/s |

---

## 의존성

### 별 카탈로그 (포함됨)
Hipparcos 기반 별 카탈로그가 `data/` 폴더에 포함되어 있습니다:
- `star_catalog_kvector.mat`: MAT 형식 (기본 사용)
- `Hipparcos_Below_6.0.csv`: CSV 형식 (백업)
- 포함 필드: RA (적경), DEC (적위), Magnitude (등급)
- 등급 제한: ≤ 6.5 (육안 가시 한계)

### MATLAB 툴박스
- Image Processing Toolbox (필수: `bwconncomp`, `regionprops`, `padarray`, `imresize`)
- Statistics and Machine Learning Toolbox (`poissrnd` 함수용)

---

## 참고 문헌

> PDF 파일은 `refs/` 폴더에 저장됨

### 1. Star Tracker & Attitude Determination

| 자료 | 링크 | 로컬 | 참고 부분 |
|------|------|------|-----------|
| Liebe, C. C. (2002). "Accuracy performance of star trackers - a tutorial." **IEEE Trans. AES, 38(2), 587-599** | [IEEE Xplore](https://ieeexplore.ieee.org/document/1008988/) / [ResearchGate](https://www.researchgate.net/publication/3003478) | `refs/Accuracy_performance_of_star_trackers_-_a_tutorial.pdf` | **Section III**: Centroid 정확도, **Eq.7-9**: NEA 계산 |
| Spratling, B. B. & Mortari, D. (2009). "A Survey on Star Identification Algorithms." **Algorithms, 2(1), 93-107** | [MDPI (Open Access)](https://www.mdpi.com/1999-4893/2/1/93) | `refs/Spratling_Mortari_2009_StarID_Survey.pdf` | **Section 2**: Lost-in-Space 알고리즘, **Table 1**: 알고리즘 비교 |

**시뮬레이션 적용:**
- `detect_stars_simple.m`: Threshold 기반 검출 (Liebe Section II-B)
- `evaluate_centroid_accuracy.m`: RMS 오차 계산 (Liebe Eq.12)

---

### 2. Pogson Formula (Magnitude-Flux)

| 자료 | 링크 | 참고 부분 |
|------|------|-----------|
| Pogson, N. (1856). MNRAS, 17, 12-15 | [ADS](https://ui.adsabs.harvard.edu/abs/1856MNRAS..17...12P) | 등급 정의 원논문 |

**수식:**
```
m₁ - m₂ = -2.5 × log₁₀(F₁/F₂)
```
- 등급 5 차이 = 100배 플럭스 차이
- 등급 1 차이 = 10^0.4 ≈ 2.512배

**시뮬레이션 적용 (`simulate_star_image_realistic.m:237`):**
```matlab
photon_flux = ref_photon_flux * 10^(-0.4 * (mag - ref_mag));
```

---

### 3. Star Catalog (Hipparcos)

| 자료 | 링크 | 로컬 | 참고 부분 |
|------|------|------|-----------|
| ESA (1997). "The Hipparcos and Tycho Catalogues." **ESA SP-1200** | [ESA Cosmos](https://www.cosmos.esa.int/web/hipparcos/catalogues) / [Vol.1 PDF](https://www.cosmos.esa.int/documents/532822/552851/vol1_all.pdf) | `refs/Hipparcos_ESA_SP1200_Vol1.pdf` | **Chapter 1.2**: 카탈로그 구조, **Appendix**: 필드 정의 |

**시뮬레이션 적용:**
- 사용 필드: RA (적경), DEC (적위), Magnitude (등급)
- 등급 제한: ≤ 6.5 (육안 가시 한계)

---

### 4. Bayer Pattern & CFA Demosaicing

| 자료 | 링크 | 로컬 | 참고 부분 |
|------|------|------|-----------|
| Bayer, B. E. (1976). "Color imaging array." **U.S. Patent 3,971,065** | [Google Patents](https://patents.google.com/patent/US3971065A/en) / [RIT PDF](https://home.cis.rit.edu/~cnspci/references/dip/demosaicking/us_patent_3971065.pdf) | `refs/Bayer_Patent_US3971065.pdf` | **Claims 1-3**: RGGB 패턴 정의, **Fig.2**: 2x2 블록 구조 |

**시뮬레이션 적용:**
- `bayer_to_rgb_cfa.m`: 3x3 bilinear interpolation (Patent Fig.2 기반)
- RGGB 패턴: R(0,0), G(0,1), G(1,0), B(1,1)

---

### 5. Sensor Specifications (OV4689)

| 자료 | 링크 | 로컬 | 참고 부분 |
|------|------|------|-----------|
| OmniVision. "OV4689 4MP Product Brief" | [OmniVision](https://www.ovt.com/wp-content/uploads/2023/08/OV4689-PB-v1.8-WEB.pdf) | `refs/OV4689_Product_Brief.pdf` | **p.1**: 해상도 2688×1520, **p.2**: 픽셀 크기 2µm |

**시뮬레이션 적용 (`main_simulation.m:41-44`):**
```matlab
sensor_params.myu = 2e-6;    % 픽셀 크기 [m] - OV4689 스펙
sensor_params.l = 1280;      % 가로 (FPGA 처리 해상도)
sensor_params.w = 720;       % 세로 (FPGA 처리 해상도)
```

---

### 6. 로컬 PDF 파일 목록

```
refs/
├── Accuracy_performance_of_star_trackers_-_a_tutorial.pdf (785 KB) - Liebe 2002 Star Tracker 튜토리얼
├── Bayer_Patent_US3971065.pdf           (324 KB) - Bayer 특허 원문
├── Spratling_Mortari_2009_StarID_Survey.pdf (370 KB) - Star ID 서베이
├── Hipparcos_ESA_SP1200_Vol1.pdf        (11 MB)  - Hipparcos 카탈로그 문서
├── NASA_StarTracker_OpenSource.pdf      (494 KB) - NASA 오픈소스 Star Tracker
└── OV4689_Product_Brief.pdf             (1.7 MB) - 센서 데이터시트
```

---
*최종 업데이트: 2026-01-31*
