# Star Tracker 별 이미지 시뮬레이션

Hipparcos 카탈로그 기반 물리적으로 정확한 Star Tracker 이미지 시뮬레이터입니다.

## 개요

### 목적
실제 별 카탈로그와 물리 모델을 사용하여 Star Tracker 센서 이미지를 시뮬레이션합니다:
- **실제 별 위치/등급**: Hipparcos 카탈로그 기반
- **물리적 정확성**: Pogson 공식, PSF 모델
- **센서 노이즈**: 샷 노이즈, 읽기 노이즈, 다크 전류

### 활용
- Star Tracker 알고리즘 검증 (별 검출, Centroid, Star ID)
- 센서/광학 파라미터 최적화
- FPGA 영상처리 파이프라인 검증

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
main_simulation
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

## 디렉토리 구조

```
bayer_comparison/
├── README.md                   # 이 문서
├── main_simulation.m           # ★ 메인 실행 파일 (별 이미지 생성)
├── sub_main_1_bayer_comparison.m  # 서브: Bayer→Gray 변환 비교
│
├── core/                       # 핵심 함수
│   ├── simulate_star_image_realistic.m   # ★ 별 이미지 생성 (Hipparcos)
│   ├── bayer_to_gray_direct.m            # 직접 변환 (4가지 방법)
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

## 시뮬레이션 기능

### 메인: 별 이미지 생성 (`main_simulation.m`)

실제 별 카탈로그 기반 물리적으로 정확한 이미지 생성:
- 관측 방향 (RA, DEC, Roll) 설정
- 센서 파라미터 (해상도, 픽셀 크기, FOV)
- 노이즈 모델 (샷 노이즈, 읽기 노이즈, 다크 전류)

### 서브: Bayer→Gray 변환 비교 (`sub_main_1_bayer_comparison.m`)

FPGA 파이프라인 vs 직접 변환 성능 비교:

| 방법 | 설명 | 용도 |
|------|------|------|
| **A. FPGA** | CFA → RGB → Gray | 현재 FPGA 구현 |
| **B1. RAW** | Bayer 값 그대로 | ★ 권장 (최단순) |
| **B2. Binning** | 2x2 평균 | 해상도↓, SNR↑ |
| **B3. Green** | G 채널만 사용 | 50% 데이터 사용 |
| **B4. Weighted** | R+2G+B/4 | 중간 복잡도 |

**결론**: RAW 직접 변환이 FPGA 방식과 동등한 성능을 제공하며, CFA + RGB2Gray IP 제거로 FPGA 리소스/전력 절감 가능.

## 물리 모델

### PSF (Point Spread Function)
```matlab
sigma = 1.2;  % 픽셀, 광학 시스템에 의해 결정 (상수)
% 2D Gaussian, 6-sigma 윈도우
```

### Pogson 공식 (등급-플럭스 관계)
```matlab
total_flux = ref_flux * 10^(-0.4 * (mag - ref_mag));
% 기준: 6등급 = 1000 ADU
% 등급 5 차이 = 100배 플럭스 차이
% 등급 1 차이 = 10^0.4 ≈ 2.512배
```

### 센서 파라미터 (OV4689)
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 해상도 | 1280 × 720 | FPGA 처리 해상도 |
| 픽셀 크기 | 2 µm | 센서 스펙 |
| 초점거리 | 10.42 mm | 렌즈 |
| FOV | 14° × 8° | 시야각 |

### 노이즈 모델
| 노이즈 | 모델 | 기본값 |
|--------|------|--------|
| 샷 노이즈 | Poisson 분포 | 신호 의존 |
| 읽기 노이즈 | Gaussian | 3 ADU |
| 다크 전류 | 상수 | 5 ADU |

## 의존성

### 별 카탈로그 (포함됨)
Hipparcos 기반 별 카탈로그가 `data/` 폴더에 포함되어 있습니다:
- `star_catalog_kvector.mat`: MAT 형식 (기본 사용)
- `Hipparcos_Below_6.0.csv`: CSV 형식 (백업)
- 포함 필드: RA (적경), DEC (적위), Magnitude (등급)
- 등급 제한: ≤ 6.5 (육안 가시 한계)

### MATLAB 툴박스
- Image Processing Toolbox (필수)
- Statistics and Machine Learning Toolbox (`poissrnd` 함수용)

## 참고 자료

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

**시뮬레이션 적용 (`simulate_star_image_realistic.m:181-186`):**
```matlab
ref_mag = 6.0;
ref_flux = 1000;
total_flux = ref_flux * 10^(-0.4 * (mag - ref_mag));
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

**시뮬레이션 적용 (`main_simulation.m:34-38`):**
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
*최종 업데이트: 2026-01-30*
