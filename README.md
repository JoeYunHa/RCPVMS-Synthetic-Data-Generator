# rcpvms-gen

> **RCP VMS 합성 고장 데이터 생성기**
> **Synthetic Fault Data Generator for RCP Vibration Monitoring System**

원자력발전소 냉각재 펌프(RCP)의 진동 모니터링 시스템(VMS)에서 수집된 실제 정상 신호를 기반으로, 수학적으로 모델링된 고장 패턴을 주입하여 합성 `.bin` 파일을 생성하는 도구입니다.

A tool that generates synthetic `.bin` vibration data files by injecting mathematically modeled fault signatures into real normal-state signals from a Reactor Coolant Pump Vibration Monitoring System.

---

## 목차 / Table of Contents

- [개요 / Overview](#개요--overview)
- [고장 유형 / Fault Types](#고장-유형--fault-types)
- [파이프라인 / Pipeline](#파이프라인--pipeline)
- [프로젝트 구조 / Project Structure](#프로젝트-구조--project-structure)
- [설치 / Installation](#설치--installation)
- [사용법 / Usage](#사용법--usage)
- [출력 파일 / Output Files](#출력-파일--output-files)
- [심각도 기준 / Severity Reference](#심각도-기준--severity-reference)
- [기술 스택 / Tech Stack](#기술-스택--tech-stack)

---

## 개요 / Overview

**한글**

실제 RCP VMS `.bin` 파일은 고장 데이터를 확보하기 매우 어렵습니다. 이 도구는 실제 정상 신호에 물리 기반 고장 파형을 수학적으로 주입하여, AI/ML 모델 학습 및 알고리즘 검증에 활용할 수 있는 레이블된 합성 데이터를 대량으로 생성합니다.

핵심 원칙:
- 실제 헤더(샘플링 레이트, 채널 수, 센서 감도)를 그대로 보존
- FFT 기반 1X 주파수 자동 감지로 실제 RPM에 동기화된 고장 신호 생성
- ISO 20816-7 진동 심각도 기준 적용
- 배치 모드로 수백 개 파일 일괄 생성 지원

**English**

Real RCP VMS `.bin` fault data is extremely difficult to obtain. This tool injects physics-based fault waveforms into real normal-state signals to mass-produce labeled synthetic data for AI/ML model training and algorithm validation.

Key principles:
- Original header bytes (sampling rate, channel count, sensor sensitivity) are preserved exactly
- Automatic 1X frequency detection via FFT ensures fault signals are synchronized to actual RPM
- ISO 20816-7 vibration severity zones applied
- Batch mode supports generating hundreds of files in a single run

---

## 고장 유형 / Fault Types

| 유형 / Type | 주파수 / Frequency | 설명 / Description |
|---|---|---|
| `unbalance` | 1X (≈30 Hz) | 질량 불균형. 회전 주파수와 동기화된 지배적 1X 성분 / Mass unbalance. Dominant synchronous 1X component |
| `misalignment` | 2X (≈60 Hz) + 1X | 축 미정렬. 지배적 2X 하모닉과 잔류 1X 성분 혼합 / Shaft misalignment. Dominant 2X harmonic with residual 1X |
| `oil_whip` | ~0.45X (≈13.5 Hz) | 오일 휩. 저널 베어링 특유의 서브싱크로너스 불안정 / Oil whip. Sub-synchronous instability specific to journal bearings |

**과도 모드 / Transient Mode**

`--transient` 플래그를 사용하면 고장 신호가 지속적으로 존재하는 대신, 특정 회전 사이클마다 나타났다 사라지는 간헐적 초기 고장을 모사합니다. 버스트 구간은 Hanning 창으로 부드럽게 처리됩니다.

Using `--transient`, the fault appears and disappears at specific rotation cycles rather than persisting continuously, simulating early-stage intermittent defects. Burst windows are shaped with a Hanning function for smooth transitions.

```
|<-- active_cycles -->|<---- silent_cycles ---->| (반복 / repeating)
|   /\  Hanning burst  |       silence (0)        |
```

---

## 파이프라인 / Pipeline

합성 데이터는 5단계 파이프라인으로 생성됩니다.

Synthetic data is produced through a 5-step pipeline.

```
[1/5] Parse     실제 .bin 파일에서 헤더 및 채널 데이터 파싱
                Parse header and channel data from a real .bin file

[2/5] Profile   FFT로 1X 주파수 감지, RMS 및 노이즈 바닥 계산
                Detect 1X frequency via FFT; compute RMS and noise floor

[3/5] Generate  고장 유형 및 심각도에 따라 고장 파형 수학적 생성
                Mathematically generate fault waveform by type and severity

[4/5] Inject    V_synthetic = V_real + (gain × V_fault), 센서 포화 클리핑 적용
                Linearly inject fault; apply sensor saturation clipping

[5/5] Save      원본 헤더 보존 + 합성 채널 데이터를 .bin으로 저장
                Preserve original header; write synthetic channels as .bin
```

X/Y 채널 쌍에서 Y 채널(홀수 인덱스)은 90° 위상 오프셋을 적용하여 베어링 플레인에서 원형 오비트를 재현합니다.

Within each X/Y bearing-plane pair, the Y channel (odd index) receives a 90° phase offset to reproduce a circular orbit.

---

## 프로젝트 구조 / Project Structure

```
rcpvms-gen/
├── main.py                     # CLI 진입점 및 파이프라인 실행 / CLI entry point & pipeline
├── requirements.txt
├── src/
│   ├── core/
│   │   ├── base_parser.py      # NIMS 이벤트 BIN 파서 기반 클래스 / NIMS event BIN parser base class
│   │   ├── rcpvms_parser.py    # RCPVMS 전용 파서 (R17 스펙) / RCPVMS-specific parser (R17 spec)
│   │   ├── generator.py        # 고장 신호 수학적 모델링 / Mathematical fault signal modeling
│   │   └── synthesizer.py      # 고장 주입 및 BIN 재패키징 / Fault injection & BIN repackaging
│   └── models/
│       └── fault_configs.py    # ISO 20816-7 고장 파라미터 / ISO 20816-7 fault parameters
└── data/
    ├── raw/
    │   ├── normal/             # 정상 상태 원본 BIN 파일 (gitignored) / Normal-state source BINs
    │   └── abnormal/           # 비정상 원본 BIN 파일 (참조용) / Abnormal reference BINs
    └── synthetic/              # 생성된 합성 BIN 파일 (gitignored) / Generated synthetic BINs
        ├── unbalance/
        ├── misalignment/
        └── oil_whip/
```

> **주의 / Note**: `data/` 디렉토리의 실제 `.bin` 파일은 용량이 크므로 `.gitignore`에 의해 버전 관리에서 제외됩니다.
> Actual `.bin` files under `data/` are excluded from version control via `.gitignore` due to their large size.

---

## 설치 / Installation

```bash
# 1. 저장소 클론 / Clone the repository
git clone <repository-url>
cd rcpvms-gen

# 2. 가상환경 생성 및 활성화 / Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. 의존성 설치 / Install dependencies
pip install -r requirements.txt
```

---

## 사용법 / Usage

### 기본 사용 / Basic Usage

```bash
# 단일 파일 생성 (정상 BIN → 불균형 고장 합성)
# Generate a single file (normal BIN → unbalance fault)
python main.py data/raw/normal/sample.bin --fault unbalance --severity WARNING --count 1

# 미정렬 고장 100개 배치 생성 (data/raw/normal/ 에서 무작위 선택)
# Batch-generate 100 misalignment fault files (randomly selected from data/raw/normal/)
python main.py --fault misalignment --severity WARNING --count 100

# 오일 휩 CRITICAL 심각도 단일 파일
# Single oil whip file at CRITICAL severity
python main.py data/raw/normal/sample.bin --fault oil_whip --severity CRITICAL --count 1
```

### 심각도 옵션 / Severity Options

```bash
# 사전 정의 레벨 사용 / Use predefined level
python main.py --fault unbalance --severity NORMAL   # 0.5
python main.py --fault unbalance --severity WARNING  # 1.5 (default)
python main.py --fault unbalance --severity CRITICAL # 3.0

# 직접 수치 지정 / Specify exact value
python main.py --fault unbalance --severity-value 2.0

# 범위 내 랜덤 샘플링 (배치 모드) / Random sampling within range (batch mode)
python main.py --fault unbalance --severity-min 0.2 --severity-max 3.0 --count 200
```

### 과도 고장 모드 / Transient Fault Mode

```bash
# 간헐적 고장 (3사이클 활성 / 10사이클 침묵) - 기본값
# Intermittent fault (3 active cycles / 10 silent cycles) - defaults
python main.py --fault unbalance --transient

# 커스텀 사이클 / Custom cycles
python main.py --fault oil_whip --transient --active-cycles 5 --silent-cycles 15
```

### 출력 디렉토리 지정 / Custom Output Directory

```bash
python main.py --fault misalignment --count 50 --output-dir /path/to/output
```

### 전체 옵션 / All Options

```
positional arguments:
  input                  실제 정상 상태 .bin 파일 경로 (생략 시 data/raw/normal/ 에서 무작위 선택)
                         Path to a normal-state .bin file (if omitted, randomly selected from data/raw/normal/)

options:
  --fault                고장 유형: unbalance, misalignment, oil_whip (default: unbalance)
                         Fault type to inject
  --severity             심각도 레벨: NORMAL, WARNING, CRITICAL (default: WARNING)
                         ISO 20816-7 severity zone
  --severity-value V     직접 심각도 수치 지정 (양수 실수)
                         Fixed severity amplitude scale (positive float)
  --severity-min V       배치 모드 심각도 범위 하한 / Lower bound for batch random severity
  --severity-max V       배치 모드 심각도 범위 상한 / Upper bound for batch random severity
  --count N              생성할 파일 수 (default: 100) / Number of files to generate
  --output-dir PATH      출력 디렉토리 / Output directory
  --transient            간헐적 과도 고장 모드 활성화 / Enable intermittent transient mode
  --active-cycles N      과도 모드 활성 사이클 수 (default: 3.0) / Rotation cycles per fault burst
  --silent-cycles N      과도 모드 침묵 사이클 수 (default: 10.0) / Silent cycles between bursts
```

---

## 출력 파일 / Output Files

합성 파일은 다음 명명 규칙을 따릅니다.

Synthetic files follow this naming convention:

```
data/synthetic/<fault>/<fault>_<severity>_<NNNN>.bin

# 예시 / Examples:
data/synthetic/unbalance/unbalance_warning_0001.bin
data/synthetic/misalignment/misalignment_1.500_0001.bin
data/synthetic/oil_whip/oil_whip_1.200-2.800_transient_0042.bin
```

각 합성 파일은 원본 `.bin`과 동일한 바이너리 구조(비인터리브 채널 블록)를 가집니다.

Each synthetic file shares the same binary structure as the original `.bin` (non-interleaved channel blocks):

```
[Original Header Bytes] [CH0 float32 LE block] [CH1 float32 LE block] ...
```

---

## 심각도 기준 / Severity Reference

ISO 20816-7 진동 심각도 구역을 기반으로 합니다.

Based on ISO 20816-7 vibration severity zones.

| 레벨 / Level | 값 / Value | ISO 구역 / ISO Zone | 설명 / Description |
|---|---|---|---|
| `NORMAL` | 0.5 | Zone A/B | 신규 설비 / 장기 운전 허용 범위 / Newly commissioned / Long-term acceptable |
| `WARNING` | 1.5 | Zone C | 조치 필요 / Action required |
| `CRITICAL` | 3.0 | Zone D | 손상 위험 / Danger of damage |

---

## 기술 스택 / Tech Stack

| 라이브러리 / Library | 버전 / Version | 용도 / Purpose |
|---|---|---|
| `numpy` | 1.26.4 | 수치 연산, 배열 처리, 바이너리 핸들링 / Numerical computation, array ops, binary handling |
| `scipy` | 1.12.0 | 신호 처리 (FFT, 필터링) / Signal processing (FFT, filtering) |
| `matplotlib` | 3.8.3 | 합성 파형 시각화 검증 / Waveform visualization & verification |
| `tqdm` | 4.66.2 | 배치 생성 진행률 표시 / Progress bars for batch generation |

---

## 라이선스 / License

내부 연구 및 개발 목적으로 제한됩니다. 실제 원전 운영 데이터는 포함되지 않습니다.

Restricted to internal research and development purposes. No actual nuclear plant operational data is included.
