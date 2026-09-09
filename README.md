# Triton 연산자 테스트 스위트

이 프로젝트는 CPU, NVIDIA GPU 및 Rebellions NPU 백엔드에서 Triton 연산자를
컴파일하고 실행합니다. 결과를 PyTorch 참조값 또는 정의된 불변 조건과 비교해
검증하고, 지원되는 성능 지표를 측정하며, 백엔드에 독립적인 단일 형식의
보고서를 생성합니다.

## 테스트 환경

| 환경 | Triton 구현 | 필수 런타임 | 테스트 범위 |
|---|---|---|---|
| CPU | [triton-cpu](https://github.com/triton-lang/triton-cpu) | 등록된 Triton CPU 드라이버 | `triton.language`만 |
| NVIDIA GPU | 업스트림 Triton | CUDA를 지원하는 PyTorch, NVIDIA 드라이버 및 지원되는 NVIDIA GPU | `triton.language`, `libdevice` 및 지원되는 `extra.cuda` API |
| RBLN NPU | `rebel-compiler`의 `rebel.triton` | RBLN 디바이스, 드라이버/런타임 및 활성화된 `rebel` 백엔드 | `rebel.triton.language` |

`--device auto`는 `torch.cuda.is_available()`이 true이면 CUDA를 선택하고,
그렇지 않으면 CPU를 선택합니다. NPU는 자동으로 선택하지 않으므로
`--device npu`를 명시적으로 사용해야 합니다.

실행 범위는 디바이스가 정합니다. CPU와 NPU는 `tl` 스위트를, CUDA는
`tl`, `libdevice`, `extra`를 모두 실행합니다.

## 테스트 대상

### Triton 언어 연산자

이 스위트는 활성 `triton.language` 구현에서 공개된 호출 가능 심볼을
런타임에 탐색합니다. 따라서 API 개수는 설치된 Triton 버전과 백엔드에
따라 달라집니다.

CPU와 CUDA는 RBLN 구현과 공통인 연산자에 동일한 표준
`@triton.jit` 커널을 실행합니다. NPU는 해당 커널을 RBLN custom
operator로 감싸고 다음과 같이 컴파일합니다.

```python
torch.compile(
    model,
    backend="rbln",
    dynamic=False,
    options={"mode": ["strict"]},
)
```

테스트 범위에는 다음 그룹이 포함됩니다.

- 단항 및 이항 산술 연산
- reduction, scan, 정렬 및 softmax
- shape, layout, memory 및 block pointer 연산
- atomic 및 정수 연산
- dot 및 dot_scaled
- 난수, 프로그램, 제어, 힌트, 타입 및 meta API

탐색된 모든 NPU callable은 결과 집합에 포함됩니다. 실행 가능한 adapter가
없는 callable은 `ERROR`로 기록하며 조용히 건너뛰지 않습니다.
선택한 `tl` 연산자만 실행하려면 `--only op1,op2`를 사용합니다.

### CUDA libdevice

CUDA 모드는 export된 `libdevice` wrapper를 런타임에 탐색합니다. 각
wrapper마다 부동소수점, 배정밀도, signed integer 및 논리적 unsigned
signature 후보를 순서대로 시도하고, 컴파일 및 실행에 처음 성공한 signature를
기록합니다. 로컬 참조값이 있는 wrapper에는 정확도 결과를 기록하고,
컴파일/실행만 확인하는 wrapper는 `accuracy=N/A`로 기록합니다.

### CUDA extra API

CUDA extra 스위트는 `triton.language.extra.cuda`에서 탐색한 API 중
지원되는 special register, GDC 및 float8 변환 API를 실행합니다. 직접
비교할 PyTorch 참조값이 없으면 launch metadata 또는 범위/유한성
round-trip 불변 조건으로 검증합니다.

## 테스트 과정

각 실행은 다음과 같은 상위 수준의 순서로 진행됩니다.

1. CLI 옵션을 해석하고 요청한 백엔드를 선택합니다.
2. 백엔드별 Triton 구현을 import하고 해당 드라이버를 사용할 수 있는지
   확인합니다.
3. callable API를 탐색하고 `tl` 및 `libdevice` 스위트에는 `--only`를
   적용합니다.
4. 연산에 맞는 입력과 PyTorch 참조값 또는 불변 조건을 생성합니다.
5. 실제 커널을 컴파일하고 launch합니다. 각 NPU 연산자는 별도의 임시
   `TRITON_HOME`을 사용하는 격리된 worker process에서 실행됩니다.
6. 출력을 검증하고 숫자 참조값이 있으면 `max_abs`와 `max_rel`을
   계산합니다.
7. 지원되는 benchmark를 실행합니다.
8. 통합 보고서를 출력하고 해당하는 경우 디스크에 기록합니다.

주요 명령어:

```bash
# 선택한 백엔드가 지원하는 스위트 전체 실행
python triton_test.py --device cuda
python triton_test.py --device cpu
python triton_test.py --device npu

# 선택한 연산자만 실행
python triton_test.py --device cuda --only exp,sum,dot
python triton_test.py --device npu --only exp,sum,dot
python triton_test.py --device cuda --only sin,cos,mul24
```

주요 CLI 기본값:

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--device` | `auto` | CUDA를 사용할 수 있으면 CUDA, 그렇지 않으면 CPU |
| `--size` | `1,048,576` | 크기를 설정할 수 있는 1차원 테스트의 길이 |
| `--block` | `256` | 크기를 설정할 수 있는 1차원 테스트의 block size |
| `--warmup` | `25` | `triton.testing.do_bench`에서는 밀리초 단위 warmup budget, fallback 및 RBLN timing에서는 launch 횟수 |
| `--rep` | `100` | `triton.testing.do_bench`에서는 밀리초 단위 측정 budget, fallback 및 RBLN timing에서는 launch 횟수 |

## PASS, FAIL 및 정확도 검증

실행 상태와 정확도는 별도로 기록합니다.

| 전체 결과 | 실행 | 정확도 | 의미 |
|---|---|---|---|
| `PASS` | `PASS` | `PASS` 또는 `N/A` | 커널이 컴파일 및 실행되었고, 적용 가능한 모든 참조값 또는 불변 조건 검증을 통과함 |
| `FAIL` | `PASS` | `FAIL` | 커널은 실행되었지만 출력이 참조값 또는 불변 조건을 충족하지 못함 |
| `ERROR` | `FAIL` | `N/A` | import, 컴파일, launch, adapter, worker 또는 결과 처리 오류로 유효한 테스트를 완료하지 못함 |

### 공통 전체 연산자 정확도 정책

공통 CPU/CUDA `tl`, CUDA libdevice 및 NPU language 스위트에서
참조값을 사용하는 모든 부동소수점 비교는 다음 기준을 사용합니다.

```python
torch.allclose(
    actual,
    expected.to(actual.dtype),
    rtol=1e-2,
    atol=1e-2,
    equal_nan=True,
)
```

원소별 판정 조건은 다음과 같습니다.

```text
abs(actual - expected) <= 1e-2 + 1e-2 * abs(expected)
```

- 정수 및 Boolean 출력은 `torch.equal`을 사용하므로 정확히 일치해야 합니다.
- 숫자 비교에는 `max_abs`와 `max_rel`을 기록합니다.
- `equal_nan=True`이므로 서로 대응하는 NaN 값은 일치하는 것으로 처리합니다.
- CUDA extra API는 공통 tensor tolerance 대신 문서화된 launch metadata
  또는 범위/유한성 불변 조건을 사용합니다.
- 의미 있는 숫자 target이 없는 API는 sentinel 또는 불변 조건 검사를
  실행하고 `accuracy=N/A`로 기록합니다.

개별 테스트의 `FAIL`이나 `ERROR`는 exit status를 바꾸지 않습니다.
스위트가 끝까지 돌면 status 0으로 종료하고, 실패는 보고서에만 기록합니다.
import, 백엔드 설정, 디바이스 초기화처럼 스위트 자체를 못 돌리는 오류는
계속 non-zero status를 반환합니다.

## 성능 측정

기본값은 `warmup=25`와 `rep=100`이지만 단위는 timing backend에 따라
달라집니다.

- `triton.testing.do_bench`는 두 값을 밀리초 단위 budget으로 해석하고
  실제 launch 횟수를 내부에서 결정합니다.
- fallback timer와 RBLN timer는 두 값을 launch 횟수로 해석합니다.

`--warmup`과 `--rep`으로 두 값을 변경할 수 있습니다. 실행과 적용 가능한
검증에 모두 성공한 테스트에만 성능 필드를 기록합니다.

### CPU 및 NVIDIA GPU

CPU/CUDA 테스트는 먼저 `triton.testing.do_bench`를 사용합니다. 이를
사용할 수 없는 경우에는 다음 방식을 사용합니다.

보고서는 호출당 실행 시간을 `ms`로 기록합니다. 주로 CUDA libdevice와
float8 변환 테스트처럼 memory traffic model을 명확히 정의할 수 있는
테스트에는 다음 값도 기록합니다.

```text
GB/s = bytes moved / elapsed seconds / 1e9
```

스위트에서 의미 있는 전송량을 정의할 수 없으면 `GB/s`는 비워 둡니다.

### RBLN NPU

NPU timing은 `rebel.capture_reports`를 통해 수집한 RBLN
`total_device` timer를 사용하며, 누적된 microsecond 값을 `rep`으로
나눕니다. timer를 사용할 수 없거나 값이 유효하지 않으면 스위트는 동기화된
host wall time을 사용합니다. 연산자마다 전송
의미가 다르므로 NPU `GB/s`는 의도적으로 비워 둡니다.

## 테스트 데이터 타입과 shape

dtype은 CLI로 고르지 않습니다. 각 스위트가 연산에 필요한 타입을 선택합니다.

| 테스트 범위 | 주요 dtype | 주요 shape | 설정 |
|---|---|---|---|
| 공통 CPU/CUDA/NPU `tl` 커널 | `fp32` | `[1, 64, 64]` | 고정 정적 shape |
| `dot` | `fp32` | `[1, 64, 64] x [1, 64, 64]` | 고정 정적 shape |
| `dot_scaled` | 인코딩된 `uint8` 입력/scale, `fp32` 출력 | 입력 `[16, 64]`, `[64, 16]`; scale `[16, 2]`; 출력 `[16, 16]` | 고정 정적 shape |
| Shape/layout 예외 | `fp32` | `expand_dims`, `permute` 및 `trans`: `[64, 64]`; `softmax`: `[64, 1, 64]`; `advance`: `[1, 64, 128]` | 고정 정적 shape |
| Tensor descriptor identity | `fp32` | `[64, 64]` | 고정 정적 shape |
| 정수/atomic/cast 관련 `tl` 케이스 | 주로 `int32` | 대체로 `[1, 64, 64]` | 연산별 설정 |
| 업스트림 전용 1차원 `tl` 케이스 | `fp32` 또는 `int32` | `[size]` | `--size`와 `--block`으로 제어 |
| CUDA libdevice | signature에 따라 `f32`, `f64`, `i32`, `u32`, `i64` 또는 `u64` | `[size]` | `--size`와 `--block`으로 제어 |
| CUDA float8 extra 테스트 | `fp32` 입력 | `[size]` | `--size`와 `--block`으로 제어 |
| CUDA register/GDC extra 테스트 | `int64` 출력 | 스칼라 `[1]` | 고정 |

입력은 연산에 따라 random, 범위 기반 또는 deterministic pattern을
사용합니다. 정의역이 제한된 함수에는 유효한 값을 전달합니다. 예를 들어
logarithm과 square root에는 양수 입력을, inverse trigonometric
function에는 범위가 제한된 입력을 사용합니다. 정수/bit 연산에는 범위가
제한된 연산별 pattern을 사용하며, 일부 pattern은 무작위로 생성합니다.

`--size`와 `--block`은 위 예외를 포함한 고정 shape CPU/CUDA/NPU
커널의 크기를 변경하지 않습니다. 두 옵션은 크기를 설정할 수 있는 1차원
경로에만 적용됩니다.

## 결과 기록

완료된 모든 테스트 실행은 terminal에 보고서를 출력합니다. 보고서에는
다음 내용이 포함됩니다.

- 생성 시각
- 전체 `PASS`, `FAIL` 및 `ERROR` 개수
- 실행 및 정확도 요약
- 디바이스 및 Triton 버전
- benchmark 설정 및 관측된 dtype
- 탐색된 API 개수
- 모듈별 합계
- `name`, `module`, `dtype`, `exec`, `accuracy`, `ms`,
  `GB/s` 및 `detail`이 포함된 상세 행
- 실패하거나 error가 발생한 테스트의 최종 목록

보고서는 terminal 출력으로만 남고 파일로 저장하지 않습니다. 남겨 두려면
직접 리다이렉트합니다.

```bash
python triton_test.py --device npu | tee report.txt
```

CI에서는 job 로그에 그대로 찍힙니다. 리포트를 실패 판정보다 먼저
출력하므로 테스트가 `FAIL`로 끝나도 로그에 남습니다.

## 설정

### 공통 저장소 설정

프로젝트 스크립트는 Python 3.9 이상을 대상으로 합니다. 현재 RBLN
compiler wheel과 Docker image는 Python 3.10을 사용합니다. 아래 wheel 설치
환경도 Python 3.10 이상을 사용하세요. 백엔드별로 별도 가상환경을 권장합니다.

```bash
git clone https://github.com/SOTA-PNU/lowlevel-api-test.git
cd lowlevel-api-test

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

아래 설명에 따라 백엔드별 PyTorch와 Triton 구현을 설치합니다. 표준
업스트림 Triton 설치에는 CPU 또는 RBLN 백엔드가 포함되지 않습니다.

### CPU 설정

CPU는 별도 `triton-lang/triton-cpu` 구현이 필요합니다. CUDA용
`pip install triton`으로는 CPU 백엔드가 설치되지 않습니다.
[triton-cpu 공식 설치 안내](https://github.com/triton-lang/triton-cpu#getting-started)에
따라 활성 환경에 설치한 다음, CPU 드라이버를 선택할 수 있는지 확인합니다.

이 프로젝트의 Triton 서브모듈은 제거했습니다. CPU Docker image는 기존처럼
`triton-cpu` 저장소를 별도로 clone하고 그 저장소 내부의 서브모듈을 초기화한 뒤
소스 빌드합니다. 테스트 코드만 바뀌면 Docker의 설치 레이어를 재사용합니다.
CPU도 소스 빌드를 없애려면 호환되는 `triton-cpu` wheel을 별도로 준비해야 합니다.

```bash
export TRITON_CPU_BACKEND=1
python triton_test.py --device cpu
```

CPU Docker image를 생성하려면 다음 명령을 사용합니다.

```bash
docker build \
  --build-arg BUILD_MODE=cpu \
  -t ghcr.io/sota-pnu/lowlevel-api-test:cpu-latest \
  -f docker/Dockerfile .

DOCKER_IMAGE_TAG=cpu-latest ./docker/run-docker.sh test-cpu
```

### NVIDIA GPU 설정

NVIDIA 드라이버와 호환되는 CUDA PyTorch wheel을 먼저 설치한 뒤
CUDA 의존성을 설치합니다. 아래는 CUDA 12.8 wheel을 사용하는 예시입니다.
드라이버에 맞는 PyTorch index를 선택하세요.

```bash
python -m pip install --only-binary=:all: torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install --only-binary=:all: torch triton numpy
python -m pip check

nvidia-smi
python -c "import torch, triton; assert torch.cuda.is_available(); print(triton.__version__, triton.__file__)"
python triton_test.py --device cuda
```

[Triton 공식 wheel](https://triton-lang.org/main/getting-started/installation.html)을
사용하므로 서브모듈 초기화나 Triton 자체의 소스 빌드는 필요하지 않습니다.
PyTorch가 요구하는 Triton 버전을 유지하도록 별도의 Triton upgrade는 하지 않습니다.
`--only-binary=:all:`은 호환 wheel이 없으면 소스 빌드 대신 설치 오류를 반환합니다.
기존 `--local-triton` 옵션은 제거했으므로 실행 명령에서 빼주세요.
테스트 실행 중 개별 커널의 JIT 컴파일은 계속 수행합니다.

GPU image는 CUDA 12.8 toolkit과 cu128 PyTorch/Triton wheel을 설치합니다
(`docker/Dockerfile:11`, `:12`).

```bash
docker build \
  --build-arg BUILD_MODE=cuda \
  -t ghcr.io/sota-pnu/lowlevel-api-test:gpu-latest \
  -f docker/Dockerfile .

DOCKER_IMAGE_TAG=gpu-latest ./docker/run-docker.sh test-cuda
```

### RBLN NPU 설정

vendor가 제공하는 RBLN 드라이버/런타임과, Docker를 사용할 경우 Container
Toolkit을 설치하고 호환되는 `rebel-compiler` 환경을 준비합니다. 드라이버,
firmware, `librbln-thunk` 및 compiler 버전은 서로 호환되어야 합니다.

현재 Dockerfile은 다음 버전을 고정합니다.

```text
rebel-compiler==0.11.1.post1
```

[공식 호환성 표](https://docs.rbln.ai/latest/supports/version_matrix.html#rbln-sdk-driver-compatibility-matrix)는
SDK 0.11.1과 드라이버 3.2.2를 호환 조합으로 지정합니다. 여전히
`librbln-thunk.so.3.0.0`을 제공하는 host에서는 호환되는 이전 SDK 환경을
사용하거나 vendor 드라이버/런타임 전체를 upgrade해야 합니다. 공유
라이브러리 symlink만 교체하는 것으로는 충분하지 않습니다. 드라이버
업데이트는 [RBLN 드라이버 설치 가이드](https://docs.rbln.ai/latest/getting_started/rbln_driver_installation_guide.html)를
따르고, 정상 동작이 검증된 RBLN 환경 안에서 `rebel-compiler` 또는
PyTorch만 따로 upgrade하지 마십시오.

환경을 확인합니다.

```bash
rbln-smi --version
rbln-smi

python -c "from rebel.triton.backends import backends; assert 'rebel' in backends; assert backends['rebel'].driver.is_active()"
```

활성화된 RBLN 환경에서 실행합니다. 업스트림 Triton 환경 override를
제거하면 잘못된 백엔드가 선택되는 것을 방지할 수 있습니다.

```bash
unset TRITON_BACKENDS_IN_TREE
PYTHONPATH="" python triton_test.py --device npu
```

NPU Docker build에서는 인증된 RBLN Python index에 접근할 수 있도록
BuildKit secret을 제공해야 합니다.

```bash
docker buildx build --load \
  --secret id=rbln_pip_config,src=/path/to/rbln-pip.conf \
  --build-arg BUILD_MODE=npu \
  -t ghcr.io/sota-pnu/lowlevel-api-test:npu-latest \
  -f docker/Dockerfile .

DOCKER_IMAGE_TAG=npu-latest ./docker/run-docker.sh test-npu
```

host에는 [RBLN Container Toolkit](https://docs.rbln.ai/latest/software/system_management/container_toolkit.html)이
설정되어 있어야 합니다. 그래야 `--device rebellions.ai/npu=all`을 통해
디바이스와 런타임 라이브러리를 container에 노출할 수 있습니다.
