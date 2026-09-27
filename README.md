# Triton 연산자 테스트 스위트

이 프로젝트는 CPU, NVIDIA GPU 및 Rebellions NPU 백엔드에서 Triton 연산자를
컴파일하고 실행합니다. 결과를 PyTorch 참조값 또는 정의된 불변 조건과 비교해
검증하고, 지원되는 성능 지표를 측정하며, 백엔드에 독립적인 단일 형식의
보고서를 생성합니다.

## 테스트 환경

| 환경 | Triton 구현 | 필수 런타임 | 테스트 범위 |
|---|---|---|---|
| CPU | [triton-cpu](https://github.com/triton-lang/triton-cpu) | 등록된 Triton CPU 드라이버 | `triton.language` |
| NVIDIA GPU | 공식 Triton | CUDA를 지원하는 PyTorch, NVIDIA 드라이버 및 지원되는 NVIDIA GPU | `triton.language`, `libdevice`, `extra.cuda` |
| RBLN NPU | `rebel-compiler`의 `rebel.triton` | RBLN 디바이스, 드라이버/런타임 및 활성화된 `rebel` 백엔드 | `rebel.triton.language` |

## 테스트 대상

### Triton 연산자

`triton.language`에서 호출 가능한 연산자를 자동으로 탐색하여 테스트합니다.
따라서 테스트되는 연산자 수는 설치된 Triton 버전과 백엔드에 따라 달라집니다.

CPU와 GPU는 RBLN 구현과 공통인 연산자에 동일한 표준
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

NPU에서는 확인된 `triton.language` 연산자를 모두 테스트 대상으로 포함합니다.
특정 `tl` 연산자만 선택적으로 실행하려면 `--only op1,op2`를 사용합니다.

### CUDA libdevice

libdevice는 NVIDIA GPU에서 사용할 수 있는 수학 및 저수준 연산 함수를 제공합니다.
NVIDIA GPU 환경에서는 이러한 libdevice 함수를 대상으로 컴파일 및 실행 여부를 확인합니다.
각 함수에 대해 지원 가능한 데이터 타입을 순차적으로 적용하며, 참조값을 계산할 수 있는 경우에는 실행 결과의 정확도도 함께 검증합니다.

### CUDA extra API

triton.language.extra.cuda는 CUDA 환경에 특화된 추가 기능을 제공합니다.
여기에는 special register, GDC, float8 변환과 같은 CUDA 전용 API가 포함됩니다.
NVIDIA GPU 환경에서는 이러한 API의 실행 여부를 확인하고, PyTorch와 직접 비교하기 어려운 경우에는 반환값의 범위, 유효성, 변환 전후의 일관성 등 각 API의 특성에 맞는 조건을 기준으로 검증합니다.

## 테스트 과정

각 테스트는 다음과 같은 순서로 진행됩니다.

1. CLI 옵션을 확인하고 실행할 백엔드를 선택합니다.
2. 선택된 백엔드에 해당하는 Triton 구현을 불러오고, 해당 드라이버가 정상적으로 사용 가능한지 확인합니다.
3. 테스트 가능한 callable API를 탐색하고, `--only` 옵션이 지정된 경우 해당 연산자만 선택합니다.
4. 각 연산에 필요한 입력 데이터와 정확도 검증을 위한 참조값 또는 검증 조건을 생성합니다.
5. Triton 커널을 컴파일하고 실행합니다. NPU 연산자는 테스트 간 영향을 최소화하기 위해 개별 worker process에서 실행합니다.
6. 실행 결과를 참조값 또는 검증 조건과 비교하고, 수치 비교가 가능한 경우 max_abs와 max_rel을 계산합니다.
7. 정상적으로 실행된 항목에 대해 성능을 측정합니다.
8. 전체 테스트 결과를 정리하여 보고서 형태로 출력합니다.

주요 명령어:

```bash
python triton_test.py --device cuda
python triton_test.py --device cpu
python triton_test.py --device npu

python triton_test.py --device cuda --only exp,sum,dot
python triton_test.py --device npu --only exp,sum,dot
```

주요 CLI 기본값:

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--device` | `auto` | CUDA를 사용할 수 있으면 CUDA, 그렇지 않으면 CPU |
| `--size` | `1,048,576` | 크기를 설정할 수 있는 1차원 테스트의 길이 |
| `--block` | `256` | 크기를 설정할 수 있는 1차원 테스트의 block size |
| `--warmup` | `25` | 모든 장치에서 워밍업 호출 횟수 |
| `--rep` | `100` | 모든 장치에서 측정 호출 횟수 |

## PASS, FAIL 및 정확도 검증

실행 상태와 정확도는 별도로 기록합니다.

| 전체 결과 | 실행 | 정확도 | 의미 |
|---|---|---|---|
| `PASS` | `PASS` | `PASS` 또는 `N/A` | 커널이 정상적으로 컴파일 및 실행되었으며, 검증 조건이 있는 경우 해당 검증을 통과한 상태 |
| `FAIL` | `PASS` | `FAIL` | 커널은 정상적으로 실행되었지만 결과값이 검증 조건을 만족하지 못한 상태 |
| `ERROR` | `FAIL` | `N/A` | 컴파일, 실행, 백엔드 초기화 또는 결과 처리 과정에서 오류가 발생하여 테스트를 정상적으로 완료하지 못한 상태 |

### 정확도 검증 기준

CPU, NVIDIA GPU 및 RBLN NPU에서 참조값과 비교할 수 있는 부동소수점 연산은 동일한 허용 오차를 적용하여 정확도를 검증합니다.

```python
torch.allclose(
    actual,
    expected.to(actual.dtype),
    rtol=1e-2,
    atol=1e-2,
    equal_nan=True
)
```

실행 결과와 참조값의 차이가 다음 허용 범위 이내인 경우 일치하는 것으로 판단합니다.

```text
abs(actual - expected) <= 1e-2 + 1e-2 * abs(expected)
```

- 부동소수점 결과는 상대 오차와 절대 오차를 각각 1e-2까지 허용합니다.
- 정수 및 Boolean 결과는 torch.equal을 사용하여 정확히 일치하는지 확인합니다.
- 수치 비교가 가능한 연산은 최대 절대 오차(max_abs)와 최대 상대 오차(max_rel)를 함께 기록합니다.
- equal_nan=True를 적용하여 동일한 위치에 발생한 NaN은 일치하는 값으로 처리합니다.
- CUDA extra API와 같이 PyTorch 참조값을 직접 정의하기 어려운 경우에는 반환값의 범위, 유효성 또는 변환 전후의 일관성 등 API 특성에 맞는 조건을 사용하여 검증합니다.
- 수치적인 정확도 비교가 적용되지 않는 항목은 accuracy=N/A로 기록합니다.

개별 연산에서 FAIL 또는 ERROR가 발생하더라도 전체 테스트는 가능한 범위까지 계속 수행합니다.
모든 테스트가 정상적으로 진행되어 결과 보고서가 생성된 경우에는 종료 코드 0을 반환하며, 각 연산의 실패 여부는 보고서에 기록합니다.
반면 백엔드 설정이나 디바이스 초기화 실패처럼 테스트 자체를 수행할 수 없는 오류가 발생한 경우에는 0이 아닌 종료 코드를 반환합니다.

## 성능 측정

CPU/CUDA/NPU 모두 같은 benchmark 함수를 사용하며, 실행과 검증에 성공한
테스트에만 `ms`를 기록합니다. `--warmup`과 `--rep`으로 측정 설정을 변경할 수
있습니다.

### GB/s

성공한 CPU/CUDA/NPU `tl` 테스트는 다음 식으로 처리량을 계산합니다.

```text
GB/s = logical I/O bytes / elapsed seconds / 1e9
```

`logical I/O bytes`는 호출에 전달한 입력과 출력 tensor의 크기를 합한 값입니다.
GB/s는 이 입출력 데이터 크기와 실행 시간을 기준으로 계산한 유효 처리량입니다.
내부 buffer, 반복 접근, padding 등은 반영하지 않으므로 실제 메모리 전송량을
측정한 값은 아닙니다.

### 성능 전용 테스트 (`perf.*`)

기능 테스트의 `[1, 64, 64]` 입력(16 KiB)은 실행 시간이 호출 오버헤드에 묻혀
처리량을 측정하기에는 너무 작습니다. 그래서 기능 테스트가 끝난 뒤, 대표 연산
6개(`copy`, `exp`, `add`, `sum`, `softmax`, `matmul`)를 큰 입력으로 다시 측정합니다.
결과는 `perf` 모듈의 `perf.<op>` 항목으로 기록되며, 출력값도 torch 결과와 비교해
검증합니다. 구현은 `perf.py`에 있습니다.

| 장치 | 원소별 연산 입력 (fp32) | tensor 1개 크기 | matmul M×N×K | 커널 구조 |
|---|---|---|---|---|
| CUDA | `[4096, 4096]` | 64 MiB | 4096³ (TF32) | 타일마다 program 하나, grid로 실행 |
| CPU | `[4096, 4096]` | 64 MiB | 2048³ (IEEE fp32) | 타일마다 program 하나, grid로 실행 |
| NPU | `[1, 2048, 1024]` | 8 MiB | 8192×256×1024 | `grid=(1,)`, 정적 타일을 `tl.static_range`로 순회 |

- `sum`은 기능 테스트의 `tl.sum`과 같은 식(`exp(x) / 행 합`)이고, `softmax`는 행 단위 softmax입니다.
- `matmul`은 K 길이의 누적 오차와 저정밀 연산 장치(TF32, NPU 행렬 연산기)를 고려해
  `atol = 1e-2 × max|기대값|`으로 검증합니다.
- NPU matmul은 256×256 출력 타일을 M 방향으로만 순회하고, 타일마다 K 전체(1024)를
  한 번에 곱합니다. 현재 RBLN 컴파일러는 M과 N을 모두 여러 타일로 나누면
  `Graph Optimization: [UNEXPECTED_GRAPH]`로, 출력 타일의 한 변이 1024 이상이면
  `DEVICE_GRAPH_CONVERSION`으로 컴파일에 실패하므로 N을 타일 폭(256)에 맞추고 M을 키웁니다.
- NPU는 CPU tensor를 입력으로 compiled model 호출 전체를 측정하므로, host↔NPU 전송 시간이 포함됩니다.
- `--only`를 주면 목록에 포함된 성능 연산만 실행합니다. `copy`, `matmul`처럼 성능
  테스트에만 있는 이름도 지정할 수 있습니다.

### FLOPS/OPS

성능 테스트는 GB/s와 함께 초당 연산 수를 기록합니다. 입력이 부동소수점이면
FLOPS, 정수면 OPS로 표시하고, 보고서에는 `179.7 TFLOPS`처럼 단위 접두사를 붙입니다.

```text
FLOPS = logical ops / elapsed seconds
```

`logical ops`는 연산 정의에서 센 값이며 실제 실행된 명령어 수는 아닙니다.

| 연산 | 원소당 연산 수 | 비고 |
|---|---|---|
| `copy` | 0 | 연산이 없으므로 FLOPS를 표시하지 않음 |
| `exp`, `add` | 1 | `exp`도 1회로 셈 |
| `sum` | 3 | exp, 덧셈, 나눗셈 |
| `softmax` | 5 | max, 뺄셈, exp, 덧셈, 나눗셈 |
| `matmul` | – | 전체 `2 × M × N × K` |

원소별 연산은 메모리 대역폭에 막히므로 GB/s로, `matmul`은 FLOPS로 해석하는 것이
적절합니다.

## 테스트 데이터 타입과 shape

데이터 타입은 CLI 옵션으로 지정하지 않습니다. 각 `tl` 테스트가 연산에 맞는
입력 타입을 코드에서 정합니다.

대부분의 입력 tensor는 `fp32`이며, 기본 입력 shape는 `[1, 64, 64]`입니다.
입력 dtype이 다른 테스트는 다음과 같습니다.

- `int32`: `cdiv`, `xor_sum`, `umulhi`, `histogram`, `atomic_and`, `atomic_or`,
  `atomic_xor`, `atomic_xchg`
- `uint8`: `dot_scaled`

shape가 다른 공통 테스트는 다음과 같습니다.

- `expand_dims`, `permute`, `trans`: `[64, 64]`
- `softmax`: `[64, 1, 64]`
- `advance`: `[1, 64, 128]`
- `dot_scaled`: 입력 `[16, 64]`, `[64, 16]`, scale `[16, 2]`, 출력 `[16, 16]`

## 결과 기록

테스트가 완료되면 결과 보고서를 터미널에 출력합니다. 보고서에는 다음 정보가 포함됩니다.

- 보고서 생성 시각화
- 전체 PASS, FAIL, ERROR 개수
- 실행 결과 및 정확도 검증 결과
- 사용한 디바이스와 Triton 버전
- 성능 측정 설정과 테스트 데이터 타입
- 테스트 대상 API 개수
- 모듈별 테스트 결과
- 각 연산자의 이름, 데이터 타입, 실행 결과, 정확도, 실행 시간(ms), 처리량(GB/s) 및 세부 정보
- FAIL 또는 ERROR가 발생한 연산자 목록

GitHub Actions에서 실행하는 경우에는 동일한 보고서가 각 job의 실행 로그에 기록됩니다.
개별 연산에서 FAIL 또는 ERROR가 발생하더라도 보고서가 출력되므로 CI 로그에서 전체 테스트 결과를 확인할 수 있습니다.

### JSON 보고서와 결과 대시보드

`--json-out <path>`를 주면 같은 결과를 JSON으로도 저장합니다. JSON에는 연산자별 결과와 함께
하드웨어 식별자(예: `npu-rbln-ca22`), 커밋과 실행 번호, Triton·torch·`rebel-compiler` 버전이 들어갑니다.

```bash
python triton_test.py --device cuda --json-out results.json
python3 report/summary.py results.json            # Markdown 요약
python3 report/build_site.py --data . --out _site # 대시보드 HTML
```

CI에서는 다음 순서로 결과를 모읍니다.

1. 각 테스트 job이 `results.json`을 만들고, 요약을 실행 페이지(Job Summary)에 표시한 뒤
   `results-<cpu|gpu|npu>` artifact로 올립니다. Docker 테스트는 `RESULTS_JSON` 환경 변수로
   `docker/run-docker.sh`가 컨테이너 안의 JSON을 꺼내 옵니다.
2. `Publish Results` 워크플로가 테스트 워크플로 완료 시 실행되어, 브랜치별 최신 실행의
   artifact를 내려받아 대시보드를 만듭니다.
3. `main`의 결과만 `results` 브랜치(`data/<하드웨어>/<실행 번호>.json`)에 누적하고 GitHub Pages에
   배포합니다. 다른 브랜치는 `dashboard-<브랜치>` artifact로 미리보기만 만듭니다.

## 설정

백엔드별로 별도 가상환경을 사용하는 것을 권장합니다. 현재 Docker 이미지와
wheel 설치는 Python 3.10을 기준으로 합니다.

### 공통 설정

```bash
git clone https://github.com/SOTA-PNU/lowlevel-api-test.git
cd lowlevel-api-test

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### CPU

[triton-cpu](https://github.com/triton-lang/triton-cpu#getting-started)를 Docker
이미지와 같은 커밋으로 설치합니다.

```bash
git clone https://github.com/triton-lang/triton-cpu.git
cd triton-cpu
git checkout 9a3dd8096b3c5b89a6dfeba012221f3fed450eb0
git submodule update --init --recursive
python -m pip install -r python/requirements.txt
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-build-isolation
cd ..

python triton_test.py --device cpu
```

```bash
docker build \
  --build-arg BUILD_MODE=cpu \
  -t ghcr.io/sota-pnu/lowlevel-api-test:cpu-latest \
  -f docker/Dockerfile .

DOCKER_IMAGE_TAG=cpu-latest ./docker/run-docker.sh test-cpu
```

### NVIDIA GPU

아래 명령은 Docker 이미지와 같은 PyTorch 2.11.0, Triton 3.6.0, CUDA 12.8
조합입니다. 다른 CUDA 버전은 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서
호스트 드라이버에 맞는 wheel을 확인하세요.

```bash
python -m pip install --only-binary=:all: "torch==2.11.0" \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --only-binary=:all: "triton==3.6.0" numpy
python -m pip check

nvidia-smi
python -c "import torch; assert torch.cuda.is_available()"
python triton_test.py --device cuda
```

```bash
docker build \
  --build-arg BUILD_MODE=cuda \
  -t ghcr.io/sota-pnu/lowlevel-api-test:gpu-latest \
  -f docker/Dockerfile .

DOCKER_IMAGE_TAG=gpu-latest ./docker/run-docker.sh test-cuda
```

### RBLN NPU

호스트에 RBLN 드라이버와 런타임을 설치하고, 다음 조합에 맞는
`rebel-compiler` 환경을 준비합니다.

```text
rebel-compiler==0.11.1.post1
RBLN Driver 3.2.2
```

버전 조합은 [RBLN 0.11.1 릴리스 노트](https://docs.rbln.ai/v0.11.1/supports/release_note.html)를
참고하세요.

```bash
rbln-smi
python -c "from rebel.triton.backends import backends; assert backends['rebel'].driver.is_active()"

unset TRITON_BACKENDS_IN_TREE
PYTHONPATH="" python triton_test.py --device npu
```

Docker를 사용하려면 호스트에
[RBLN Container Toolkit](https://docs.rbln.ai/latest/software/system_management/container_toolkit.html)을
설치하고 CDI를 설정합니다.

```bash
sudo rbln-ctk cdi generate
sudo rbln-ctk runtime configure
rbln-ctk cdi list
```

NPU 이미지 빌드에는 인증된 RBLN Python index 설정이 필요합니다.

```bash
docker buildx build --load \
  --secret id=rbln_pip_config,src=/path/to/rbln-pip.conf \
  --build-arg BUILD_MODE=npu \
  -t ghcr.io/sota-pnu/lowlevel-api-test:npu-latest \
  -f docker/Dockerfile .

DOCKER_IMAGE_TAG=npu-latest ./docker/run-docker.sh test-npu
```
