# Triton Docker Build Guide

이 문서는 Triton 테스트 환경을 Docker로 빌드하고 실행하는 방법을 설명합니다. 기존 `TROUBLESHOOTING.md`의 내용도 이 문서에 합쳤습니다.

## Docker 빌드 과정

### 1. Docker 시작

```bash
sudo systemctl start docker
sudo systemctl enable docker
```

Docker Desktop을 쓰는 환경에서는 Docker Desktop을 먼저 실행하세요.

### 2. Docker 이미지 빌드

프로젝트 루트에서 실행합니다. CI가 쓰는 이미지 이름·태그와 같게 붙여야 실행 단계에서 그대로 쓸 수 있습니다.
러너에서는 CI가 이미 만들어 둔 이미지가 있으므로 이 단계를 건너뛸 수 있습니다.

```bash
# CPU
docker build \
  --build-arg BUILD_MODE=cpu \
  -t ghcr.io/sota-pnu/lowlevel-api-test:cpu-latest \
  -f docker/Dockerfile .

# GPU (CUDA)
docker build \
  --build-arg BUILD_MODE=cuda \
  -t ghcr.io/sota-pnu/lowlevel-api-test:gpu-latest \
  -f docker/Dockerfile .
```

NPU 빌드는 `rebel-compiler` 설치를 위해 BuildKit secret `rbln_pip_config`가 필요합니다.

```bash
docker build \
  --secret id=rbln_pip_config,src=/path/to/pip.conf \
  --build-arg BUILD_MODE=npu \
  -t ghcr.io/sota-pnu/lowlevel-api-test:npu-latest \
  -f docker/Dockerfile .
```

`--secret`은 BuildKit 기능입니다. Docker 23 미만이면 앞에 `DOCKER_BUILDKIT=1`을 붙이세요.

`BUILD_MODE`에 따라 설치 내용이 갈립니다.

| BUILD_MODE | 설치 내용 | 근거 |
|---|---|---|
| `cpu` | CPU용 torch + triton-cpu 소스 빌드 | `docker/Dockerfile:100`, `:110` |
| `cuda` | CUDA toolkit 12.8 + cu128 torch/triton wheel | `docker/Dockerfile:53`, `:83` |
| `npu` | CPU용 torch + `rebel-compiler` | `docker/Dockerfile:97`, `:122` |

`rebel-compiler` 버전은 `docker/Dockerfile:13`의 `RBLN_COMPILER_VERSION`에서 정합니다.

### 3. Docker 컨테이너 실행

CI와 로컬이 같은 이미지 이름을 씁니다. 태그로 디바이스를 고릅니다.

```bash
DOCKER_IMAGE_TAG=cpu-latest ./docker/run-docker.sh test-cpu
DOCKER_IMAGE_TAG=gpu-latest ./docker/run-docker.sh test-cuda
DOCKER_IMAGE_TAG=npu-latest ./docker/run-docker.sh test-npu
```

`DOCKER_IMAGE_TAG`는 필수입니다. 빠뜨리면 어떤 값을 써야 하는지 알려주고 종료합니다.

## CI가 실패했을 때 러너에서 재현하기

NPU 카드와 GPU는 러너(`giga-w773-s-3`)에만 있습니다. 따라서 재현은 그 서버에 접속해서 합니다.
CI가 만들어 둔 이미지가 그대로 남아 있으므로 다시 빌드할 필요가 없습니다.

```bash
# 1. CI 이미지가 있는지 확인
docker images --filter reference='ghcr.io/sota-pnu/lowlevel-api-test:*'

# 2. CI와 똑같은 명령을 손으로 실행
cd ~/lowlevel-api-test
DOCKER_IMAGE_TAG=npu-latest ./docker/run-docker.sh test-npu
```

테스트가 아니라 컨테이너 안에서 직접 만져보려면 `docker run`을 씁니다.
플래그는 `run_test`가 컨테이너를 띄울 때 쓰는 것과 같습니다
(`docker/run-docker.sh:62`, `:82`, `:100`).

```bash
# NPU
docker run -it --rm \
  --device rebellions.ai/npu=all --ipc=host \
  -w /workspace -e PYTHONPATH="" \
  ghcr.io/sota-pnu/lowlevel-api-test:npu-latest bash

# GPU
docker run -it --rm \
  --runtime=nvidia --gpus all \
  -w /workspace -e PYTHONPATH="" \
  ghcr.io/sota-pnu/lowlevel-api-test:gpu-latest bash

# CPU
docker run -it --rm \
  -w /workspace -e TRITON_CPU_BACKEND=1 -e PYTHONPATH="" \
  ghcr.io/sota-pnu/lowlevel-api-test:cpu-latest bash
```

★ NPU는 `--device rebellions.ai/npu=all`이 없으면 컨테이너 안에서 카드가 보이지 않습니다.

워크플로의 테스트 스텝도 같은 스크립트를 부릅니다(`.github/workflows/stable_test.yml:214`).
차이는 `DOCKER_IMAGE`/`DOCKER_IMAGE_TAG`를 워크플로가 넘긴다는 것뿐이고,
`DOCKER_IMAGE` 기본값이 CI와 같아서 로컬에서는 태그만 주면 됩니다.

`.py` 파일은 이미지에 구워진 것이 아니라 컨테이너 기동 후 복사됩니다
(`docker/run-docker.sh`의 `sync_test_sources`). 코드를 고치고 바로 다시 돌리면 반영됩니다.

## 사용 가능한 명령어

### 빌드

빌드는 `docker build`를 직접 씁니다. 명령은 "2. Docker 이미지 빌드" 참고.

### 실행

- `./docker/run-docker.sh test-cpu` - CPU 테스트 실행
- `./docker/run-docker.sh test-cuda` - CUDA 테스트 실행
- `./docker/run-docker.sh test-npu` - Rebellions NPU 테스트 실행

모두 `DOCKER_IMAGE_TAG`가 필요합니다. 다른 이미지를 쓰려면 `DOCKER_IMAGE`도 함께 지정합니다.

## Dockerfile 특징

현재 Dockerfile은 다음 구성을 지원합니다.

1. Ubuntu 22.04 베이스 이미지
2. `/opt/triton-venv` 가상환경 생성
3. 빌드 모드별 PyTorch 설치
4. CUDA 모드에서 CUDA toolkit 설치
5. CPU 모드에서 별도로 clone한 `triton-cpu` 소스 빌드 (해당 저장소 내부 서브모듈 포함)
6. CUDA 모드에서 pip로 Triton wheel 설치 (PyTorch의 버전 제약 유지)
7. NPU 모드에서 `rebel-compiler`/`rebel.triton` pip 설치
8. `pip check`와 선택된 백엔드 import/등록 상태 검증
9. 패키지 설치 후 테스트 소스 복사: 테스트 코드 변경 시 설치 레이어 재사용

프로젝트의 `.gitmodules`와 CUDA용 `triton` 서브모듈은 제거했습니다.
CUDA는 소스 빌드 없이 wheel만 설치하며, CPU 소스 빌드는 유지합니다.
실행 명령에서 `--local-triton`을 제거하고 가상환경에 설치된 Triton을 사용합니다.
`.dockerignore`는 이전 `triton` checkout, `.git`, host 가상환경을 build context에서 제외합니다.
패키지 설치 명령은 Dockerfile에서 관리하며, CI가 Dockerfile 변경을 감지해 이미지를 다시 빌드합니다.

## 주의사항

1. Docker가 실행 중이어야 합니다.
2. CUDA 테스트에는 NVIDIA Container Toolkit과 NVIDIA runtime이 필요합니다.
3. CPU의 첫 소스 빌드는 30-60분 정도 걸릴 수 있습니다. CUDA는 wheel과 CUDA toolkit을 다운로드·설치하는 시간이 필요합니다.
4. Docker 이미지와 빌드 캐시를 위해 충분한 디스크 공간이 필요합니다.
5. NPU 빌드는 인증된 RBLN Python index 설정을 BuildKit secret으로 전달해야 합니다.

## 문제 해결

### Docker가 실행되지 않는 경우

```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

정상 동작 여부를 확인합니다.

```bash
docker --version
docker run hello-world
```

### Docker 권한 오류가 나는 경우

다음과 같은 오류가 나오면 현재 사용자가 Docker daemon socket에 접근하지 못하는 상태입니다.

```text
permission denied while trying to connect to the Docker daemon socket
```

사용자를 `docker` 그룹에 추가한 뒤 새 세션을 열거나 `newgrp docker`를 실행하세요.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

일회성으로는 `sudo`를 붙여 실행할 수 있습니다.

```bash
sudo docker build --build-arg BUILD_MODE=cpu -t ghcr.io/sota-pnu/lowlevel-api-test:cpu-latest -f docker/Dockerfile .
```

### NVIDIA runtime이 감지되지 않는 경우

CUDA 테스트를 실행하려면 NVIDIA Container Toolkit이 필요합니다. 설치 후 Docker를 재시작하세요.

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

GPU가 컨테이너에서 보이는지 확인합니다.

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Docker 이미지가 없다는 오류가 나는 경우

`run-docker.sh`는 실행 전에 이미지 존재 여부를 확인합니다.

```text
Docker image 'ghcr.io/sota-pnu/lowlevel-api-test:cpu-latest' not found.
```

먼저 이미지를 빌드하세요.

```bash
docker build --build-arg BUILD_MODE=cpu -t ghcr.io/sota-pnu/lowlevel-api-test:cpu-latest -f docker/Dockerfile .
```

태그가 다른 이미지를 사용할 때는 실행 시 같은 태그를 지정하세요.

```bash
DOCKER_IMAGE_TAG=npu-latest ./docker/run-docker.sh test-npu
```

### 빌드가 실패하는 경우

캐시 문제를 의심할 수 있으면 Docker 캐시를 정리하고 다시 빌드합니다.

```bash
docker system prune -a
docker build --no-cache --build-arg BUILD_MODE=cpu -t ghcr.io/sota-pnu/lowlevel-api-test:cpu-latest -f docker/Dockerfile .
```

Docker 서비스 로그도 확인할 수 있습니다.

```bash
sudo journalctl -u docker.service
```
