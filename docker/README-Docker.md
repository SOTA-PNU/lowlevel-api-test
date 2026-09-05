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

프로젝트 루트에서 실행합니다.

```bash
./docker/build-docker.sh
```

`build-docker.sh`는 호스트에서 CUDA 사용 가능 여부를 확인한 뒤 빌드 모드를 고릅니다.

- CUDA가 감지되면 `BUILD_MODE=cuda`로 빌드합니다.
- CUDA가 없으면 `BUILD_MODE=cpu`로 빌드합니다.
- NPU 이미지는 Dockerfile의 `BUILD_MODE=npu`를 직접 지정해서 빌드합니다.

NPU 빌드는 `rebel-compiler` 설치를 위해 BuildKit secret `rbln_pip_config`가 필요합니다.

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=rbln_pip_config,src=/path/to/pip.conf \
  --build-arg BUILD_MODE=npu \
  -t triton-local-build:npu-latest \
  -f docker/Dockerfile .
```

### 3. Docker 컨테이너 실행

프로젝트 루트에서 실행합니다.

```bash
./docker/run-docker.sh test
./docker/run-docker.sh dev
./docker/run-docker.sh jupyter
```

다른 이미지 태그를 실행하려면 `DOCKER_IMAGE_TAG`를 지정합니다.

```bash
DOCKER_IMAGE_TAG=npu-latest ./docker/run-docker.sh test-npu
DOCKER_IMAGE_TAG=cpu-latest ./docker/run-docker.sh test-cpu
DOCKER_IMAGE_TAG=gpu-latest ./docker/run-docker.sh test-cuda
```

## 사용 가능한 명령어

### 빌드

- `./docker/build-docker.sh` - Docker 이미지 빌드

### 실행

- `./docker/run-docker.sh test` - 이미지 태그나 Docker 런타임 기준으로 디바이스를 자동 선택해 테스트 실행
- `./docker/run-docker.sh test-cpu` - CPU 테스트 실행
- `./docker/run-docker.sh test-cuda` - CUDA 테스트 실행
- `./docker/run-docker.sh test-npu` - Rebellions NPU 테스트 실행
- `./docker/run-docker.sh test-detailed` - 상세 테스트 실행
- `./docker/run-docker.sh dev` - 개발용 interactive bash 실행
- `./docker/run-docker.sh jupyter` - Jupyter Lab 서버 실행
- `./docker/run-docker.sh bash` - 가상환경이 활성화된 bash 실행
- `./docker/run-docker.sh clean` - Docker 리소스 정리
- `./docker/run-docker.sh logs` - 실행 중인 컨테이너 로그 보기

### Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up triton-dev
docker-compose -f docker/docker-compose.yml up triton-test
docker-compose -f docker/docker-compose.yml up triton-test-detailed
docker-compose -f docker/docker-compose.yml up triton-jupyter
```

`docker-compose.yml`은 기본적으로 `triton-local-build:latest`와 NVIDIA runtime을 사용합니다.

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
sudo ./docker/build-docker.sh
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
Docker image 'triton-local-build:latest' not found.
```

먼저 이미지를 빌드하세요.

```bash
./docker/build-docker.sh
```

태그가 다른 이미지를 사용할 때는 실행 시 같은 태그를 지정하세요.

```bash
DOCKER_IMAGE_TAG=npu-latest ./docker/run-docker.sh test-npu
```

### 빌드가 실패하는 경우

캐시 문제를 의심할 수 있으면 Docker 캐시를 정리하고 다시 빌드합니다.

```bash
docker system prune -a
./docker/build-docker.sh
```

Docker 서비스 로그도 확인할 수 있습니다.

```bash
sudo journalctl -u docker.service
```
