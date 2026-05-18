# Triton Docker Build Guide

이 가이드는 성공적으로 테스트된 로컬 빌드 과정을 Docker로 재현하는 방법을 설명합니다.

## 🐳 Docker 빌드 과정

### 1. Docker 시작
```bash
# Docker 서비스 시작 (Ubuntu/Debian)
sudo systemctl start docker
sudo systemctl enable docker

# 또는 Docker Desktop이 설치된 경우
# Docker Desktop을 실행하세요
```

### 2. Docker 이미지 빌드
```bash
# 프로젝트 루트에서 실행
./docker-build.sh

# 또는 docker 디렉토리에서 직접 실행
cd docker && ./build-docker.sh
```

### 3. Docker 컨테이너 실행
```bash
# 프로젝트 루트에서 실행
./docker-run.sh test
./docker-run.sh dev
./docker-run.sh jupyter

# 또는 docker 디렉토리에서 직접 실행
cd docker && ./run-docker.sh test
cd docker && ./run-docker.sh dev
cd docker && ./run-docker.sh jupyter
```

## 📋 사용 가능한 명령어

### 빌드 스크립트
- `./docker-build.sh` - Docker 이미지 빌드 (프로젝트 루트에서)
- `./docker/build-docker.sh` - Docker 이미지 빌드 (docker 디렉토리에서)

### 실행 스크립트
- `./docker-run.sh test` - 기본 테스트 실행 (프로젝트 루트에서)
- `./docker-run.sh test-cpu` - CPU 전용 테스트
- `./docker-run.sh test-cuda` - CUDA 테스트
- `./docker-run.sh test-detailed` - 상세 테스트
- `./docker-run.sh dev` - 개발 환경 (interactive bash)
- `./docker-run.sh jupyter` - Jupyter Lab 서버
- `./docker-run.sh bash` - Interactive bash 세션
- `./docker-run.sh clean` - Docker 리소스 정리
- `./docker-run.sh logs` - 컨테이너 로그 보기

### Docker Compose
```bash
# 프로젝트 루트에서 실행
docker-compose -f docker/docker-compose.yml up triton-dev
docker-compose -f docker/docker-compose.yml up triton-test
docker-compose -f docker/docker-compose.yml up triton-test-detailed
docker-compose -f docker/docker-compose.yml up triton-jupyter

# 또는 docker 디렉토리에서 실행
cd docker && docker-compose up triton-dev
cd docker && docker-compose up triton-test
```

## 🔧 Dockerfile 특징

이 Dockerfile은 성공적으로 테스트된 로컬 빌드 과정을 정확히 재현합니다:

1. **Ubuntu 22.04 + CUDA 12.1** 베이스 이미지
2. **가상환경 생성** (`python3 -m venv`)
3. **PyTorch + CUDA 설치** (CUDA 12.1 지원)
4. **Triton 빌드 의존성 설치**
5. **로컬 Triton 개발 모드 설치** (`pip install -e .`)
6. **가상환경이 기본 환경으로 설정**

## 🚀 예상 결과

성공적으로 빌드되면 다음과 같은 결과를 얻을 수 있습니다:

```
🎉 ALL 324 ALL MODULES OPERATORS PASSED! 🎉

SUMMARY:
--------
Total Tests:  324
Passed:       324 (100.0%)
Failed:       0 (0.0%)
Errors:       0 (0.0%)
Skipped:      0 (0.0%)

BREAKDOWN BY MODULE:
-------------------
tl              117 tests | 117 passed (100.0%)
libdevice       198 tests | 198 passed (100.0%)
cuda              8 tests |   8 passed (100.0%)
hip               1 tests |   1 passed (100.0%)
```

## ⚠️ 주의사항

1. **Docker 실행 필요**: Docker가 실행 중이어야 합니다
2. **NVIDIA Docker**: CUDA 지원을 위해 NVIDIA Docker runtime이 필요합니다
3. **빌드 시간**: 첫 빌드는 10-30분 정도 소요될 수 있습니다
4. **디스크 공간**: Docker 이미지는 약 5-10GB 정도 필요합니다

## 🛠️ 문제 해결

### Docker가 실행되지 않는 경우
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
# 로그아웃 후 다시 로그인
```

### NVIDIA Docker가 없는 경우
```bash
# NVIDIA Docker 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 빌드 실패 시
```bash
# Docker 캐시 정리
docker system prune -a

# 다시 빌드
./build-docker.sh
```
