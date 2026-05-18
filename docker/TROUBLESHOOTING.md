# Docker 빌드 문제 해결 가이드

## 🚨 현재 발생한 문제들

### 1. NVIDIA CUDA 이미지를 찾을 수 없음
```
ERROR: nvidia/cuda:12.1-devel-ubuntu22.04: not found
```

**해결 방법:**
- Dockerfile을 CUDA 11.8 버전으로 수정했습니다
- 더 안정적이고 널리 사용되는 CUDA 11.8 이미지를 사용합니다

### 2. Docker 권한 문제
```
permission denied while trying to connect to the Docker daemon socket
```

**해결 방법:**
```bash
# 1. Docker 서비스 시작
sudo systemctl start docker

# 2. 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# 3. 로그아웃 후 다시 로그인하거나 다음 명령어 실행
newgrp docker

# 4. Docker가 정상 작동하는지 확인
docker --version
docker run hello-world
```

## 🔧 수정된 Dockerfile

현재 Dockerfile은 다음 설정을 사용합니다:
- **Base Image**: `nvidia/cuda:11.8-devel-ubuntu22.04`
- **PyTorch**: CUDA 11.8 지원 버전
- **Ubuntu**: 22.04 LTS

## 🚀 빌드 시도

Docker 권한 문제를 해결한 후:

```bash
# 1. Docker 서비스 시작
sudo systemctl start docker

# 2. 사용자를 docker 그룹에 추가 (한 번만 실행)
sudo usermod -aG docker $USER

# 3. 새 그룹 권한 적용
newgrp docker

# 4. Docker 빌드 시도
./docker-build.sh
```

## 🔍 대안 방법들

### 방법 1: sudo로 실행
```bash
sudo ./docker-build.sh
```

### 방법 2: Docker Compose 사용
```bash
sudo docker-compose -f docker/docker-compose.yml build
```

### 방법 3: 직접 Docker 명령어 사용
```bash
sudo docker build -t triton-local-build:latest -f docker/Dockerfile .
```

## ⚠️ 주의사항

1. **CUDA 버전 호환성**: CUDA 11.8은 대부분의 GPU에서 지원됩니다
2. **PyTorch 호환성**: CUDA 11.8용 PyTorch를 사용합니다
3. **빌드 시간**: 첫 빌드는 10-30분 정도 소요될 수 있습니다
4. **디스크 공간**: 최소 5GB의 여유 공간이 필요합니다

## 🆘 추가 도움이 필요한 경우

1. **Docker 설치 확인**:
   ```bash
   docker --version
   docker-compose --version
   ```

2. **NVIDIA Docker 확인**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

3. **시스템 로그 확인**:
   ```bash
   sudo journalctl -u docker.service
   ```
