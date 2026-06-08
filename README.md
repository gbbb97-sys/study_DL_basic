# study_DL_basic

수치해석 백그라운드 기반 딥러닝 기초 학습 정리  
PyTorch 2.11.0 · CUDA 12.8 · WSL2 Ubuntu

---

## 개념 발전 순서 + 약자

### 1. 퍼셉트론 (Perceptron) → [세부](./01_forward_backward/README.md)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| Perceptron | — | 입력 → 가중합 → 0 or 1 출력. 딥러닝의 시작점 |
| W, b | Weight, Bias | 학습 대상 파라미터 |

---

### 2. 다층 퍼셉트론 (MLP) → [세부](./01_forward_backward/README.md)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| MLP | Multi-Layer Perceptron | 은닉층 추가한 신경망 |
| FC / Linear | Fully Connected | 모든 뉴런이 연결된 층 |
| Hidden layer | — | 입력·출력 사이의 중간 층 |
| ReLU / Sigmoid | Activation function | 비선형성 주입. flux limiter와 같은 역할 |

---

### 3. 순전파 / 역전파 (FP / BP) → [세부](./01_forward_backward/README.md)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| FP | Forward Pass | 입력 → 레이어 통과 → 예측값 ŷ |
| BP | Backpropagation | chain rule로 ∂L/∂W 계산. ODE adjoint와 동일 구조 |
| Autograd | Automatic Differentiation | PyTorch가 BP 자동 계산해주는 엔진 |
| ∂L/∂W | Gradient | loss의 W에 대한 편미분. "어느 방향으로 움직일까" |

---

### 4. Loss function + Optimizer → [세부](./02_loss_optimizer/README.md)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| MSE | Mean Squared Error | 회귀용. (y−ŷ)² 평균 → BP 예측에 사용 |
| MAE | Mean Absolute Error | \|y−ŷ\| 평균. 이상치에 덜 민감 |
| CE / BCE | Cross Entropy / Binary CE | 분류용. −Σ y log(ŷ) |
| GD / SGD | Gradient Descent / Stochastic GD | W ← W − α·∂L/∂W |
| LR (α) | Learning Rate | 업데이트 보폭. RK45의 step size h |
| Adam | Adaptive Moment Estimation | 가장 많이 쓰는 optimizer. lr 자동 조절 |

---

### 5. 정규화 & 일반화 → [세부](./02_loss_optimizer/README.md)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| BN | Batch Normalization | 미니배치 단위 activation 정규화 |
| LN | Layer Normalization | 샘플 단위 정규화. Transformer에서 사용 |
| DO | Dropout | 뉴런 랜덤 비활성화. overfitting 방지 |
| WD / L2 | Weight Decay | 파라미터 크기 억제. L2 정규화 |
| BS / EP | Batch Size / Epoch | 미니배치 크기 / 전체 데이터 반복 횟수 |

---

### 6. 아키텍처 선택 (데이터 종류에 따라)

#### 6a. CNN — 이미지 · 공간 패턴 → [세부](./03_architectures/cnn/README.md)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| CNN | Convolutional Neural Network | 공간 패턴 추출 |
| Conv | Convolution layer | 필터로 특징 추출 |
| Pool | Pooling layer | 공간 크기 축소 |
| ResNet | Residual Network | skip connection으로 깊은 CNN 가능 |

#### 6b. RNN / LSTM — 시계열 · 순서 데이터 → [세부](./03_architectures/rnn_lstm/README.md)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| RNN | Recurrent Neural Network | 순서 데이터 처리 |
| LSTM | Long Short-Term Memory | RNN 개선판. 장기 의존성 학습 가능 |
| GRU | Gated Recurrent Unit | LSTM 경량화 버전 |
| BPTT | Backpropagation Through Time | RNN의 역전파 방식 |

#### 6c. Transformer — 범용 → [세부](./03_architectures/transformer/README.md)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| Attn | Attention mechanism | 토큰 간 가중 보간 |
| MHA | Multi-Head Attention | Attention을 여러 관점으로 병렬 수행 |
| PE | Positional Encoding | 토큰 순서 정보 주입 |
| FFN | Feed-Forward Network | Transformer 내부 MLP 블록 |

---

### 7. 학습 실전 & 디버깅 → [세부](./04_debugging/README.md)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| Overfit | Overfitting | 훈련 데이터만 잘 맞고 새 데이터에 실패 |
| LR schedule | Learning Rate Scheduler | 학습 진행에 따라 lr 자동 감소 |
| CV / LOSO | Cross Validation / Leave-One-Subject-Out | 모델 성능 신뢰성 검증 |
| W&B | Weights & Biases | 학습 로그·시각화 툴 |

---

### 8. 전이학습 (TL) / 파인튜닝 (FT)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| TL | Transfer Learning | 사전학습 모델을 새 태스크에 재활용 |
| FT | Fine-Tuning | 사전학습 모델을 소량 데이터로 추가 학습 |
| Pretrained | Pretrained model | 대규모 데이터로 미리 학습된 모델 |

---

### 9. PINN — 최종 목표 (기범 직결)

| 약자 | 풀네임 | 설명 |
|------|--------|------|
| PINN | Physics-Informed Neural Network | ODE/PDE residual을 loss에 포함 |
| ODE / PDE | Ordinary / Partial Differential Equation | Windkessel(ODE) · SWE 논문(PDE) |
| LPM | Lumped Parameter Model | 혈관 → RC 회로 모델 |
| L_data / L_E | Data loss / Physics loss | PMB-NN의 L_tot = L_data + L_E1 + L_E2 |

---

## 수치해석 → 딥러닝 연결 고리

| 수치해석 (내 것) | 딥러닝 (새 것) |
|-----------------|----------------|
| RK45 step size h | Learning rate α |
| MUSCL 가중 보간 | Attention 메커니즘 |
| Adaptive Moving Mesh | Adam optimizer |
| Flux limiter (TVD) | Activation function |
| PDE residual 최소화 | PINN physics loss |

---

## 현재 진행 상황

- [x] 환경 세팅 (WSL2, venv, PyTorch 2.11.0+cu128)
- [x] 폴더 구조 생성
- [x] README.md 작성
- [ ] 01_forward_backward/forward_pass.py
- [ ] 01_forward_backward/backprop_manual.py
- [ ] 02_loss_optimizer/
- [ ] 03_architectures/cnn/
- [ ] 03_architectures/rnn_lstm/
- [ ] 03_architectures/transformer/
- [ ] 04_debugging/