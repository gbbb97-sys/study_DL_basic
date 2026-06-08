# 01. 순전파 / 역전파 (Forward Pass / Backpropagation)

---

## 1. 퍼셉트론 (Perceptron)

뉴럴넷의 최소 단위. 입력에 가중치를 곱하고 더한 뒤 활성화 함수를 통과시킨다.
z = w₁x₁ + w₂x₂ + b = Wx + b
a = σ(z)

- `W` : Weight (가중치) — 학습 대상
- `b` : Bias (편향) — 학습 대상
- `σ` : Activation function (비선형성 주입)

---

## 2. 다층 퍼셉트론 (MLP)

퍼셉트론을 여러 층으로 쌓은 것. 은닉층(Hidden layer)이 추가된다.
입력층 → [Linear → Activation] × N → 출력층

2층 MLP 수식:
z¹ = W¹x + b¹        # 선형 변환
a¹ = ReLU(z¹)        # 비선형 변환
z² = W²a¹ + b²       # 선형 변환
ŷ  = z²              # 출력 (회귀면 activation 없이)

---

## 3. 순전파 (Forward Pass)

입력 x가 레이어를 차례로 통과해 예측값 ŷ을 만드는 과정.  
수치해석의 `u^{n+1} = F(u^n)` 시간 업데이트와 구조가 동일하다.

---

## 4. Loss function

예측값 ŷ과 정답 y의 차이를 숫자 하나로 압축.

| Loss | 수식 | 용도 |
|------|------|------|
| MSE | (1/n) Σ(y − ŷ)² | 회귀 (BP 예측 등) |
| MAE | (1/n) Σ\|y − ŷ\| | 회귀, 이상치에 강함 |
| CE  | −Σ y log(ŷ) | 분류 |

---

## 5. 역전파 (Backpropagation)

loss를 줄이기 위해 W, b를 어느 방향으로 얼마나 바꿀지 계산.  
핵심 도구: **chain rule**
∂L/∂W¹ = ∂L/∂ŷ · ∂ŷ/∂a¹ · ∂a¹/∂z¹ · ∂z¹/∂W¹

출력 쪽 loss부터 입력 쪽으로 거꾸로 미분을 전파.  
수치해석의 ODE adjoint method와 수학적으로 동일한 구조.

---

## 6. 파라미터 업데이트 (Gradient Descent)
W ← W - α · ∂L/∂W
b ← b - α · ∂L/∂b

- `α` : Learning rate — RK45의 step size h와 동일한 역할
- 너무 크면 발산, 너무 작으면 수렴이 느림

---

## 7. PyTorch에서의 흐름

```python
y_hat = model(x)        # 순전파
loss = loss_fn(y_hat, y) # loss 계산
optimizer.zero_grad()    # gradient 초기화 (누적 방지)
loss.backward()          # 역전파 (chain rule 자동 계산)
optimizer.step()         # 파라미터 업데이트
```

---

## 실습 파일

- [forward_pass.py](./forward_pass.py) : 2층 MLP 순전파 → 역전파 → 학습 루프
- [backprop_manual.py](./backprop_manual.py) : autograd 없이 gradient 직접 계산