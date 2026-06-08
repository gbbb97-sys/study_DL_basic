import torch
import torch.nn as nn

# =====================
# 순전파
# =====================

# 입력, 정답
x = torch.tensor([[1.0]])         # (1×1)
y = torch.tensor([[1.0]])         # 정답

# 가중치 직접 지정 (손계산이랑 같은 값)
W1 = torch.tensor([[0.5],         # (2×1)
                   [0.3]], requires_grad=True)
b1 = torch.tensor([[0.1],         # (2×1)
                   [0.2]], requires_grad=True)

W2 = torch.tensor([[0.4, 0.6]],   # (1×2)
                  requires_grad=True)
b2 = torch.tensor([[0.0]],        # (1×1)
                  requires_grad=True)

# 순전파 직접 계산
z1 = W1 @ x + b1           # (2×1)
a1 = torch.relu(z1)         # (2×1)
z2 = W2 @ a1 + b2           # (1×1)
y_hat = z2                  # activation 없음

print("=== 순전파 ===")
print(f"z1     = {z1.detach().numpy().flatten()}")
print(f"a1     = {a1.detach().numpy().flatten()}")
print(f"z2     = {z2.item():.4f}")
print(f"y_hat  = {y_hat.item():.4f}")

# =====================
# Loss
# =====================

loss = ((y - y_hat) ** 2)
print(f"\n=== Loss ===")
print(f"L = {loss.item():.4f}")

# =====================
# 역전파
# =====================

loss.backward()

print(f"\n=== 역전파 (gradient) ===")
print(f"∂L/∂W2 = {W2.grad.numpy().flatten()}")
print(f"∂L/∂b2 = {b2.grad.item():.4f}")
print(f"∂L/∂W1 = {W1.grad.numpy().flatten()}")
print(f"∂L/∂b1 = {b1.grad.numpy().flatten()}")

# =====================
# 파라미터 업데이트
# =====================

lr = 0.1

with torch.no_grad():
    W2 -= lr * W2.grad
    W1 -= lr * W1.grad
    b2 -= lr * b2.grad
    b1 -= lr * b1.grad

print(f"\n=== 업데이트 후 ===")
print(f"W2 = {W2.detach().numpy().flatten()}")
print(f"W1 = {W1.detach().numpy().flatten()}")

# =====================
# 학습 루프 (100번 반복)
# =====================

# 가중치 초기화 (다시 원래값으로)
W1 = torch.tensor([[0.5],
                   [0.3]], requires_grad=True)
b1 = torch.tensor([[0.1],
                   [0.2]], requires_grad=True)
W2 = torch.tensor([[0.4, 0.6]], requires_grad=True)
b2 = torch.tensor([[0.0]], requires_grad=True)

print("\n=== 학습 루프 ===")
for step in range(100):
    # 순전파
    z1 = W1 @ x + b1
    a1 = torch.relu(z1)
    z2 = W2 @ a1 + b2
    y_hat = z2

    # loss
    loss = ((y - y_hat) ** 2)

    # 역전파
    loss.backward()

    # 업데이트
    with torch.no_grad():
        W1 -= lr * W1.grad
        W2 -= lr * W2.grad
        b1 -= lr * b1.grad
        b2 -= lr * b2.grad

    # gradient 초기화
    W1.grad.zero_()
    W2.grad.zero_()
    b1.grad.zero_()
    b2.grad.zero_()

    if step % 10 == 0:
        print(f"step {step:3d} | loss: {loss.item():.6f} | y_hat: {y_hat.item():.4f}")