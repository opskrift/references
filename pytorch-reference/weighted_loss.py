import torch as T

batch_size = 32
num_classes = 4

class_weights = T.tensor([0.1, 0.5, 0.2, 0.2])

x = T.rand(batch_size, num_classes)
true = T.randint(low=0, high=num_classes, size=(batch_size,))
pred = T.log(nn.Softmax(dim=1)(x))

# pytorch
loss = nn.NLLLoss(weight=class_weights)
print(f"PyTorch: {loss(pred, true)}")

# manual
l = ws = 0

for b in range(batch_size):
    w = class_weights[true[b]]
    l += -w * pred[b, true[b]]
    ws += w

print(f"manual: {l / ws}")
