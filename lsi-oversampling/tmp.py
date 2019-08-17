import torch
import torchvision

from lsi_torch import lsi_torch

import matplotlib.pyplot as plt

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./datasets/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=128,
    shuffle=True,
)

x, y = next(iter(test_loader))

x_s, y_s = lsi_torch(x, y, params={"p": 1.2})

# print(x_s.shape)
# print(y)
val, idx = torch.max(y_s[-1], 0)
print(val, idx)
print(y_s[-1])
plt.imshow(x_s[-1, 0, :, :], cmap="Greys")
plt.show()
