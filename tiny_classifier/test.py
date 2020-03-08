import torch
import os
import torchvision

from model import TinyModel
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from classifier import test


if __name__ == "__main__":
    model = TinyModel()
    model.load_state_dict(torch.load("checkpoints/epoch_90_0.955.pt"))

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(1, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_datasets = ImageFolder(os.path.join("ROI_data", "test"),
                                transform=test_trans)

    print(test_datasets.classes)
    test_dataloader = DataLoader(test_datasets, batch_size=32, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    acc = test(model, test_dataloader, criterion)

