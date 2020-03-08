import argparse
import torch
import torchvision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--sources', type=str)
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            "ROI_data/ALL", transform=torchvision.transforms.ToTensor()),
        batch_size=6)

    print('Computing mean and std ...')
    mean = 0.
    std = 0.
    n_samples = 0.
    for data, label in train_loader:
        batch_size = data.size(0)
        data = data.view(batch_size, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        n_samples += batch_size

    mean /= n_samples
    std /= n_samples
    print('Mean: {}'.format(mean))
    print('Std: {}'.format(std))


if __name__ == '__main__':
    main()
