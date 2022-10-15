import torch

from datasets import CubDataset
from models import FineGrainedModel, ResNet18
from torchvision import transforms
from torch.utils.data import DataLoader


train_transforms = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4819, 0.4976, 0.4321),
                             (0.1878, 0.1864, 0.1992))
    ]
)

train_ds = CubDataset(transform=train_transforms, trainable=True)
test_ds = CubDataset(transform=train_transforms, trainable=False)

train_dataloader = DataLoader(
    train_ds, batch_size=3, shuffle=True, collate_fn=train_ds.collect_fn, drop_last=True)
test_dataloader = DataLoader(
    test_ds, batch_size=3, shuffle=False, collate_fn=train_ds.collect_fn)

model = ResNet18()
# model2 = ResNet18()

if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=30,
                                            gamma=0.1)

if __name__ == "__main__":
    EPOCH = 1
    for e in range(EPOCH):
        # train
        model.train()
        print("="*10, "Train %d EPOCH" % e, "="*10)
        for itr, (img, label) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            output = model(img)

            label = torch.squeeze(label).long()

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print("step: %03d | train loss:%.3f" % (itr, loss.item()))

        print("="*10, "Test %d EPOCH" % e, "="*10)
        n_loss = 0.
        n_sample = 0
        n_right = 0

        # model.eval()
        # for iter, (img, label) in enumerate(test_dataloader):
        #     bs = img.shape[0]
        #     output = model(img)

        #     ######################
        #     output = output.topk(1, dim=1)[1].view(bs, 1)
        #     diff = (output != label)
        #     diff = diff.sum(1)
        #     diff = (diff != 0)

        #     res = diff.sum(0).item()
        #     n_right += (bs - res)
        #     ######################

        #     label = torch.squeeze(label).long()

        #     loss = criterion(output, label)
        #     n_sample += bs
        #     n_loss += loss.item()

        # print("step: %03d | Test Loss: %.3f | Acc: %.3f" %
        #       (iter, n_loss/n_sample, n_right * 1./n_sample))
