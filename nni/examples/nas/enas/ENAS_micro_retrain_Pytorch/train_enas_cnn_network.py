from enas_cnn_network import MicroNetwork
import json
import os
import sys
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn

def resolve_json_file(path):
    file = open(path)
    file_json = json.load(file)
    normal_cell_input = list(file_json.values())[0:20:2]
    normal_cell_op = list(file_json.values())[1:20:2]
    reduce_cell_input = list(file_json.values())[20:40:2]
    reduce_cell_op = list(file_json.values())[21:40:2]
    file_json = [{'cell_input': normal_cell_input,
                  'cell_op': normal_cell_op},
                 {'cell_input': reduce_cell_input,
                  'cell_op': reduce_cell_op}]
    return file_json

def train(net, epoch, f):
    print('\nEpoch: %d' % epoch)
    f.write('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs1, outputs2 = net(inputs)
        loss1 = criterion(outputs1, targets)
        loss2 = criterion(outputs2, targets)
        loss = 0.5 * (loss1 + loss2)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs1.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(batch_idx, 'Acc: %.3f%%' % (100. * correct / total))
            f.write(str(batch_idx) + 'Acc: %.3f%%\n' % (100. * correct / total))


def test(net, epoch, best_acc, f, save_path):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print(batch_idx, 'Acc: %.3f%%' % (100. * correct / total), best_acc)
                f.write(str(batch_idx) + '  Acc: %.3f%%\n' % (100. * correct / total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving A Better Result: acc:' + str(acc) + '*' * 20)
        f.write('Saving A Better Result: acc:' + str(acc) + '*' * 20 + '\n')
        state = net.state_dict(),
        if not os.path.isdir('checkpoint_pth'):
            os.mkdir('checkpoint_pth')
        torch.save(state, save_path)
        best_acc = acc
    return best_acc


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    lr = 0.1
    batch_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('加载数据')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    if not os.path.isdir('logs'):
        os.mkdir('logs')

    acc_f = open('./logs/acc_f.txt', 'w+')
    for i in range(1, 150):
        best_acc = 0
        f = open("./logs/log_{0}.txt".format(i), "w+")
        f.write('开始训练\n')
        path = os.getcwd() + '/checkpoint/epoch_{0}.json'.format(i)
        file_json = resolve_json_file(path)
        print('建立模型{0}.....'.format(i))
        net = MicroNetwork(num_layers=6, out_channels=20, num_nodes=5, dropout_rate=0.1, use_aux_heads=True,
                           file_json=file_json)
        save_path = os.getcwd() + '/checkpoint_pth/net_{0}_ckpt.pth'.format(i)
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        milestones = [150, 225]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        for epoch in range(300):
            train(net, epoch, f)
            best_acc = test(net, epoch, best_acc, f, save_path)
            scheduler.step()
        print("net {0}: best accuracy = {1}%%".format(i, best_acc))
        f.write("net {0}: best accuracy = {1}%%\n".format(i, best_acc))
        acc_f.write("net {0}: best accuracy = {1}%%\n".format(i, best_acc))
        f.close()
    acc_f.close()
