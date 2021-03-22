import torch

from imagenet_dataset import get_train_dataprovider, get_val_dataprovider
import tqdm

assert torch.cuda.is_available()

train_dataprovider, val_dataprovider = None, None


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, cand, args):
    global train_dataprovider, val_dataprovider

    if train_dataprovider is None:
        use_gpu = False
        train_dataprovider = get_train_dataprovider(
            args.train_batch_size, use_gpu=False, num_workers=8)
        val_dataprovider = get_val_dataprovider(
            args.test_batch_size, use_gpu=False, num_workers=8)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    max_train_iters = args.max_train_iters
    max_test_iters = args.max_test_iters

    print('clear bn statics....')
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)

    print('train bn with training set (BN sanitize) ....')
    model.train()

    for step in tqdm.tqdm(range(max_train_iters)):
        # print('train step: {} total: {}'.format(step,max_train_iters))
        data, target = train_dataprovider.next()
        # print('get data',data.shape)

        target = target.type(torch.LongTensor)

        data, target = data.to(device), target.to(device)

        output = model(data, cand)

        del data, target, output

    top1 = 0
    top5 = 0
    total = 0

    print('starting test....')
    model.eval()

    for step in tqdm.tqdm(range(max_test_iters)):
        # print('test step: {} total: {}'.format(step,max_test_iters))
        data, target = val_dataprovider.next()
        batchsize = data.shape[0]
        # print('get data',data.shape)
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)

        logits = model(data, cand)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))

        # print(prec1.item(),prec5.item())

        top1 += prec1.item() * batchsize
        top5 += prec5.item() * batchsize
        total += batchsize

        del data, target, logits, prec1, prec5

    top1, top5 = top1 / total, top5 / total

    top1, top5 = 1 - top1 / 100, 1 - top5 / 100

    print('top1: {:.2f} top5: {:.2f}'.format(top1 * 100, top5 * 100))

    return top1, top5


def main():
    pass
