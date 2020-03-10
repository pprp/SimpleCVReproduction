import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils import data
from tqdm import tqdm


from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    #数据
    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = t_loader.n_classes

    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        #shuffle=True,
        shuffle=False,
    )
    valloader = data.DataLoader(
        v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    print("Num of classes:", n_classes)

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    #模型
    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    #优化
    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    #optimizer = torch.optim.Adam(model.parameters(), lr = cfg["training"]["optimizer"]["lr"])
    #weight_decay = cfg["training"]["optimizer"]["weight_decay"]
    
    logger.info("Using optimizer {}".format(optimizer))
    print("Using optimizer: {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])
    #损失函数
    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))
    print("Using loss: {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    #loyo = 1
    while i <= cfg["training"]["train_iters"] and flag:
        #loyo += 1
        #print("119train!!!!!!!!!!!!!!", loyo)
        for (images, labels) in trainloader:
            i += 1
            #print("124traini:", i)
            start_ts = time.time()
            optimizer.step()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

             #输出训练模型图片
            '''
            print("开始输出模型图片")
            plt.ion()
            with torch.no_grad():
                for x, _ in trainloader:
                    x = x.to(device)
                    y = model(x)
                    img_y = torch.squeeze(y).numpy()
                    plt.imshow(img_y)
                    plt.savefig("./results/output_%d.jpg" % random.randint(0, 100))
                    plt.pause(0.01)
                plt.show()
            print("模型图片输出完毕")
            '''


            optimizer.zero_grad()
            outputs = model(images)


            # print("outputs:", outputs.shape, "\n labels:", labels.shape)
            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            #报错说optimizer.step()应该在scheduler.step()前面
            #optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"
            ]:
                model.eval()


                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                #print("train190score:", score)
                #print("train190class_io:", class_iou)
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("class{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)
            

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/unet.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(3700, 3799)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")
    print("Let the games begin")

    train(cfg, writer, logger)
