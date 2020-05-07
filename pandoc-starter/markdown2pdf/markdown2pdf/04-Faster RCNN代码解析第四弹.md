# 1. 前言
经过前面三节，我们已经大概上讲清楚了如何构造一个完整的Faster RCNN模型以及里面的代码实现细节，这一节呢主要来解析一下工程中更外围一点的东西，即`train.py`和`trainer.py`，这将教会我们如何使用已经搭建好的Faster RCNN网络。解析代码地址为：https://github.com/BBuf/simple-faster-rcnn-explain 。

# 2. 回顾
首先从**三年一梦这个博主的博客**里面看到了一张对Faster RCNN全过程总结的图，地址为：https://www.cnblogs.com/king-lps/p/8995412.html 。它是针对Chainner实现的一个Faster RCNN工程所做的流程图，但我研究了一下过程和本文介绍的陈云大佬的代码流程完全一致，所以在这里贴一下这张图，再熟悉一下Faster RCNN的整体流程。

![Faster RCNN 整体流程图](https://img-blog.csdnimg.cn/20200504204915246.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

这张图把整个Faster RCNN的流程都解释的比较清楚，注意一下图中出现的`Conv(512，512，3，1，1)`类似的语句里面的最后一个参数表示`padding`。



# 3. 代码解析
这一节我们主要是对`train.py`和`trainer.py`的代码进行解析，我们首先来看`trainer.py`，这个脚本定义了类**FasterRCNNTrainer** ，在初始化的时候用到了之前定义的类**FasterRCNNVGG16** 为`faster_rcnn`。  此外在初始化中有引入了其他`creator、vis、optimizer`等。

另外，还定义了四个损失函数以及一个总的联合损失函数：`rpn_loc_loss`、`rpn_cls_loss`、`roi_loc_loss`、`roi_cls_loss`,`total_loss`。

首先来看一下**FasterRCNNTrainer**类的初始化函数：

```python
class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn):
        # 继承父模块的初始化
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        # 下面2个参数是在_faster_rcnn_loc_loss调用用来计算位置损失函数用到的超参数
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        # 用于从20000个候选anchor中产生256个anchor进行二分类和位置回归，也就是
        # 为rpn网络产生的预测位置和预测类别提供真正的ground_truth标准
        self.anchor_target_creator = AnchorTargetCreator()
        # AnchorTargetCreator和ProposalTargetCreator是为了生成训练的目标
        # （或称ground truth），只在训练阶段用到，ProposalCreator是RPN为Fast
        #  R-CNN生成RoIs，在训练和测试阶段都会用到。所以测试阶段直接输进来300
        # 个RoIs，而训练阶段会有AnchorTargetCreator的再次干预
        self.proposal_target_creator = ProposalTargetCreator()
        # (0., 0., 0., 0.)
        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        # (0.1, 0.1, 0.2, 0.2)
        self.loc_normalize_std = faster_rcnn.loc_normalize_std
        # SGD
        self.optimizer = self.faster_rcnn.get_optimizer()
        # 可视化，vis_tool.py
        self.vis = Visualizer(env=opt.env)

        # 混淆矩阵，就是验证预测值与真实值精确度的矩阵ConfusionMeter
        # (2)括号里的参数指的是类别数
        self.rpn_cm = ConfusionMeter(2)
        # roi的类别有21种（20个object类+1个background）
        self.roi_cm = ConfusionMeter(21)
        # 平均损失
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

```

接下来是`Forward`函数，因为只支持batch_size等于1的训练，因此n=1。每个batch输入一张图片，一张图片上所有的bbox及label，以及图片经过预处理后的scale。

然后对于两个分类损失（RPN和ROI Head）都使用了交叉熵损失，而回归损失则使用了`smooth_l1_loss`。

还需要注意的一点是例如ROI回归输出的是$128\times 84$，然而真实位置参数是$128\times 4$和真实标签$128\times 1$，我们需要利用真实标签将回归输出索引为$128\times 4$，然后在计算过程中只计算**前景类的回归损失**。具体实现与Fast-RCNN略有不同（$\sigma$设置不同）。

代码解析如下：


```python
def forward(self, imgs, bboxes, labels, scale):
        # 获取batch个数
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        # （n,c,hh,ww）
        img_size = (H, W)

        # vgg16 conv5_3之前的部分提取图片的特征
        features = self.faster_rcnn.extractor(imgs)

        # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2），
        #  rois的维度为（2000,4），roi_indices用不到，anchor的维度为
        # （hh*ww*9，4），H和W是经过数据预处理后的。计算（H/16）x(W/16)x9
        # (大概20000)个anchor属于前景的概率，取前12000个并经过NMS得到2000个
        # 近似目标框G^的坐标。roi的维度为(2000,4)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        # bbox维度(N, R, 4)
        bbox = bboxes[0]
        # labels维度为（N，R）
        label = labels[0]
        #hh*ww*9
        rpn_score = rpn_scores[0]
        # hh*ww*9
        rpn_loc = rpn_locs[0]
        # (2000,4)
        roi = rois

        # Sample RoIs and forward
        # 调用proposal_target_creator函数生成sample roi（128，4）、
        # gt_roi_loc（128，4）、gt_roi_label（128，1），RoIHead网络
        # 利用这sample_roi+featue为输入，输出是分类（21类）和回归
        # （进一步微调bbox）的预测值，那么分类回归的groud truth就
        # 是ProposalTargetCreator输出的gt_roi_label和gt_roi_loc。

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        # roi回归输出的是128*84和128*21，然而真实位置参数是128*4和真实标签128*1
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        # 输入20000个anchor和bbox，调用anchor_target_creator函数得到
        # 2000个anchor与bbox的偏移量与label
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        # 下面分析_fast_rcnn_loc_loss函数。rpn_loc为rpn网络回归出来的偏移量
        # （20000个），gt_rpn_loc为anchor_target_creator函数得到2000个anchor
        # 与bbox的偏移量，rpn_sigma=1.
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        # rpn_score为rpn网络得到的（20000个）与anchor_target_creator
        # 得到的2000个label求交叉熵损失
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1] #不计算背景类
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        # roi_cls_loc为VGG16RoIHead的输出（128*84）， n_sample=128
        n_sample = roi_cls_loc.shape[0]
        # roi_cls_loc=（128,21,4）
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        # proposal_target_creator()生成的128个proposal与bbox求得的偏移量
        # dx,dy,dw,dh
        gt_roi_label = at.totensor(gt_roi_label).long()
        # 128个标签
        gt_roi_loc = at.totensor(gt_roi_loc)
        # 采用smooth_l1_loss
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)
        # 求交叉熵损失
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())
        # 四个loss加起来
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)
```

下面我们来解析一下代码中的`_fast_rcnn_loc_loss`函数，它用到了smooth_l1_loss。其中`in_weight`代表权重，只将那些不是背景的Anchor/ROIs的位置放入到损失函数的计算中来，方法就是只给不是背景的Anchor/ROIs的`in_weight`设置为1，这样就可以完成`loc_loss`的求和计算。

代码解析如下：

```python
# 输入分别为rpn回归框的偏移量和anchor与bbox的偏移量以及label
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    # sigma设置为1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    # 除去背景类
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
```

接下来就是`train_step`函数，整个函数实际上就是进行了一次参数的优化过程，首先`self.optimizer.zero_grad()`将梯度数据全部清零，然后利用刚刚介绍`self.forward(imgs,bboxes,labels,scales)`函数将所有的损失计算出来，接着依次进行`losses.total_loss.backward()`反向传播计算梯度，`self.optimizer.step()`进行一次参数更新过程，`self.update_meters(losses)`就是将所有损失的数据更新到可视化界面上,最后将`losses`返回。代码如下：

```python
def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses
```

接下来还有一些函数比如`save()`，`load()`，`update_meters()`，`reset_meters()`，`get_meter_data()`等。其中`save()`和`load()`就是根据输入参数来选择保存和解析`model`模型或者`config`设置或者`other_info`其他`vis_info`可视化参数等等，代码如下：

```python
# 模型保存
    def save(self, save_optimizer=False, save_path=None, **kwargs):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path
    # 模型加载
    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self
```

而`update_meters`,`reset_meters`以及`get_meter_data()`就是负责将数据向可视化界面更新传输获取以及重置的函数。

OK，`trainer.py`大概就解析到这里，接下来我们来看看`train.py`，详细解释如下：

```python
def train(**kwargs):
    # opt._parse(kwargs)#将调用函数时候附加的参数用，
    # config.py文件里面的opt._parse()进行解释，然后
    # 获取其数据存储的路径，之后放到Dataset里面！
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    # #Dataset完成的任务见第二次推文数据预处理部分，
    # 这里简单解释一下，就是用VOCBboxDataset作为数据
    # 集，然后依次从样例数据库中读取图片出来，还调用了
    # Transform(object)函数，完成图像的调整和随机翻转工作
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    # 将数据装载到dataloader中，shuffle=True允许数据打乱排序，
    # num_workers是设置数据分为几批处理，同样的将测试数据集也
    # 进行同样的处理，然后装载到test_dataloader中
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    # 定义faster_rcnn=FasterRCNNVGG16()训练模型
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')

    # 设置trainer = FasterRCNNTrainer(faster_rcnn).cuda()将
    # FasterRCNNVGG16作为fasterrcnn的模型送入到FasterRCNNTrainer
    # 中并设置好GPU加速
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    # 用一个for循环开始训练过程，而训练迭代的次数
    # opt.epoch=14也在config.py文件中预先定义好，属于超参数
    for epoch in range(opt.epoch):
        # 首先在可视化界面重设所有数据
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            # 然后从训练数据中枚举dataloader,设置好缩放范围，
            # 将img,bbox,label,scale全部设置为可gpu加速
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # 调用trainer.py中的函数trainer.train_step
            # (img,bbox,label,scale)进行一次参数迭代优化过程
            trainer.train_step(img, bbox, label, scale)

            # 判断数据读取次数是否能够整除plot_every
            # (是否达到了画图次数)，如果达到判断debug_file是否存在，
            # 用ipdb工具设置断点，调用trainer中的trainer.vis.
            # plot_many(trainer.get_meter_data())将训练数据读取并
            # 上传完成可视化
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                # 将每次迭代读取的图片用dataset文件里面的inverse_normalize()
                # 函数进行预处理，将处理后的图片调用Visdom_bbox可视化 
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                # 调用faster_rcnn的predict函数进行预测，
                # 预测的结果保留在以_下划线开头的对象里面
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                # 利用同样的方法将原始图片以及边框类别的
                # 预测结果同样在可视化工具中显示出来
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                # 调用trainer.vis.text将rpn_cm也就是
                # RPN网络的混淆矩阵在可视化工具中显示出来
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                # 可视化ROI head的混淆矩阵
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        # 调用eval函数计算map等指标
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        # 可视化map
        trainer.vis.plot('test_map', eval_result['map'])
        # 设置学习的learning rate
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        # 将损失学习率以及map等信息及时显示更新
        trainer.vis.log(log_info)
        # 用if判断语句永远保存效果最好的map
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            # if判断语句如果学习的epoch达到了9就将学习率*0.1
            # 变成原来的十分之一
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay
        # 判断epoch==13结束训练验证过程
        if epoch == 13: 
            break
```

在`train.py`里面还有一个函数为`eval()`，具体解释如下：

```python
def eval(dataloader, faster_rcnn, test_num=10000):
    # 预测框的位置，预测框的类别和分数
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    # 真实框的位置，类别，是否为明显目标
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    # 一个for循环，从 enumerate(dataloader)里面依次读取数据，
    # 读取的内容是: imgs图片，sizes尺寸，gt_boxes真实框的位置
    #  gt_labels真实框的类别以及gt_difficults
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        # 用faster_rcnn.predict(imgs,[sizes]) 得出预测的pred_boxes_,
        # pred_labels_,pred_scores_预测框位置，预测框标记以及预测框
        # 的分数等等
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break
    # 将pred_bbox,pred_label,pred_score ,gt_bbox,gt_label,gt_difficult
    # 预测和真实的值全部依次添加到开始定义好的列表里面去，如果迭代次数等于测
    # 试test_num，那么就跳出循环！调用 eval_detection_voc函数，接收上述的
    # 六个列表参数，完成预测水平的评估！得到预测的结果
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result
```


关于如何计算map我就不再赘述了，感兴趣可以去看我这篇推文，自认为写的是很清楚的，也有源码解释：[目标检测算法之常见评价指标(mAP)的详细计算方法及代码解析](https://mp.weixin.qq.com/s/6D1OcHXJC2uJ7CLXnZ0wJQ) 。



# 4. 总结
今天是5/5号，也是五一的最后一天假期，算是完成了对Faster RCNN代码的全部解读，另外不久后我也修改一些内容并将整理一个PDF版本（包括NMS和mAP的计算也准备放到PDF里），并且目前所有的代码注释都放在了这个github工程：https://github.com/BBuf/simple-faster-rcnn-explain 。

# 5. 参考
- https://blog.csdn.net/qq_32678471/article/details/85678921
- https://www.cnblogs.com/king-lps/p/8995412.html

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)