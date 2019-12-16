import torch 

@torch.no_grad()
def generate():
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    # for k_, v_ in kwargs.items():
    #     setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')

    netg, netd = NetG(opt).eval(), NetD(opt).eval()

    noises = t.randn(opt.gen_search_num, opt.nz, 1,
                     1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    netd.to(device)
    netg.to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).detach()

    # 挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    tv.utils.save_image(t.stack(result),
                        opt.gen_img,
                        normalize=True,
                        range=(-1, 1))

if __name__ == "__main__":
    generate()