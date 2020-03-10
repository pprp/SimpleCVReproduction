'''
python test.py [-h] [--model_path [MODEL_PATH]] [--dataset [DATASET]]
               [--dcrf [DCRF]] [--img_path [IMG_PATH]] [--out_path [OUT_PATH]]
 
  --model_path          Path to the saved model
  --dataset             Dataset to use ['pascal, camvid, ade20k etc']
  --dcrf                Enable DenseCRF based post-processing
  --img_path            Path of the input image
  --out_path            Path of the output segmap

python test.py --model_path runs/unet_pascal/3744/unet_pascal_best_model.pkl --dataset pascal --img_path datasets/GRAPE2020/JPEGImages/139_96.png --out_path datasets/GRAPE2020/modelresults/139_96_model.png
''' 
import os
import torch
import argparse
import numpy as np
import scipy.misc as misc
import imageio
from PIL import Image


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )


def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #以下两句结果:model_name=unet
    model_file_name = os.path.split(args.model_path)[1]  #去掉路径,返回文件名
    model_name = model_file_name[: model_file_name.find("_")] #从unet_pascal_best_model.pkl提取unet，即model_name

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    #img = misc.imread(args.img_path) #s输入的是图片
    img = imageio.imread(args.img_path) #替代方案

    data_loader = get_loader(args.dataset)
    loader = data_loader(root=None, is_transform=True, img_norm=args.img_norm, test_mode=True)
    n_classes = loader.n_classes

    #resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp="bicubic") 
    resized_img = np.array(Image.fromarray(img).resize((loader.img_size[0], loader.img_size[1])))

    orig_size = img.shape[:-1]
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        # uint8 with RGB mode, resize width and height which are odd numbers
        #img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
        img = np.array(Image.fromarray(img).resize((orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1)))
    else:
        #img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))  #调整图片大小
        img = np.array(Image.fromarray(img).resize((loader.img_size[0], loader.img_size[1])))

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=args.dataset)
    #将保存在pkl文件中的参数放回unet网络中
    state = convert_state_dict(torch.load(args.model_path)["model_state"]) 
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = img.to(device)
    outputs = model(images)
    print("shape of output:", outputs.shape)

    if args.dcrf:
        unary = outputs.data.cpu().numpy()
        unary = np.squeeze(unary, 0)
        unary = -np.log(unary)
        unary = unary.transpose(2, 1, 0)
        w, h, c = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        unary = np.ascontiguousarray(unary)

        resized_img = np.ascontiguousarray(resized_img)

        d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        q = d.inference(50)
        mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0) #每列最大值
        decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        dcrf_path = args.out_path[:-4] + "_drf.png"  #平滑后的图片
        #misc.imsave(dcrf_path, decoded_crf)
        imageio.imsave(dcrf_path, decoded_crf)
        print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    print("pre shape:", pred.shape)
    '''
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        pred = pred.astype(np.float32)
        # float32 with F mode, resize back to orig_size
        #pred = misc.imresize(pred, orig_size, "nearest", mode="F")
        pred = np.array(Image.fromarray(pred))
    '''

    decoded = loader.decode_segmap(pred,plot=True)
    # print("test.py_123_decoded:", decoded.shape)  #[512, 512]
    # print("Classes found: ", np.unique(pred))  #[0 1]
    #misc.imsave(args.out_path, decoded)
    imageio.imsave(args.out_path, decoded)
    print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="unet_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
        #不是路径，是图片
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output segmap"
        #不是路径，是图片
    )
    args = parser.parse_args()
    test(args)
