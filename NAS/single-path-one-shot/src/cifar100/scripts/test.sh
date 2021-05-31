gpu=0
CUDA_VISIBLE_DEVICES=${gpu} \
python test_sample.py \
    --eval_json_path=data/benchmark.json \
    --convbn_type=sample_channel \
    --alpha_type=sample_uniform \
    --model_path=weights/2021Y_05M_31D_21H_0208/checkpoint-latest.pth.tar \
    --save_dir=eval \
    --save_file=eval-final \
    --affine \
    --sameshortcut 