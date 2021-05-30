gpu=0
CUDA_VISIBLE_DEVICES=${gpu} \
python test_supernet.py \
    --eval_json_path=files/benchmark.json \
    --convbn_type=sample_channel \
    --alpha_type=sample_uniform \
    --model_path=files/supernet.th \
    --save_dir=eval \
    --save_file=eval-final \
    --affine \
    --sameshortcut \