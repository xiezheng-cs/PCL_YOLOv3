#!/usr/bin/env bash
############################################ full precision inference ##################################################

# run inference on example images in data/images:
python detect.py --source data/images --weights yolov3-tiny.pt --conf 0.25

############################################ full precision training ###################################################

# gpu023:exp  双倍bs 双倍lr
CUDA_VISIBLE_DEVICES=6,7 python train.py --data coco.yaml --cfg yolov3-tiny.yaml --weights '' --hyp data/hyp.scratch_doubleLR.yaml --batch-size 128
# resume
CUDA_VISIBLE_DEVICES=6,7 python train.py --data coco.yaml --cfg yolov3-tiny.yaml --weights '' --hyp data/hyp.scratch_doubleLR.yaml --batch-size 128 --resume runs/train/exp/weights/last.pt

# gpu023:exp2 正常bs 正常lr
CUDA_VISIBLE_DEVICES=4,5 python train.py --data coco.yaml --cfg yolov3-tiny.yaml --weights '' --batch-size 64
# resume
CUDA_VISIBLE_DEVICES=4,5 python train.py --data coco.yaml --cfg yolov3-tiny.yaml --weights '' --batch-size 64 --resume runs/train/exp2/weights/last.pt

############################################ training aware quantization ################################################

# gpu023:exp3   8bit模拟量化  单卡
CUDA_VISIBLE_DEVICES=5 python train_quantized.py --data coco.yaml --cfg yolov3-tiny.yaml --weights 'yolov3-tiny.pt' --batch-size 32 --quantization
# resume
CUDA_VISIBLE_DEVICES=0 python train_quantized.py --data coco.yaml --cfg yolov3-tiny.yaml --weights 'yolov3-tiny.pt' --batch-size 32 --quantization --resume runs/train/exp3/weights/last.pt

# gpu023:exp4   6bit模拟量化  单卡
CUDA_VISIBLE_DEVICES=2 python train_quantized.py --data coco.yaml --cfg yolov3-tiny.yaml --weights 'yolov3-tiny.pt' --batch-size 64 --quantization --quantization_bits 6

# gpu023:exp8   6bit 多卡并行模拟量化   使用了陈总的代码，自动选择端口   scale_bits=10
python -m tool.launch --gpus 4,0 train_quantized.py --data coco.yaml --cfg yolov3-tiny.yaml --weights 'yolov3-tiny.pt'  --quantization --sync-bn --batch-size 64 --quantization_bits 6 --scale_bits 10

# gpu023:exp6   6bit 多卡并行模拟量化   使用了陈总的代码，自动选择端口   scale_bits=12
python -m tool.launch --gpus 2,5 train_quantized.py --data coco.yaml --cfg yolov3-tiny.yaml --weights 'yolov3-tiny.pt'  --quantization --sync-bn --batch-size 64 --quantization_bits 6 --scale_bits 12

# gpu023:xxx   8bit 多卡并行模拟量化
python -m tool.launch --gpus 2,4 train_quantized.py --data coco.yaml --cfg yolov3-tiny.yaml --weights 'yolov3-tiny.pt'  --quantization --sync-bn --batch-size 64 --quantization_bits 8 --scale_bits 12

# 8bit量化 scale_bits最少需要12
# 6bit量化 scale_bits最少需要10
# 4bit量化 scale_bits最少需要8


################################################　需要鹏城志鹏跑的baseline #####################################################
# 全精度 双卡 100epoch
python -m tool.launch --gpus 0,1 train_quantized.py --data coco.yaml --cfg yolov3-tiny.yaml --weights '' --sync-bn --batch-size 64 --epochs 100

# 全精度 双卡 300epoch
python -m tool.launch --gpus 0,1 train_quantized.py --data coco.yaml --cfg yolov3-tiny.yaml --weights '' --sync-bn --batch-size 64 --epochs 300