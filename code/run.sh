# CUDA_VISIBLE_DEVICES=7 nohup python -u prune_train_resnet.py > /home2/pengyifan/pyf/freq-lite/logs/resnet56/0.24/1/nohup.out 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python -u prune_train_vgg.py > /home2/pengyifan/pyf/freq-lite/logs/vgg16/1.9/1/1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u prune_train_resnet.py > /home2/pengyifan/pyf/freq-lite/logs/resnet56/0.47/1/1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6,7 nohup python -u prune_train_resnet50.py >  /home2/pengyifan/pyf/freq-lite/logs/resnet50/1/1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u prune_train_resnet110.py > /home2/pengyifan/pyf/freq-lite/logs/resnet110/405060/1.log  2>&1 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u prune_train_resnet50.py > /home2/pengyifan/pyf/freq-lite/logs/resnet50/3/1.log  2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_resnet110.py > /home2/pengyifan/pyf/freq-lite/logs/resnet110/405060/finetuned/46_72/1.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python -u trigger.py >/home2/pengyifan/pyf/freq-lite/code/Ablation_expriment/1/only_tail/1.log 2>&1 &