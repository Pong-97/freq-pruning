运行：

    1. 把raw——model放到code/models/raw_model下
   
    2. CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -u code/prune_train_resnet50.py > 1.log  2>&1 &



配置参数：（在prune_train_resnet50.py中）

    L52: --student_epochs  小epoch

    L57: --save            保存地址

    L299: compress_rate    每一层的剪枝率
