# baseline 实验

## 运行
```bash
python train.py --config config.yaml --seed 42 --device 0
```

## 评测
```bash
python eval.py --config config.yaml --ckpt runs/seed_42/<run>/ckpts/best.pt
```


