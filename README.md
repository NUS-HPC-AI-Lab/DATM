# DD
## Notation
I use CFG to convey the value of keywords, which are enrolled in /utils/cfg.py

Preset hyper-parameters are contained in /configs. (Most of them are cleaned).

## run the distillation
```
cd distill
python test.py --cfg xxx.yaml
```

## distilled datasets
Backed up here first

https://drive.google.com/drive/folders/1kZlYgiVrmFEz0OUyxnww3II7FBPQe7W0?usp=sharing

## evaluation
Use ZCA and batch size 128

Have written an evaluation program, contained in /distill/evaluation.py
