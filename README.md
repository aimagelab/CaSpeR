
Repository of [Latent Spectral Regularization for Continual Learning](https://www.sciencedirect.com/science/article/pii/S0167865524001909)


## Run Experiments
```
python utils/main.py --model <model_name> --lr <lr> [...additional args]
```

**Available models**

+ `sgd` (finetune)
+ `joint`
+ `er_ace`
+ `icarl`
+ `derpp`
+ `podnet`
+ `xder_rpc`
