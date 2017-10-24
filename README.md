# joint training

Adjust arguments in *joint_training.ini*. To assgin a GPU device, use
```
export CUDA_VISIBLE_DEVICES="0"
```

Start training
```
python execute.py
```

Run a TensorBoard server in a separate process for real-time monitoring of training progress and evaluation metrics.
```
tensorboard --logdir=log_dir/ --port=6364
```
