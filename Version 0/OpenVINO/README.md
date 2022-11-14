```
# line 62
exec_net = ie.load_network(net, "CPU")
```

Three processing units can be used for OpenVINO inference:
- "CPU"
> if you have Intel CPU
- "GPU"
> if you have Intel GPU
- "MYRIAD"
> if you have Intel VPU eg. Intel Neural Compute Stick 2
  
<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/76240694/201554750-1534a000-4126-414b-84ab-ce8777e69fe6.png">
</p>
<p align="center">
Intel Neural Compute Stick 2 (Intel NCS2)
  </p>
