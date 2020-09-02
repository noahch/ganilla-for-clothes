# ganilla-for-clothes
Modification of ganilla project to apply style transfer on clothes

Run test.py
``` bash
python test.py --dataroot .\datasets\recomm --name recomm_cyclegan --model cycle_gan --netG resnet_fpn
```

Run train.py
``` bash
python test.py --dataroot .\datasets\recomm --name recomm_cyclegan --model cycle_gan --netG resnet_fpn --niter 9 --niter_decay 1 --continue_train
```


Run train.py ablation model2
``` bash
python test.py --dataroot .\datasets\recomm --name recomm_cyclegan --model cycle_gan --netG ablation_model2 --niter 9 --niter_decay 1 --continue_train
```
