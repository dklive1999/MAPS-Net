# MAPS-Net
Paper code: Image tampering localization network based on multi-class attention and progressive subtraction
<div align="center">
  <img src="https://github.com/dklive1999/MAPS-Net/blob/main/img/MAPS-Net.jpg">
</div>


# Environment

- Ubuntu18.04
- Python 3.8
- PyTorch  1.9.0 + Cuda  11.1
- Detail python librarys can found in [requirements.txt](./requirements.txt)

# Dataset

### An example of the dataset index file is given as  [datasets/Casiav1.txt](./datasets/Casiav1.txt), where each line contains:

```
 img_path mask_path label
```

- 0 represents the authentic and 1 represents the manipulated.
- For an authentic image, the mask_path is "None".
- For wild images without mask groundtruth, the index should at least contain "img_path" per line.

### For example, each line in txt file should like this:`

  - Authentic image:

    ```
    ./Casiav2/authentic/Au_ani_00001.jpg None None 0
    ```

  - Manipulated image with pre-generated edge mask: 

    ```
    ./Casiav2/tampered/Tp_D_CND_M_N_ani00018_sec00096_00138.tif ./Casiav2/mask/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png ./Casiav2/edge/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png 1
    ```

  - Manipulated image without pre-generated edge mask: 

    ```
    ./Casiav2/tampered/Tp_D_CND_M_N_ani00018_sec00096_00138.tif ./Casiav2/mask/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png None 1
    ```

  - You should follow the format and generate your own "path file" in a `xxxx.txt`.

> Limits: At this time, the edge mask can only be generated during training and cannot be pre generated.   This will be a little bit slow. Since every Epoch you will generate a edge mask for each image, however, they are always the same edge mask. Better choice should be generate the edge mask from the ground truth mask before start training. Script for pre-generate the edge mask will release later.

## Training sets

- [CASIAv2](./datasets/Casiav2.txt)

## Test sets

- [CASIAv1](./datasets/Casiav1.txt)
- [Columbia](./datasets/Columbia.txt)
- [COVERAGE](./datasets/COVERAGE.txt)
- [nist16](./datasets/nist16.txt)

# Training

Please set the train image path in [train_base.py](./train_base.py), then run [train_lanch.py](./train_launch.py) with Python.

# Evaluation

Please set the test image path in [inference.py](./inference.py) and  run [inference.py](./inference.py) with Python, then run [evaluate.py](./evaluate.py) with Python.
