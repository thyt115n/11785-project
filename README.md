# Instruction for running baseline models


## Environment
We are running our baseline models on AWS EC2 G4 instances with Pytorch 1.6.0. Below are some packages you might need to install for running the baseline models

```bash
python -m pip install cityscapesscripts
pip install mmcv-full
```

## Dataset
The dataset we are currently using is Cityscapes, you can download it with from https://www.cityscapes-dataset.com/ or you can simply run the following commands.

```bash
wget --keep-session-cookies --save-cookies=cookies.txt \
    --post-data 'username=yourusername&password=yourpassword&submit=Login'https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip leftImg8bit_trainvaltest.zip && rm leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip && rm gtFine_trainvaltest.zip
```
The final file structure will be:
```
    .
    ├── ...
    ├── data                    
    │   ├── cityscapes
    │   │   ├── getFine 
    │   │   ├── leftImg8bit
    └── ...
```  
   
## Running

### Train DeepLabv3+ with ResNet50 backbone:
```bash
./tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py 1
```

### Evaluate DeepLabv3+:
```bash
python tools/test.py configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py \
    checkpoints/FileGeneratedInTraining.pth\
    --eval mIoU cityscapes
```
For running DeepLabv3+ with ResNet101 backbone and FCN-8s, you can also run similar commands above.
