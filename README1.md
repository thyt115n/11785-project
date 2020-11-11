# Instruction for running baseline models


## Environment
We are running our baseline models on AWS EC2 G4 instances with Pytorch 1.6.0.

## Dataset
The dataset we are currently using is Cityscapes, you can download it with from https://www.cityscapes-dataset.com/ or you can simply run the following commands.

```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=yourusername&password=yourpassword&submit=Login' https://www.cityscapes-dataset.com/login/
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

### Train FCN-8s with ResNet50 backbone:
```bash
./tools/dist_train.sh configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py 1
```

### Evaluate FCN-8s with ResNet50 backbone with the .pth file generated from training:
```bash
python tools/test.py configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py \
    checkpoints/FileGeneratedInTraining.pth\
    --eval mIoU cityscapes
```
For running FCN-8s with ResNet101 backbone, you can also run similar commands above for fcn_r101-d8_512x1024_40k_cityscapes.py.

## FCN-8s Model Structure
ResNetV1c(backbone) --> MaxPool2d --> ResLayer * 4 --> FCNHead(decode head) --> FCNHead(auxiliary head)
SGD, lr=0.01, momentum=0.9, weight_decay=0.0005, epoch=108
