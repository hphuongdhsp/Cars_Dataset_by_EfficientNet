# Cars_Dataset_by_EfficientNet

This work  is to do car recognition by fine-tuning [EfficientNet](https://arxiv.org/pdf/1905.11946v2.pdf) with [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). 
### About EfficientNet
EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models. EfficientNets are developed  based on AutoML and Compound Scaling. 
In particular, one first uses [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:

 
### Dataset

We use only the train part of the Cars Dataset, which contains 8,144 training images of 196 classes of cars. 
 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/random.jpg). 
 
The data is split into 7326 training images and 814 validation images, where each class has been split roughly in a 90-10 split.
 
 ### Our approaching 
 There are a lot of workings with this dataset. Our approaching is in particular with: 
  * Using the bounding boxes
  * [Using the state-of-the-art model (EfficientNet)](https://arxiv.org/pdf/1905.11946v2.pdf)
  * Having a good fine-tuning, here 60 first epochs, we use Step learning rate. Then next 60 epochs, we use [Cyclical Learning Rate](https://arxiv.org/abs/1506.01186)
  * We also use the Apex package. The intention of Apex is to make up-to-date utilities available to users as quickly as possible.
 
### Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Pytorch](https://github.com/pytorch/pytorch)
- [Apex](https://github.com/NVIDIA/apex)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)
- [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [Tensorboard](https://github.com/tensorflow/tensorboard)

### Usage
 #### Step 1: Clone my git and download data 
 ```bash
$ git clone https://github.com/hphuongdhsp/Cars_Dataset_by_EfficientNet
$ cd Cars_Dataset_by_EfficientNet
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```
#### Step 2: Split training validation by labels stratify

 ```bash
$ cd Cars_Dataset_by_EfficientNet
$ python3 data_processing.py
```
The aim of this step is spitting the training and validation set. We also use the bounding box columns to crop our images.

#### Step 3: Training
 ```bash
$ cd Cars_Dataset_by_EfficientNet
$ python3 train.py 
```
#### Maybe Step 4.
The goal of step 4 is to predict the data from testing set. For that, we shall create a csv file, namely, submitssion.csv that is stored in ./Cars_Dataset_by_EfficientNet/data/
To do that, run 
 ```bash
$ cd Cars_Dataset_by_EfficientNet
$ python3 predict_and_submission.py 
```
#### Tips
Because of limiting of resource (only 11GB of GPU), here we use only batch-size 16 for "efficientnet-b3" model. We get 92,2% accuracy scores. To get more accuracy scores, you can change these parameters in the "parser.py". It will make a better performance. 


The pretrained weights of[EfficientNet-b4,EfficientNet-b4 ](https://arxiv.org/pdf/1905.11946v2.pdf) are released on 18-June. 
To use them, please upgrade the pip package with 
 ```bashpip install --upgrade efficientnet-pytorch
```
But there is a bug, u need to change the file" /efficientnet_pytorch/utils.py" as: 

    - random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
    
    + random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    
   Access (https://github.com/lukemelas/EfficientNet-PyTorch/commit/939d4abdeefc07e63d8bd42e7223365a4bc67942#diff-57e79865f7111e0dd0165032e805d446) to get more details. 





