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
 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/random.jpg)
## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)


