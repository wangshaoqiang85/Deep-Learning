# 从图像中提取CNN特征，VGG模型是首选算法。
# 它的缺点在于，参数量有140M之多，需要更大的存储空间。
# scipy是一个用于数学、科学、工程领域的常用软件包，
# 可以处理插值、积分、优化、图像处理、常微分方程数值解的求解、信号处理等问题。
# 它用于有效计算Numpy矩阵，使Numpy和Scipy协同工作，高效解决问题。

# 基于VGG实现tensorflow卷积层的可视化

import scipy.io  # 引入scipy
import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf


# 卷积计算层
def _conv_layer(input, weigths, bias):
    conv = tf.nn.conv2d(input, tf.constant(weigths), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


# 池化层
def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def preporcess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


# 读图片
def imread(path):
    return scipy.misc.imread(path).astype(np.float)


# 保存图片
def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


print('Function for VGG ready')


def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    )
    # 载入数据
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    # 数据的平均值
    mean_pixel = np.mean(mean, axis=(0, 1))
    # 取出数据的权重值列表
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]  # 取name字符串的前4个字符
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width,height,in_channels,out_channels]
            # tensorflow: weights are [height,width,in_channels,out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            # 多维转换成一维
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
    # 断言
    assert len(net) == len(layers)
    return net, mean_pixel, layers


print('Network for VGG ready')

cwd = os.getcwd()
# 获得图片在VGGNet19模型中的某些隐藏层上的特征矩阵。所以我们这里使用已经训练好的VGGNet19模型的参数
VGG_PATH = cwd + '/data/imagenet-vgg-verydeep-19.mat'
IMG_PATH = cwd + '/data/cat.jpg'
input_image = imread(IMG_PATH)
shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
with tf.Session() as sess:
    image = tf.placeholder('float', shape=shape)
    nets, mean_pixel, all_layers = net(VGG_PATH, image)
    input_image_pre = np.array([preporcess(input_image, mean_pixel)])
    layers = all_layers

    for i, layer in enumerate(layers):
        print('[%d / %d] %s' % (i + 1, len(layers), layer))
        features = nets[layer].eval(feed_dict={image: input_image_pre})

        print("Type of 'features' is ", type(features))
        print("Shape of 'features' is  %s" % (features.shape,))
        # Plot response
        if 1:
            plt.figure(i + 1, figsize=(10, 5))
            plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray, fignum=i + 1)
            plt.title('' + layer)
            plt.colorbar()
            plt.show()
