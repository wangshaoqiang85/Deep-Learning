import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 下载数据，存储到data中
mnist = input_data.read_data_sets('data2/', one_hot=True)
# input_data会调用一个maybe_download函数，确保数据下载成功
# 这个函数会判断数据的是否下载，如果已经下载完成，则不会重新下载
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print('Mnist Ready')

# 隐藏层的设置
# 设定一个随机数量的值
n_input = 784  # 28*28
# 设定输出值的随机数的值
n_output = 10  # 10个数字

# 权重
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),  # 平方差间距0.1
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),  # 平方差间距0.1
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev=0.1)),  # 平方差间距0.1
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))  # 平方差间距0.1
}
# 偏置
biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
}


# CNN Ready
# _input 输入数据
# _w 权重
# _b 偏置
# _keepratio 保持比列
def conv_basic(_input, _w, _b, _keepratio):
    # INPUT
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    # Conv Layer1
    # nn.conv2d 使用cnn的模式
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    # _means,_var=nn.moments(_conv1,[0,1,2])
    # 激励层
    _conv_relu1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    # 池化层，最大池化
    _conv_po1 = tf.nn.max_pool(_conv_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 使用dropout进行防止过拟合化,保留比例
    _pool_dr1 = tf.nn.dropout(_conv_po1, _keepratio)

    # Conv layer2
    # nn.conv2d 使用cnn的模式
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    # _means,_var=nn.moments(_conv1,[0,1,2])
    # 激励层
    _conv_relu2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    # 池化层，最大池化
    _conv_po2 = tf.nn.max_pool(_conv_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 使用dropout进行防止过拟合化,保留比例
    _pool_dr2 = tf.nn.dropout(_conv_po2, _keepratio)

    # vectorize 矢量化
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])

    # fc层1 矩阵相乘 matmul
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # fc层2 out
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    out = {
        'input_r': _input_r,
        'conv1': _conv1, 'pool1': _conv_po1, 'pool_dr1': _pool_dr1,
        'conv2': _conv2, 'pool2': _conv_po2, 'pool_dr2': _pool_dr2,
        'dense1': _dense1, 'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
    }
    print('CNN Ready')
    return out


# 模型训练
# 使用tensorflow在默认的图中创建节点，这个节点是一个变量
a = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1))
# 打印tensor的值
a = tf.Print(a, [a], 'a:')
# 变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# GRAPH ready
# 声明float节点
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keeparatio = tf.placeholder(tf.float32)

# FUNCTIONS
# 调用CNN函数，返回运算完的结果
_pred = conv_basic(x, weights, biases, keeparatio)['out']
# 求平均值
# softmax_cross_entropy_with_logits 交叉熵
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_pred, labels=y))
# Adam 优化算法，一个寻求全局最优点的优化算法，引入二次方tidings矫正
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# 对比这个两个矩阵或者向量相等的元素，如果是相等发返回True,反之为false
# 返回的值的矩阵维度要和A是一样
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
# 转换数据类型，将x或者x.value转换成float32
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
# 初始化
init = tf.global_variables_initializer()
# SAVER
print('GRAPH Ready')

sess = tf.Session()
sess.run(init)

# 训练次数
trainimg_epochs = 15
# batch
batch_size = 16
# 执行到第几次显示运行结果
display_step = 1  # 每次都显示结果
for epoch in range(trainimg_epochs):
    # 平均误差
    avg_cost = 0
    total_batch = 10
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keeparatio: 0.7})
        # Compute averagr loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keeparatio: 1.}) / total_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print('Epoch: %03d/%03d cost: %.9f' % (epoch, trainimg_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keeparatio: 1.})
        print('Traininig accuracy: %.3f' % (train_acc))

print('OPTIMIZATION FINISHED')
