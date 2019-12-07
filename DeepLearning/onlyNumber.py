
# 验证码图像识别

from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import tensorflow as tf
import matplotlib.pyplot as plt

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#指定数据集和验证码长度
def random_captcha_text(char_set=number,captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    # 把list类型转换成str类型
    captcha_text = ''.join(captcha_text)

    # 把验证码画成图像信息
    captcha = image.generate(captcha_text)
    # image.write(captcha_text,captcha_text+'.jpg')

    captcha_image = Image.open(captcha)
    # 图片转换成矩阵形式
    captcha_image = np.array(captcha_image)
    return captcha_text,captcha_image

# 把图片转换成灰度值
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img,-1)
        # # 上面的转法较快，正规转法如下：
        # r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
        # gray = 0.2989 *r + 0.5870 * 0.1140 * b
        return gray
    else:
        return img

# 文字转换
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')
    # 生成 MAX_CAPTCHA * CHAR_SER_LEN 大小的0矩阵
    vector = np.zeros(MAX_CAPTCHA * CHAR_SER_LEN)

    for i,c in enumerate(text):
        idx = i * CHAR_SER_LEN + int(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    text = []
    char_pos = vec.nonzero()[0]
    for i,c in enumerate(char_pos):
        number = i % 10
        text.append(str(number))
    return ''.join(text)


# 生成一个训练的batch
def get_next_batch(batch_size=128,):
    batch_x = np.zeros([batch_size,IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size,MAX_CAPTCHA*CHAR_SER_LEN])

    # 有时生成的图像大小不是(60,160,3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text,image = gen_captcha_text_and_image()
            if image.shape == (60,160,3):
                return text,image

    for i in range(batch_size):
        text,image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i,:] = image.flatten() / 255
        batch_y[i,:] = text2vec(text)

    return batch_x,batch_y


# 定义cnn
def crack_captcha_cnn(w_alpha=0.01,b_alpha=0.1):
    x = tf.reshape(X,shape=[-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH))
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_aplha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3,3,1,32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1))
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1 = tf.nn.dropout(conv1,keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3,3,32,64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2 = tf.nn.dropout(conv2,keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3,3,64,64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3))
    conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3 = tf.nn.dropout(conv3,keep_prob)

    w_d = tf.Variable(w_alpha * tf.random_normal([8*20*64,1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3,[-1,w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense,w_d),b_d))
    dense = tf.nn.dropout(dense,keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024,MAX_CAPTCHA*CHAR_SER_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SER_LEN]))
    out = tf.add(tf.matmul(dense,w_out),b_out)
    return out


# 训练
def train_crack_captcha_cnn():
    # CNN训练过程
    output = crack_captcha_cnn()
    # 损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=Y))
    # Adam函数
    optimzer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # 转换矩阵形状
    predict = tf.reshape(output,[-1,MAX_CAPTCHA,CHAR_SER_LEN])
    max_idx_p = tf.argmax(predict,2)
    max_idx_1 = tf.argmax(tf.reshape(Y,[-1,MAX_CAPTCHA,CHAR_SER_LEN]),2)
    # 值相等的判断
    correct_pred = tf.equal(max_idx_p,max_idx_1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x,batch_y = get_next_batch(64)
            _,loss_ = sess.run([optimzer,loss],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            print(step,loss_)

            # 每100次step计算一次准确率
            if step % 10 == 0:
                batch_x_test,batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy,feed_dict={X:batch_x_test,Y:batch_y_test,keep_prob:1.})
                print(step,acc)
                # 如果准确率大于50%保存模型，完成训练
                if acc > 0.5:
                    # 持久化
                    saver.save(sess,'./model/crack_captcha.model',global_step=step)
                    break

            step += 1


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'./model/crack_captcha.model-1590')  # '1590'为模型最终的训练次数

        predict = tf.argmax(tf.reshape(output,[-1,MAX_CAPTCHA,CHAR_SER_LEN]),2)
        text_list = sess.run(predict,feed_dict={X:[captcha_image],keep_prob:1})
        text = text_list[0].tolist()
        return text


if __name__ == '__main__':
    train = 1   # train==0是训练模型，train是应用模型
    if train == 0:
        number = ['0','1','2','3','4','5','6','7','8','9']

        # 生成验证的值和图像
        text,image = gen_captcha_text_and_image()
        print('验证码图像chanel：',image.shape)  #(60,160,3)
        # 图像大小
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text) # 验证码长度是4
        print('验证码文本最长字符数',MAX_CAPTCHA)
        # 文本转向量
        char_set = number
        CHAR_SER_LEN = len(char_set)

        X = tf.placeholder(tf.float32,[None,IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32,[None,MAX_CAPTCHA * CHAR_SER_LEN])
        keep_prob = tf.placeholder(tf.float32) #dropout
        
        train_crack_captcha_cnn()
    if train == 1:
        number = ['0','1','2','3','4','5','6','7','8','9']
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        char_set = number
        CHAR_SER_LEN = len(char_set)  # 验证码长度是4

        text ,image = gen_captcha_text_and_image()

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1,0.9,text,ha='center',va='center',transform=ax.transAxes)
        plt.imshow(image)

        plt.show()

        MAX_CAPTCHA = len(text)
        image = convert2gray(image)
        image = image.flatten()/255

        X = tf.placeholder(tf.float32,[None,IMAGE_HEIGHT*IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32,[None,MAX_CAPTCHA*CHAR_SER_LEN])
        keep_prob = tf.placeholder(tf.float32)
        # 调用模型
        predict_text = crack_captcha(image)

        print('正确：{}  预测： {}'.format(text,predict_text))
