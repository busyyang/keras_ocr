from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import densenet

dataset_path = r'E:\03personal\DeepLearning\data\keras_ocr_data\images'


def get_session(gpu_fraction=0.6):
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


K.set_session(get_session())


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


char = ''
with open('char_std_5990.txt', encoding='utf-8') as f:
    for ch in f.readlines():
        ch = ch.strip('\r\n')
        char = char + ch

# caffe_ocr中把0作为blank，但是tf 的CTC  the last class is reserved to the blank label.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/ctc/ctc_loss_calculator.h
char = char[1:] + '卍'
print('nclass:', len(char))

id_to_char = {i: j for i, j in enumerate(char)}
print(id_to_char[5988])

maxlabellength = 20
img_h = 32
img_w = 280
nclass = len(char)
rnnunit = 256
batch_size = 64


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """

    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if self.index + batchsize > self.total:
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)

        else:
            r_n = self.range[self.index:self.index + batchsize]
            self.index = self.index + batchsize
        return r_n


def readtrainfile(filename):
    with open(os.path.join(dataset_path, filename), 'r') as f:
        lines = f.readlines()
    res = [line.strip() for line in lines]
    dic = {}
    for i in res:
        p = i.split(' ')
        # caffe_ocr中把0作为blank，但是tf 的CTC  the last class is reserved to the blank label.
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/ctc/ctc_loss_calculator.h
        dic[p[0]] = [int(i) - 1 for i in p[1:]]
    return dic


def gen3(trainfile, batchsize=64, maxlabellength=10, imagesize=(32, 280)):
    image_label = readtrainfile(trainfile)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    print('total image in {}'.format(trainfile), len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(dataset_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape',img.shape)
            index = image_label[j]
            label_length[i] = len(index)

            if len(index) <= 0:
                print("len<0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(index)] = index

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


if __name__ == '__main__':
    input = Input(shape=(img_h, None, 1), name='the_input')

    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[maxlabellength], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

    adam = Adam()

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        r'E:\03personal\DeepLearning\data\keras_ocr_data\models\weights-densent-{epoch:02d}.hdf5')
    earlystop = EarlyStopping(patience=10)
    tensorboard = TensorBoard(r'E:\03personal\DeepLearning\data\keras_ocr_data\models\tflog-densent', write_graph=True)

    print('beginfit'.center(98, '-'))
    cc1 = gen3(r'E:\03personal\DeepLearning\data\keras_ocr_data\train.txt', batchsize=batch_size,
               maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    cc2 = gen3(r'E:\03personal\DeepLearning\data\keras_ocr_data\test.txt', batchsize=batch_size,
               maxlabellength=maxlabellength, imagesize=(img_h, img_w))

    hist = model.fit_generator(cc1, steps_per_epoch=3279601 // batch_size, epochs=100,
                               validation_data=cc2, validation_steps=364400 // batch_size,
                               callbacks=[earlystop, checkpoint, tensorboard], verbose=1)
