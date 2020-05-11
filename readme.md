<center><font size=6>Keras的OCR</font></center>
项目fork自：https://github.com/xiaomaxiao/keras_ocr

原项目中的readme文件中有数据集下载地址，不过在`train.txt`中有一个错误的地方，先用`python densenet_ocr.py`排除一下，都是文件不存在的问题，把相应的行删除，还有一个文件后缀写成了`jpjpg`需要改为`jpg`.
`densenet_ocr.py`文件是根据`densenet_ocr.ipynb`文件改的，基本没有大的改动，主要是一些路径改了，然后一个str转int的位置改了，这样可以提前检查出`train.txt`文件的一些错误，不用等到训练快完成一个epoch才跳出错误。
下载了数据集后，需要在
~~~py
    cc1 = gen3(r'E:\03personal\DeepLearning\data\keras_ocr_data\train.txt', batchsize=batch_size,
               maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    cc2 = gen3(r'E:\03personal\DeepLearning\data\keras_ocr_data\test.txt', batchsize=batch_size,
               maxlabellength=maxlabellength, imagesize=(img_h, img_w))
~~~
这里指定`train.txt`和`test.txt`的地方，以及`dataset_path = r'E:\03personal\DeepLearning\data\keras_ocr_data\images'`是所有的图片的位置。打开`train.txt`可以看到，图片的标记是数字，这个数字是在`'char_std_5990.txt'`字的行数。可以直接进行训练。

下载了一个weights文件，不清楚为什么（可能是训练者）改了一些东西，在输出节点处多了一个，所以在`y_pred = densenet.dense_cnn(input, nclass + 1)`这里需要多加一个输出节点。模型文件在release可以查看。替换`densenet_ocr_test.py`代码中`modelPath = r'model\weights-densent-32-0.9846.hdf5'`.测试效果还不错，测试图片的高度应为32。