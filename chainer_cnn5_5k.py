#coding: utf-8
import os
#os.environ["CHAINER_TYPE_CHECK"] = "0" #ここでオフに  オンにしたかったら1にするかコメントアウト
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import time
import pylab
import matplotlib.pyplot as plt
import cv2
import logging
import random
from chainer import cuda, optimizers, serializers
from os import path
from TDLBB.log_tool import start_logging

gpu_flag = 0

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

annotation_file1 = '/123/maya/honda/20161213_testdata/annotation/101/P101S03T01/P101S03T01_01.txt'
annotation_file2 = '/123/maya/honda/20161213_testdata/annotation/101/P101S03T03/P101S03T03_01.txt'
annotation_file3 = '/123/maya/honda/20161213_testdata/annotation/101/P101S03T01/P101S03T01_01/face2.txt'
annotation_file4 = '/123/maya/honda/20161213_testdata/annotation/101/P101S03T03/P101S03T03_01/face2.txt'
n_epoch          = 300
result_folder    = '/123/maya/honda/Train/CNN_chainer/jikken04/trial05_5k' ##変更する!!
batch_size       = 5
frame            = 4
margin           = 100


#ConvolutionND( ndim, in_channels, out_channels,       ksize,      stride=1,         pad=0, initialW=None)
#ConvolutionND(次元数, 入力チャンネル,  出力チャンネル, フィルタサイズ, 畳み込み適用間隔, 縁に付け加える値, initialW=None)
model = chainer.FunctionSet(conv1=L.ConvolutionND(2,   3,  32, 3, 1, 1, initial_bias=None),         # 入力  5枚、出力 16枚、フィルタサイズ5ピクセル
                            conv2=L.ConvolutionND(2,  32,  64, 3, 1, 1, initial_bias=None),        # 入力 16枚、出力 32枚、フィルタサイズ5ピクセル
                            conv3=L.ConvolutionND(2,  64,  64, 3, 1, 1, initial_bias=None),        # 入力 32枚、出力 32枚、フィルタサイズ5ピクセル
                            conv4=L.ConvolutionND(2,  64, 128, 3, 1, 1, initial_bias=None),        # 入力 32枚、出力 64枚、フィルタサイズ5ピクセル
                            conv5=L.ConvolutionND(2, 128, 128, 3, 1, 1, initial_bias=None),   # 入力 64枚、出力 64枚、フィルタサイズ4x5ピクセル
                            conv6=L.ConvolutionND(2, 128, 128, 3, 1, 1, initial_bias=None),       # 入力 64枚、出力128枚、フィルタサイズ5ピクセル
                            conv7=L.ConvolutionND(2, 128, 128, 3, 1, 1, initial_bias=None), # 入力 64枚、出力128枚、フィルタサイズ5x4ピクセル
                            conv8=L.ConvolutionND(2, 128, 128, 3, 1, 1, initial_bias=None),      # 入力128枚、出力128枚、フィルタサイズ5ピクセル
                            l1=F.Linear(2048, 768),                 # 入力6656ユニット、出力3328ユニット
                            l2=F.Linear(768, 2))                    # 入力3328ユニット、出力   2ユニット

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()

def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.relu(model.conv1(x))
    h = F.max_pooling_nd(F.relu(model.conv2(h)), 2, 2) ##max_pooling_nd(x, ksize) : x =input variable, ksize =size of pooling window. ksize=k, ksize=(k, k, ..., k)
    h = F.relu(model.conv3(h))
    h = F.max_pooling_nd(F.relu(model.conv4(h)), 2, 2)
    h = F.relu(model.conv5(h))
    h = F.max_pooling_nd(F.relu(model.conv6(h)), 2, 2)
    h = F.relu(model.conv7(h))
    h = F.max_pooling_nd(F.relu(model.conv8(h)), 2, 2)
    h = F.dropout(F.relu(model.l1(h)), train=train)
    y = model.l2(h)
    if train:
        print('y: {}\n t: {}'.format(y.data, t.data))
        return F.softmax_cross_entropy(y, t)
    else:
        print('y: {}, t: {}'.format(y.data, t.data))
        return F.accuracy(y, t)

optimizer = optimizers.Adam()
optimizer.setup(model)



fp1 = open("accuracy.txt", "w") ##評価結果
fp2 = open("loss.txt", "w") ##学習誤差を記述するファイル
fp3 = open("sample.txt", "w") ##サンプル数で評価結果を表示

fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")

##画像名とラベルの読み込み----------------------------------------
f = open(annotation_file1, 'r')
line = f.readlines()
f.closed

f4 = open(annotation_file3, 'r')
line3 = f4.readline() ##1行めだけ読み込む
right, left, bottom, top = line3[:-1].split(' ')
f4.closed
##-----------------------------------------------------------


##訓練ループ-----------------------------------------------------------------------------------------------------
print('Learning...')
start_logging()
start_time = time.clock()
for epoch in range(1, n_epoch + 1):
    print "epoch: %d" % epoch

    sample   = 0
    sum_loss = 0
    NOD      = 0
    OTHER    = 0
    NOD_g    = 0
    OTHER_g  = 0
    x        = []
    y        = []
    len_f    = len(line)-46
    shuffle_f = np.random.permutation(len_f)

    for n, l in enumerate(shuffle_f):
        X = []
        for i in xrange(l, l+46, 15): ##4フレーム
            i_name, nod = line[i][:-1].split(' ') ##最後のフレームのラベルを使う
            image = cv2.imread(path.join('/123', 'maya', 'honda', '20161213_testdata', 'annotation', '101', 'P101S03T01', 'P101S03T01_01_shomen', i_name), cv2.IMREAD_GRAYSCALE).astype('float32')
            image = image[int(top)-margin:int(bottom)+margin, int(left)-margin:int(right)+margin] ##顔領域の切り出し(定位置で)
            image = cv2.resize(image, (int(image.shape[1]/9), int(image.shape[0]/9)), interpolation=cv2.INTER_CUBIC)
            col   = image.shape[0]
            row   = image.shape[1]
            X.append(image)

        if nod == '0':
           Y        = nod
           OTHER_g += 1
        else:
           Y        = nod
           NOD_g   += 1

        x.append(X)
        y.append(Y)

        if len(x) == batch_size: ##バッチサイズ分収集できたら
           x_batch  = xp.asarray(x, dtype = 'float32')
           x_batch /= 255. #正規化
           x_batch  = x_batch.reshape(batch_size, 1, frame, col, row) ##3フレームずつにまとめる
           y_batch  = xp.asarray(y, dtype = 'int32') ##ソフトマックス関数が'int32'しか受け付けないらしい
           y_batch  = y_batch.reshape(batch_size)

           optimizer.zero_grads()
           loss = forward(x_batch, y_batch)
           loss.backward()
           optimizer.update()
           sum_loss += float(loss.data)
           x = []
           y = []
           sample += batch_size

    print "sample: {}".format(sample)
    print "train mean loss: %f" % (sum_loss / sample)
    fp2.write("%d\t%f\n" % (epoch, sum_loss / sample))
    fp2.flush() ##即書き出しする関数らしい


##画像名とラベルの読み込み----------------------------------------
    f2 = open(annotation_file2, 'r')
    line2 = f2.readlines()
    f.closed

    f3 = open(annotation_file4, 'r')
    line4 = f3.readline() ##1行めだけ読み込む
    right2, left2, bottom2, top2 = line4[:-1].split(' ')
    f3.closed
##-----------------------------------------------------------

    sum_accuracy = 0
    sample       = 0
    sum_loss     = 0
    NOD          = 0
    OTHER        = 0
    NOD_t        = 0
    OTHER_t      = 0
    len_f2       = len(line2)-46
    shuffle_f2 = np.random.permutation(len_f2)

    for t2, l2 in enumerate(shuffle_f2):
        X = []
        for i in xrange(l2, l2+46, 15): ##4フレーム分
            i_name, nod = line2[i][:-1].split(' ')
            image = cv2.imread(path.join('/123', 'maya', 'honda', '20161213_testdata', 'annotation', '101', 'P101S03T03', 'P101S03T03_01_shomen', i_name), cv2.IMREAD_GRAYSCALE).astype('float32')
            image = image[int(top2)-margin:int(bottom2)+margin, int(left2)-margin:int(right2)+margin]
            image = cv2.resize(image, (int(image.shape[1]/9), int(image.shape[0]/9)), interpolation=cv2.INTER_CUBIC)
            col   = image.shape[0]
            row   = image.shape[1]
            X.append(image)

        x_batch  = xp.asarray(X, dtype = 'float32')
        x_batch /= 255. #正規化
        x_batch  = x_batch.reshape(1, 1, frame, col, row) ##3フレームずつにまとめる
        if nod == '0':
           Y        = nod
           OTHER_t += 1
        else:
           Y        = nod
           NOD_t   += 1
        y_batch  = xp.asarray(Y, dtype = 'int32') ##ソフトマックス関数が'int32'しか受け付けないらしい
        y_batch  = y_batch.reshape(1)

        acc           = forward(x_batch, y_batch, train=False)
        if acc.data == 1.0:
           if Y == '1':
              NOD   += 1
           else:
              OTHER += 1
        print('acc: {}'.format(acc.data))
        sum_accuracy += float(acc.data)
        sample += 1

    print("sample: {}".format(sample))
    print "test accuracy: %f" % (sum_accuracy / sample)
    fp1.write("Epoch %d %f\n" % (epoch, sum_accuracy / sample))
    fp1.flush() ##即書き出しする関数らしい
    fp3.write("NOD: %d/%d, OTHER: %d/%d\n" % (NOD, NOD_t, OTHER, OTHER_t))
    fp3.flush() ##即書き出しする関数らしい

    #パラメータの保存
    if (epoch == 1) or (epoch % 5 == 0):
       out_model = '{}/epoch-{}.model'.format(result_folder, epoch)
    #   out_opt = '{}/epoch-{}.state'.format(result_folder, epoch)
    #   serializers.save_hdf5(out_model, src_model)
       serializers.save_hdf5(out_model, optimizer)

end_time = time.clock()
print end_time - start_time

print('OVER')

fp2.write("NOD: %d, OTHER: %d\n" % (NOD_g, OTHER_g))
fp2.write("Training_time : %f\n" % (end_time - start_time))
fp2.flush() #即書き出しする関数らしい

fp1.close()
fp2.close()
fp3.close()
