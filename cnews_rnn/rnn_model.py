import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
from collections import Counter
import time
from datetime import timedelta


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


embedding_dim = 64      # 词向量维度
seq_length = 600        # 序列长度
num_classes = 10        # 类别数
vocab_size = 5000       # 词汇表达小

num_layers= 2           # 隐藏层层数
hidden_dim = 128        # 隐藏层神经元
rnn = 'gru'             # lstm 或 gru

dropout_keep_prob = 0.8 # dropout保留比例
learning_rate = 1e-3    # 学习率

batch_size = 128         # 每批训练大小
num_epochs = 10          # 总迭代轮次

print_per_batch = 100    # 每多少轮输出一次结果

input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)


def gru_cell():
    return tf.contrib.rnn.GRUCell(hidden_dim)


def dropout():
    if rnn == 'lstm':
        cell = lstm_cell()
    else:
        cell = gru_cell()
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def rnn_model(x):
    # 词向量映射
    embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(embedding, x)

    # 多层rnn网络
    cells = [dropout() for _ in range(num_layers)]
    rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
    last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

    # 全连接层，后面接dropout以及relu激活
    fc = tf.layers.dense(last, hidden_dim, name='fc1')
    fc = tf.contrib.layers.dropout(fc, keep_prob)
    fc = tf.nn.relu(fc)

    logits = tf.layers.dense(fc, num_classes, name='fc2')
    return logits


logits = rnn_model(input_x)
y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1)  # 预测类别

# 交叉熵，损失函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=input_y)
loss = tf.reduce_mean(cross_entropy)

# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 准确率
correct_pred = tf.equal(tf.argmax(input_y, 1), y_pred_cls)
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        loss_, acc_ = sess.run([loss, acc], feed_dict={input_x: x_batch,
                                                       input_y: y_batch,
                                                       keep_prob: 1.0})
        total_loss += loss_ * batch_len
        total_acc += acc_ * batch_len

    return total_loss / data_len, total_acc / data_len


# 载入训练集与验证集
train_dir = 'D:/cnews/cnews.train.txt'
val_dir = 'D:/cnews/cnews.val.txt'
categories, cat_to_id = read_category()
vocab_dir = 'D:/cnews/cnews.vocab.txt'
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, vocab_size)
words, word_to_id = read_vocab(vocab_dir)
start_time = time.time()
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    total_batch = 0  # 总批次

    for epoch in range(num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, batch_size)
        for x_batch, y_batch in batch_train:

            if total_batch % print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                loss_train, acc_train = sess.run([loss, acc], feed_dict={input_x: x_batch,
                                                                         input_y: y_batch,
                                                                         keep_prob: 1.0})
                loss_val, acc_val = evaluate(sess, x_val, y_val)  # todo

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif))

            sess.run(optimizer, feed_dict={input_x: x_batch,
                                           input_y: y_batch,
                                           keep_prob: dropout_keep_prob})
            total_batch += 1
