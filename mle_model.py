#-*- coding: UTF-8 -*-
import collections
import numpy as np
import tensorflow as tf
import data_utils
from keras.engine.training import _make_batches
import sys

'''
mle_model.py 生成古诗模型  python3.6 tensorflow1.2.1
#TODO 没有使用验证集合，无法检测是否已经过拟合
#TODO 没有使用BLEU量化性能指标，无法直观地比较两个模型的优劣
#TODO 可以使用beam search、hierachy softmax加快速度，避免在概率值很小的生成样本上浪费时间
'''



class MLE_Model(object):
    def def_model(self, num_word, batch_size, model = 'lstm', rnn_size = 128, num_layers = 2):
        self.input_data = tf.placeholder(tf.int32, [batch_size, None])
        self.output_targets = tf.placeholder(tf.int32, [batch_size, None])
        if model == 'rnn':
            cell_fun = tf.contrib.rnn.BasicRNNCell
        elif model == 'gru':
            cell_fun = tf.contrib.rnn.GRUCell
        elif model == 'lstm':
            cell_fun = tf.contrib.rnn.BasicLSTMCell
        cell = cell_fun(rnn_size, state_is_tuple = True)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple = True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, num_word])
            softmax_b = tf.get_variable("softmax_b", [num_word])
            embedding = tf.get_variable("embedding", [num_word, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        #dynamic_rnn可以使用不同长度的序列作为输入，在训练是长度为N；而在生成序列的时候则是1，即一个字一个字的输入
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state = initial_state, scope = 'rnnlm')
        output = tf.reshape(outputs, [-1, rnn_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        probs = tf.nn.softmax(logits)
        return logits, last_state, probs, cell, initial_state

    def load_model(self,sess, saver, ckpt_path):
        latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        if latest_ckpt:
            print ('resume from', latest_ckpt)
            saver.restore(sess, latest_ckpt)
            return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
        else:
            print ('building model from scratch')
            sess.run(tf.global_variables_initializer())
            return -1

    #训练
    def train_neural_network(self, words, vocab_dict, batch_size ):
        num_word = len( vocab_dict )
        logits, last_state, _, _, _ = self.def_model( num_word, batch_size)
        targets = tf.reshape(self.output_targets, [-1])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], \
            [tf.ones_like(targets, dtype = tf.float32)], len(words))
        cost = tf.reduce_mean(loss)
        learning_rate = tf.Variable(0.0, trainable = False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        Session_config = tf.ConfigProto(allow_soft_placement = True)
        Session_config.gpu_options.allow_growth = True

        batches = _make_batches(len(words), batch_size)

        with tf.Session(config = Session_config) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())
            last_epoch = self.load_model(sess, saver, 'model/')

            for epoch in range(last_epoch + 1, 100):
                sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
                #sess.run(tf.assign(learning_rate, 0.01))
                index_array = np.arange(len(words))
                np.random.shuffle( index_array)

                all_loss = 0.0
                for batch_index, (batch_start, batch_end) in enumerate(batches):
                    if batch_end - batch_start != batch_size:
                        # print('skip batch {} {}'.format(batch_start, batch_end))
                        continue
                    batch_ids = index_array[batch_start:batch_end]
                    xdata = words[batch_ids]
                    ydata = np.copy(xdata)
                    ydata[:, :-1] = xdata[:, 1:]
                    train_loss, _, _ = sess.run([cost, last_state, train_op], feed_dict={self.input_data: xdata, self.output_targets: ydata})
                    all_loss = all_loss + train_loss

                    if batch_index % 50 == 1:
                        print(epoch, batch_index, 0.002 * (0.97 ** epoch),train_loss)

                saver.save(sess, 'model/poetry.module', global_step = epoch)
                print (epoch,' Loss: ', all_loss * 1.0 / len(batches))

    def gen_poetry(self, vocab_dict, vocab_dict_res ):
        def to_word(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            sample = int(np.searchsorted(t, np.random.rand(1) * s))
            return vocab_dict_res[sample]
        num_word = len(vocab_dict)
        _, last_state, probs, cell, initial_state = self.def_model(num_word, batch_size= 1)
        Session_config = tf.ConfigProto(allow_soft_placement=True)
        Session_config.gpu_options.allow_growth = True

        with tf.Session(config=Session_config) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())
            # saver.restore(sess, 'model/poetry.module-99')
            ckpt = tf.train.get_checkpoint_state('./model/')
            checkpoint_suffix = ""
            if tf.__version__ > "0.12":
                checkpoint_suffix = ".index"
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
                # print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                return None
            for _ in range(5):
                state_ = sess.run(cell.zero_state(1, tf.float32))
                x = np.array([[data_utils.GO_ID]])#list(map(vocab_dict.get, data_utils._GO))
                [probs_, state_] = sess.run([probs, last_state], feed_dict={self.input_data: x, initial_state: state_})
                word = to_word(probs_)
                # word = words[np.argmax(probs_)]
                poem = ''
                while word != data_utils._EOS:
                    poem += word
                    x = np.zeros((1, 1))
                    x[0, 0] = vocab_dict[word]
                    [probs_, state_] = sess.run([probs, last_state], feed_dict={self.input_data: x, initial_state: state_})
                    word = to_word(probs_)
                    # word = words[np.argmax(probs_)]
                print( poem )

if __name__ == '__main__':
    if len( sys.argv) > 1 and sys.argv[1] == 'train':
        print( 'train ')
        model = MLE_Model()
        vocab_dict, vocab_res = data_utils.load_vocab('./vocab.txt')
        data = data_utils.load_data( 'data.pkl' )
        # data = data[:1000]
        model.train_neural_network( data, vocab_dict, batch_size = 512 )
    else:
        print('start generation')
        model = MLE_Model( )
        vocab_dict, vocab_res = data_utils.load_vocab('./vocab.txt')
        model.gen_poetry( vocab_dict, vocab_res )
