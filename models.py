# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from termcolor import colored
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.framework import dtypes
import numpy as np
import time
import os
import utils
import data_utils

def reduce_print(x):
  if x == "_PAD": return colored("空", "green")
  elif x == "_EOS": return colored("完", "yellow")
  else: return x

def postprocess(x):
  x = list(x)
  x.reverse()
  i = 0
  while(x[i] == 0):
    i += 1
  if(x[i] != 1):
    if(i == 0):
      x[i] = 1
    else:
      x[i - 1] = 1
  x.reverse()
  return x

class CompQAModel(object):
    def __init__(self, vocab_size, embedding_size, max_src_len, max_des_len,
                 max_fact_num, state_size, num_layers, num_samples,
                 max_grad_norm, is_train, cell_type,
                 optimizer_name, learning_rate):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_src_len = max_src_len
        self.max_des_len = max_des_len
        self.max_fact_num = max_fact_num
        self.state_size = state_size
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.max_grad_norm = max_grad_norm
        self.cell_type = cell_type
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.load_prev_train = False
        self.is_train = is_train
        self.global_step = tf.Variable(0, trainable=False)
        self.mode_loss_ratio = 10.0 # 1.0, 2.0, 5.0, 10.0

        self._define_embedding()
        self._define_graph()

    def _define_embedding(self):
        # put this variable to cpu0
        with tf.device("/cpu:0"):
            # embeddings = tf.get_variable("vocab_embeddings", [self.vocab_size, self.embedding_size])
            embeddings = tf.get_variable("vocab_embeddings", [self.vocab_size + self.max_src_len + self.max_fact_num,
              self.embedding_size])
            embeddings = tf.nn.l2_normalize(embeddings, 1)
            self.embeddings = embeddings

    def _define_graph(self):
        # why max_src_len is the first argument? -- because this code performs time_major, a little bit faster than batch_major
        self.encoder_inputs = tf.placeholder(tf.int32, [self.max_src_len, None], 'encoder_inputs')
        self.encoder_lengths = tf.placeholder(tf.int32, [None], 'encoder_lengths')
        # what is this? -- facts 
        self.facts_inputs = tf.placeholder(tf.int32, [self.max_fact_num, 2, None], 'input_facts')

        self.decoder_inputs = tf.placeholder(tf.int32, [self.max_des_len + 2, None], 'decoder_inputs')
        # what is this? -- if copy from source sentence, decide which position
        self.decoder_sources = tf.placeholder(tf.int32,
            [self.max_des_len + 2, self.max_src_len, None], 'decoder_sources')
        # what is this? -- if copy from kb, decide which position
        self.decoder_kbkbs = tf.placeholder(tf.int32,
            [self.max_des_len + 2, self.max_fact_num, None], 'decoder_kbkbs')
        self.decoder_modes = tf.placeholder(tf.int32,
            [self.max_des_len + 2, 3, None], 'decoder_modes')

        self.decoder_weights = tf.placeholder(tf.float32, [self.max_des_len + 2, None], 'decoder_weights')
        self.keep_prob = tf.placeholder(tf.float32, shape = ())

        ###################对问句进行处理##########################
        # -- TB test 
        with tf.device("/cpu:0"):
            embedded_encoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            print("size of encoder inputs: ", embedded_encoder_inputs.shape)
            # embedded_encoder_inputs = tf.unstack(embedded_encoder_inputs)
            # print("encoder inputs: ", len(embedded_encoder_inputs), embedded_encoder_inputs[0].shape)
        self.input_output_size = self.state_size * 2
        self.input_state_size = self.state_size * 2
        if self.cell_type.lower() == 'lstm':
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.state_size, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.state_size, state_is_tuple=True)
            self.input_state_size *= 2
        elif self.cell_type.lower() == 'gru':
            cell_fw = rnn_cell.GRUCell(self.state_size)
            cell_bw = rnn_cell.GRUCell(self.state_size)
        else:
            cell_fw = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
            cell_bw = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
        (output_fw, output_bw), ((output_state_fw_c, output_state_fw_h), (output_state_bw_c, output_state_bw_h)) =\
            tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedded_encoder_inputs,
                sequence_length=self.encoder_lengths, dtype=dtypes.float32, time_major = True)
        print("output size: ", output_fw.shape, output_fw.shape)
        self.encoder_outputs = tf.concat([output_fw, output_bw], 2)
        print("after concat: ", self.encoder_outputs.shape)
        self.encoder_outputs = tf.unstack(self.encoder_outputs)
        print("after unstack, length = ", len(self.encoder_outputs), "shape: ", self.encoder_outputs[0].shape)
        print("state size: ", output_state_fw_c.shape)
        self.encoder_states = tf.concat([output_state_fw_c, output_state_fw_h,
          output_state_bw_c, output_state_bw_h], 1)
        print("states after concat: ", self.encoder_states.shape)
        print("question processing finished .... ")
        #####################################################################

        ###################对知识库中的事实进行处理##########################
        #  -- TB test
        # self.facts_inputs = tf.placeholder(tf.int32, [self.max_fact_num, 2, None], 'input_facts')
        with tf.device("/cpu:0"):
            facts_inputs = tf.transpose(self.facts_inputs, [1, 0, 2]) #[2, fact, batch]
            print("shape of facts_inputs: ", facts_inputs.shape)
            facts_rel, facts_obj = tf.unstack(facts_inputs) #[fact, batch], [fact, batch]
            print("shape of facts_rel: ", facts_rel.shape)
            embedded_fact_rels = tf.unstack(tf.nn.embedding_lookup(self.embeddings, facts_rel))
            print("after embedding, tuple length: ", len(embedded_fact_rels), "shape:", embedded_fact_rels[0].shape)
            embedded_fact_objs = tf.unstack(tf.nn.embedding_lookup(self.embeddings, facts_obj))
            embedded_facts = [tf.concat([one, two], 1) for one, two in
                              zip(embedded_fact_rels, embedded_fact_objs)] #4个元素，每个元素都是[batch, emb*2]
            print("after concat, tuple length: ", len(embedded_facts), "shape:", embedded_facts[0].shape)
            avg_embedded_facts = embedded_facts[-1] #性别属性的表示
        #####################################################################

        self.base_loss = 0.0
        self.norm_loss = 0.0

        #######################解码准备工作################################
        with tf.device("/cpu:0"):
            embedded_decoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
            embedded_decoder_inputs = tf.unstack(embedded_decoder_inputs) #max_des_num个元素，每个元素都是[batch, emb]
        self.output_output_size = self.input_output_size
        self.output_state_size = self.input_output_size
        if self.cell_type.lower() == 'lstm':
            print("cell_out, input size = %d" % self.input_output_size)
            cell_out = tf.nn.rnn_cell.BasicLSTMCell(self.input_output_size, state_is_tuple=True)
            self.output_state_size *= 2
        elif self.cell_type.lower() == 'gru':
            cell_out = rnn_cell.GRUCell(self.input_output_size)
        else:
            cell_out = tf.nn.rnn_cell.BasicRNNCell(self.input_output_size)

        ###用于对输出状态预测目标词
        W1 = tf.get_variable("proj_w1", [self.output_output_size, self.vocab_size])
        B1 = tf.get_variable("proj_b1", [self.vocab_size])
        # W2 = tf.get_variable("proj_w2", [self.output_output_size, self.max_src_len + self.max_fact_num])
        # B2 = tf.get_variable("proj_b2", [self.max_src_len + self.max_fact_num])
        # W_all = tf.concat([W1, W2], 1)
        # B_all = tf.concat([B1, B2], 0)
        # output_projection = (W_all, B_all)
        output_projection = (W1, B1)
        self.norm_loss += tf.nn.l2_loss(W1)
        self.norm_loss += tf.nn.l2_loss(B1)
        # self.norm_loss += tf.nn.l2_loss(W_all)
        # self.norm_loss += tf.nn.l2_loss(B_all)

        # 输入(prev, loc1, loc2)表示上一步预测的词ID和从源句子和KB中copy的位置
        # obsolete

        # two types of new loop functions
        def loop_same(prev):
            return prev

        def loop_projection(prev, mode_logit):
            # prev shape: [batch size, state size]
            # mode shape: [batch size, 3]
            prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
            prev_symbol = tf.argmax(prev, 1)
            embedded_prev = tf.nn.embedding_lookup(self.embeddings, prev_symbol)
            pad_embeddings = embedded_decoder_inputs[-1]
            mode = tf.cast(tf.one_hot(tf.argmax(mode_logit, axis = 1), 3), tf.float32)
            # is_common shape: [batch size]
            is_common = tf.unstack(tf.transpose(mode))[0]
            prev = embedded_prev * tf.expand_dims(is_common, 1) + pad_embeddings * tf.expand_dims(
                tf.ones(tf.shape(is_common), tf.float32) - is_common, 1)
            return prev

        # loop_function = loop_same
        loop_function = loop_projection

        #对源句子进行attention
        source_atten_state_size = self.output_state_size + self.input_output_size + self.max_src_len
        source_atten_mlp_w1 = tf.Variable(tf.truncated_normal([source_atten_state_size, 200], stddev=0.1),
                                          name="SOURCE_ATTEN_MLP_W1")
        source_atten_mlp_w2 = tf.Variable(tf.truncated_normal([200, 1], stddev=0.1),
                                          name="SOURCE_ATTEN_MLP_W2")
        self.norm_loss += tf.nn.l2_loss(source_atten_mlp_w1)
        self.norm_loss += tf.nn.l2_loss(source_atten_mlp_w2)
        def source_atten_comp(_state, verbose = False): #输入的是隐状态[batch, state_size]
            # _one.shape = batch_size * state_size 
            # questions: is self.encoder_outputs iteratable?
            if(verbose): print("state shape: ", _state.shape)
            _inputs = [tf.concat([_state, _one], 1) for _one in self.encoder_outputs]
            if(verbose): print("inputs length ", len(_inputs), " shape ", _inputs[0].shape)
            _inputs = [tf.tanh(tf.matmul(_one, source_atten_mlp_w1)) for _one in _inputs]
            if(verbose): print("inputs length ", len(_inputs), " shape ", _inputs[0].shape)
            _inputs = [tf.matmul(_one, source_atten_mlp_w2) for _one in _inputs]
            if(verbose): print("inputs length ", len(_inputs), " shape ", _inputs[0].shape)
            _inputs = tf.stack(_inputs)  #[seq, batch]
            if(verbose): print("in attention, after stack: ", _inputs.shape)
            _inputs = tf.reshape(_inputs, [self.max_src_len, -1]) #没有进行softmax
            if(verbose): print("in attention, after reshape: ", _inputs.shape)
            _inputs = tf.transpose(_inputs)  # [batch_size, seq]
            if(verbose): print("in attention, after transpose: ", _inputs.shape)
#            _inputs = tf.reshape(_inputs, [-1, self.max_src_len]) #没有进行softmax
#            if(verbose): print("in attention, after reshape: ", _inputs.shape) # Why we need to reshape?
            return _inputs

        #对知识库进行attention
        kbkb_atten_state_size = self.output_state_size + self.input_state_size + \
                                self.embedding_size * 2 + self.max_fact_num
        kbkb_atten_mlp_w1 = tf.Variable(tf.truncated_normal([kbkb_atten_state_size, 200], stddev=0.1),
                                        name="KBKB_ATTEN_MLP_W1")
        kbkb_atten_mlp_w2 = tf.Variable(tf.truncated_normal([200, 1], stddev=0.1),
                                        name="KBKB_ATTEN_MLP_W2")
        self.norm_loss += tf.nn.l2_loss(kbkb_atten_mlp_w1)
        self.norm_loss += tf.nn.l2_loss(kbkb_atten_mlp_w2)

        def kbkb_atten_comp(_state, verbose):  # 输入的是隐状态[batch, state_size]
            if(verbose): print("kbkb state shape: ", _state.shape)
            _inputs = [tf.concat([_state, _one, self.encoder_states], 1) for _one in embedded_facts]
            if(verbose): print("inputs length ", len(_inputs), " shape ", _inputs[0].shape)
            _inputs = [tf.tanh(tf.matmul(_one, kbkb_atten_mlp_w1)) for _one in _inputs]
            if(verbose): print("inputs length ", len(_inputs), " shape ", _inputs[0].shape)
            _inputs = [tf.matmul(_one, kbkb_atten_mlp_w2) for _one in _inputs]
            if(verbose): print("inputs length ", len(_inputs), " shape ", _inputs[0].shape)
            _inputs = tf.stack(_inputs)  # [fact, batch]
            if(verbose): print("in kbkb attention, after stack: ", _inputs.shape)
            _inputs = tf.reshape(_inputs, [self.max_fact_num, -1])  # 没有进行softmax
            if(verbose): print("in kbkb attention, after reshape: ", _inputs.shape)
            _inputs = tf.transpose(_inputs)  # [batch_size, fact]
            if(verbose): print("in kbkb attention, after transpose: ", _inputs.shape)
            return _inputs

        #对使用mode进行预测
        mode_comp_state_size = self.output_state_size + self.embedding_size
        # mode_comp_state_size = self.embedding_size
        mode_mlp_w1 = tf.Variable(tf.truncated_normal([mode_comp_state_size, 200], stddev=0.1),
                                  name="MODE_MLP_W1")
        mode_mlp_w2 = tf.Variable(tf.truncated_normal([200, 3], stddev=0.1),
                                  name="MODE_MLP_W2")
        self.norm_loss += tf.nn.l2_loss(mode_mlp_w1)
        self.norm_loss += tf.nn.l2_loss(mode_mlp_w2)
        def mode_comp(_state):  # 输入的是隐状态[batch, state_size]
            _inputs = tf.tanh(tf.matmul(_state, mode_mlp_w1))
            _inputs = tf.matmul(_inputs, mode_mlp_w2)
            _inputs = tf.nn.softmax(_inputs) #[batch, 3]
            return _inputs

        print("start decoding ... ")
        ###################开始解码##################################
        with tf.variable_scope("rnn_decoder"):
            decoder_truths = [tf.unstack(self.decoder_inputs)[i + 1] for i in xrange(self.max_des_len + 1)]
            print("decoder_truth, len ", len(decoder_truths), "shape ", decoder_truths[0].shape)
            # question: what's the use of weight?
            decoder_weights = tf.unstack(self.decoder_weights)[:-1] #每个weight都是[batch]
            print("decoder_weights, len ", len(decoder_weights), "shape ", decoder_weights[0].shape)

            sous_locs_seq = [tf.transpose(_one) for _one in tf.unstack(self.decoder_sources)][:-1] #[batch, seq]
            print("sous_locs_seq, len ", len(sous_locs_seq), "shape ", sous_locs_seq[0].shape)

            kbkb_locs_seq = [tf.transpose(_one) for _one in tf.unstack(self.decoder_kbkbs)][:-1] #[batch, fact]
            print("kbkb_locs_seq, len ", len(kbkb_locs_seq), "shape ", kbkb_locs_seq[0].shape)

            modes_seq = [tf.transpose(_one) for _one in tf.unstack(self.decoder_modes)][:-1] #[batch, 3]
            print("modes_seq, len ", len(modes_seq), "shape ", modes_seq[0].shape)

            decoder_state_c = tf.concat([output_state_fw_c, output_state_bw_c], 1)# [batch, out_state_size]
            decoder_state_h = tf.concat([output_state_fw_h, output_state_bw_h], 1)# [batch, out_state_size]
            state = (decoder_state_c, decoder_state_h)

            source_state = tf.zeros((tf.shape(self.encoder_inputs)[1], self.max_src_len))
            kb_state = tf.zeros((tf.shape(self.encoder_inputs)[1], self.max_fact_num))

            hist_source_logit = tf.zeros((tf.shape(self.encoder_inputs)[1], self.max_src_len))
            hist_kbkb_logit = tf.zeros((tf.shape(self.encoder_inputs)[1], self.max_fact_num))

            decoder_logits = []
            decoder_predicts = []
            decoder_gold_logits = []
            decoder_pred_logits = []
            outputs = []
            # look up embeddings for _GO
            prev = tf.nn.embedding_lookup(self.embeddings, tf.unstack(self.decoder_inputs)[0])
            # size = batch_size * output_size
            #decoder_input_dim = tf.stack([tf.shape(self.decoder_inputs)[1], self.input_output_size])
            #prev = tf.fill(decoder_input_dim, 0.0) 
            print("size of first decoder inputs: ", prev.shape)
            decoder_loss, decoder_mode_loss, decoder_predict_modes = [], [], []
            print("starting end2end decoding ... ")
            for i, inp in enumerate(embedded_decoder_inputs[:-1]):  # 输出端的每个输入词，第一个是GOprojection
                # if not self.is_train and loop_function is not None and prev is not None and i >= 1:  # 测试的时候才用
                if not self.is_train and i >= 1:  # 测试的时候才用
                    # question: why I should use this "with" statement?
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, pred_modes)
                        # inp = prev
                if(self.is_train):
                    if(i == 0): print("inp shape:", inp.shape)
                    if(i != 0):
                        # inp = loop_function(prev, pred_modes)
                        if(i == 1): print("after loop function:", inp.shape)
                # test here
                # if(i != 0):
                #     inp = loop_function(prev, pred_modes)

                # important!
                # What if I comment this?
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # output, state = cell_out(inp, state)
                if(i == 0): print("source state shape: ", source_state.shape)
                source_state = tf.reshape(source_state, [-1, self.max_src_len])
                if(i == 0): print("after reshape: ", source_state.shape)
                if(i == 0): print("kb state shape: ", kb_state.shape)
                kb_state = tf.reshape(kb_state, [-1, self.max_fact_num])
                if(i == 0): print("after reshape: ", kb_state.shape)
                if(i == 0): print("state shape: ", state[0].shape)
                concated_inp = tf.concat([inp, source_state, kb_state, avg_embedded_facts], 1)
                if(i == 0): print("concated input shape: ", concated_inp.shape)
                cell_out = tf.contrib.rnn.DropoutWrapper(cell_out, output_keep_prob = self.keep_prob)
                output, state = cell_out(concated_inp, state)
                if(i == 0): print("output shape: ", output.shape)
                if loop_function is not None:
                    prev = output
                outputs.append(output)

                #真实的模识
                modes, weights = modes_seq[i], decoder_weights[i]
                if(i == 0): print("shape of modes: ", modes.shape)
                common_mode, source_mode, kb_mode = tf.unstack(tf.transpose(modes))
                common_mode = tf.cast(common_mode, tf.float32)
                source_mode = tf.cast(source_mode, tf.float32)
                kb_mode = tf.cast(kb_mode, tf.float32)

                #预测的模识
                # pred_modes = mode_comp(state) #[batch, 3]
                pred_modes = mode_comp(tf.concat([inp, state[0], state[1]], 1))  # [batch, 3]
                decoder_predict_modes.append(pred_modes)
                # pred_modes = mode_comp(inp)  # [batch, 3]

                # pred_mode_crossent = tf.nn.softmax_cross_entropy_with_logits(pred_modes, tf.cast(modes, tf.float32))
                # pred_mode_crossent = tf.nn.sigmoid_cross_entropy_with_logits(pred_modes, tf.cast(modes, tf.float32))
                pred_mode_crossent = -1 * tf.log(pred_modes) * tf.cast(modes, tf.float32) #直接计算交叉熵
                pred_mode_crossent = tf.reduce_sum(pred_mode_crossent, 1)
                if(i == 0): print("shape of pred_mode_crossent", pred_mode_crossent.shape)
                if(i == 0): print("shape of weights", weights.shape)
                decoder_mode_loss.append(pred_mode_crossent * weights)

                if(i == 0): print("shape of pred_modes: ", pred_modes.shape)
                pred_common_mode, pred_source_mode, pred_kb_mode = tf.unstack(tf.transpose(pred_modes))

                # 训练的时候
                # 真实结果
                common_truths = tf.one_hot(decoder_truths[i], self.vocab_size)
                common_truths = tf.cast(common_truths, tf.float32)
                if(i == 0): print("common_truth shape: ", common_truths.shape)
                sources_locs = tf.cast(sous_locs_seq[i], tf.float32)
                if(i == 0): print("source_loc shape: ", sources_locs.shape)
                kbkb_locs = tf.cast(kbkb_locs_seq[i], tf.float32)
                if(i == 0): print("kbkb_locs shape: ", kbkb_locs.shape)

                #用这时的mode进行预测
                # common_truths = tf.transpose(tf.transpose(common_truths) * common_mode)
                # sources_locs = tf.transpose(tf.transpose(sources_locs) * source_mode)
                # kbkb_locs = tf.transpose(tf.transpose(kbkb_locs) * kb_mode)
                #用预测测mode进行计算
                common_truths = tf.transpose(tf.transpose(common_truths) * pred_common_mode)
                sources_locs = tf.transpose(tf.transpose(sources_locs) * pred_source_mode)
                kbkb_locs = tf.transpose(tf.transpose(kbkb_locs) * pred_kb_mode)
                entire_truths = tf.concat([common_truths, sources_locs, kbkb_locs], 1)

                # 预测结果
                common_logit = tf.matmul(output, W1) + B1  # 通用的预测 [batch, vocab_size]

                verbose = False
                if(i == 0): verbose = True
                # Size = [batch, seq]
                source_logit = source_atten_comp(tf.concat([state[0], state[1], hist_source_logit], 1), verbose)
                hist_source_logit += source_logit
                # hist_source_logit = source_logit

                # Size = [batch, fact]
                kbkb_logit = kbkb_atten_comp(tf.concat([state[0], state[1], hist_kbkb_logit], 1), verbose)
                hist_kbkb_logit += kbkb_logit
                # hist_kbkb_logit = kbkb_logit

                # common_logit = tf.transpose(tf.transpose(common_logit) * common_mode)
                # source_logit = tf.transpose(tf.transpose(source_logit) * source_mode)
                # kbkb_logit = tf.transpose(tf.transpose(kbkb_logit) * kb_mode)
                common_logit = tf.transpose(tf.transpose(common_logit) * pred_common_mode)
                source_logit = tf.transpose(tf.transpose(source_logit) * pred_source_mode)
                kbkb_logit = tf.transpose(tf.transpose(kbkb_logit) * pred_kb_mode)
                entire_logit = tf.concat([common_logit, source_logit, kbkb_logit], 1)
                decoder_logits.append(entire_logit)

                #计算交叉熵
                entire_truths = tf.cast(entire_truths, tf.float32) #需要行和为1
                entire_truths = tf.transpose(entire_truths) / (tf.reduce_sum(entire_truths, 1) + 1e-12)
                entire_truths = tf.transpose(entire_truths)
                source_crossent = tf.nn.softmax_cross_entropy_with_logits(
                    labels = entire_truths, logits = entire_logit)
                decoder_loss.append(source_crossent * weights)

                # 预测的时候
                decoder_gold_logits.append(entire_truths)
                decoder_pred_logits.append(entire_logit)
                decoder_predicts.append(tf.arg_max(entire_logit, 1))

                source_state = source_logit
                kb_state = kbkb_logit

            self.decoder_loss = tf.reduce_mean(decoder_loss)
            self.mode_loss = tf.reduce_mean(decoder_mode_loss)
            self.base_loss += tf.reduce_mean(decoder_loss)
            self.base_loss += self.mode_loss_ratio * tf.reduce_mean(decoder_mode_loss)

            self.decoder_outputs = outputs
            self.predict_truths = tf.stack(decoder_predicts)
            self.predict_modes = tf.one_hot(tf.argmax(tf.stack(decoder_predict_modes), axis = 2), 3)
            self.pred_logits = tf.stack(decoder_pred_logits)
            self.gold_logits = tf.stack(decoder_gold_logits)
            print("size of predict modes: ", self.predict_modes.shape)
        #####################################################################

        self.loss = self.base_loss + 0.01 * self.norm_loss# + 0.01 * tf.nn.l2_loss(self.embeddings)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)

        if self.optimizer_name.lower() == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'adadelta':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

       #  with tf.device("/cpu:0"):
            # self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            # self.saver = tf.train.Saver()

    def initilize(self, model_dir, session=None):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        self.saver = tf.train.Saver()
        if(self.is_train and (not self.load_prev_train)):
            print("Creating model with fresh parameters.")
            session.run(tf.global_variables_initializer())
            # test
            # print("saving model ... ")
            # self.saver.save(session, "model.ckpt")
            return
        else:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            print("test restoring model ... ")
            self.saver.restore(session, ckpt.model_checkpoint_path)

    def test(self, dataloader, model_dir):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        batch_size = 120
        with tf.Session(config=config) as session:
            self.initilize(model_dir, session)
            gold_questions, pred_answers = list(), list()
            for ques, ques_lens, facts, resp, source, kbkb, modes, weights, ques_words, real_facts in \
                    dataloader.get_batchs(data=dataloader.test_data, batch_size = batch_size, shuffle=False):
                feed = dict()
                feed[self.encoder_inputs] = ques
                feed[self.encoder_lengths] = ques_lens
                feed[self.facts_inputs] = facts
                feed[self.decoder_inputs] = resp
                feed[self.decoder_sources] = source
                feed[self.decoder_kbkbs] = kbkb
                feed[self.decoder_modes] = modes
                feed[self.decoder_modes] = np.ones(np.shape(modes), dtype=int)#全部一样
                feed[self.decoder_weights] = weights
                feed[self.keep_prob] = 1.0
                decoder_predicts = session.run([self.predict_truths], feed)
                decoder_predicts = decoder_predicts[0]
                # test here
                # self.answer_measure(decoder_predicts, resp, 20, dataloader, True)
                # return

                predicts = np.transpose(decoder_predicts)
                resp = resp[1:]
                gold_ouputs = np.transpose(resp)
                gold_inputs = np.transpose(ques)

                print 'vocab size: ', len(dataloader.vocab_list)
                print 'max ques len: ', dataloader.max_q_len
                for i in range(len(predicts)):
                    input, output, predict = gold_inputs[i], gold_ouputs[i], predicts[i]
                    q_words, facts = ques_words[i], real_facts[i]
                    words = list()
                    predict = postprocess(predict)
                    for id in predict:
                        if id == data_utils.EOS_ID:
                          break
                        if id < dataloader.vocab_size:
                            words.append(dataloader.vocab_list[id])
                        # else:
                        #     words.append(dataloader.vocab_list[0])
                        elif id < dataloader.vocab_size + dataloader.max_q_len:
                            q_id = id - dataloader.vocab_size
                            words.append(q_words[q_id])
                        else:
                            f_id = id - dataloader.vocab_size - dataloader.max_q_len
                            words.append(facts[f_id][1])

                    gold_questions.append(''.join(ques_words[i])) #真实问题
                    pred_answers.append(''.join(words)) #预测结果
                    pred_tmp = " ".join([reduce_print(j) for j in words])
                    gold_tmp = " ".join([reduce_print(dataloader.vocab_list[j]) for j in output])
                    # print('question: %s\ngolden:  %s\npredict: %s' % (
                    #   utils.ids_to_sentence(input.tolist(), dataloader.vocab_list, data_utils.EOS_ID, ''),
                    #   gold_tmp, pred_tmp))
                      # utils.ids_to_sentence(output.tolist(), dataloader.vocab_list, data_utils.EOS_ID, ''),
                      # ''.join(words)))
                # break
            #开始测试
            import evaluation
            evaluation.eval(pred_answers)

    def step(self, ques, ques_lens, facts, resp, source, kbkb, modes, weights, is_train, session=None, dataloader=None):
        feed = dict()
        feed[self.encoder_inputs] = ques
        feed[self.encoder_lengths] = ques_lens
        feed[self.facts_inputs] = facts
        feed[self.decoder_inputs] = resp
        feed[self.decoder_sources] = source
        feed[self.decoder_kbkbs] = kbkb
        feed[self.decoder_modes] = modes
        feed[self.decoder_weights] = weights
        if is_train: feed[self.keep_prob] = 0.8
        else: feed[self.keep_prob] = 1.0

        if is_train:
            predict_modes, predict_truths, loss, _, decoder_loss, mode_loss = session.run(
                [self.predict_modes, self.predict_truths, self.loss, self.update, self.decoder_loss, self.mode_loss],
                feed)
            return predict_modes, predict_truths, loss, decoder_loss, mode_loss
        else:
            predict_modes, predict_truths, loss, gold_logits, pred_logits = session.run(
                [self.predict_modes, self.predict_truths, self.loss, self.gold_logits, self.pred_logits], feed)
            # predict_truths = np.transpose(predict_truths)
            # print '\n'.join(utils.ids_to_sentence(predict_truths.tolist(),
            #                                       dataloader.vocab_list, data_utils.EOS_ID, ''))
            return predict_modes, predict_truths, loss, gold_logits, pred_logits

    def get_answer_lens(self, ans):
      ans = np.transpose(ans)
      # print(ans)
      anslens = []
      for ian in ans:
        endpos = [i for i, w in enumerate(ian) if w == 1]
        anslens.append(endpos[0])
      return anslens

    def mode_measure(self, pred, gold, anslens, batch_size, verbose = False):
      pred = np.transpose(pred, [1, 0, 2]).astype(int)
      gold = np.transpose(gold, [2, 0, 1])
      if(verbose):
        # print("---- Measure predicated mode:")
        # print("batch size = %d" % batch_size)
        # print("anslens: ")
        # qlen size: [batchsize]
        # print(anslens)
        # pred size: [timestep, batch size, 3] -> [batch size, timestep, 3]
        # print("pred shape", pred.shape)
        # gold size: [timestep, 3, batch size] -> [batch size, timestep, 3]
        # print("gold shape", gold.shape)
        # print("pred: ")
        # print(pred)
        # print("gold: ")
        # print(gold)
        pass
      # common
      ctp = 0
      cfn = 0
      cfp = 0
      ctn = 0
      ctotal = 0
      # question
      qtp = 0
      qfn = 0
      qfp = 0
      qtn = 0
      qtotal = 0
      # kb
      ktp = 0
      kfn = 0
      kfp = 0
      ktn = 0
      ktotal = 0
      confusion_matrix = np.zeros([3, 3], dtype = np.int)
      batch_size = gold.shape[0]
      for i in range(batch_size):
        for j in range(int(anslens[i])):
          # common
          predicted = np.argmax(pred[i][j])
          if(gold[i][j][0] == 1):
            ctotal += 1
            if(pred[i][j][0] == 1):
              ctp += 1
              confusion_matrix[0][0] += 1
            else:
              cfn += 1
              confusion_matrix[0][predicted] += 1
          else:
            if(pred[i][j][0] == 1): cfp += 1
            else: ctn += 1
          # question
          if(gold[i][j][1] == 1):
            qtotal += 1
            if(pred[i][j][1] == 1):
              qtp += 1
              confusion_matrix[1][1] += 1
            else:
              qfn += 1
              confusion_matrix[1][predicted] += 1
          else:
            if(pred[i][j][1] == 1): qfp += 1
            else: qtn += 1
          # kb
          if(gold[i][j][2] == 1):
            ktotal += 1
            if(pred[i][j][2] == 1):
              ktp += 1
              confusion_matrix[2][2] += 1
            else:
              kfn += 1
              confusion_matrix[2][predicted] += 1
          else:
            if(pred[i][j][2] == 1): kfp += 1
            else: ktn += 1
      def precfunc(tp, fp):
        if(tp + fp == 0): return 0
        else: return float(tp) / float(tp + fp)
      def reclfunc(tp, fn):
        if(tp + fn == 0): return 0
        else: return float(tp) / float(tp + fn)
      def f1msfunc(prec, recl):
        if(prec + recl == 0): return 0
        else: return 2 * prec * recl / (prec + recl)
      # print("-- measurement for validation mode")
      # common
      c_prec = precfunc(ctp, cfp)
      c_recl = reclfunc(ctp, cfn)
      c_f1ms = f1msfunc(c_prec, c_recl)
      c_accu = float(ctp + ctn) / float(ctp + ctn + cfp + cfn)
      # print("common mode: prec = %.4f, recl = %.4f, f1 = %.4f, acc = %.4f" % 
      #     (c_prec, c_recl, c_f1ms, c_accu))
      # question
      q_prec = precfunc(qtp, qfp)
      q_recl = reclfunc(qtp, qfn)
      q_f1ms = f1msfunc(q_prec, q_recl)
      q_accu = float(qtp + qtn) / float(qtp + qtn + qfp + qfn)
      # print("question:    prec = %.4f, recl = %.4f, f1 = %.4f, acc = %.4f" % 
      #     (q_prec, q_recl, q_f1ms, q_accu))
      # kb
      k_prec = precfunc(ktp, kfp)
      k_recl = reclfunc(ktp, kfn)
      k_f1ms = f1msfunc(k_prec, k_recl)
      k_accu = float(ktp + ktn) / float(ktp + ktn + kfp + kfn)
      # print("kb mode:     prec = %.4f, recl = %.4f, f1 = %.4f, acc = %.4f" % 
      #     (k_prec, k_recl, k_f1ms, k_accu))
      ret = dict()
      ret["c_prec"] = c_prec
      ret["c_recl"] = c_recl
      ret["c_f1ms"] = c_f1ms
      ret["c_accu"] = c_accu
      ret["q_prec"] = q_prec
      ret["q_recl"] = q_recl
      ret["q_f1ms"] = q_f1ms
      ret["q_accu"] = q_accu
      ret["k_prec"] = k_prec
      ret["k_recl"] = k_recl
      ret["k_f1ms"] = k_f1ms
      ret["k_accu"] = k_accu
      ret["c"] = np.array((ctp, ctn, cfp, cfn, ctotal))
      ret["q"] = np.array((qtp, qtn, qfp, qfn, qtotal))
      ret["k"] = np.array((ktp, ktn, kfp, kfn, ktotal))
      ret["confusion_matrix"] = confusion_matrix
      return ret

    def answer_measure(self, pred, gold, batch_size, dataloader, verbose = False):
      pred = np.transpose(pred)
      gold = gold[1:]
      gold = np.transpose(gold)
      indices = pred >= self.vocab_size
      pred[indices] = 0
      correct_predicted = np.sum(pred == gold)
      total_ans = pred.shape[0] * pred.shape[1]
      # print("pred: ", pred.shape)
      # print(pred)
      # print("gold: ", gold.shape)
      # print(gold)
      if(verbose):
        # print("pred / gold")
        # print '\n'.join(utils.ids_to_sentence(total_ans.tolist(), dataloader.vocab_list))
        for i in range(pred.shape[0]):
          preds = " ".join([reduce_print(dataloader.vocab_list[j]) for j in pred[i]])
          golds = " ".join([reduce_print(dataloader.vocab_list[j]) for j in gold[i]])
          print("pred: %s" % preds)
          print("gold: %s" % golds)
        print("accuracy: %.4f" % (float(correct_predicted) / total_ans))
        # print("predicted ")
        # print(pred)
        # print '\n'.join(utils.ids_to_sentence(pred.tolist(), dataloader.vocab_list))
        # print("golden ")
        # print(gold)
        # print '\n'.join(utils.ids_to_sentence(gold.tolist(), dataloader.vocab_list))
      # size = [maxlen, batch_size]
      return

    # if I predict this mode, whether I get it right
    def knowledge_measure(self, pred_truth, pred_mode, gold_src, gold_kb, gold_mode, gold_logits, pred_logits):
      def parse_pred_kb(pred_truth):
        # return pred_truth
        return pred_truth - self.vocab_size - self.max_src_len
      def parse_pred_src(pred_truth):
        # return pred_truth
        return pred_truth - self.vocab_size
      print("---- measurement for knowledge and source location prediction")
      # Size: [timestep, batch_size, totallen] -> [batch_size, timestep, totallen]
      gold_logits = np.transpose(gold_logits, [1, 0, 2])
      pred_logits = np.transpose(pred_logits, [1, 0, 2])
      # After: size = [batch_size, time_step]
      pred_truth = np.transpose(pred_truth)
      # Size: [timestep, input_length, batch_size] -> [batch_size, time_step, input_length]
      gold_src = gold_src[: -1]
      gold_kb = gold_kb[: -1]
      gold_src = np.transpose(gold_src, [2, 0, 1])
      gold_kb = np.transpose(gold_kb, [2, 0, 1])
      # transform to index
      gold_src = np.argmax(gold_src, axis = 2)
      gold_kb = np.argmax(gold_kb, axis = 2)
      # Size: [batch_size, timestep]
      pred_kb = parse_pred_kb(pred_truth)
      pred_src = parse_pred_src(pred_truth)
      # After size: [batch_size, timestep]
      pred_mode = np.argmax(np.transpose(pred_mode, [1, 0, 2]).astype(int), axis = 2)
      gold_mode = gold_mode[: -1]
      gold_mode = np.argmax(np.transpose(gold_mode, [2, 0, 1]), axis = 2)
      # test
      #print(" ---- pred_kb")
      #print(pred_kb)
      #print(" ---- pred_src")
      #print(pred_src)
      #print(" ---- pred_mode")
      #print(pred_mode)
      #print(" ---- gold_src")
      #print(gold_src)
      #print(" ---- gold_kb")
      #print(gold_kb)
      #print(" ---- gold_mode")
      #print(gold_mode)
      total_kb = 0
      total_src = 0
      correct_src = 0
      correct_kb = 0
      def show_pred_sentence(pred_mode, gold_mode, pred_src, gold_src, pred_kb, gold_kb):
        print("pred mode: ", pred_mode)
        print("gold mode: ", gold_mode)
        len_sent = len(pred_mode)
        tmp_pred_src = np.array(pred_src)
        tmp_gold_src = np.array(gold_src)
        tmp_pred_kb = np.array(pred_kb)
        tmp_gold_kb = np.array(gold_kb)
        for j in range(len_sent):
          if(pred_mode[j] == 0):
            tmp_pred_src[j] = -1
            tmp_pred_kb[j] = -1
          if(gold_mode[j] == 0):
            tmp_gold_src[j] = -1
            tmp_gold_kb[j] = -1
        print("    pred src: ", tmp_pred_src)
        print("    gold src: ", tmp_gold_src)
        print("    pred kb: ", tmp_pred_kb)
        print("    gold kb: ", tmp_gold_kb)
        return
      for i in range(np.shape(pred_mode)[0]):
        show_pred_sentence(pred_mode[i], gold_mode[i], pred_src[i], gold_src[i], pred_kb[i], gold_kb[i])
        # print("pred mode: ", pred_mode[i])
        # print("gold mode: ", gold_mode[i])
        # print("    pred src: ", pred_src[i])
        # print("    gold src: ", gold_src[i])
        # print("    pred kb:  ", pred_kb[i])
        # print("    gold kb:  ", gold_kb[i])
        # print("pred logits: ")
        # print(pred_logits[i])
        # print("gold logits: ")
        # print(gold_logits[i])
        # print("")
        for j in range(np.shape(pred_mode)[1]):
          # src
          if(gold_mode[i][j] == 1):
            total_src += 1
            if(pred_mode[i][j] == 1 and pred_src[i][j] == gold_src[i][j]): correct_src += 1
          # kb
          elif(gold_mode[i][j] == 2):
            total_kb += 2
            if(pred_mode[i][j] == 2 and pred_kb[i][j] == gold_kb[i][j]): correct_kb += 1
      print("accuracy for src prediction: %.4f" % (float(correct_src) / total_src))
      print("accuracy for kb prediction: %.4f" % (float(correct_kb) / total_kb))
      return

    def fit(self, dataloader, batch_size, epoch_size, step_per_checkpoint, model_dir):
        print("start training ... ")
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            self.initilize(model_dir, session)
            print "epoch size is : ", epoch_size
            print "batch size is : ", batch_size
            for epoch in xrange(1, epoch_size + 1):
                epoch_start = time.time()
                loss_sum = 0
                validation_loss_sum = 0
                run_id, loss_run = 0, 0
                decoder_loss_run = 0
                mode_loss_run = 0
                start = time.time()
                train_measure = dict()
                train_measure["c_prec"] = 0
                train_measure["c_recl"] = 0
                train_measure["c_f1ms"] = 0
                train_measure["c_accu"] = 0
                train_measure["q_prec"] = 0
                train_measure["q_recl"] = 0
                train_measure["q_f1ms"] = 0
                train_measure["q_accu"] = 0
                train_measure["k_prec"] = 0
                train_measure["k_recl"] = 0
                train_measure["k_f1ms"] = 0
                train_measure["k_accu"] = 0
                train_measure["c"] = np.zeros([5], dtype = np.int)
                train_measure["q"] = np.zeros([5], dtype = np.int)
                train_measure["k"] = np.zeros([5], dtype = np.int)
                train_measure["confusion_matrix"] = np.zeros([3, 3], dtype = np.int)
                for ques, ques_lens, facts, resp, source, kbkb, modes, weights, _, _ in \
                        dataloader.get_train_batchs(batch_size):
                    predict_modes, predict_truths, step_loss, decoder_loss, mode_loss = self.step(ques,
                        ques_lens, facts, resp, source, kbkb, modes, weights, True, session)
                    loss_sum += step_loss
                    loss_run += step_loss
                    decoder_loss_run += decoder_loss
                    mode_loss_run += mode_loss
                    run_id += 1
                    anslens = self.get_answer_lens(resp)
                    step_measure = self.mode_measure(predict_modes, modes, anslens, batch_size)
                    for mr_ in train_measure: train_measure[mr_] += step_measure[mr_]
                    if run_id % 100 == 0:
                        print "---- run %d loss %.5f decoder %.5f mode %.5f time %.2f" % (run_id, loss_run,
                            decoder_loss_run, mode_loss_run, time.time() - start)
                        print("-- measurement for train mode")
                        for mr_ in train_measure: train_measure[mr_] /= 100
                        print("common mode: prec = %.4f, recl = %.4f, f1 = %.4f, acc = %.4f" %
                            (train_measure["c_prec"], train_measure["c_recl"],
                              train_measure["c_f1ms"], train_measure["c_accu"]))
                        print("question:    prec = %.4f, recl = %.4f, f1 = %.4f, acc = %.4f" %
                            (train_measure["q_prec"], train_measure["q_recl"],
                              train_measure["q_f1ms"], train_measure["q_accu"]))
                        print("kb mode:     prec = %.4f, recl = %.4f, f1 = %.4f, acc = %.4f" %
                            (train_measure["k_prec"], train_measure["k_recl"],
                              train_measure["k_f1ms"], train_measure["k_accu"]))
                        total_case_num = train_measure["c"][4] + train_measure["q"][4] + train_measure["k"][4]
                        print("c - tp/ tn/ fp/ fn/ total/ frag", train_measure["c"],
                            float(train_measure["c"][4]) / total_case_num)
                        print("q - tp/ tn/ fp/ fn/ total/ frag", train_measure["q"],
                            float(train_measure["q"][4]) / total_case_num)
                        print("k - tp/ tn/ fp/ fn/ total/ frag", train_measure["k"],
                            float(train_measure["k"][4]) / total_case_num)
                        print("confusion matrix:")
                        print(train_measure["confusion_matrix"])
                        for mr_ in train_measure: train_measure[mr_] *= 0
                        start = time.time()
                        loss_run = 0
                        decoder_loss_run = 0
                        mode_loss_run = 0
                        # self.saver.save(session, os.path.join(model_dir, 'checkpoint'), 
                        # global_step = self.global_step)
                        for _ques, _ques_lens, _facts, _resp, _source, _kbkb, _modes, _weights, _, _ in \
                          dataloader.get_valid_batchs(20):
                            predict_modes, predict_truths, step_loss, gold_logits, pred_logits = self.step(
                                _ques, _ques_lens, _facts, _resp, _source, _kbkb, _modes, _weights,
                                False, session, dataloader)
                            anslens = self.get_answer_lens(_resp)
                            validation_measure = self.mode_measure(
                                np.array(predict_modes), np.array(_modes), anslens, 20, True)
                            print("-- measurement for validation mode")
                            print("common mode: prec = %.4f, recl = %.4f, f1 = %.4f, acc = %.4f" %
                                (validation_measure["c_prec"], validation_measure["c_recl"],
                                  validation_measure["c_f1ms"], validation_measure["c_accu"]))
                            print("question:    prec = %.4f, recl = %.4f, f1 = %.4f, acc = %.4f" %
                                (validation_measure["q_prec"], validation_measure["q_recl"],
                                  validation_measure["q_f1ms"], validation_measure["q_accu"]))
                            print("kb mode:     prec = %.4f, recl = %.4f, f1 = %.4f, acc = %.4f" %
                                (validation_measure["k_prec"], validation_measure["k_recl"],
                                  validation_measure["k_f1ms"], validation_measure["k_accu"]))
                            print("c - tp/ tn/ fp/ fn/ total", validation_measure["c"])
                            print("q - tp/ tn/ fp/ fn/ total", validation_measure["q"])
                            print("k - tp/ tn/ fp/ fn/ total", validation_measure["k"])
                            print("confusion matrix:")
                            print(validation_measure["confusion_matrix"])
                            self.answer_measure(
                                np.array(predict_truths), np.array(_resp), 20, dataloader, True)
                            self.knowledge_measure(
                                np.array(predict_truths), np.array(predict_modes), np.array(_source),
                                np.array(_kbkb), np.array(_modes), np.array(gold_logits), np.array(pred_logits))
                            print "validation-loss %.5f" % step_loss
                            print("")
                            # self.saver.save(session, model_dir + 'model.ckpt', global_step = run_id)
                            validation_loss_sum += step_loss
                            break
#                if epoch % step_per_checkpoint == 0:
#                    self.saver.save(session, os.path.join(model_dir, 'checkpoint'), global_step=self.global_step)
                self.saver.save(session, model_dir + 'model.ckpt', global_step=epoch)
                epoch_finish = time.time()
                print "\n---- epoch %d loss %.5f, validation loss = %.5f, time %.2f\n" % (epoch,
                    loss_sum, validation_loss_sum,epoch_finish - epoch_start)
