# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.framework import dtypes
import numpy as np
import time
import os
import utils
import data_utils

class CompQAModel(object):
    def __init__(self, vocab_size = 10000, embedding_size = 256, max_src_len = 7, max_des_len = 11, max_fact_num = 4,
                 state_size=500, num_layers=1, num_samples=64,
                 max_grad_norm=5.0, is_train=True, cell_type='LSTM',
                 optimizer_name='Adam', learning_rate=0.05):
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
        self.is_train = is_train
        self.global_step = tf.Variable(0, trainable=False)

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
        print("shape of encoder inputs: ", self.encoder_inputs.shape)
        self.encoder_lengths = tf.placeholder(tf.int32, [None], 'encoder_lengths')
        # what is this?
        self.facts_inputs = tf.placeholder(tf.int32, [self.max_fact_num, 2, None], 'input_facts')

        self.decoder_inputs = tf.placeholder(tf.int32, [self.max_des_len + 2, None], 'decoder_inputs')
        # what is this?   
        self.decoder_sources = tf.placeholder(tf.int32, [self.max_des_len + 2, self.max_src_len, None], 'decoder_sources')
        # what is this?   
        self.decoder_kbkbs = tf.placeholder(tf.int32, [self.max_des_len + 2, self.max_fact_num, None], 'decoder_kbkbs')
        self.decoder_modes = tf.placeholder(tf.int32, [self.max_des_len + 2, 3, None], 'decoder_modes')

        self.decoder_weights = tf.placeholder(tf.float32, [self.max_des_len + 2, None], 'decoder_weights')

        ###################对问句进行处理##########################
        # -- TB test 
        with tf.device("/cpu:0"):
            embedded_encoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            print("size of encoder inputs, before unstack: ", embedded_encoder_inputs.shape)
            # embedded_encoder_inputs = tf.unstack(embedded_encoder_inputs)
            # print("encoder inputs: ", len(embedded_encoder_inputs), embedded_encoder_inputs[0].shape)
        self.input_output_size = self.state_size * 2
        self.input_state_size = self.state_size * 2
        if self.cell_type.lower() == 'lstm':
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.state_size, state_is_tuple=False)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.state_size, state_is_tuple=False)
            self.input_state_size *= 2
        elif self.cell_type.lower() == 'gru':
            cell_fw = rnn_cell.GRUCell(self.state_size)
            cell_bw = rnn_cell.GRUCell(self.state_size)
        else:
            cell_fw = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
            cell_bw = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
        (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, embedded_encoder_inputs, sequence_length=self.encoder_lengths, dtype=dtypes.float32, time_major = True)
        print("output size: ", output_fw.shape, output_fw.shape)
        self.encoder_outputs = tf.concat([output_fw, output_bw], 2)
        print("after concat: ", self.encoder_outputs.shape)
        self.encoder_outputs = tf.unstack(self.encoder_outputs)
        print("after unstack, length = ", len(self.encoder_outputs), "shape: ", self.encoder_outputs[0].shape)
        self.encoder_states = tf.concat([output_state_fw, output_state_bw], 1)
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
            cell_out = tf.nn.rnn_cell.BasicLSTMCell(self.input_output_size, state_is_tuple=False)
            self.output_state_size *= 2
        elif self.cell_type.lower() == 'gru':
            cell_out = rnn_cell.GRUCell(self.input_output_size)
        else:
            cell_out = tf.nn.rnn_cell.BasicRNNCell(self.input_output_size)

        ###用于对输出状态预测目标词
        W1 = tf.get_variable("proj_w1", [self.output_output_size, self.vocab_size])
        B1 = tf.get_variable("proj_b1", [self.vocab_size])
        W2 = tf.get_variable("proj_w2", [self.output_output_size, self.max_src_len + self.max_fact_num])
        B2 = tf.get_variable("proj_b2", [self.max_src_len + self.max_fact_num])
        W_all = tf.concat([W1, W2], 1)
        B_all = tf.concat([B1, B2], 0)
        output_projection = (W_all, B_all)
        self.norm_loss += tf.nn.l2_loss(W_all)
        self.norm_loss += tf.nn.l2_loss(B_all)

        #输入(prev, loc1, loc2)表示上一步预测的词ID和从源句子和KB中copy的位置
        # What is this?
        def loop_function(prev, _):
            prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
            prev_symbol = tf.argmax(prev, 1) #[batch_size]
            emb_prev = tf.nn.embedding_lookup(self.embeddings, prev_symbol) #[batch_size, embedding_size]
            return emb_prev

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
#            _inputs = [tf.squeeze(_one) for _one in _inputs] # [batch, 1]
#            if(verbose): print("inputs length ", len(_inputs), " shape ", _inputs[0].shape)
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

            state = self.encoder_states  # [batch, out_state_size]

            source_state = tf.zeros((tf.shape(self.encoder_inputs)[1], self.max_src_len))
            kb_state = tf.zeros((tf.shape(self.encoder_inputs)[1], self.max_fact_num))

            hist_source_logit = tf.zeros((tf.shape(self.encoder_inputs)[1], self.max_src_len))
            hist_kbkb_logit = tf.zeros((tf.shape(self.encoder_inputs)[1], self.max_fact_num))

            decoder_logits = []
            decoder_predicts = []
            outputs = []
            prev = None
            decoder_loss, decoder_mode_loss = [], []
            print("starting end2end decoding ... ")
            for i, inp in enumerate(embedded_decoder_inputs[:-1]):  # 输出端的每个输入词，第一个是GO
                if not self.is_train and loop_function is not None and prev is not None:  # 测试的时候才用
                    # question: why I should use this "with" statement?
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, i)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # output, state = cell_out(inp, state)
                if(i == 1): print("source state shape: ", source_state.shape)
                source_state = tf.reshape(source_state, [-1, self.max_src_len])
                if(i == 1): print("after reshape: ", source_state.shape)
                if(i == 1): print("kb state shape: ", kb_state.shape)
                kb_state = tf.reshape(kb_state, [-1, self.max_fact_num])
                if(i == 1): print("after reshape: ", kb_state.shape)
                output, state = cell_out(tf.concat([inp, source_state, kb_state, avg_embedded_facts], 1), state)
                if(i == 1): print("output shape: ", output.shape)
                if loop_function is not None:
                    prev = output
                outputs.append(output)

                #真实的模识
                modes, weights = modes_seq[i], decoder_weights[i]
                if(i == 1): print("shape of modes: ", modes.shape)
                common_mode, source_mode, kb_mode = tf.unstack(tf.transpose(modes))
                common_mode = tf.cast(common_mode, tf.float32)
                source_mode = tf.cast(source_mode, tf.float32)
                kb_mode = tf.cast(kb_mode, tf.float32)

                #预测的模识
                # pred_modes = mode_comp(state) #[batch, 3]
                pred_modes = mode_comp(tf.concat([inp, state], 1))  # [batch, 3]
                # pred_modes = mode_comp(inp)  # [batch, 3]

                # pred_mode_crossent = tf.nn.softmax_cross_entropy_with_logits(pred_modes, tf.cast(modes, tf.float32))
                # pred_mode_crossent = tf.nn.sigmoid_cross_entropy_with_logits(pred_modes, tf.cast(modes, tf.float32))
                pred_mode_crossent = -1 * tf.log(pred_modes) * tf.cast(modes, tf.float32) #直接计算交叉熵
                pred_mode_crossent = tf.reduce_sum(pred_mode_crossent, 1)
                if(i == 1): print("shape of pred_mode_crossent", pred_mode_crossent.shape)
                if(i == 1): print("shape of weights", weights.shape)
                decoder_mode_loss.append(pred_mode_crossent * weights)

                pred_modes = tf.reshape(pred_modes, [-1, 3])
                pred_common_mode, pred_source_mode, pred_kb_mode = tf.unstack(tf.transpose(pred_modes))

                # 训练的时候
                # 真实结果
                common_truths = tf.one_hot(decoder_truths[i], self.vocab_size)
                common_truths = tf.cast(common_truths, tf.float32)
                if(i == 1): print("common_truth shape: ", common_truths.shape)
                sources_locs = tf.cast(sous_locs_seq[i], tf.float32)
                kbkb_locs = tf.cast(kbkb_locs_seq[i], tf.float32)

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
                if(i == 1): verbose = True
                source_logit = source_atten_comp(tf.concat([state, hist_source_logit], 1), verbose)  # [batch, seq]
                hist_source_logit += source_logit
                # hist_source_logit = source_logit

                kbkb_logit = kbkb_atten_comp(tf.concat([state, hist_kbkb_logit], 1), verbose)  # [batch, fact]
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
                source_crossent = tf.nn.softmax_cross_entropy_with_logits(labels = entire_truths, logits = entire_logit)
                decoder_loss.append(source_crossent * weights)

                # 预测的时候
                decoder_predicts.append(tf.arg_max(entire_logit, 1))

                source_state = source_logit
                kb_state = kbkb_logit

            self.base_loss += tf.reduce_mean(decoder_loss)
            self.base_loss += 2.0 * tf.reduce_mean(decoder_mode_loss)

            self.decoder_outputs = outputs
            self.predict_truths = tf.stack(decoder_predicts)
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

        with tf.device("/cpu:0"):
            self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

    def initilize(self, model_dir, session=None):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters.")
            session.run(tf.initialize_all_variables())

    def test(self, dataloader, model_dir):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            self.initilize(model_dir, session)
            gold_questions, pred_answers = list(), list()
            for ques, ques_lens, facts, resp, source, kbkb, modes, weights, ques_words, real_facts in \
                    dataloader.get_batchs(data=dataloader.test_data, shuffle=False):
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
                decoder_predicts = session.run([self.predict_truths], feed)
                decoder_predicts = decoder_predicts[0]

                predicts = np.transpose(decoder_predicts)
                gold_ouputs = np.transpose(resp)
                gold_inputs = np.transpose(ques)

                print 'vocab size: ', len(dataloader.vocab_list)
                print 'max ques len: ', dataloader.max_q_len
                for i in range(len(predicts)):
                    input, output, predict = gold_inputs[i], gold_ouputs[i], predicts[i]
                    q_words, facts = ques_words[i], real_facts[i]
                    words = list()
                    for id in predict:
                        if id in [data_utils.EOS_ID, data_utils.PAD_ID]:
                            break
                        if id < dataloader.vocab_size:
                            words.append(dataloader.vocab_list[id])
                        elif id < dataloader.vocab_size + dataloader.max_q_len:
                            q_id = id - dataloader.vocab_size
                            words.append(q_words[q_id])
                        else:
                            f_id = id - dataloader.vocab_size - dataloader.max_q_len
                            words.append(facts[f_id][1])

                    gold_questions.append(''.join(ques_words[i])) #真实问题
                    pred_answers.append(''.join(words)) #预测结果
                    print '%s\t%s\t%s' % (utils.ids_to_sentence(input.tolist(), dataloader.vocab_list, data_utils.EOS_ID, ''),
                                          utils.ids_to_sentence(output.tolist(), dataloader.vocab_list, data_utils.EOS_ID, ''),
                                          ''.join(words))
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

        if is_train:
            loss, _ = session.run([self.loss, self.update], feed)
            return None, loss
        else:
            predict_truths, loss = session.run([self.predict_truths, self.loss], feed)
            # predict_truths = np.transpose(predict_truths)
            # print '\n'.join(utils.ids_to_sentence(predict_truths.tolist(),
            #                                       dataloader.vocab_list, data_utils.EOS_ID, ''))
            return predict_truths, loss

    def fit(self, dataloader, batch_size, epoch_size, step_per_checkpoint, model_dir):
        print("start training ... ")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            self.initilize(model_dir, session)
            print "epoch size is : ", epoch_size
            print "batch size is : ", batch_size
            for epoch in xrange(1, epoch_size + 1):
                epoch_start = time.time()
                loss_sum = 0
                run_id, loss_run = 0, 0
                start = time.time()
                for ques, ques_lens, facts, resp, source, kbkb, modes, weights, _, _ in \
                        dataloader.get_train_batchs(batch_size):
                    _, step_loss = self.step(ques, ques_lens, facts, resp, source, kbkb, modes, weights, True, session)
                    loss_sum += step_loss
                    loss_run += step_loss
                    run_id += 1
                    if run_id % 100 == 0:
                        print "run %d loss %.5f time %.2f" % (run_id, loss_run, time.time() - start)
                        start = time.time()
                        loss_run = 0
                        self.saver.save(session, os.path.join(model_dir, 'checkpoint'), global_step=self.global_step)
                        for _ques, _ques_lens, _facts, _resp, _source, _kbkb, _modes, _weights, _, _ in \
                                dataloader.get_valid_batchs(20):
                            predict_truths, step_loss = self.step(_ques, _ques_lens, _facts,
                                                                  _resp, _source, _kbkb, _modes, _weights,
                                                                  False, session, dataloader)
                            print "validation-loss %.5f" % step_loss
                            break
                if epoch % step_per_checkpoint == 0:
                    self.saver.save(session, os.path.join(model_dir, 'checkpoint'), global_step=self.global_step)
                epoch_finish = time.time()
                print "epoch %d loss %.5f time %.2f" % (epoch, loss_sum, epoch_finish - epoch_start)
