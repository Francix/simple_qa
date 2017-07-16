# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')


import os
import cPickle
import random
import jieba
import codecs
import numpy as np
from collections import Counter
import re
from utils import *

PAD, EOS, UNK, GO = ('_PAD', '_EOS', '_UNK', '_GO')
PAD_ID, EOS_ID, UNK_ID, GO_ID = range(4)


def is_digit_word(word):
    return re.match(r'\d+', word)

def relation_str(rel):
    return "REL:%s" % rel
def is_relation_str(rel_str):
    return rel_str.startswith('REL:')
def relation(rel_str):
    return rel_str[4:]

#以<开头为实体，从问句中拷贝
#以_开头为数字，从知识库中拷贝

def tokenizer(sent):
    if sent.find('<') != -1 and sent.find('>') != -1:
        s, e = sent.find('<'), sent.find('>')
        words = list(jieba.cut(sent[:s])) + [sent[s:e + 1]] + list(jieba.cut(sent[e + 1:]))
        ent = sent[s:e + 1]
    else:
        words = jieba.cut(sent)
        ent = ''
    new_words = list()
    for w in words:
        if is_digit_word(w):
            new_words.append('_' + w)
        else:
            new_words.append(w)
    words = [w.strip() for w in new_words]
    return [w for w in words if len(w) > 0], ent


class DataLoader(object):
    def __init__(self, data_path, min_frq=0, max_vocab_size=0):
        print 'deal with: ', data_path
        self.data_path = data_path
        self.min_frq = min_frq
        self.max_vocab_size = max_vocab_size
        self.max_fact_num = 4
        # self.max_q_len = 10
        # self.max_a_len = 10
        self.create_vocbulary(min_frq, max_vocab_size)
        self.load_kb()
        self.load_data()

    def create_vocbulary(self, min_frq=5, max_vocab_size=0):
        word_counts = Counter()
        with codecs.open(os.path.join(self.data_path, 'train_data'), 'rb', 'utf-8') as f:
            ques_word_lens, answ_word_lens = list(), list()
            for line in f:
                terms = [w.strip() for w in line.split('\t')]
                if len(terms) < 2:
                    continue
                ques, answ = terms[:2]
                ques_words, ques_ent = tokenizer(ques)
                answ_words, answ_ent = tokenizer(answ)
                word_counts.update(ques_words)
                word_counts.update(answ_words)
                ques_word_lens.append(len(ques_words))
                answ_word_lens.append(len(answ_words))
            self.max_q_len = max(ques_word_lens)
            self.max_a_len = max(answ_word_lens)
            print "max question tokens: ", self.max_q_len
            print "max answer tokens: ", self.max_a_len

        ori_words = [w for w in word_counts.keys()]
        print "orginal word size: ", len(ori_words)
        word_frqs = filter_vocab(word_counts, min_frq, max_vocab_size)
        fil_words = [w for w in word_frqs.keys()
                     if not w.startswith('_') and not w.startswith('<')]
        print "basic word size: ", len(fil_words)

        # 导入知识库
        entities, relations = set(), set()
        with codecs.open(os.path.join(self.data_path, 'kb_facts'), 'rb', 'utf-8') as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                sub, rel, obj = [w.strip() for w in parts[:3]]
                entities.add(sub)
                if rel == '性别':
                    entities.add(obj)
                else:
                    entities.add('_' + obj) #数字
                relations.add(relation_str(rel))
        self.kb_relations = list(relations)
        self.kb_entities = list(entities)
        print "kb relation size: ", len(self.kb_relations)
        print "kb entity size: ", len(self.kb_entities)
        fil_kb_entities = [w for w in self.kb_entities if not w.startswith('_') and not w.startswith('<')]

        total_words = fil_words + self.kb_relations + fil_kb_entities

        self.vocab_list = [PAD, EOS, UNK, GO] + total_words
        self.vocab_size = len(self.vocab_list)
        self.vocab = dict([(x, y) for (y, x) in enumerate(self.vocab_list)])
        print "vocab size: ", self.vocab_size

        # with codecs.open(os.path.join(self.data_path, "syngenqa_vocab_%d" % min_frq), 'wb', 'utf-8') as vocab_file:
        #     for w in self.vocab_list:
        #         vocab_file.write(w + '\n')

    def load_kb(self):
        ent_facts = dict()
        for name in ['kb_facts_zero_shot_data', 'kb_facts']:
            with codecs.open(os.path.join(self.data_path, name), 'rb', 'utf-8') as f:
                for line in f:
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue
                    sub, rel, obj = [w.strip() for w in parts[:3]]
                    sub = '_' + sub
                    if rel != '性别':
                        obj = '_' + obj
                    rel = relation_str(rel)
                    facts = ent_facts.get(sub, list())
                    facts.append((rel, obj))
                    ent_facts[sub] = facts
        print "ent num: ", len(ent_facts)
        print "fact num: ", sum([len(x) for x in ent_facts.values()])
        # 按照关系进行排序
        self.entity_facts = dict()
        for sub in ent_facts.keys():
            self.entity_facts[sub] = sorted(ent_facts[sub], key=lambda x: x[0])
        # 不排序
        # self.entity_facts = ent_facts

    def load_data(self):
        def filter_data(data):
            new_data = list()
            for ques_ids, answ_ids, answ_modes, answ4ques_locs, answ4kbkb_locs, \
                cand_fact_ids, ques_words, real_facts in data:
                if len(ques_ids) > self.max_q_len:
                    continue
                if len(answ_ids) > self.max_a_len:
                    continue
                new_data.append((ques_ids, answ_ids, answ_modes, answ4ques_locs, answ4kbkb_locs,
                                 cand_fact_ids, ques_words, real_facts))
            return new_data

        datas = list()
        for name in ['train', 'valid', 'qa_pairs_zero_shot']:
            data = list()
            print "load data from ", name
            with codecs.open(os.path.join(self.data_path, '%s_data' % name), 'rb', 'utf-8') as f:
                debug = 1
                for line in f:
                    terms = [w.strip() for w in line.split('\t')]
                    if len(terms) < 2:
                        continue
                    ques, answ = terms[:2]
                    ques_words, ques_ent = tokenizer(ques)
                    answ_words, answ_ent = tokenizer(answ)
                    ques_ids = [self.vocab.get(w, UNK_ID) for w in ques_words]  # 没出现的用UNK代替

                    ent_facts = self.entity_facts.get('_' + ques_ent, list()) #没是四个
                    real_facts = list()
                    cand_fact_ids = list()
                    for rel, obj in ent_facts:
                        rel_id = self.vocab.get(rel, PAD_ID)
                        obj_id = self.vocab.get(obj, PAD_ID)
                        if(debug == 1):
                            print("obj = %s, padid = %d" % (obj, PAD_ID))
                            debug += 1
                        cand_fact_ids.append((rel_id, obj_id))
                        real_facts.append((rel, obj))
                    for i in range(self.max_fact_num - len(ent_facts)):
                        cand_fact_ids.append((PAD_ID, PAD_ID))
                        real_facts.append((PAD, PAD))
                    fact_objs = [x[1] for x in ent_facts]

                    answ_ids, answ_modes = list(), list()
                    answ4ques_locs, answ4kbkb_locs = list(), list()

                    for word in answ_words:
                        if word.startswith('<'): #来源为句子
                            answ_ids.append(PAD_ID)
                            answ_modes.append(1)
                            ques_locs = list()
                            for q_w in ques_words:
                                if q_w == word:
                                    ques_locs.append(1)
                                else:
                                    ques_locs.append(0)
                            answ4ques_locs.append(ques_locs)
                            answ4kbkb_locs.append(list())
                        elif word.startswith('_'):#来源为知识库
                            answ_ids.append(PAD_ID)
                            answ_modes.append(2)
                            kb_locs = list()
                            for k_w in fact_objs:
                                if k_w == word:
                                    kb_locs.append(1)
                                else:
                                    kb_locs.append(0)
                            answ4ques_locs.append(list())
                            answ4kbkb_locs.append(kb_locs)
                        else: #来源为推理
                            answ_ids.append(self.vocab[word]) #保持就要检查了
                            answ_modes.append(0)
                            answ4ques_locs.append(list())
                            answ4kbkb_locs.append(list())

                    data.append((ques_ids, answ_ids, answ_modes, answ4ques_locs, answ4kbkb_locs,
                                 cand_fact_ids, ques_words, real_facts))
            datas.append(data)
            print "entire line : ", len(data)
        train_data, valid_data, test_data = datas
        self.train_data = filter_data(train_data)
        print "train instances: ", len(self.train_data)
        self.valid_data = filter_data(valid_data)
        print "valid instances: ", len(self.valid_data)
        self.test_data = filter_data(test_data)
        print "test instances: ", len(self.test_data)

    def get_train_batchs(self, batch_size=120, shuffle=True):
        return self.get_batchs(batch_size, self.train_data, shuffle)

    def get_valid_batchs(self, batch_size=50):
        return self.get_batchs(batch_size, self.valid_data)

    def get_batchs(self, batch_size=120, data=None, shuffle=True):
        # print 'batch size is : ', batch_size
        if data is None:
            data = self.train_data
        if shuffle:
            random.shuffle(data)
        batch_num = len(data) // batch_size
        for kk in xrange(batch_num + 1):
            begin, end = batch_size * kk, batch_size * (kk + 1)
            if begin >= end:
                continue
            if end > len(data):
                end = len(data)
            batch_data = data[begin:end]
            inst_size = len(batch_data)

            encoder_inputs = np.zeros((inst_size, self.max_q_len), dtype=int)
            encoder_lengths = np.zeros(inst_size)
            fact_inputs = np.zeros((inst_size, self.max_fact_num, 2), dtype=int)

            decoder_inputs = np.zeros((inst_size, self.max_a_len + 2), dtype=int)
            decoder_sources = np.zeros((inst_size, self.max_a_len + 2, self.max_q_len), dtype=int)
            decoder_kbkbs = np.zeros((inst_size, self.max_a_len + 2, self.max_fact_num), dtype=int)
            decoder_modes = np.zeros((inst_size, self.max_a_len + 2, 3), dtype=int) #0,1,2三种模式
            decoder_weights = np.ones((inst_size, self.max_a_len + 2), dtype=float)
            inst_ques_words, inst_real_facts = list(), list()
            for i in xrange(0, inst_size):
                ques_ids, answ_ids, answ_modes, answ4ques_locs, answ4kbkb_locs, ent_facts, \
                ques_words, real_facts = batch_data[i]

                inst_ques_words.append(ques_words + [PAD] * (self.max_q_len - len(ques_words)))
                inst_real_facts.append(real_facts)

                encoder_inputs[i] = np.array(ques_ids + [PAD_ID] * (self.max_q_len - len(ques_ids)))
                encoder_lengths[i] = len(ques_ids)
                for f_id in range(len(ent_facts)):
                    fact_inputs[i, f_id, 0:2] = np.array(ent_facts[f_id])

                decoder_inputs[i] = np.array([GO_ID] + answ_ids + [EOS_ID] +
                                             [PAD_ID] * (self.max_a_len - len(answ_ids)))
                #######################################
                for m_id, source in enumerate(answ4ques_locs):
                    if len(source) == len(ques_ids):
                        decoder_sources[i, m_id, 0:len(source)] = source
                for m_id, kbkb in enumerate(answ4kbkb_locs):
                    if len(kbkb) == self.max_fact_num:
                        decoder_kbkbs[i, m_id, 0:len(kbkb)] = kbkb
                #######################################
                for m_id, mode in enumerate(answ_modes):
                    if mode == 1:
                        tmp = [0, 1, 0]
                    elif mode == 2:
                        tmp = [0, 0, 1]
                    else:
                        tmp = [1, 0, 0]
                    decoder_modes[i, m_id, 0:3] = np.array(tmp)
                #######################################
                # decoder_weights[i, 0:(len(answ_ids) + 1)] = 1.0  # 最后一个权重为0
                # decoder_weights[i, (len(answ_ids) + 1):] = 0.0  # 最后一个权重为0

            # this np.transpose() in fact put the batch size into the last dimension 
            yield np.transpose(encoder_inputs), encoder_lengths, np.transpose(fact_inputs, (1, 2, 0)), \
                  np.transpose(decoder_inputs), np.transpose(decoder_sources, (1, 2, 0)), \
                  np.transpose(decoder_kbkbs, (1, 2, 0)), np.transpose(decoder_modes, (1, 2, 0)),\
                  np.transpose(decoder_weights), inst_ques_words, inst_real_facts
            # yield encoder_inputs, encoder_lengths, fact_inputs, \
            #       decoder_inputs, decoder_sources, decoder_kbkbs, decoder_modes,
            #       decoder_weights, inst_ques_words, inst_real_facts

if __name__ == '__main__':
    data_path = '../syndata/'
    min_frq = 0

    # dataloader = DataLoader(data_path, min_frq)
    # ques_ids, answ_ids, answ_modes, answ4ques_locs, answ4kbkb_locs, ent_facts = random.choice(dataloader.train_data)
    # print ques_ids
    # print ids_to_sentence(ques_ids, dataloader.vocab_list)
    # print answ_ids
    # print ids_to_sentence(answ_ids, dataloader.vocab_list)
    # print answ_modes
    # print answ4ques_locs
    # print answ4kbkb_locs
    # print ent_facts

    dump_path = os.path.join(data_path, "syncompqa_temp_data_%d.pkl" % min_frq)
    if not os.path.exists(dump_path):
        dataloader = DataLoader(data_path, min_frq)
        cPickle.dump(dataloader, open(dump_path, "wb"))
    else:
        dataloader = cPickle.load(open(dump_path, "rb"))

    for ques, ques_lens, facts, resp, source, kbkb, modes, weights, ques_words, real_facts in dataloader.get_train_batchs(10):
        print("---- questions:")
        print ques
        print np.shape(ques)
        print '\n'.join(ids_to_sentence(ques.tolist(), dataloader.vocab_list))
        print("---- question length:")
        print ques_lens
        print np.shape(ques_lens)
        print("---- facts:")
        print facts
        print np.shape(facts)
        print '\n'.join(ids_to_sentence(facts.tolist(), dataloader.vocab_list))
        print("---- resp:")
        print resp
        print np.shape(resp)
        print '\n'.join(ids_to_sentence(resp.tolist(), dataloader.vocab_list))
        print("---- now source: ")
        print source
        print np.shape(source)
        print("---- now kbkb: ")
        print kbkb
        print np.shape(kbkb)
        print("---- now modes: ")
        print modes
        print np.shape(modes)
        print("---- now weights: ")
        print weights
        print np.shape(weights)
        break





