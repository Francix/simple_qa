# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import argparse
import os, cPickle
import data_utils
import models
from utils import *



parser = argparse.ArgumentParser()
arg = parser.add_argument
# this --data-path is expecting a file, not a directory 
arg('--data-path', default="../syndata/")
arg('--cell', default='lstm', help='cell type: lstm, gru')
arg('--model-path', default="../saved/")

arg('--embedding-size', type=int, default=300)
arg('--vocab-min-frq', type=int, default=0)
arg('--state-size', type=int, default=1024, help='size of hidden states')
arg('--num-layers', type=int, default=1, help='number of hidden layers') #暂时不支持多层处理
arg('--num-samples', type=int, default=128, help='number of sampled softmax')
arg('--max-gradient-norm', type=float, default=5.0, help='gradient norm is commonly set as 5.0 or 15.0')
arg('--optimizer', default='adam', help='Optimizer: adam, adadelta')
arg('--learning-rate', type=float, default=0.01)
arg('--batch-size', type=int, default=128)
arg('--epoch-size', type=int, default=10)
arg('--checkpoint-step', type=int, default=1, help='do validation and save after each this many of steps.')
arg('--mode', type = str, default = "train")
arg('--gpu', type = str, default = "0")
args = parser.parse_args()


def main():
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    print("delete previous model")
    os.system("rm -rf " + args.model_path + "*")
    print "save model to : ", args.model_path

    dump_path = os.path.join(args.data_path, "syncompqa_data_%d.pkl" % args.vocab_min_frq)
    if not os.path.exists(dump_path):
        dataloader = data_utils.DataLoader(args.data_path, args.vocab_min_frq)
        cPickle.dump(dataloader, open(dump_path, "wb"))
    else:
        dataloader = cPickle.load(open(dump_path, "rb"))
    print "train instance: ", len(dataloader.train_data)
    print("vocab size: %d" % dataloader.vocab_size)
    print("max fact num: %d" % dataloader.max_fact_num)

    print("start to build the model .. ")
    model = models.CompQAModel(vocab_size = dataloader.vocab_size,
                               embedding_size = args.embedding_size,
                               max_src_len = dataloader.max_q_len,
                               max_des_len = dataloader.max_a_len,
                               max_fact_num = dataloader.max_fact_num,
                               state_size = args.state_size,
                               num_layers = args.num_layers,
                               num_samples = args.num_samples,
                               max_grad_norm = args.max_gradient_norm,
                               is_train = True,
                               cell_type = args.cell,
                               optimizer_name = args.optimizer,
                               learning_rate = args.learning_rate)
    print "begining to train model .. "
    model.fit(dataloader, args.batch_size, args.epoch_size, args.checkpoint_step, args.model_path)


def test():
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    print "load mode from : ", args.model_path

    dump_path = os.path.join(args.data_path, "syncompqa_data_zero_%d.pkl" % args.vocab_min_frq)
    if not os.path.exists(dump_path):
        dataloader = data_utils.DataLoader(args.data_path, args.vocab_min_frq)
        cPickle.dump(dataloader, open(dump_path, "wb"))
    else:
        dataloader = cPickle.load(open(dump_path, "rb"))
    print "test instance: ", len(dataloader.test_data)
    print "vocab size: ", dataloader.vocab_size

    # model = models.CompQAModel(dataloader.vocab_size, args.embedding_size,
    #                            dataloader.max_q_len, dataloader.max_a_len,
    #                            dataloader.max_fact_num,
    #                            args.state_size, args.num_layers, args.num_samples,
    #                            args.max_gradient_norm, False, args.cell,
    #                            args.optimizer, args.learning_rate)
    model = models.CompQAModel(vocab_size = dataloader.vocab_size,
                               embedding_size = args.embedding_size,
                               max_src_len = dataloader.max_q_len,
                               max_des_len = dataloader.max_a_len,
                               max_fact_num = dataloader.max_fact_num,
                               state_size = args.state_size,
                               num_layers = args.num_layers,
                               num_samples = args.num_samples,
                               max_grad_norm = args.max_gradient_norm,
                               is_train = False,
                               cell_type = args.cell,
                               optimizer_name = args.optimizer,
                               learning_rate = args.learning_rate)
    print "begining to test model"
    model.test(dataloader, args.model_path)

if __name__ == '__main__':
  if(args.gpu == "1"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  if(args.mode == "train"): main()
  else: test()
