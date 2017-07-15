#encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import codecs
import os
import random

def load_kb(file_name="/home/hesz/complexqa/syndata/kb_facts_zero_shot_data"):
    ent_facts = dict() #key为实体，facts为dict
    with codecs.open(file_name, encoding='utf-8') as f:
        for line in f:
            terms = line.strip().split('\t')
            if len(terms) < 3:
                continue
            sub, prop, obj = [s.strip() for s in terms[:3]]
            facts = ent_facts.get(sub, dict())
            facts[prop] = obj
            ent_facts[sub] = facts
    return ent_facts


def extract_ent(line):
    s, e = line.find('<'), line.find('>')
    if s != -1 and e != -1:
        return line[s:e + 1]
    else:
        return ''

import re
year_re = re.compile(u'.*?(\d+)年.*?')
month_re = re.compile(u'.*?(\d+)月.*?')
day_re = re.compile(u'.*?(\d+)[号|日].*?')

def extract_prop_value(line, re_patt):
    matchObj = re.match(re_patt, line)
    if matchObj:
        return matchObj.group(1)
    else:
        return ''

def extract_facts(line):
    facts = dict()
    #抽取性别
    ent = extract_ent(line)
    if ent != '':
        if ent in [u'<李牧>', u'<何世柱>']:
            gender = u'男'
        elif ent in [u'<李芳>', u'<余巧玲>']:
            gender = u'女'
        elif int(ent[5:-1]) > 40000:
            gender = u'女'
        else:
            gender = u'男'
    else:
        if line.find(u'他') != -1:
            gender = u'男'
        else:
            gender = u'女'
    facts[u'性别'] = gender

    year = extract_prop_value(line, year_re)
    month = extract_prop_value(line, month_re)
    day = extract_prop_value(line, day_re)
    if year != '':
        facts[u'出生年'] = year
    if month != '':
        facts[u'出生月'] = month
    if day != '':
        facts[u'出生日'] = day
    return facts

#判断各个属性的正确情况
def correct_props(preds, golds):
    props = [u'性别', u'出生年', u'出生月', u'出生日']
    values = list()
    for prop in props:
        if prop not in preds:
            if prop not in golds:
                values.append(None)
            else:
                values.append(0.0)
            continue
        if prop not in golds or preds[prop] == golds[prop]:
            values.append(1.0)
        else:
            values.append(0.0)
    return values


def eval(predicts, kbs=None):
    if kbs is None:
        kbs = load_kb()
    questions, answers = list(), list()
    with codecs.open("/home/hesz/complexqa/syndata/qa_pairs_zero_shot_data", encoding='utf-8') as f:
        for line in f:
            terms = line.strip().split('\t')
            if len(terms) < 2:
                continue
            questions.append(terms[0].strip())
            answers.append(terms[1].strip())

    g_w, y_w, m_w, d_w = 0, 0, 0, 0
    g_t, y_t, m_t, d_t = 0, 0, 0, 0
    r_num, t_num = 0, 0
    recall = 0
    for ques, answ, pred in zip(questions, answers, predicts):
        kb_facts = kbs[extract_ent(ques)]  # 这是的事实
        real_facts = extract_facts(answ)
        pred_facts = extract_facts(pred)
        g, y, m, d = correct_props(pred_facts, real_facts)
        prec_all = 1.0  # 有一个不对就不对
        recall_inst = 0
        if g is not None:
            g_t += 1
            g_w += g
            prec_all *= g
            recall_inst += g
        if y is not None:
            y_t += 1
            y_w += y
            prec_all *= y
            recall_inst += y
        if m is not None:
            m_t += 1
            m_w += m
            prec_all *= m
            recall_inst += m
        if d is not None:
            d_t += 1
            d_w += d
            prec_all *= d
            recall_inst += d
        if g is None and y is None and m is None and d is None: #没预测一个
            prec_all = 0
        t_num += 1
        r_num += prec_all
        curr_recall = recall_inst / len(real_facts)
        recall += 1.0 if curr_recall > 1.0 else curr_recall

    def f(a, b):
        if b == 0:
            return 0.0
        return a * 1.0 / b
    def f1(a, b):
        if a + b > 0:
            return 2 * a * b / (a + b)
        else:
            return 0
    print '预测正确次数: %d/%d/%d/%d\t(g/y/m/d)' % (g_w, y_w, m_w, d_w)
    print '总共出现次数: %d/%d/%d/%d\t(g/y/m/d)' % (g_t, y_t, m_t, d_t)
    print '预测正确率: %f/%f/%f/%f\t(g/y/m/d)' % (f(g_w, g_t), f(y_w, y_t), f(m_w, m_t), f(d_w, d_t))
    prec = f(r_num, t_num)
    recall = recall / t_num
    print '整体正确率/召回率/F值: %f/%f/%f' % (prec, recall, f1(prec, recall))

if __name__ == '__main__':

    sent = u"她的出生年月日是_1950年_12月_20日"
    # sent = u"<ent_6033>是1962年4月5日出生的"
    facts = extract_facts(sent)
    for k, v in facts.items():
        print k, '\t', v


    pass
