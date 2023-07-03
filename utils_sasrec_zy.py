import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import scipy.sparse as sp


random.seed(2023)

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

# sampler for batch generation
def random_neq2(l, r, s):
    t = random.randint(l, r)
    while t in s:
        t = random.randint(l, r)
    return t


class early_stoper(object):
    def __init__(self,refer_metric='auc',stop_condition=5):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric

    def update_and_isbest(self,eval_metric,epoch):
        if self.init_flag:
            self.best_epoch = epoch
            self.init_flag = False
            self.best_eval_result = eval_metric
            return True
        else:
            if eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]: # update the best results
                self.best_eval_result = eval_metric
                self.not_change = 0
                self.best_epoch = epoch
                return True              # best
            else:                        # add one to the maker for not_change information 
                self.not_change += 1     # not best
                return False

    def is_stop(self):
        if self.not_change > self.stop_condition:
            return True
        else:
            return False


def dict_to_csr(data_dict,user_num,item_num):
    row = []
    col = []
    for k,v in data_dict.items():
        col.extend(v)
        row.extend([k]*len(v))
    data = np.ones(len(row))
    return sp.coo_matrix((data, (row, col)),
                            shape=(user_num, item_num)).tocsr()

    


def valid_and_test_seqs(user_train, user_valid, user_test, user_num, maxlen):
    user_num_ = user_num 
    valid_sequence = np.zeros([user_num_, maxlen], dtype=np.int32)
    test_sequence = np.zeros([user_num_, maxlen], dtype=np.int32)
    for u in user_train.keys():
        u_len = min(maxlen,len(user_train[u]))
        valid_sequence[u][-u_len:] = np.array(user_train[u])[-u_len:]
        test_sequence[u][0:-1] = valid_sequence[u][1:]+0
        test_sequence[u][-1] = user_valid[u][0]+0
    return valid_sequence, test_sequence


        
def sample_function2(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample2():
        # usernum=1
        # itemnum=1
        user = random.randint(1, usernum)
        while len(user_train[user]) <= 1: user = random.randint(1, usernum)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq2(1, itemnum, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    # np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample2())

        result_queue.put(zip(*one_batch))

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler2(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        sub_seeds = np.random.randint(0,2e9,size=n_workers)
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      sub_seeds
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
    
    def close2(self):
        for p in self.processors:
            p.terminate()
            # p.join()

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 1)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function2, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
    
    def close2(self):
        for p in self.processors:
            p.terminate()
            # p.join()


# train/val/test data generation
def data_partition_new_amazon(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(fname+"train.txt", 'r')
    for line in f:
        u, i, t = line.rstrip().split('  ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        try:
            user_train[u].append(i)
        except:
            user_train[u] = [i]
    
    f = open(fname+"valid.txt", 'r')
    for line in f:
        u, i, t = line.rstrip().split('  ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_valid[u] = [i]
    
    f = open(fname+"test.txt", 'r')
    for line in f:
        u, i, t = line.rstrip().split('  ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_test[u] = [i]
        
    return [user_train, user_valid, user_test, usernum, itemnum]


# train/val/test data generation
def data_partition_new(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(fname+"train.txt", 'r')
    for line in f:
        u, i, t = line.rstrip().split('  ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        try:
            user_train[u].append(i)
        except:
            user_train[u] = [i]
    
    f = open(fname+"valid.txt", 'r')
    for line in f:
        u, i, t = line.rstrip().split('  ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_valid[u] = [i]
    
    f = open(fname+"test.txt", 'r')
    for line in f:
        u, i, t = line.rstrip().split('  ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_test[u] = [i]
        
    return [user_train, user_valid, user_test, usernum, itemnum]

# train/val/test data generation
def data_partition_backward(fname,maxlen):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(fname+"train.txt", 'r')
    for line in f:
        u, i, t = line.rstrip().split('   ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    User_ = defaultdict(list)
    for u,v in User.items():
        v_ = list(reversed(v))
        v_len = min(maxlen+2,len(v_))
        User_[u] = v_[0:v_len]
    User = User_
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


# train/val/test data generation
def data_partition_backward_new(fname,maxlen):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(fname+"train.txt", 'r')
    for line in f:
        u, i, t = line.rstrip().split('   ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    User_ = defaultdict(list)
    for u,v in User.items():
        v_ = list(reversed(v))
        v_len = min(maxlen+2,len(v_))
        User_[u] = v_[0:v_len]
    User = User_
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            # user_train[user] = User[user]
            # user_valid[user] = []
            # user_test[user] = []
            pass
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user