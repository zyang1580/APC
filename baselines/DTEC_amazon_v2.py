import numpy
from tqdm import tqdm
import argparse
from time import time
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import pandas as pd

import torch.optim as optim
from torch.autograd import Variable
import torch.autograd
import torch

from model_adam import SASRec
from evaluation import evaluate_ranking
from interactions import Interactions
from utils import *
import tqdm
import time


from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



import os
import time
import torch
import argparse

from model_sasrec_zy import SASRec
from utils_sasrec_zy import *
from ray import tune


def set_seed(seed, cuda=True):
    seed = 2023
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def arg_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='/home/zyang/S2SRec/datasets/Beauty/')
    parser.add_argument('--train_dir',default='train_log')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--seed', default=1024, type=int)
    parser.add_argument('--mode', default='backward')
    parser.add_argument('--save_dir', default='/data/zyang/pc-records/')
    return parser.parse_args()

args = arg_para() 

def conver_fkseq_to_bkseq(input_sequences,target_position):
    temp_seqs = copy.deepcopy(input_sequences)
    targets = temp_seqs[:, -target_position]
    temp_seqs[:,:-1] = temp_seqs[:,1:]
    temp_seqs[:,-1] = -1
    seq_lens = (temp_seqs!=0).sum(axis=-1)
    L_ = temp_seqs.shape[1]
    for i in range(temp_seqs.shape[0]):
        i_len = seq_lens[i]
        temp_seqs[i,-i_len:] = temp_seqs[i,-i_len:][-1::-1]
    target_position_ = np.minimum(seq_lens,target_position)-1
    prediction_position = L_ - seq_lens + target_position_
    return temp_seqs, prediction_position









def batch_evaluate(rec_array, target_array, device="cpu"):
    """
    please note that, each user only contain one target user if you use this function
    """
    if len(rec_array.shape) < 2:
        rec_array = rec_array.unsqueeze(0)
    if len(target_array.shape) < 2:
        target_array = target_array.squeeze()
        target_array = target_array.unsqueeze(1)  # add one dimension
    assert rec_array.shape[0] == target_array.shape[0]
    hit_array = torch.eq(rec_array, target_array).float()
    sum_hr = hit_array.sum()  # the sum of hit ratio of the batch
    kk = rec_array.shape[-1]
    k_index = torch.arange(2, kk + 2).float().cuda()
    log_rank = 1.0 / torch.log2(k_index)
    idcg = log_rank[0]
    sum_ndcg = (
        hit_array * log_rank.unsqueeze(0)
    ).sum() / idcg  # the sum of ndcg of the batch
    return sum_hr.item(), sum_ndcg.item()


def batch_evaluate_return_array(rec_array, target_array, device="cpu"):
    """
    please note that, each user only contain one target user if you use this function
    """
    if len(rec_array.shape) < 2:
        rec_array = rec_array.unsqueeze(0)
    if len(target_array.shape) < 2:
        target_array = target_array.squeeze()
        target_array = target_array.unsqueeze(1)  # add one dimension
    assert rec_array.shape[0] == target_array.shape[0]
    hit_array = torch.eq(rec_array, target_array).float()
    sum_hr = hit_array  # the sum of hit ratio of the batch
    kk = rec_array.shape[-1]
    k_index = torch.arange(2, kk + 2).float().cuda()
    log_rank = 1.0 / torch.log2(k_index)
    idcg = log_rank[0]
    sum_ndcg = (
        hit_array * log_rank.unsqueeze(0)
    ) / idcg  # the sum of ndcg of the batch
    return sum_hr, sum_ndcg


def batch_evaluate_test(rec_array, target_array, device="cpu"):
    """
    please note that, each user only contain one target user if you use this function
    """
    if len(rec_array.shape) < 2:
        rec_array = rec_array.unsqueeze(0)
    if len(target_array.shape) < 2:
        target_array = target_array.squeeze()
        target_array = target_array.unsqueeze(1)  # add one dimension
    assert rec_array.shape[0] == target_array.shape[0]
    hit_array = torch.eq(rec_array, target_array).float()
    sum_hr = hit_array.sum()  # the sum of hit ratio of the batch
    kk = rec_array.shape[-1]
    k_index = torch.arange(2, kk + 2).float().cuda()
    log_rank = 1.0 / torch.log2(k_index)
    idcg = log_rank[0]
    t_ndcg = (
        hit_array * log_rank.unsqueeze(0)
    ) / idcg
    sum_ndcg = (
        hit_array * log_rank.unsqueeze(0)
    ).sum() / idcg  # the sum of ndcg of the batch
    return sum_hr.item(), sum_ndcg.item(), hit_array, t_ndcg



def batch_user_data_generate(user_list, batch_size=256):
    for i in range(0, user_list.shape[0], batch_size):
        start_idx = i
        end_idx = min(start_idx + batch_size, user_list.shape[0])
        yield user_list[start_idx:end_idx]
        
        
def get_item_dict(path):
    train = pd.read_csv(path,sep=' ',names=['user','item','click'])
    train.head(2)
    train_info = train.groupby('item').agg({'click':['count','mean','sum']})
    train_info.head(2)
    item_dict = dict(zip(train_info.index.to_list(),train_info.values[:,0]))
    return item_dict




def evaluate_reverse_batch_rules(
    model,
    model_reverse,
    train_csr,
    test_sequences,
    test_data,
    test_sequences_bk,
    position_bk,
    config,
    out_file,
    valid_user_list=None,
    batch_size=1024,
    top=20,
    re_top=10,
    lr=1e-1,
    lr2=1e-1,
    lr3=1e-1,
    T=5,
    alpha=1,
    num=1,
    weight=2
):
    # 模型主体，利用reverse对正向的模型的预测进行精选
    # x = torch.FloatTensor()
    
    model.eval()
    # model_reverse.eval()

    hit_ratio_adj_ori = np.zeros(num)
    hit_ratio_adj = np.zeros(num)
    ndcg_adj = np.zeros(num)

    hit_ratio_adj_init_n = np.zeros(num)
    ndcg_adj_init_n = np.zeros(num)
    out_loss_list = np.zeros(num)

    user_num_n = np.zeros(num)

    hit_ratio_init = 0
    ndcg_init = 0
    valid_user = 0
    hit_user = 0
    if valid_user_list is None:
        raise NotImplementedError(
            "please only input the valid users, \ie, users with more than 1 interacted items in testing set"
        )

    s_time = time.time()
    loss_init = []
    
    for k_batch, batch_data in tqdm.tqdm(
        enumerate(batch_user_data_generate(test_data, batch_size))
    ):
       
        target_item = torch.from_numpy(
            batch_data[:,1]
        ).cuda()  # batch_size
        batch_user = batch_data[:,0]
        batch_seqs = test_sequences[batch_user]
        batch_postion = position_bk[batch_user]

        with torch.no_grad():
            pre = (
                model.predict_all(batch_user, batch_seqs).cpu().numpy() - 1e9 * train_csr[batch_user].toarray()
            )  # 交互过的item的分数变的很低，排除掉，避免影响softmax
            
        pre_var = Variable(torch.empty_like(torch.from_numpy(pre),requires_grad=True)).cuda()
        with torch.no_grad():
             pre_var.copy_(torch.from_numpy(pre).data)
        optimizer = torch.optim.Adam([pre_var], lr=lr)
        pre_var.requires_grad=True


        origin = []
        for i in range(1):
            origin.append(0)
        try:
            p_num_ = position_bk.shape[1]
        except:
            p_num_ = 1
        
        for n_ in range(num):
            out_loss = 0
            for p_num in range(p_num_):
                # pre_var = learn_val[p_num]
                pre_var_ori = pre_var.clone()
                val, ind = torch.topk(pre_var, top, dim=-1)
                if n_ == 0 and p_num == 0:
                    hr_n0, ndcg_n0 = batch_evaluate_return_array(ind[:, 0:re_top], target_item)
                    hit_ratio_init += hr_n0.sum()
                    ndcg_init += ndcg_n0.sum()
                # soft = torch.softmax(val, dim=-1).unsqueeze(-1) # batch_size * top * 1
                # val = torch.sigmoid(val)
                # soft = (val ** weight) / (val ** weight).sum(dim=1).unsqueeze(1)
                val = torch.sigmoid(val)
                soft = (val ** weight) / (val ** weight).sum(dim=1).unsqueeze(1)
                # soft = val / val.sum(dim=1).unsqueeze(1)  # sum: batch_size
                t_batch, _ = soft.size()
                soft = soft.reshape(t_batch, -1, 1)  # batch_size * top * 1
                emb_top_list = model_reverse.item_emb(ind)  # batch_size * top * emb_size

                emb = (emb_top_list * soft).sum(-2)  # batch_size * emb_size
                emb = emb.float()

                truth_in_pre_flag = torch.eq(ind, target_item.unsqueeze(-1)).nonzero()
                hit_user += truth_in_pre_flag.shape[0]



                target_bk  = test_sequences_bk[batch_user, batch_postion]
                u_tensor = torch.from_numpy(batch_user).long().cuda()
                s = test_sequences_bk[batch_user]

                output = model_reverse.predict_position(
                    copy.deepcopy(s), batch_postion-1, emb_replace=emb
                )[:,target_bk]
                output_ = model_reverse.predict_position(
                    copy.deepcopy(s), batch_postion-1, emb_replace=None
                )[:,target_bk] # not input the predicted one

                loss = -torch.log(
                    torch.sigmoid(torch.diag(output))
                )  # 可以换成BEC
                # loss = loss.sum()
                # grad1 = torch.autograd.grad(loss,emb,retain_graph=True)[0].max()
                # loss2 = output.sum()
                # grad2 = torch.autograd.grad(loss2,emb,retain_graph=True)[0].max()
                # loss3 = emb.sum()
                # grad3 = torch.autograd.grad(loss3,emb,retain_graph=True)[0].max()

                

                loss_ = -torch.log(
                    torch.sigmoid(torch.diag(output_))
                )  # 可以换成BEC
                loss_origin = loss.clone().detach()
                
                # neg_t = []
                # for i in range(config.max_list_length):
                #     neg_t.append(output[:, i, target[:, i, :]].squeeze(-1))
                # out_neg = torch.stack(neg_t, dim=1)
                # out_neg = output[:, :, target].squeeze(-1)
                
                
                neg_flag = torch.ones_like(output)
                neg_flag = neg_flag - torch.diag(torch.ones(output.shape[0])).cuda()
                neg_flag = neg_flag.cuda()
                cmp_loss = -torch.log(torch.sigmoid(torch.diag(output).unsqueeze(-1) - output))
                cmp_loss = torch.mul(cmp_loss, neg_flag).sum(dim=-1)/(output.shape[1]-1)

                
                cmp_loss_ = -torch.log(torch.sigmoid(torch.diag(output_).unsqueeze(-1) - output_))
                cmp_loss_ = torch.mul(cmp_loss_, neg_flag).sum(dim=-1)/(output.shape[1]-1)

                loss += T * cmp_loss
                loss_ += T * cmp_loss_
                if n_ == 0 and p_num == 0:
                    loss_init.append(loss.detach().cpu().numpy())
                if(n_ == 0):
                    origin[p_num] = loss
                    loss_user = loss
                else:
                    loss_user = origin[p_num] - loss
                loss_user = loss_user.detach().cpu().numpy()
                
                loss_init_k = loss_init[k_batch]
                batch_lr = np.zeros_like(loss_init_k)
                batch_lr[loss_.detach().cpu().numpy()<loss.detach().cpu().numpy()+0.000] = 1
                # batch_lr[loss_.detach().cpu().numpy()<loss_origin.detach().cpu().numpy()] -= lr
                batch_lr = torch.from_numpy(batch_lr).cuda()    

                loss = loss.sum()
                # torch.autograd.grad(loss,emb,retain_graph=True)[0].max()
                # grad = torch.autograd.grad(loss,soft,retain_graph=True)

                out_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # with torch.no_grad():
                #     pre_var[batch_lr==0] = pre_var_ori[batch_lr==0]
                    # for b in range(batch_lr.size()[0]):
                    #     if batch_lr[b, 0] == 1:
                    #         params[0][b, :] = pre_var_ori[b, :]
            
            soft = torch.zeros_like(ind)
            soft = torch.gather(pre_var, -1, ind)
            
            out_loss_list[n_] += out_loss
            _, ind_rerank = torch.topk(soft, re_top, dim=-1)
            top_rerank = torch.gather(ind, -1, ind_rerank)  # ind[ind_rerank]

            hr, ndcg = batch_evaluate(top_rerank, target_item)
            hit_ratio_adj_ori[n_] += hr
            ndcg_adj[n_] += ndcg

            # print("batch_lr shape:",batch_lr.shape)
            s_idx = torch.nonzero(batch_lr!=1)[:,0]
            user_num_n[n_] += s_idx.shape[0]
            

            hr, ndcg = batch_evaluate(top_rerank[s_idx], target_item[s_idx])
            hit_ratio_adj[n_] += hr 
            # ndcg_adj[n_] += ndcg

            hit_ratio_adj_init_n[n_] += hr_n0[s_idx].sum()





    #     print(model._net.item_embeddings.weight[1].sum())
    #     print(model_reverse._net.item_embeddings.weight[1].sum())

    hit_ri_n = (hit_ratio_adj - hit_ratio_adj_init_n)/hit_ratio_adj_init_n

    user_num = valid_user_list.shape[0]
    hit_ratio_adj /= user_num_n
    hit_ratio_adj_ori /= user_num
    hit_ratio_init /= user_num
    ndcg_init /= user_num
    ndcg_adj /= user_num
    out_loss /= 3
    out_loss /= user_num

    out_loss_list /= user_num

    hit_ratio_adj_init_n /= user_num_n

    hit_ri_n = (hit_ratio_adj - hit_ratio_adj_init_n)/hit_ratio_adj_init_n

    

    # print("original Recall@", hit_ratio_init, "\nndcg:", ndcg_init, file=out_file)
    # print("adjust Recall@", hit_ratio_adj, ndcg_adj, file=out_file)

    print(
        "normal loss :",
        out_loss_list,
    )

    print("valid user number:", valid_user_list.shape[0])
    print("original Recall@", hit_ratio_init, "\noriginal NDCG", ndcg_init)
    print(
        "adjust Recall@",
        hit_ratio_adj_ori,
        "max recall:",
        hit_ratio_adj_ori.max(),
        "improve:",
        hit_ratio_adj_ori.max() - hit_ratio_init,
        'RI:',
        (hit_ratio_adj_ori.max() - hit_ratio_init)/hit_ratio_init,
        "RI max:",
        hit_ri_n.max(),hit_ratio_adj[np.argmax(hit_ri_n)],hit_ratio_adj_init_n[np.argmax(hit_ri_n)],
        "RI max index:",
        np.argmax(hit_ri_n),
        user_num_n[np.argmax(hit_ri_n)], user_num
    )
    print(
        "\nadjust NDCG",
        ndcg_adj,
        "max ndcg:",
        ndcg_adj.max(),
        "improve:",
        ndcg_adj.max() - ndcg_init,
        "RI:",
        (ndcg_adj.max() - ndcg_init)/ndcg_init
    )

    print(
        "\nnormal loss :",
        out_loss,
    )
    print("time cost:", time.time() - s_time)
    # tune.report (ndcg=ndcg_adj.max(), recall_10=hit_ratio_adj.max(), recall_improve=hit_ratio_adj.max() - hit_ratio_init, ndcg_improve=ndcg_adj.max() - ndcg_init)




class conf(object):
    def __init__(self, flag="original"):
        self.hidden_pos = 1  # start with 0
        data_path = "./datasets/ML1M/"
        if flag == "original":
            self.train_root = data_path+"train.txt"
            self.valid_root = data_path+"valid.txt"
            self.test_valid_root = data_path+"test_valid.txt"
            self.test_test_root = data_path+"test_test.txt"
        else:
            self.train_root = '/home/huangyl/S2SRec/datasets/ml1m/test/train.txt'
            self.test_root = '/home/huangyl/S2SRec/datasets/ml1m/test/test_reverse.txt'

        # if flag == "original":
        #     # self.train_root = '/home/huangyl/seq_rec/datasets/ml1m/test/train.txt'
        #     self.train_root = '/home/huangyl/S2SRec/datasets/ml1m/test/train.txt'
        # else:
        #     # self.train_root = '/home/huangyl/seq_rec/datasets/ml1m/test/train_reverse.txt'
        #     self.train_root = '/home/huangyl/S2SRec/datasets/ml1m/test/train_reverse.txt'
        # if flag == "original":
        #     # self.test_root = '/home/huangyl/seq_rec/datasets/ml1m/test/test.txt'
        #     self.test_root = '/home/huangyl/S2SRec/datasets/ml1m/test/test.txt'
        # else:
        #     # self.test_root = '/home/huangyl/seq_rec/datasets/ml1m/test/test_reverse.txt'
        #     self.test_root = '/home/huangyl/S2SRec/datasets/ml1m/test/test_reverse.txt'
    
        self.n_iter = 200
        self.seed = 2020
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.l2 = 1e-3
        self.neg_samples = 1
        self.use_cuda = True
        self.device = 0
        self.hidden_size = 128
        self.max_list_length = 200
        self.n_layers = 2
        self.n_heads = 2
        if flag == "original":
            self.drop = 0.8
        else:
            self.drop = 0.9



if __name__ == '__main__':
    # global dataset
    #Best config is: {'learning_rate': 0.01, 'learning_rate2': -1, 'learning_rate3': -1, 'num': 100, 'hid_pos': 3, 'top_': (50, 10), 'weight': 1, 'w': 0.5}
    config = {
        'learning_rate':0.01, #tune.grid_search([100, 10, 1, 0.1, 0.01, 0.001]),
        'learning_rate2':-1, #tune.grid_search([-1]),
        'learning_rate3':-1, #tune.grid_search([-1]),
        # 'learning_rate':tune.grid_search([10, 15, 20, 50, 100]),
        'num':100, #tune.grid_search([100]),
        'hid_pos':3, #tune.grid_search([100, 80, 20, 10]),
        'top_':(50,10),#tune.grid_search([(50, 10)]),
        'weight':1, #tune.grid_search([100, 10, 1, 0.1, 0.01]),
        'w':0.5, #tune.grid_search([1, 2, 3]),
        # 'weight':tune.grid_search([-1]),
    }
    set_seed(args.seed)
    config_forward = conf("original")
    config_reverse = conf("reverse")
    # dataset = 
    estop = early_stoper(refer_metric='hr_v',stop_condition=25)

    bk_position = config['hid_pos']

    print("********************************\nNOTE: *******forward mode***********")
    [user_train, user_valid, user_test, usernum, itemnum] = data_partition_new(args.dataset)
    train_csr = dict_to_csr(user_train,usernum+1,itemnum+1)
    valid_csr = dict_to_csr(user_valid,usernum+1,itemnum+1)
    valid_sequences, test_sequences = valid_and_test_seqs(user_train,user_valid,user_test,usernum+1,args.maxlen)
    test_data = [[u,v[0]] for u, v in user_test.items()]
    test_data = np.array(test_data)
    valid_data = [[u,v[0]] for u, v in user_valid.items()]
    valid_data = np.array(valid_data)
    
     
    valid_sequences_bk, valid_tp_bk = conver_fkseq_to_bkseq(valid_sequences, target_position=bk_position) # 'tp' short for 'target position'
    bk_position += 1 # test need to add another 1
    test_sequences_bk, test_tp_bk = conver_fkseq_to_bkseq(test_sequences, target_position=bk_position)

    # forward {'lr': 0.001, 'l2': 1e-06, 'drop': 0.2, 'n_heads': 1}
    args.lr = 0.001
    args.l2_emb = 1e-6
    args.dropout_rate = 0.2
    args.mode = '0215forward'
    save_forward = "amazon-SASRec"+args.mode+'lr' + str(args.lr)+"l2"+str(args.l2_emb)+"drop"+str(args.dropout_rate)+"h"+str(args.num_heads)+".pth"
    model_forward = SASRec(usernum+1, itemnum+1, args).to(args.device) # no ReLU activation in original SASRec implementation?
    model_forward.load_state_dict(torch.load(args.save_dir+save_forward))

    # Best config is: {'lr': 0.001, 'l2': 0.01, 'drop': 0.2, 'n_heads': 1}
    # args.lr = 0.001
    # args.l2_emb = 0.01
    # args.dropout_rate = 0.2
    # args.mode = 'backward-0109'
    # save_backward = "SASRec"+args.mode+'lr' + str(args.lr)+"l2"+str(args.l2_emb)+"drop"+str(args.dropout_rate)+"h"+str(args.num_heads) 
    # model_backward = SASRec(usernum+1, itemnum+1, args).to(args.device)
    # model_backward.load_state_dict(torch.load(args.save_dir + save_backward))


    #generate top-k recommendation list:
    total_csr = train_csr + valid_csr
    hit_ratio_init = 0
    ndcg_init = 0
    hit_ratio = 0
    ndcg = 0
    model_forward.eval()
    for k_batch, batch_data in tqdm.tqdm(
        enumerate(batch_user_data_generate(test_data, 128))
    ):
       
        target_item = torch.from_numpy(batch_data[:,1]).cuda()  # batch_size
        batch_user = batch_data[:,0]
        batch_seqs = test_sequences[batch_user]
        # batch_postion = position_bk[batch_user]
        
        with torch.no_grad():
            # # print(batch_seqs[0])
            # pre = model_forward.predict_all(batch_user, batch_seqs).cpu().numpy()
            # # print(pre[0])
            # pre = model_forward.predict_all(batch_user, batch_seqs).cpu().numpy()
            # # print(pre[0])

            pre = model_forward.predict_all(batch_user, batch_seqs).cpu().numpy()  - 1e9 * total_csr[batch_user].toarray()  # 交互过的item的分数变的很低，排除掉，避免影响softmax
            # pre = torch.from_numpy(pre).cuda()
        pre_var = Variable(torch.empty_like(torch.from_numpy(pre),requires_grad=True)).cuda()
        with torch.no_grad():
            pre_var.copy_(torch.from_numpy(pre).data)
        val, ind = torch.topk(pre_var, 50, dim=-1)
        hr_n0, ndcg_n0 = batch_evaluate_return_array(ind[:,:10], target_item)
        hit_ratio_init += hr_n0.sum()
        ndcg_init += ndcg_n0.sum()
        rec_embs = model_forward.item_emb(ind)
        seq_embs = model_forward.item_emb(torch.LongTensor(batch_seqs).to(model_forward.dev))
        idx_raw, idx_col = np.where(batch_seqs==0)
        seq_embs[idx_raw,idx_col] = 0
        
        sim_matrix = torch.matmul(rec_embs,torch.transpose(seq_embs,-1,-2))
        # v3
        sim_matrix[idx_raw,:,idx_col] = -1e12
        sim_matrix = torch.softmax(sim_matrix,dim=-1)
        # print("sim matrix shape:",sim_matrix.shape)
        with torch.no_grad():
            log_feats = model_forward.log2feats(batch_seqs)[:,:-1] # user_ids hasn't been used yet
            pos_embs = model_forward.item_emb(torch.LongTensor(batch_seqs[:,1:]).to(model_forward.dev))
            pos_logits = torch.sigmoid((log_feats * pos_embs).sum(dim=-1))
            pos_errors = torch.unsqueeze(1 - pos_logits,-2)    # batch_size * 1 * (L-1)
            sim_matrix = sim_matrix[:,:,1:] 
            # sim_matrix = sim_matrix/sim_matrix.sum(-1).unsqueeze(-1) # bacth_size * K * (L-1)
            correction = torch.mul(sim_matrix,pos_errors).sum(dim=-1)
        pre_correct = val + correction

        val_new, ind_new = torch.topk(pre_correct, config['top_'][1],dim=-1)
        re_rank = torch.gather(ind,-1,ind_new)
        hr_n1, ndcg_n1 = batch_evaluate_return_array(re_rank, target_item)
        hit_ratio += hr_n1.sum()
        ndcg += ndcg_n1.sum()
    user_num = test_data.shape[0]
    print("before adjust----recall{%.4f} NDCG:{%.4f}"%(hit_ratio_init/user_num, ndcg_init/user_num))
    print("after adjust----recall{%.4f} NDCG:{%.4f}"%(hit_ratio/user_num, ndcg/user_num))
    


        





        


 