import os
import time
import torch
import argparse
import random
import numpy as np


from model_sasrec_zy import SASRec
from utils_sasrec_zy import *

def set_seed_all():
    seed = 2023
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark=False
# torch.backends.cudnn.deterministic=True

# from ray import tune


def set_seed(seed, cuda=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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
    del k_index
    return sum_hr.item(), sum_ndcg.item(), hit_array, t_ndcg



def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# Best config is: {'lr': 0.001, 'l2': 0.0001, 'drop': 0.2, 'n_heads': 1}
def arg_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='/home/zyang/S2SRec/datasets/Beauty/')
    parser.add_argument('--train_dir',default='train_log')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
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
# if not os.path.isdir(args.dataset + '_' + args.train_dir):
#     os.makedirs(args.dataset + '_' + args.train_dir)
# with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
#     f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()
def batch_evaluate(model,valid_data,valid_sequences,train_csr,batch_size=128,topk=10):
    max_id = valid_data.shape[0]
    hr_v = 0
    ndcg_v = 0
    for i in range(0,max_id,batch_size):
        end_id = min(max_id,batch_size+i)
        batch_data = valid_data[i:end_id]
        batch_seqs = valid_sequences[i:end_id]
        prediction = model.predict_all(batch_data[:,0],batch_seqs).cpu().numpy()
        prediction -= 1e9*train_csr[batch_data[:,0]].toarray()
        _, rec_array = torch.topk(torch.from_numpy(prediction).cuda(), topk)
        hr, ndcg,_,_ = batch_evaluate_test(rec_array,torch.from_numpy(batch_data[:,1]).cuda())
        hr_v += hr
        ndcg_v += ndcg
    hr_v /= valid_data.shape[0]
    ndcg_v /= valid_data.shape[0]
    return hr_v,ndcg_v
    

if __name__ == '__main__':
    set_seed_all()
    # global dataset
    # eval('setattr(torch.backends.cudnn, "benchmark", True)')
    # eval('setattr(torch.backends.cudnn, "deterministic", True)')
    # dataset = 
    estop = early_stoper(refer_metric='hr_v',stop_condition=30)
    if 'forward' in args.mode:
        print("********************************\nNOTE: *******forward mode***********")
        [user_train, user_valid, user_test, usernum, itemnum] = data_partition_new(args.dataset)
    else:
        print("NOTE: *******backward mode**********")
        [user_train, user_valid, user_test, usernum, itemnum] = data_partition_backward(args.dataset,args.maxlen)
   
    train_csr = dict_to_csr(user_train,usernum+1,itemnum+1)
    valid_csr = dict_to_csr(user_valid,usernum+1,itemnum+1)
    valid_sequences, test_sequences = valid_and_test_seqs(user_train,user_valid,user_test,usernum+1,args.maxlen)
    test_data, valid_data = None, None

    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    max_cc = 0
    for u in user_train:
        cc_ = len(user_train[u])
        cc += cc_
        max_cc = max(max_cc, cc_)
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    # to make the results reproducible, set n_workers=1
    sampler = WarpSampler2(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1)
    model = SASRec(usernum+1, itemnum+1, args).to(args.device) # no ReLU activation in original SASRec implementation?
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    # set_seed_all()
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    # if args.state_dict_path is not None:
    #     try:
    #         model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
    #         tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
    #         epoch_start_idx = int(tail[:tail.find('.')]) + 1
    #     except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
    #         print('failed loading state_dicts, pls check file path: ', end="")
    #         print(args.state_dict_path)
    #         print('pdb enabled for your quick check, pls type exit() if you do not need it')
    #         import pdb; pdb.set_trace()
            
    
    # if args.inference_only:
    #     model.eval()
    #     t_test = evaluate(model, dataset, args)
    #     print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating...', end='')
            if valid_data is None:
                valid_data = [[u,v[0]] for u,v in user_valid.items()]
                valid_data = np.array(valid_data)
                valid_sequences_cuda = valid_sequences[valid_data[:,0]]
            with torch.no_grad():
                # prediction = model.predict_all(valid_data[:,0],valid_sequences_cuda).cpu().numpy()
                # prediction -= 1e9*train_csr[valid_data[:,0]].toarray()
                # _, rec_array = torch.topk(torch.from_numpy(prediction).cuda(), 10)
                # hr_v, ndcg_v,_,_ = batch_evaluate_test(rec_array,torch.from_numpy(valid_data[:,1]).cuda())
                # hr_v /= valid_data.shape[0]
                # ndcg_v /= valid_data.shape[0]
                hr_v,ndcg_v = batch_evaluate(model,valid_data,valid_sequences_cuda,train_csr,batch_size=128,topk=10)
            

            print('Testing...', end='')
            if test_data is None:
                test_data = [[u,v[0]] for u,v in user_test.items()]
                test_data = np.array(test_data)
                test_sequences_cuda = test_sequences[test_data[:,0]]
            with torch.no_grad():
                # prediction = model.predict_all(test_data[:,0],test_sequences_cuda).cpu().numpy()
                # prediction = prediction - 1e9*train_csr[test_data[:,0]].toarray() - 1e9 * valid_csr[test_data[:,0]].toarray()
                # _,rec_array = torch.topk(torch.from_numpy(prediction).cuda(), 10)
                # hr_t, ndcg_t,_,_ = batch_evaluate_test(rec_array,torch.from_numpy(test_data[:,1]).cuda())
                # hr_t /= test_data.shape[0]
                # ndcg_t /= test_data.shape[0]
                hr_t,ndcg_t = batch_evaluate(model,test_data,test_sequences_cuda,train_csr+valid_csr,batch_size=128,topk=10)
            
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, ndcg_v, hr_v, ndcg_t, hr_t))
            is_best = estop.update_and_isbest({'hr_v':hr_v,'ndcg_v':ndcg_v,'hr_t':hr_t,'ndcg_t':ndcg_t},epoch)
            if is_best:
                save_name = "amazon-SASRec"+args.mode+'lr' + str(args.lr)+"l2"+str(args.l2_emb)+"drop"+str(args.dropout_rate)+"h"+str(args.num_heads)+".pth"
                torch.save(model.state_dict(),args.save_dir+save_name)
                
            if estop.is_stop():
                break
            # f.write(str(hr_v) + ' ' + str(hr_t) + '\n')
            # f.flush()
            t0 = time.time()
            model.train()
           
        # if epoch == args.num_epochs:
        #     folder = args.dataset + '_' + args.train_dir
        #     fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        #     fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        #     torch.save(model.state_dict(), os.path.join(folder, fname))
    print(estop.best_eval_result)
    # tune.report(ndcg_v=estop.best_eval_result['ndcg_v'],hr_v=estop.best_eval_result['hr_v'], ndcg_t=estop.best_eval_result['ndcg_t'], hr_t=estop.best_eval_result['hr_t'],epoch=epoch)
    
    # f.close()
    sampler.close2()
    print("Done")
    del model 
    torch.cuda.empty_cache()
