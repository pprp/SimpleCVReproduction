from scipy.stats import stats
import random
import os
import sys
from coder import GeneCoder
from mlp_predictor import AccuracyPredictor
from kmeans import kmeans_select
from generation import next_generation
import train_predictor
import time,argparse
import torch
import individual_train
#

# class Logger():
#     rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
#     log_path = './Logs/'
#     log_name = log_path + rq + '.log'
#     logfile = log_name
#     if not os.path.exists(logfile):
#         os.system(r"touch {}".format(logfile))  # 调用系统命令行来创建文件
#     def __init__(self, filename=logfile):
#         self.terminal = sys.stdout
#         self.log = open(filename, "w")
# 
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
# sys.stdout = Logger()





def gen1(coder,xargs):

    popl1 = coder.gen_popl(xargs.popl_size)
    print('***********************************************')
    print('Generation 1 individual training')
    pred_set,score_set= [],[]
    for idx, individual in enumerate(popl1):
        t1 = time.time()
        acc = individual_train.train_arch(individual, xargs.gen1_epoch, idx+1, xargs, pred_set)
        t2 = time.time()
        t = t2 - t1
        pred_set.append(individual + [acc])
        score_set.append(acc)
        print('The {}-th individual {}'.format(idx + 1, individual))
        print('Accuracy: {}, Time Cost: {}'.format(acc, t))
    print('Generation 1 finished!')
    print('Gen 1 trained results: {}'.format(score_set))
    print('Gen 1 best acc: {}'.format(max(score_set)))
    print('Saving..')
    gen_state = {
        'pred' : pred_set,
        'arch' : popl1
    }
    torch.save(gen_state, os.path.join(xargs.g_path, '{}.pth'.format(1)))
    print('***********************************************')


def gen2_8(coder,xargs):
    if xargs.resume_gen<=2 or xargs.resume_gen==None:
        start=2
    else:
        start=xargs.resume_gen
    for i in range(start,xargs.gene_num+1):
        gen_path=os.path.join(xargs.g_path, '{}.pth'.format(i-1))
        last_save=torch.load(gen_path)
        pred_set = last_save['pred']
        arch_set = last_save['arch']
        llast_len=len(pred_set)
        print('all trained models before: ')
        print('length:',len(pred_set))
        for l in pred_set:
            print(l[:-1],'  ',l[-1])
        print('Derive next generation:')

        popl_new = next_generation(coder, xargs, pred_set[-xargs.popl_size:], arch_set)

        print('***********************************************')
        print('Generation {} individual training:'.format(i))
        truth_set=[]
        for idx, indiv in enumerate(popl_new):
            t1 = time.time()
            acc = individual_train.train_arch(indiv, xargs.other_epoch, idx+llast_len+1, xargs, arch_set)
            t2 = time.time()
            t = t2 - t1
            truth_set.append(acc)
            pred_set.append(indiv + [acc])
            arch_set.append(indiv)
            print('The {}-th individual {}'.format(idx +llast_len+ 1, indiv))
            print('Accuracy: {}, Time Cost: {}'.format(acc, t))

        print('***********************************************')
        print('Generation {} finished!'.format(i))
        print('Gen {} trained results: {}'.format(i,truth_set))
        print('Gen {} best acc: {}'.format(i,max(truth_set)))
        print('Saving..')
        gen_state = {
            'pred' : pred_set,
            'arch' : arch_set
        }
        torch.save(gen_state, os.path.join(xargs.g_path, '{}.pth'.format(i)))
        print('***********************************************')




def main(xargs):
    coder = GeneCoder(xargs.search_space)
    if xargs.start_gen == 1:
        gen1(coder,xargs)
        gen2_8(coder,xargs)

    elif xargs.resume_gen >= 2:
        gen2_8(coder,xargs)


    final = torch.load(os.path.join(xargs.g_path,'{}.pth'.format(xargs.gene_num)))
    final_predset=final['pred']
    final_score=[]
    for fi in final_predset:
        final_score.append(fi[-1])
    print('Finished!')
    print('Final best acc is: ',max(final_score))
    print('Best arch index :', final_score.index(max(final_score)))


#
if __name__ == "__main__":
    parser = argparse.ArgumentParser("evolution search with online-predictor")
    #  paths
    parser.add_argument('--search_space', default='search_space.json',type=str, help='search_space path')
    parser.add_argument('--m_path', default='model_checkpoint', type=str, help='individual save path')
    parser.add_argument('--g_path', default='generation_checkpoint', type=str, help='generation save path')

    # evolution
    parser.add_argument('--popl_size', default=210, type=int, help='be divided by 3')
    parser.add_argument('--gene_num', default=8, type=int, help='generation number')
    parser.add_argument('--tour_size', default=5, type=int, help='toursize of selTournament')
    parser.add_argument('--incross_prob', default=0.7, type=float, help='inside cross probability of spring1 or spring2')
    parser.add_argument('--cross_prob', default=0.5, type=float, help='inter cross probability')
    parser.add_argument('--m_prob', default=0.08, type=float, help='mutation probability within a individual')


    # resume
    parser.add_argument('--start_gen', default=2, type=int, help='generation to resume')

    # individual
    parser.add_argument('--gen1_epoch', default=100, type=int, help='epoch number of generation 1')
    parser.add_argument('--other_epoch', default=30, type=int, help='epoch number of generation 2-8')

    parser.add_argument('--rand_seed', type=int, help='manual seed')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    # print arguments
    for k, v in sorted(vars(args).items()):
        print(k,' = ',v)
    main(args)





