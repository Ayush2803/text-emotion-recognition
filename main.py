from src.config import ConfigRegression
from data.dataloader import MMDataLoader
from src.train import WMSATrain
from models.model import WMSA
import gc
import time
import random
import torch
#import pynvml
import logging
import argparse
import os
import os.path
import numpy as np
import pandas as pd

localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def setup_seed(seed):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.enabled = False

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(
        args.model_save_dir, f'{args.unitask}-{args.modelname}-{args.name}.pth')
    args.best_model_save_path = os.path.join(
        args.model_save_dir, f'{args.unitask}-{args.modelname}-{args.name}-best.pth')

    using_cuda = True

    # using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    print(torch.version.cuda)
    print(len(args.gpu_ids) > 0)
    print(torch.cuda.is_available())
    print(using_cuda)

    # if not torch.cuda.is_available():
    #     return None

    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(
        args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = MMDataLoader(args)

    train_loader, valid_loader, test_loader = dataloader

    model = WMSA(args).to(device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(
        f'The model has {count_parameters(model)} trainable parameters')
    trainmodel = WMSATrain(args)  # .getTrain(args)
    args.train_epoch = 0
    # do train
    if not args.test_only:
        trainmodel.do_train(model, dataloader)
    # load pretrained model
    assert os.path.exists(args.best_model_save_path)
    model.load_state_dict(torch.load(args.best_model_save_path))
    model.to(device)

    # do test
    if args.tune_mode:
        # using valid dataset to debug hyper parameters
        results = trainmodel.do_test(model, valid_loader, mode="VALID")
    else:
        results = trainmodel.do_test(model, test_loader, mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def run_normal(args):
    args.res_save_dir = args.res_save_dir
    init_args = args
    model_results = []
    seeds = args.seeds
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        config = ConfigRegression(args)
        args = config.get_config()
        setup_seed(seed)
        args.seed = seed

        logger.info('Start running %s...' % (args.modelname))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        test_results = run(args)
        # restore results
        if test_results is not None:
            model_results.append(test_results)

    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(args.res_save_dir,
                             f'{args.modelname}-{args.name}-{str_time}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    # res = [args.modelname]
    res = []
    res_temp = []
    for c in criterions:
        values = [r[c] for r in model_results]
        # mean = round(np.mean(values)*100, 2)
        # std = round(np.std(values)*100, 2)
        res_temp.append(values)
    res_temp = np.array(res_temp)
    print(res_temp)

    for i in range(len(seeds)):
        res_tmp_tmp = [str(i)]
        res_tmp_tmp += list(res_temp[:, i])
        res.append(res_tmp_tmp)
        df.loc[len(df)] = res[i]
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' % (save_path))


def set_log(args):

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    log_file_path = os.path.join(
        args.logs_dir, f'{args.modelname}-{args.name}-{str_time}.log')

    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter(
        '%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='wmsa',
                        help='support wmsa')
    parser.add_argument('--name', type=str, default='Train',
                        help='the model name details')
    parser.add_argument('--dataset', type=str, default='iemocap',
                        help='support the dataset iemocap')
    parser.add_argument('--test_only', action='store_true',
                        help='train+test or test only')
    parser.add_argument('--MAX_Epoch', type=int, default=20,
                        help='MAX_Epoch')
    parser.add_argument('--MIN_Epoch', type=int, default=0,
                        help='MIN_Epoch')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    # parser.add_argument('--lr_bert', type=float, default=2e-5,  # 2e-5
    #                     help='lr_text_bert')
    parser.add_argument('--lr_roberta', type=float, default=2e-6,  # 2e-5
                        help='lr_text_roberta')
    parser.add_argument('--lr_other', type=float, default=5e-5,  # 1e-2, #1e-2
                        help='lr_text_other')
    # parser.add_argument('--weight_decay_text', type=float, default=1e-3,  # 1e-2, #1e-3
    #                     help='weight_decay_text')
    parser.add_argument('--weight_decay_text', type=float, default=0.01,  # 1e-2, #1e-3
                        help='weight_decay_text')
    parser.add_argument('--weight_decay_other', type=float, default=1e-3,  # 1e-2,
                        help='weight_decay_other')
    parser.add_argument('--post_text_dropout', type=float, default=0.0,  # 0.0,
                        help='post_text_dropout')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='support wmsa')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='path to tensorboard results.')
    parser.add_argument('--logs_dir', type=str, default='results/logs',
                        help='path to log results.')  # NO
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    # parser.add_argument('--seeds', nargs='+', type=int,
    #                    help='set seeds for multiple runs!')
    parser.add_argument('--nargs-int-type', nargs='+', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger = set_log(args)
    args.seeds = [0]
    print(args)

    run_normal(args)
