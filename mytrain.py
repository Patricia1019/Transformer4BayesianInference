import sys,os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
from torch import nn

from train import train, my_train
import priors
import encoders
import positional_encodings
import utils
import bar_distribution
import transformer

import gpytorch
import numpy as np, scipy.stats as st
import logging

def get_log(file_name):
     logger = logging.getLogger('train')  # 设定logger的名字
     logger.setLevel(logging.INFO)  # 设定logger得等级
 
     ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
     ch.setLevel(logging.INFO)  # 设定输出hander的level
 
     fh = logging.FileHandler(file_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
     fh.setLevel(logging.INFO)  # 设定文件hander得lever
 
     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
     ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
     fh.setFormatter(formatter)
     logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
     logger.addHandler(ch)
     return logger

def compute_mean_and_conf_interval(accuracies, confidence=.95):
    accuracies = np.array(accuracies)
    n = len(accuracies)
    m, se = np.mean(accuracies), st.sem(accuracies)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h

@torch.inference_mode()
def run_test(model,device='cuda:0',step_size=100, start_pos=1, batch_size=1000, sub_batch_size=10, seq_len=2000):
    assert batch_size % sub_batch_size == 0
    model.to(device)

    model.eval()
    nlls = []
    nll_confidences = []
    mses = []
    max_mses = []
    eval_positions = []
    
    def get_metrics(model, eval_pos, batch_size):
        x,y, target_y = priors.fast_gp.get_batch(batch_size=batch_size, seq_len=eval_pos+1, num_features=5,hyperparameters=hps, device=device)
        logits = model((x,y), single_eval_pos=eval_pos)
        if isinstance(model.criterion,nn.GaussianNLLLoss):
            nll = model.criterion(logits[0][...,0], target_y[eval_pos], var=logits[0][...,1].abs())
            return nll, 0., 0.
        means = model.criterion.mean(logits) # num_evals x batch_size
        maxs = (model.criterion.borders[logits.argmax(-1)] + model.criterion.borders[logits.argmax(-1)+1])/2
        mse = nn.MSELoss()
        nll = model.criterion(logits[0], target_y[eval_pos])
        return nll, mse(means[0], target_y[eval_pos]), mse(maxs[0], target_y[eval_pos])
        
    
    for eval_pos in range(start_pos, seq_len, step_size):
        eval_positions.append(eval_pos)
        print(eval_pos)
        
        nll = []
        mean_mse = []
        max_mse = []
        for i in range(batch_size//sub_batch_size):
            batch_nll, batch_mean_mse, batch_max_mse = get_metrics(model, eval_pos, sub_batch_size)
            nll.append(batch_nll)
            mean_mse.append(batch_mean_mse)
            max_mse.append(batch_max_mse)
        
        nll = torch.cat(nll)
        mean_mse = torch.tensor(mean_mse).mean()
        max_mse = torch.tensor(max_mse).mean()
        
        
        mses.append(mean_mse)
        max_mses.append(max_mse)
        nlls.append(nll.mean())
        nll_confidences.append(compute_mean_and_conf_interval(nll.to('cpu'))[1])
    return eval_positions, torch.stack(mses).to('cpu'), torch.stack(max_mses).to('cpu'), torch.stack(nlls).to('cpu'), torch.tensor(nll_confidences).to('cpu')

if __name__ == "__main__":
    num_features = 5
    hps = {'noise': 1e-4, 'outputscale': 1., 'lengthscale': .6, 'fast_computations': (False,False,False)}
    ys = priors.fast_gp.get_batch(100000,20,num_features, hyperparameters=hps)[1]
    kwargs = {'nlayers': 6, 'dropout': 0.0, 'steps_per_epoch': 100, }
    device_ids = [0, 1, 2, 3]
    batch_fraction = 8
    out_dir = './myresults/GPfitting_parallel_test'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    num_border_list = [1000]
    epoch_list = [50,100,200]
    data_augment =False
    for num_borders in num_border_list:
        for lr in [.0001*batch_fraction]:
            for epochs in [int(x*25/batch_fraction) for x in epoch_list]:
                print(f'num_borders={num_borders}, lr={lr}, epochs={epochs}')
                total_loss, total_positional_losses, model = my_train(priors.fast_gp.DataLoader_batch_first, bar_distribution.FullSupportBarDistribution(bar_distribution.get_bucket_limits(num_borders, ys=ys)), encoder_generator=encoders.Linear, emsize=512, nhead=4, warmup_epochs=epochs//4, y_encoder_generator=encoders.Linear, pos_encoder_generator=positional_encodings.NoPositionalEncoding,
                            batch_size=4*batch_fraction, scheduler=utils.get_cosine_schedule_with_warmup, extra_prior_kwargs_dict={'num_features': num_features, 'fuse_x_y': False, 'hyperparameters': hps},device_ids=device_ids,
                            epochs = epochs, 
                            data_augment = data_augment,
                            lr=lr, nhid=2*512, input_normalization=False, bptt=2010, single_eval_pos_gen=utils.get_weighted_single_eval_pos_sampler(2000),aggregate_k_gradients=25, **kwargs)
                torch.save(model.state_dict(), f'{out_dir}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting.pth')
                logger = get_log(f'{out_dir}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting_log.txt')
                logger.info(f'loss={total_loss}, positional_losses={total_positional_losses}')




