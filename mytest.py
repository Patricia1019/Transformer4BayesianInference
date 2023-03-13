import sys,os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import numpy as np, scipy.stats as st
import priors
import torch
import torch.nn as nn
import encoders
import positional_encodings
import pdb
from transformer import TransformerModel, MyTransformerModel
import bar_distribution
import matplotlib.pyplot as plt
import pdb
from mytrain import get_log

def compute_mean_and_conf_interval(accuracies, confidence=.95):
    accuracies = np.array(accuracies)
    n = len(accuracies)
    m, se = np.mean(accuracies), st.sem(accuracies)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h

@torch.inference_mode()
def run_test(model,data_augment=True,device='cuda:0',step_size=100, start_pos=50, batch_size=1000, sub_batch_size=10, seq_len=2000):
    assert batch_size % sub_batch_size == 0
    model.to(device)

    model.eval()
    nlls = []
    nll_confidences = []
    mses = []
    max_mses = []
    eval_positions = []
    
    def get_metrics(model, eval_pos, batch_size):
        x,y, target_y = priors.fast_gp.get_batch_first(batch_size=batch_size, seq_len=eval_pos+1, num_features=5,hyperparameters=hps, device=device)
        target_y = target_y.transpose(0,1)
        logits,_ = model((x,y), single_eval_pos=eval_pos, data_augment=data_augment)
        if isinstance(model.criterion,nn.GaussianNLLLoss):
            nll = model.criterion(logits[0][...,0], target_y[eval_pos], var=logits[0][...,1].abs())
            return nll, 0., 0.
        logits = logits.transpose(0,1)
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
    emsize = 512
    num_features = 5
    encoder = encoders.Linear(num_features,emsize)
    bptt = 2010
    hps = {'noise': 1e-4, 'outputscale': 1., 'lengthscale': .6, 'fast_computations': (False,False,False)}
    ys = priors.fast_gp.get_batch_first(100000,20,num_features, hyperparameters=hps)[1]
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    root_dir = './myresults/GPfitting_parallel'
    logger = get_log(f'{root_dir}/test_log.txt')
    # num_border_list = [1000,10000]
    num_border_list = [1000]
    # epoch_list = [50,100,200,400]
    epoch_list = [200]
    batch_fraction = 8
    draw_flag = False
    data_augment = False
    for num_borders in num_border_list:
        model = MyTransformerModel(encoder, num_borders, emsize, 4, 2*emsize, 6, 0.0,
                                        y_encoder=encoders.Linear(1, emsize), input_normalization=False,
                                        pos_encoder=positional_encodings.NoPositionalEncoding(emsize, bptt*2),
                                        decoder=None
                                        )
        model.criterion = bar_distribution.FullSupportBarDistribution(bar_distribution.get_bucket_limits(num_borders, ys=ys))
        for lr in [.0001*batch_fraction]:
        # for lr in [.0001]:
            for epochs in [int(x*25/batch_fraction) for x in epoch_list]:
                try:
                    model_path = f'{root_dir}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting.pth'
                    checkpoint = torch.load(model_path)
                    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
                    model.eval()
                    eval_positions, mses, max_mses, nlls, nll_confidences = run_test(model,data_augment=data_augment)
                    logger.info("*"*50)
                    logger.info(f"num_borders={num_borders}, lr={lr}, epochs={epochs}")
                    logger.info("mses")
                    logger.info(mses)
                    logger.info("max_mses")
                    logger.info(max_mses)
                    logger.info("nlls")
                    logger.info(nlls)
                    logger.info("nll_confidences")
                    logger.info(nll_confidences)
                    logger.info(' ')
                    if draw_flag:
                        ax1.set_title('mses')
                        ax1.plot(eval_positions,mses,label=f'num_borders{num_borders},epochs{epochs}')
                        ax2.set_title('max_mses')
                        ax2.plot(eval_positions,max_mses,label=f'num_borders{num_borders},epochs{epochs}')
                        ax3.set_title('nlls')
                        ax3.plot(eval_positions,nlls,label=f'num_borders{num_borders},epochs{epochs}')
                        ax4.set_title('nll_confidences')
                        ax4.plot(eval_positions,nll_confidences,label=f'num_borders{num_borders},epochs{epochs}')
                except:
                    print(f'{root_dir}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting.pth not found!')
    if draw_flag:
        ax1.legend() 
        ax2.legend()
        ax3.legend()
        ax4.legend()
        plt.show() # 图形可视化
        save_dir = f'{root_dir}/curves'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig1.savefig(f"{save_dir}/mses_curves.png")
        fig2.savefig(f"{save_dir}/max_mses_curves.png")
        fig3.savefig(f"{save_dir}/nll_curves.png")
        fig4.savefig(f"{save_dir}/nll_conf_curves.png")


                
