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
import pandas as pd

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
    mse_confidences = []
    max_mses = []
    max_mse_confidences = []
    eval_positions = []
    
    def get_metrics(model, eval_pos, batch_size):
        x,y, target_y = priors.fast_gp.get_batch_first(batch_size=batch_size, seq_len=eval_pos+1, num_features=5,hyperparameters=hps, device=device)
        target_y = target_y.transpose(0,1)
        logits,_ = model((x,y), single_eval_pos=eval_pos, data_augment=data_augment)
        if isinstance(model.criterion,nn.GaussianNLLLoss):
            nll = model.criterion(logits[0][...,0], target_y[eval_pos], var=logits[0][...,1].abs())
            return nll, 0., 0.
        logits = logits.transpose(0,1) # [seq_len-single_eval_pos,bs,border_num]
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
        # mean_mse = torch.tensor(mean_mse).mean()
        # max_mse = torch.tensor(max_mse).mean()
        mean_mse = torch.tensor(mean_mse)
        max_mse = torch.tensor(max_mse)        

        mses.append(torch.tensor(mean_mse).mean())
        mse_confidences.append(compute_mean_and_conf_interval(mean_mse.to('cpu'))[1])
        max_mses.append(torch.tensor(max_mse).mean())
        max_mse_confidences.append(compute_mean_and_conf_interval(max_mse.to('cpu'))[1])
        nlls.append(nll.mean())
        nll_confidences.append(compute_mean_and_conf_interval(nll.to('cpu'))[1])
    return eval_positions, torch.stack(mses).to('cpu'),torch.tensor(mse_confidences).to('cpu'), torch.stack(max_mses).to('cpu'), torch.tensor(max_mse_confidences).to('cpu'), torch.stack(nlls).to('cpu'), torch.tensor(nll_confidences).to('cpu')

if __name__ == "__main__":
    emsize = 512
    num_features = 5
    encoder = encoders.Linear(num_features,emsize)
    bptt = 2010
    hps = {'noise': 1e-4, 'outputscale': 1., 'lengthscale': .6, 'fast_computations': (False,False,False)}
    ys = priors.fast_gp.get_batch_first(100000,20,num_features, hyperparameters=hps)[1]
    noaugment_root_dir = './myresults/GPfitting_parallel'
    augment_root_dir = f'./myresults/GPfitting_augmentTrue_{num_features}feature'
    # num_border_list = [1000,10000]
    num_border_list = [1000]
    epoch_list = [50,100,200]
    # epoch_list = [50]
    batch_fraction = 8
    draw_flag = True
    for num_borders in num_border_list:
        model_noaugment = MyTransformerModel(encoder, num_borders, emsize, 4, 2*emsize, 6, 0.0,
                                        y_encoder=encoders.Linear(1, emsize), input_normalization=False,
                                        pos_encoder=positional_encodings.NoPositionalEncoding(emsize, bptt*2),
                                        decoder=None
                                        )
        model_noaugment.criterion = bar_distribution.FullSupportBarDistribution(bar_distribution.get_bucket_limits(num_borders, ys=ys))
        model_augment = MyTransformerModel(encoder, num_borders, emsize, 4, 2*emsize, 6, 0.0,
                                        y_encoder=encoders.Linear(1, emsize), input_normalization=False,
                                        pos_encoder=positional_encodings.NoPositionalEncoding(emsize, bptt*2),
                                        decoder=None
                                        )
        model_augment.criterion = bar_distribution.FullSupportBarDistribution(bar_distribution.get_bucket_limits(num_borders, ys=ys))
        for lr in [.0001*batch_fraction]:
        # for lr in [.0001]:
            for epochs in [int(x*25/batch_fraction) for x in epoch_list]:
                # try:
                    out_data_root = './test_data/ifaugment'
                    if not os.path.exists(out_data_root):
                        os.makedirs(out_data_root)
                    out_data_path = f'{out_data_root}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting.xlsx'

                    out_curve_root = f'./test_curves/ifaugment/numborder{num_borders}_lr{lr}_epoch{epochs}'
                    if not os.path.exists(out_curve_root): 
                        os.makedirs(out_curve_root)
                    
                    noaugment_model_path = f'{noaugment_root_dir}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting.pth'
                    noaugment_checkpoint = torch.load(noaugment_model_path)
                    model_noaugment.load_state_dict({k.replace('module.',''):v for k,v in noaugment_checkpoint.items()})
                    model_noaugment.eval()
                    eval_positions_noaugment, mses_noaugment,mse_confidences_noaugment, max_mses_noaugment, max_mse_confidences_noaugment, nlls_noaugment, nll_confidences_noaugment = run_test(model_noaugment,data_augment=False)
                    
                    augment_model_path = f'{augment_root_dir}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting.pth'
                    augment_checkpoint = torch.load(augment_model_path)
                    model_augment.load_state_dict({k.replace('module.',''):v for k,v in augment_checkpoint.items()})
                    model_augment.eval()
                    eval_positions_augment, mses_augment,mse_confidences_augment, max_mses_augment, max_mse_confidences_augment,nlls_augment, nll_confidences_augment = run_test(model_augment,data_augment=True)

                    assert eval_positions_noaugment == eval_positions_augment
                    eval_positions = eval_positions_noaugment
                    result = {}
                    result['eval_positions'] = eval_positions_augment
                    result['mses_noaugment'] = mses_noaugment
                    result['mses_augment'] = mses_augment
                    result['mse_confidences_noaugment'] = mse_confidences_noaugment
                    result['mse_confidences_augment'] = mse_confidences_augment
                    result['max_mses_noaugment'] = max_mses_noaugment  
                    result['max_mses_augment'] = max_mses_augment
                    result['max_mse_confidences_noaugment'] = max_mse_confidences_noaugment
                    result['max_mse_confidences_augment'] = max_mse_confidences_augment
                    result['nlls_noaugment'] = nlls_noaugment
                    result['nlls_augment'] = nlls_augment
                    result['nll_confidences_noaugment'] = nll_confidences_noaugment
                    result['nll_confidences_augment'] = nll_confidences_augment
                    df = pd.DataFrame(result)
                    df.to_excel(out_data_path,index=False)

                    if draw_flag:
                        fig1, ax1 = plt.subplots()
                        fig2, ax2 = plt.subplots()
                        fig3, ax3 = plt.subplots()
                        fig4, ax4 = plt.subplots(3,1)
                        ax1.set_title('Mses')
                        ax1.plot(eval_positions,mses_noaugment,label=f'no_augment')
                        ax1.plot(eval_positions,mses_augment,label=f'augment')
                        ax1.fill_between(eval_positions,mses_noaugment-mse_confidences_noaugment,mses_noaugment+mse_confidences_noaugment,alpha=0.2)
                        ax1.fill_between(eval_positions,mses_augment-mse_confidences_augment,mses_augment+mse_confidences_augment,alpha=0.2)
                        ax1.set_xlabel('Number of revealed data points (n)')
                        ax1.set_ylabel('Mses')

                        ax2.set_title('max_mses')
                        ax2.plot(eval_positions,max_mses_noaugment,label=f'no_augment')
                        ax2.plot(eval_positions,max_mses_augment,label=f'augment')
                        ax2.fill_between(eval_positions,max_mses_noaugment-max_mse_confidences_noaugment,max_mses_noaugment+max_mse_confidences_noaugment,alpha=0.2)
                        ax2.fill_between(eval_positions,max_mses_augment-max_mse_confidences_augment,max_mses_augment+max_mse_confidences_augment,alpha=0.2)
                        ax2.set_xlabel('Number of revealed data points (n)')
                        ax2.set_ylabel('Max_mses')

                        ax3.set_title('Nlls with 0.95 confidence')
                        ax3.plot(eval_positions,nlls_noaugment,label=f'no_augment')
                        ax3.plot(eval_positions,nlls_augment,label=f'augment')
                        ax3.fill_between(eval_positions,nlls_noaugment-nll_confidences_noaugment,nlls_noaugment+nll_confidences_noaugment,alpha=0.2)
                        ax3.fill_between(eval_positions,nlls_augment-nll_confidences_augment,nlls_augment+nll_confidences_augment,alpha=0.2)
                        ax3.set_xlabel('Number of revealed data points (n)')
                        ax3.set_ylabel('Nlls')

                        ax4[0].set_title('curves for different metrics')
                        ax4[0].plot(eval_positions,mses_noaugment,label=f'no_augment')
                        ax4[0].plot(eval_positions,mses_augment,label=f'augment')
                        ax4[0].fill_between(eval_positions,mses_noaugment-mse_confidences_noaugment,mses_noaugment+mse_confidences_noaugment,alpha=0.2)
                        ax4[0].fill_between(eval_positions,mses_augment-mse_confidences_augment,mses_augment+mse_confidences_augment,alpha=0.2)
                        ax4[0].set_ylabel('Mses')
                        ax4[1].plot(eval_positions,max_mses_noaugment,label=f'no_augment')
                        ax4[1].plot(eval_positions,max_mses_augment,label=f'augment')
                        ax4[1].fill_between(eval_positions,max_mses_noaugment-max_mse_confidences_noaugment,max_mses_noaugment+max_mse_confidences_noaugment,alpha=0.2)
                        ax4[1].fill_between(eval_positions,max_mses_augment-max_mse_confidences_augment,max_mses_augment+max_mse_confidences_augment,alpha=0.2)
                        ax4[1].set_ylabel('Max_mses')
                        ax4[2].plot(eval_positions,nlls_noaugment,label=f'no_augment')
                        ax4[2].plot(eval_positions,nlls_augment,label=f'augment')
                        ax4[2].fill_between(eval_positions,nlls_noaugment-nll_confidences_noaugment,nlls_noaugment+nll_confidences_noaugment,alpha=0.2)
                        ax4[2].fill_between(eval_positions,nlls_augment-nll_confidences_augment,nlls_augment+nll_confidences_augment,alpha=0.2)
                        ax4[2].set_xlabel('Number of revealed data points (n)')
                        ax4[2].set_ylabel('Nlls')

                        ax1.legend() 
                        ax2.legend()
                        ax3.legend()
                        ax4[0].legend()
                        ax4[1].legend()
                        ax4[2].legend()
                        plt.show() # 图形可视化
                        fig1.savefig(f"{out_curve_root}/mses_curves.png")
                        fig2.savefig(f"{out_curve_root}/max_mses_curves.png")
                        fig3.savefig(f"{out_curve_root}/nll_curves.png")
                        fig4.savefig(f"{out_curve_root}/all_curves.png")
                        plt.close()
                # except:
                #     print(f'{root_dir}/numborder{num_borders}_lr{lr}_epoch{epochs}_GPfitting.pth not found!')
    # if draw_flag:
    #     ax1.legend() 
    #     ax2.legend()
    #     ax3.legend()
    #     ax4.legend()
    #     plt.show() # 图形可视化
    #     save_dir = f'{root_dir}/curves'
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     fig1.savefig(f"{save_dir}/mses_curves.png")
    #     fig2.savefig(f"{save_dir}/max_mses_curves.png")
    #     fig3.savefig(f"{save_dir}/nll_curves.png")
    #     fig4.savefig(f"{save_dir}/nll_conf_curves.png")


                
