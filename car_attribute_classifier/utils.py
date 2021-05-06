import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
def load_model(model_name):
    print(f'load pre-trained encoder of {model_name}')
    if model_name == 'resnet18'
    elif model_name == 'resnet50':
    elif model_name == 'vgg16':
    else:
        raise KeyError(f'Not permitted model name `{model_name}`!')
'''

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.mse = nn.MSELoss()
        self.crossEntropy = nn.BCELoss()

    def forward(self, pred_attribute, pred_landmark, attributes, landmark):
        loss0 = self.crossEntropy(pred_attribute, attributes)
        loss1 = self.mse(pred_landmark, landmark)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]
        
        return loss0 + loss1

fn_tonumpy = lambda x : x.to('cpu').detach().numpy()

def save(ckpt_dir,net,optimizer,epoch,model_name):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict()},'%s/%s_epoch%d.pth'%(ckpt_dir,model_name,epoch))

def load(ckpt_dir,net,optimizer):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optimizer, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    # ckpt_lst.sort(key = lambda f : int(''.join(filter(str,isdigit,f))))
    print(ckpt_lst[-1]) # 맨 마지막에 있는거 뽑아오도록,,,
    dict_model = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optimizer.load_state_dict(dict_model['optimizer'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net,optimizer,epoch