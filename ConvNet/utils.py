'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
term_width = 80#int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = 0.03)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean = 0, std = 0.03)


class AdamW_sto(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, th = 1e-8,
                 weight_decay=1e-2, Gran = 200, beta_noise=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, th=th,
                        weight_decay=weight_decay, Gran=Gran, beta_noise=beta_noise, amsgrad=amsgrad)
        super(AdamW_sto, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW_sto, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:

                    # Initialization of max/min/Betas/G values
#                    fichier = open('updates.csv', 'w')
                    #state['m'] = torch.zeros_like(grad)
                    state['nb_up'] = 0
                    state['GmaxBL'] = 1.3e-4 + 1e-6*torch.abs(torch.randn_like(p.data))
                    state['GmaxBLb'] = 1.3e-4 + 1e-6*torch.abs(torch.randn_like(p.data))
                    state['GminBL'] = 1.3e-5 + 1e-5*torch.abs(torch.randn_like(p.data))
                    state['GminBLb'] = 1.3e-5 + 1e-5*torch.abs(torch.randn_like(p.data))
                    state['GBL'] = 1.3e-4 - 1e-6*torch.abs(torch.randn_like(p.data))
                    state['GBLb'] = 1.3e-4 - 1e-6*torch.abs(torch.randn_like(p.data))
                    state['betaBL'] = torch.abs(2 + group['beta_noise']*torch.randn_like(p.data))
                    state['betaBLb'] = torch.abs(2 + group['beta_noise']*torch.randn_like(p.data))
                    p.data = state['GBL'] - state['GBLb']


                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
    
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                
                if len(p.size())==1: # True if p is bias, false if p is weight
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    up_W = -step_size*exp_avg / denom
                    #up_W = torch.floor(1e4*Pup_W)
                    state['m'] = (1 - group['lr'])*state['m'] + group['lr'] * grad
                    threshold = group['th']
                    Bi_W = torch.sign(p.data) * torch.gt(torch.abs(state['m']), threshold).type(torch.float) * (torch.sign(p.data)*torch.sign(state['m'])+1)
                    p.data = torch.sign(p.data - Bi_W)

                    #First reading of Wb values
                    Wb = torch.sign(p.data)
                   #Prog. devices BL & BLb
                    GBL = non_linear_RRAM(state['GBL'], up_W*torch.lt(up_W,0).type(torch.float), state['GminBL'], state['GmaxBL'], state['betaBL'], group['Gran'], group['SNR'])
                    GBLb = non_linear_RRAM(state['GBLb'], -up_W*torch.gt(up_W,0).type(torch.float), state['GminBLb'], state['GmaxBLb'], state['betaBLb'], group['Gran'], group['SNR'])
                   #Second reading of devices GBL & GBLb
                    Wb_new = torch.sign(GBL - GBLb)
                   #Apply SET on devices when Wb_new != W then apply weak RESET to the specific device
                    GBL_set = non_linear_RRAM(1.3e-4 - 1e-6*torch.abs(torch.randn_like(p.data)), up_W*torch.lt(up_W,0).type(torch.float), state['GminBL'], state['GmaxBL'], state['betaBL'], group['Gran'], group['SNR'])
                    GBLb_set = non_linear_RRAM(1.3e-4 - 1e-6*torch.abs(torch.randn_like(p.data)), -up_W*torch.gt(up_W,0).type(torch.float), state['GminBLb'], state['GmaxBLb'], state['betaBLb'], group['Gran'], group['SNR'])
                   #Define new conductance values
                    state['GBL'] = GBL*torch.eq(Wb_new, Wb).type(torch.float) + (1-torch.eq(Wb_new, Wb).type(torch.float))*GBL_set
                    state['GBLb'] = GBLb*torch.eq(Wb_new, Wb).type(torch.float) + (1-torch.eq(Wb_new, Wb).type(torch.float))*GBLb_set
                   #Define new weight value
                    p.data = state['GBL'] - state['GBLb']

        return loss

def Bernoulli(x):
    bit_x = torch.gt(x,torch.rand_like(x))
    bit_x = bit_x.type(torch.float)
    return bit_x


def non_linear_RRAM(Gprev, up_W, Gmin, Gmax, beta, Gran, SNR):

    C = ((Gmax-Gmin)/beta)*(torch.exp(beta)-1)
    Bi_W  = torch.sign(up_W)*(torch.floor(torch.abs(up_W)))# + Bernoulli(torch.abs(up_W) - torch.floor(torch.abs(up_W))))
    N  = Bi_W * C * torch.lt(up_W,0).type(torch.float)
    gamma = SNR
    delta_t = N/Gran
    noise = gamma*torch.sqrt(torch.abs(N))*torch.randn_like(Gprev)/Gran
    d_G = -((Gmax-Gmin)/beta)*torch.log(1+((beta*torch.abs(delta_t))/(Gmax-Gmin))*torch.exp(-beta*(Gmax-Gprev)/(Gmax-Gmin)))+noise
    Gnext = (Gprev + d_G*torch.lt(N,0).type(torch.float))
    Gnext = (1-torch.eq(N,0).type(torch.float))*torch.max(torch.min(Gnext, Gmax+noise), Gmin+noise) + Gprev*torch.eq(N,0).type(torch.float)

    return Gnext

class Bop(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, th = 1e-6, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, th=th, weight_decay = weight_decay, amsgrad=amsgrad)
        super(Bop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Bop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Bop does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['m'] = torch.zeros_like(grad)
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
    
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                bs = 50

                if len(p.size())==1: # True if p is bias, false if p is weight
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                        
                else:
                    if state['step'] % bs == 0 :
                        state['m'] = (1 - group['lr'])*state['m'] + (group['lr']/bs)*grad
                        Bi_W = torch.sign(p.data) * torch.gt(torch.abs(state['m']), group['th']).type(torch.float) * (torch.sign(p.data)*torch.sign(state['m'])+1)
                        p.data = torch.sign(p.data - Bi_W)
                    else:
                        state['m'] = state['m'] + (group['lr']/bs)*grad


        return loss
