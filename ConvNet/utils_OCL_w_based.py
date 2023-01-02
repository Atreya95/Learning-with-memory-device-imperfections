import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
import scipy

import subprocess

pole=15
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def RTN(state_RTN, npulses, all_Tn, amp):
    ''' Takes as input the state of each device(state_RTN) and
        the number of pulses(npulse) to be applied to each device 
        and returns the new state. '''
    torch.cuda.empty_cache()    
    r = torch.rand_like(npulses, requires_grad=False)
    Phigh_net = all_Tn[0, 2*npulses.type(torch.long)+1]
    Plow_net = all_Tn[1, 2*npulses.type(torch.long)]
    state_RTN = torch.ceil(state_RTN)
    condition =  r < Plow_net * state_RTN + Phigh_net * (1 - state_RTN)
    low_lim = 0
    up_lim = torch.rand_like(npulses, requires_grad=False)*amp
    new_state = torch.where(condition, 1 - state_RTN, state_RTN) * (torch.rand_like(npulses) * (up_lim - low_lim) + low_lim)             
    del r, state_RTN, condition    
           
    return new_state
    
def pink_noise(hold_array, npulses, pole, b_tensor):
    ''' Takes as input the array that stores the generated pink noise
        sequence and the number of pulses to be applied to each device
        and returns the output pink noise value and the updates hold_array. '''
    
    expanded = torch.cat([hold_array, torch.rand_like(hold_array)], dim = -1)
    pulses_number = torch.unique(npulses).type(torch.int).tolist()

    for p in pulses_number:
        expanded[npulses==p] = torch.roll(expanded[npulses==p], p, -1)
    
    
    hold_array_new = expanded.index_select(dim=-1, index=torch.arange(0, hold_array.shape[-1], device=device_selected))

    
    out = torch.sum(hold_array_new * b_tensor[0,:pole], -1)
       
    return out, hold_array_new
    
def mean_w(t, npulses, m1_all, t_star_all, c1_all, m2_all):
    ''' Takes as input the array with the cumulative pulses applied to each device
        and the number of pulses to be applied to each device and returns the 
        mean model values as well as the updated cumulative number of pulses for each device. '''
    torch.cuda.empty_cache()
    npulses = npulses
    t = t
    t = t + npulses

    out = torch.where(t < t_star_all, m1_all*t + c1_all , m2_all*t + t_star_all*(m1_all-m2_all) + c1_all)

    return out, t



def w_based_reset_model(M_tensor, npulses, R0, all_Tn, pole, alpha, b_tensor, amp, m1_all, t_star_all, c1_all, m2_all):
    ''' Takes as input the M_tensor - the tensor holding all the values necessary
        to compute the model, the number of pulses to be applied to each device,
        and the initial resistances of all the devices and outputs the next value
        of resistance as well as the updated M_tensor. ''' 
    
    torch.cuda.empty_cache()
    state_RTN = M_tensor.index_select(dim=-1, index=torch.tensor(pole, device=device_selected)).squeeze()
    hold_array = M_tensor.index_select(dim=-1, index=torch.arange(start=0, end=pole, device=device_selected))
    t = M_tensor.index_select(dim=-1, index=torch.tensor(pole+1, device=device_selected)).squeeze()
    
    new_state = RTN(state_RTN=state_RTN, npulses=npulses, all_Tn=all_Tn, amp=amp)
    w_RTN = new_state
    w_noise, hold_array = pink_noise(hold_array=hold_array, npulses=npulses, pole=pole, b_tensor=b_tensor)
    w_noise = w_noise * alpha
    w_mean, t = mean_w(t=t, npulses=npulses, m1_all=m1_all, t_star_all=t_star_all, c1_all=c1_all, m2_all=m2_all)
    del state_RTN
    w = w_mean  + w_RTN + w_noise
    
    Rnext = R0 * torch.exp(w)
    state_RTN = new_state

    M_tensor = torch.cat((state_RTN.unsqueeze(-1), hold_array, t.unsqueeze(-1)), dim=-1)     

    
    return Rnext, M_tensor




class Adam_with_OCL_w_based(torch.optim.Optimizer):
     
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    
    def __init__(self, params, lr=8, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False): #Gamme converts the update to number of pulses.
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_with_OCL_w_based, self).__init__(params, defaults)
    
    

    def __setstate__(self, state):
        super(Adam_with_OCL_w_based, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    

  

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
        
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                global device_selected

                if p.get_device() != -1:
                    device_selected = torch.device('cuda',p.get_device())
                else:
                    device_selected = torch.device('cpu')
                
                # Device initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)    
                    print("Initializing")
                    state['step'] = 0
                    #state['m'] = torch.zeros_like(p.data)                                       		  			   # The latent weight
                    state['M_tensor_BL'] = torch.zeros(list(p.data.size())+[(pole+2)], device = device_selected, requires_grad=False)       # Initializing the M_tensor with zeros for both BL and BLb
                    state['M_tensor_BLb'] = torch.zeros(list(p.data.size())+[(pole+2)], device = device_selected, requires_grad=False)

                    
                    ######################
                    # RTN initialization #
                    ######################
                    
                    Phigh = 0.0008 
                    Plow = 0.002
                    T = torch.tensor([[1-Phigh, Phigh], [Plow, 1-Plow]], requires_grad=False)       # Creating the transition matrix
                    
                    state['all_Tn'] = torch.zeros([2, (pole+1)*2], device = device_selected, requires_grad=False)                           # Creating all the possible transition matrices 
                                                                               # for applying 0 to 126 pulses for RTN generation
                    for i in torch.arange(0, (pole+1)):
                        state['all_Tn'][:, 2*i:2*i+2] = torch.matrix_power(T, int(i))
                        
                    
                    state['amp_BL'] = torch.rand_like(p.data, requires_grad=False) * 0.5               # The D2D variability for the RTN
                    state['amp_BLb'] = torch.rand_like(p.data, requires_grad=False) * 0.5              # for both BL and BLb
                    
                    #############################
                    # Pink noise initialization #
                    #############################
                    
                    
                    state['alpha'] = 2.5e-2                       		# The pink noise coupling term 
                    state['pole'] =  pole                                        # The pole for the pink noise generation
                                                                               # b_tensor is the coefficient matrix    
                    state['b_tensor'] = torch.tensor([[1., 0.5, 0.375, 0.3125, 0.2734375, 0.24609375, 0.22558594, 0.20947266, 0.19638062, 0.18547058, 0.17619705, 0.1681881, 0.16118026, 0.15498102, 0.14944598, 0.14446445, 0.13994993, 0.13583376, 0.1320606, 0.12858532, 0.12537069, 0.12238567, 0.11960418, 0.11700409, 0.1145665, 0.11227517, 0.11011603, 0.10807685, 0.10614691, 0.10431679,0.10257817, 0.10092369, 0.09934675, 0.0978415, 0.09640265, 0.09502547, 0.09370568,0.09243938, 0.09122307, 0.09005355, 0.08892788, 0.08784339, 0.08679764, 0.08578836,0.0848135,  0.08387112, 0.08295948, 0.08207693, 0.08122196, 0.08039317, 0.07958924, 0.07880895, 0.07805117, 0.07731484, 0.07659896, 0.07590261, 0.07522491, 0.07456504, 0.07392224, 0.07329578, 0.07268498, 0.0720892,  0.07150784, 0.07094031, 0.07038609,0.06984466, 0.06931553, 0.06879825, 0.06829238, 0.06779751, 0.06731324, 0.06683921,0.06637505, 0.06592042, 0.06547501, 0.06503851, 0.06461063, 0.06419108, 0.0637796,  0.06337593, 0.06297983, 0.06259107, 0.06220941, 0.06183466, 0.06146659, 0.06110503, 0.06074977, 0.06040063, 0.06005744, 0.05972004, 0.05938826, 0.05906195, 0.05874096,0.05842515, 0.05811438, 0.05780851, 0.05750743, 0.057211, 0.05691911, 0.05663164,0.05634848, 0.05606954, 0.05579469, 0.05552385, 0.05525691, 0.05499378, 0.05473436,0.05447858, 0.05422634, 0.05397757, 0.05373219, 0.05349013, 0.05325133, 0.05301572,0.05278323, 0.05255381, 0.05232739, 0.05210389, 0.05188324, 0.05166536, 0.05145014,0.05123749, 0.05102732, 0.05081955, 0.05061414, 0.05041107]], device=device_selected, requires_grad=False)
                    
                    #############################
                    # Mean model initialization #
                    #############################
                    
                    # From experimental distribution
                    x1, y1 = 3.7415768289771944e-05, 0.0006564678011593196 #params for m1
                    a, b, c = 0.8033817700570505, 0.0, 542.5385224079308   #params for t_star
                    m, s = 0.005293141020227579, 0.05320804175540371       #params for c1
                    x2, y2 = 1.644496812189661e-34, 2.8890516951651312e-05 #params for m2

                    # Generating and constraining the values of m1, t_star, c1 and m2
                    num_dev = p.data.numel()
                    
                    # Generating and constraining the values of m1, t_star, c1 and m2

                    # m1
                    low_m1 = scipy.stats.expon.cdf(3.74e-5, x1, y1) #F(a)
                    high_m1 = scipy.stats.expon.cdf(0.0076, x1, y1) #F(b)
                    u_m1_BL = scipy.stats.uniform.rvs(loc=low_m1, scale=high_m1-low_m1, size=num_dev)
                    u_m1_BLb = scipy.stats.uniform.rvs(loc=low_m1, scale=high_m1-low_m1, size=num_dev)
                    
                    state['m1_all_BL'] = torch.tensor(scipy.stats.expon.ppf(u_m1_BL, x1, y1), device=device_selected, requires_grad=False).view(p.data.shape)
                    state['m1_all_BLb'] = torch.tensor(scipy.stats.expon.ppf(u_m1_BLb, x1, y1), device=device_selected, requires_grad=False).view(p.data.shape)
    

                    # t_star
                    low_t_star=scipy.stats.lognorm.cdf(41, a, b, c) #F(a)
                    high_t_star=scipy.stats.lognorm.cdf(2000, a, b, c) #F(b)
                    u_t_star_BL = scipy.stats.uniform.rvs(loc=low_t_star, scale=high_t_star-low_t_star, size=num_dev)
                    u_t_star_BLb = scipy.stats.uniform.rvs(loc=low_t_star, scale=high_t_star-low_t_star, size=num_dev)
                    
                    state['t_star_all_BL'] = torch.tensor(scipy.stats.lognorm.ppf(u_t_star_BL, a, b, c), device=device_selected, requires_grad=False).view(p.data.shape)
                    state['t_star_all_BLb'] = torch.tensor(scipy.stats.lognorm.ppf(u_t_star_BLb, a, b, c), device=device_selected, requires_grad=False).view(p.data.shape)


                    # c1        
                    low_c1 = scipy.stats.norm.cdf(-0.15, m, s) #F(a)
                    high_c1 = scipy.stats.norm.cdf(0.15, m, s) #F(b)
                    u_c1_BL = scipy.stats.uniform.rvs(loc=low_c1, scale=high_c1-low_c1, size=num_dev)
                    u_c1_BLb = scipy.stats.uniform.rvs(loc=low_c1, scale=high_c1-low_c1, size=num_dev)
                    
                    state['c1_all_BL'] = torch.tensor(scipy.stats.norm.ppf(u_c1_BL, m, s), device=device_selected, requires_grad=False).view(p.data.shape)
                    state['c1_all_BLb'] = torch.tensor(scipy.stats.norm.ppf(u_c1_BLb, m, s), device=device_selected, requires_grad=False).view(p.data.shape)


                    # m2
                    low_m2 = scipy.stats.expon.cdf(0, x2, y2) #F(a)
                    high_m2 = scipy.stats.expon.cdf(0.00013, x2, y2) #F(b)
                    u_m2_BL = scipy.stats.uniform.rvs(loc=low_m2, scale=high_m2-low_m2, size=num_dev)
                    u_m2_BLb = scipy.stats.uniform.rvs(loc=low_m2, scale=high_m2-low_m2, size=num_dev)
                    
                    state['m2_all_BL'] = torch.tensor(scipy.stats.expon.ppf(u_m2_BL, x2, y2), device=device_selected, requires_grad=False).view(p.data.shape)
                    state['m2_all_BLb'] = torch.tensor(scipy.stats.expon.ppf(u_m2_BLb, x2, y2), device=device_selected, requires_grad=False).view(p.data.shape)


                    #######################################
                    # Initializating the resistance at LRS#
                    #######################################
                    
                    m_R0, s_R0 = 6988, 381.7
                    low_R0 = scipy.stats.norm.cdf(6252, m_R0, s_R0) #F(a)
                    high_R0 = scipy.stats.norm.cdf(7897, m_R0, s_R0) #F(b)
                    u_R0_BL = scipy.stats.uniform.rvs(loc=low_R0, scale=high_R0-low_R0, size=num_dev)
                    u_R0_BLb = scipy.stats.uniform.rvs(loc=low_R0, scale=high_R0-low_R0, size=num_dev)
                    
                    R0_tensor_BL = torch.tensor(scipy.stats.norm.ppf(u_R0_BL, m_R0, s_R0), device=device_selected, requires_grad=False).view(p.data.shape)
                    R0_tensor_BLb = torch.tensor(scipy.stats.norm.ppf(u_R0_BLb, m_R0, s_R0), device=device_selected, requires_grad=False).view(p.data.shape)
                    
                    
                    state['R0_BL'] =  R0_tensor_BL
                    state['R0_BLb'] = R0_tensor_BLb
                    state['RBL'] = state['R0_BL'].view(p.data.size())
                    state['RBLb'] = state['R0_BLb'].view(p.data.size())

                    
                    # State initialization
                
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        
                

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1


                # Get the update given by the Adam algorithm
                update = torch.mul(torch.div(exp_avg, denom),-step_size)

                #update = -group['lr']*grad
                
                
                if len(p.size())==1: # This ensures that the device simulation only happens for the weight parameters
                    p.addcdiv_(exp_avg, denom, value=-8e-3*step_size/group['lr']) 
                
                else:
                    
                    # Convert the update value to number of pulses (to be applied) between -126 and 126
                    npulses = torch.round(torch.clamp(update, -(pole-1.), (pole-1.)))
                    #if (state['step'] % 300 == 0):

                    
                    # Get the previous binary weights so that they can be compared for the sign chance
                    old_Bi_W = torch.sign(state['RBL'] - state['RBLb'])
                    
                    # Apply pulses to the corresponding device
                    zero = torch.zeros_like(npulses)

                    RBL_reset, M_tensor_BL_reset = w_based_reset_model(state['M_tensor_BL'], npulses=torch.where(npulses>0, torch.abs(npulses), zero), R0=state['R0_BL'], all_Tn=state['all_Tn'], pole=state['pole'], alpha=state['alpha'], b_tensor=state['b_tensor'], amp=state['amp_BL'], m1_all=state['m1_all_BL'], t_star_all=state['t_star_all_BL'], c1_all=state['c1_all_BL'], m2_all=state['m2_all_BL'])
                    RBLb_reset, M_tensor_BLb_reset = w_based_reset_model(state['M_tensor_BLb'], npulses=torch.where(npulses<=0, torch.abs(npulses), zero), R0=state['R0_BLb'], all_Tn=state['all_Tn'], pole=state['pole'], alpha=state['alpha'], b_tensor=state['b_tensor'], amp=state['amp_BLb'], m1_all=state['m1_all_BLb'], t_star_all=state['t_star_all_BLb'], c1_all=state['c1_all_BLb'], m2_all=state['m2_all_BLb'])
                    
                    RBL_reset = RBL_reset.view(p.data.size())
                    RBLb_reset = RBLb_reset.view(p.data.size())

                    # Get the new binary weights so that they can be compared for the sign chance
                    new_Bi_W = torch.sign(RBL_reset - RBLb_reset)  
                    set_pulses = (torch.ones_like(npulses)).clamp(0, (pole-1.))
                    
                   
                    old_Bi_W = old_Bi_W.unsqueeze(-1)
                    new_Bi_W = new_Bi_W.unsqueeze(-1)

                                       
                    state['M_tensor_BL'] = torch.where(old_Bi_W == new_Bi_W, M_tensor_BL_reset, M_tensor_BL_reset)
                    state['M_tensor_BLb'] = torch.where(old_Bi_W == new_Bi_W, M_tensor_BLb_reset, M_tensor_BLb_reset)

                    old_Bi_W = old_Bi_W.squeeze()
                    new_Bi_W = new_Bi_W.squeeze()  

                    state['RBL'] = torch.where(old_Bi_W == new_Bi_W, RBL_reset, RBL_reset)
                    state['RBLb'] = torch.where(old_Bi_W == new_Bi_W, RBLb_reset, RBLb_reset)

                    p.data = torch.sign(state['RBL'] - state['RBLb'])
                    
        
        return loss
