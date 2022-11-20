import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
#I should probably move the FP stuff to here.
import numpy as np

def max_power(general_para, channel,alpha):
    n_layouts = channel.shape[0]
    sr_one = compute_rates(general_para,np.ones((n_layouts,channel.shape[-1])),channel)*alpha
    sr_one = np.sum(sr_one,axis=-1)
    return sr_one

"""
WMMSE
"""
def WMMSEBenchmark(var_noise,channel,alpha=None):
    n_layouts = channel.shape[0]
    init_x = np.ones((n_layouts, channel.shape[-1],1))
    p_opt = WMMSE(var_noise,channel,init_x = init_x,alpha=alpha)
    rates = np.expand_dims(compute_rates(var_noise,p_opt,channel), axis = 0) * alpha
    sr = np.sum(rates,axis=-1)
    return sr


def WMMSE(var_noise, channel,init_x,alpha, iter_n = 100):
    p_int = init_x
    H = channel
    Pmax = 1
    N = p_int.shape[0]
    K = p_int.shape[1]

    b = np.sqrt(p_int)
    f = np.zeros((N,K,1) )
    w = np.zeros( (N,K,1) )
    

    mask = np.eye(K)
    rx_power = np.multiply(H, b)
    rx_power_s = np.square(rx_power)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
    
    interference = np.sum(rx_power_s, 2) + var_noise
    f = np.divide(valid_rx_power,interference)
    w = 1/(1-np.multiply(f,valid_rx_power)+1e-10)
    
    for ii in range(iter_n):
        fp = np.expand_dims(f,1)
        rx_power = np.multiply(H.transpose(0,2,1), fp)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        bup = np.multiply(alpha,np.multiply(w,valid_rx_power))
        rx_power_s = np.square(rx_power)
        wp = np.expand_dims(w,1)
        alphap = np.expand_dims(alpha,1)
        bdown = np.sum(np.multiply(alphap,np.multiply(rx_power_s,wp)),2)
        btmp = bup/bdown
        b = np.minimum(btmp, np.ones((N,K) )*np.sqrt(Pmax)) + np.maximum(btmp, np.zeros((N,K) )) - btmp
        
        bp = np.expand_dims(b,1)
        rx_power = np.multiply(H, bp)
        rx_power_s = np.square(rx_power)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        interference = np.sum(rx_power_s, 2) + var_noise
        f = np.divide(valid_rx_power,interference)
        w = 1/(1-np.multiply(f,valid_rx_power)+1e-10)
    p_opt = np.square(b)
    return p_opt


def compute_rates(var_noise, p_opt, channel):
    Hsq = np.square(channel)
    signal = p_opt * np.diagonal(Hsq, axis1=1, axis2=2)
    inter_channel = Hsq * ((np.identity(Hsq.shape[-1]) < 1).astype(float))
    interference = np.squeeze(np.matmul(inter_channel, np.expand_dims(p_opt, axis=-1))) + var_noise
    SNR = signal / interference
    rates = np.log2(1 + SNR) 
    return rates