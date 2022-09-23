'''
module to generate random signals
- sine 
- cosine
- exp
- AM (amplitude modulation)
- FM (frequency modulation)
- self-impose signals
- add noise.
'''
import numpy as np
import scipy
import scipy.signal as signal
random_seed = 42
def gen_time(tmin,tmax,N,endpoint):
    return np.linspace(tmin, tmax, N, endpoint=endpoint)
def gen_sine(t,f,phi0):
    return np.sin((2 * np.pi * f * t ) + phi0)

def gen_cos(t,f,phi0):
    return np.cos((2 * np.pi * f * t ) + phi0)

def gen_exp(t,f,phi0):
    return np.exp((2 * np.pi * f * t ) + phi0)


def amp_mod(carrier,t,fa,ac,ka):
    modulator = np.sin(2 * np.pi * fa * t)
    envelope = ac * (1.0 + ka * modulator)
    modulated = envelope * carrier
    return modulated

def freq_mod(carrier,t,fc,kf):
    return np.cos((2*np.pi*fc*t + kf*np.cumsum(carrier))) # modulated signal



def add_noise(sig,snr,r_seed=random_seed):
    np.random.seed(r_seed)
    noise_var = sig.var() / (10**(snr/10))
    noise = np.sqrt(noise_var) * np.random.randn(len(sig))
    noisy_sig = sig + noise
    return noisy_sig

def gen_syn_signal(type='sine',n=1000,t=(0,1),f=1,phi0=0,snr=None,am=False,ac=1.0, fa=50,
                    ka=0.25,fm=False,fc=50,kf=0.25,s_imp=False,seed=random_seed,endpoint=False):
    if not isinstance(type,str):
        raise TypeError('Signal Type must be a string.')

    if isinstance(t,(int,float)):
        time = gen_time(0,t,n,endpoint)
    elif isinstance(t,(list,tuple)):
        time = gen_time(min(t),max(t),n,endpoint)
    else:
        raise TypeError('Time must be a number(tmax) or list of two numbers(tmin,tmax).')

    if type=='sine':
        sig = gen_sine(time,f,phi0)
    elif type == 'cos':
        sig = gen_cos(time,f,phi0)
    elif type == 'exp':
        sig = gen_exp(time,f,phi0)
    else:
        raise ValueError("This signal type isn't supported. Please enter 'sine' , 'cos', or 'exp'")
    
    if s_imp:
        sig += sig[::-1]
    if snr:
        sig = add_noise(sig,snr,r_seed = random_seed)
    if fm:
        sig = freq_mod(sig,time,fc,kf)
    if am:
        sig = amp_mod(sig,time,fa,ac,ka)
    
    return sig,time