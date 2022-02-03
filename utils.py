import re
import yaml
import socket
import getpass
import numpy as np
from config import _C as C


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")

def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)
    
def _combine_std(std_x, std_y):
  return np.sqrt(std_x**2 + std_y**2)

def update_metadata(metadata, device):
    updated_metadata = {}
    for key in metadata:
        if key == 'vel_std':
            updated_metadata[key] = _combine_std(metadata[key], C.NET.NOISE).to(device)
        elif key == 'acc_std':
            updated_metadata[key] = _combine_std(metadata[key], C.NET.NOISE).to(device)
        else:
            updated_metadata[key] = metadata[key].to(device)

    return updated_metadata

def time_diff(input_seq):
    return input_seq[:, 1:] - input_seq[:, :-1]

def get_random_walk_noise(pos_seq, idx_timestep, noise_std):
    noise_shape = (pos_seq.shape[0], pos_seq.shape[1]-1, pos_seq.shape[2])
    n_step_vel = noise_shape[1]
    acc_noise = np.random.normal(0, noise_std / n_step_vel ** 0.5, size=noise_shape).astype(np.float32)
    vel_noise = np.cumsum(acc_noise, axis=1)
    pos_noise = np.cumsum(vel_noise, axis=1)
    pos_noise = np.concatenate([np.zeros_like(pos_noise[:, :1]),
                                pos_noise], axis=1)

    return pos_noise

def get_data_root():
    hostname = socket.gethostname()
    username = getpass.getuser()
    paths_yaml_fn = 'configs/paths.yaml'
    with open(paths_yaml_fn, 'r') as f: 
        paths_config = yaml.load(f, Loader=yaml.Loader)

    for hostname_re in paths_config:
        if re.compile(hostname_re).match(hostname) is not None:
            for username_re in paths_config[hostname_re]:
                if re.compile(username_re).match(username) is not None:
                    return paths_config[hostname_re][username_re]['data_dir']

    raise Exception('No matching hostname or username in config file')
