import os
import os.path as osp

from .smpl_layer import SMPL
from .smplh_layer import SMPLH
from .smplx_layer import SMPLX


def create(model_path, model_type = 'smpl', **kwargs) :
    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)

    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return SMPLH(model_path, **kwargs)
    elif model_type.lower() == 'smplx':
        return SMPLX(model_path, **kwargs)
    else:
        raise ValueError('Unknown model type {}, exiting!'.format(model_type))
    
def get_num_vertics(model_type = 'smpl'):

    if model_type.lower() == 'smpl':
        return 6890
    elif model_type.lower() == 'smplx':
        return 10475
    else:
        raise ValueError('Unknown model type {}, exiting!'.format(model_type))