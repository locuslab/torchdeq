import torch
import torch.nn as nn

from modules.models import *


def get_model(model_args):
    model_type = model_args['model_type']
    if model_type == 'implicit':
        print(model_args)
        model = DEQINR(
            **model_args,
        )
    elif model_type == 'siren':
        model = SIREN(
            in_channels=model_args['in_channels'],
            interm_channels=model_args['interm_channels'],
            out_channels=model_args['out_channels'],
            scale=model_args['input_scale'],
            n_layers=model_args['n_layers'],
        )
    elif model_type == 'ffn':
        model = FFN(
            in_channels=model_args['in_channels'],
            interm_channels=model_args['interm_channels'],
            out_channels=model_args['out_channels'],
            n_layers=model_args['n_layers'],
        )

    return model


def construct_model_args(
        model_type,
        args,
        n_layers, 
        in_channels, 
        interm_channels, 
        output_channels,
        input_scale=256.,
        use_implicit=False, 
        filter_type='fourier', 
        filter_options={},
        norm_type='none'
        ):
    _dict = {
        'model_type': model_type,
        'args': args,
        'n_layers': n_layers,
        'in_channels': in_channels,
        'interm_channels': interm_channels,
        'output_channels': output_channels,
        'input_scale': input_scale,
        'one_pass': not use_implicit,
        'filter_type': filter_type,
        'filter_options': filter_options,
        'norm_type': norm_type,
        'init': 'default' if not use_implicit else 'deq',
    }
    return _dict


def get_summary_dict(args, in_channels, out_channels=None, filter_options=['gabor', 'fourier', 'siren_like'], size_options=[(1, 256), (1, 512), (4, 256)], input_scale=256.):
    if out_channels is None:
        out_channels = in_channels

    deq_options = [True, False]
    _dict = {}

    for size in size_options:
        for filter_t in filter_options:
            for deq in deq_options:
                n_layer, interm_channels = size

                deq_tag = 'DEQ-' if deq else ''
                filter_tag = filter_t.capitalize() + '-'

                interm_channel_tag = '{:d}D'.format(interm_channels)
                tag = deq_tag + filter_tag + 'MFN' + str(n_layer) + 'L' + interm_channel_tag

                _dict[tag] = {
                    'config': construct_model_args(
                        'implicit', 
                        args,
                        n_layer, 
                        in_channels, 
                        interm_channels, 
                        out_channels, 
                        # input_scale=input_scale / 2 if filter_t == 'siren_like' else input_scale,
                        input_scale=input_scale,
                        use_implicit=deq, 
                        filter_type=filter_t, 
                        filter_options={'alpha': 3.0},
                        norm_type='spectral_norm'
                        ), 
                    'results': {}, 'state_dicts': [], 'misc': {}, 'finished': False
                }
    return _dict
