"""Kick off the StyleGAN2 training run

Slight adaptation to this script:
https://raw.githubusercontent.com/NVlabs/stylegan2/master/run_training.py

Which provides more defaults and should work with existing notebooks

This needs a checkout of stylegan2 in PYTHONPATH
"""
import copy
import os

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults

# ----------------------------------------------------------------------------

_valid_configs = [
    # Table 1
    'config-a',  # Baseline StyleGAN
    'config-b',  # + Weight demodulation
    'config-c',  # + Lazy regularization
    'config-d',  # + Path length regularization
    'config-e',  # + No growing, new G & D arch.
    'config-f',  # + Large networks (default)

    # Table 2
    'config-e-Gorig-Dorig', 'config-e-Gorig-Dresnet', 'config-e-Gorig-Dskip',
    'config-e-Gresnet-Dorig', 'config-e-Gresnet-Dresnet', 'config-e-Gresnet-Dskip',
    'config-e-Gskip-Dorig', 'config-e-Gskip-Dresnet', 'config-e-Gskip-Dskip',
]

# ----------------------------------------------------------------------------


def train(dataset='tfrecords',
          data_dir=None,
          resume_from=None,
          result_dir='results',
          config_id='config-f',
          num_gpus=1,
          gamma=None,
          mirror_augment=True,
          metrics='fid50k',
          total_kimg=25000,
          save_ticks=1):

    # Options for training loop.
    train = EasyDict(run_func_name='training.training_loop.training_loop')
    # Options for generator network.
    G = EasyDict(func_name='training.networks_stylegan2.G_main')
    # Options for discriminator network.
    D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')
    # Options for generator optimizer.
    G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)
    # Options for discriminator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)
    # Options for generator loss.
    G_loss = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')
    # Options for discriminator loss.
    D_loss = EasyDict(func_name='training.loss.D_logistic_r1')
    # Options for TrainingSchedule.
    sched = EasyDict()
    # Options for setup_snapshot_image_grid().
    grid = EasyDict(size='8k', layout='random')
    # Options for dnnlib.submit_run().
    sc = dnnlib.SubmitConfig()
    # Options for tflib.init_tf().
    tf_config = {'rnd.np_random_seed': 1000}

    if not data_dir:
        data_dir = os.getcwd()

    if resume_from:
        train.resume_pkl = resume_from

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = save_ticks

    sched.G_lrate_base = sched.D_lrate_base = 0.002
    sched.minibatch_size_base = 32
    sched.minibatch_gpu_base = 4
    D_loss.gamma = 10
    metrics = [metric_defaults[x] for x in metrics]
    desc = 'stylegan2'

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset)

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    assert config_id in _valid_configs
    desc += '-' + config_id

    # Configs A-E: Shrink networks to match original StyleGAN.
    if config_id != 'config-f':
        G.fmap_base = D.fmap_base = 8 << 10

    # Config E: Set gamma to 100 and override G & D architecture.
    if config_id.startswith('config-e'):
        D_loss.gamma = 100
        if 'Gorig' in config_id:
            G.architecture = 'orig'
        if 'Gskip' in config_id:
            G.architecture = 'skip'  # (default)
        if 'Gresnet' in config_id:
            G.architecture = 'resnet'
        if 'Dorig' in config_id:
            D.architecture = 'orig'
        if 'Dskip' in config_id:
            D.architecture = 'skip'
        if 'Dresnet' in config_id:
            D.architecture = 'resnet'  # (default)

    # Configs A-D: Enable progressive growing and switch to networks that
    # support it.
    if config_id in ['config-a', 'config-b', 'config-c', 'config-d']:
        sched.lod_initial_resolution = 8
        sched.G_lrate_base = sched.D_lrate_base = 0.001
        sched.G_lrate_dict = sched.D_lrate_dict = {
            128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        sched.minibatch_size_base = 32  # (default)
        sched.minibatch_size_dict = {8: 256, 16: 128, 32: 64, 64: 32}
        sched.minibatch_gpu_base = 4  # (default)
        sched.minibatch_gpu_dict = {8: 32, 16: 16, 32: 8, 64: 4}
        G.synthesis_func = 'G_synthesis_stylegan_revised'
        D.func_name = 'training.networks_stylegan2.D_stylegan'

    # Configs A-C: Disable path length regularization.
    if config_id in ['config-a', 'config-b', 'config-c']:
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns')

    # Configs A-B: Disable lazy regularization.
    if config_id in ['config-a', 'config-b']:
        train.lazy_regularization = False

    # Config A: Switch to original StyleGAN networks.
    if config_id == 'config-a':
        G = EasyDict(func_name='training.networks_stylegan.G_style')
        D = EasyDict(func_name='training.networks_stylegan.D_basic')

    if gamma is not None:
        D_loss.gamma = gamma

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(
        G_args=G,
        D_args=D,
        G_opt_args=G_opt,
        D_opt_args=D_opt,
        G_loss_args=G_loss,
        D_loss_args=D_loss)
    kwargs.update(
        dataset_args=dataset_args,
        sched_args=sched,
        grid_args=grid,
        metric_arg_list=metrics,
        tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)
