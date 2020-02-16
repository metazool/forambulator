"""Kick off the StyleGAN training loop
Code lifted from the demo notebook forked at
https://github.com/metazool/stylegan-art/blob/master/styleganportraits.ipynb

This needs to run with a checkout of stylegan-art in its PYTHONPATH
"""
import copy
import dnnlib
from dnnlib import EasyDict


def train(tfrecord_dir=None,
          resume_from=None,
          result_dir='results',
          save_ticks=1):

    # Description string included in result subdir name.
    desc = 'sgan'
    # Options for training loop.
    train = EasyDict(run_func_name='training.training_loop.training_loop')
    # Options for generator network.
    G = EasyDict(func_name='training.networks_stylegan.G_style')
    # Options for discriminator network.
    D = EasyDict(func_name='training.networks_stylegan.D_basic')
    # Options for generator optimizer.
    G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)
    # Options for discriminator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)
    # Options for generator loss.
    G_loss = EasyDict(func_name='training.loss.G_logistic_nonsaturating')
    # Options for discriminator loss.
    D_loss = EasyDict(
        func_name='training.loss.D_logistic_simplegp',
        r1_gamma=10.0)
    # Options for load_dataset().
    dataset = EasyDict()
    # Options for TrainingSchedule.
    sched = EasyDict()
    # Options for setup_snapshot_image_grid().
    grid = EasyDict(size='4k', layout='random')
    # metrics       = [metric_base.fid50k]
    # # Options for MetricGroup.
    # Options for dnnlib.submit_run().
    submit_config = dnnlib.SubmitConfig()
    # Options for tflib.init_tf().
    tf_config = {'rnd.np_random_seed': 1000}

    # Ddefault options.
    train.total_kimg = 25000
    sched.lod_initial_resolution = 8
    sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)
    desc += '-custom'
    dataset = EasyDict(tfrecord_dir=tfrecord_dir)
    train.mirror_augment = True
    # Number of GPUs.
    desc += '-1gpu'
    submit_config.num_gpus = 1
    sched.minibatch_base = 4
    sched.minibatch_dict = {
        4: 128,
        8: 128,
        16: 128,
        32: 64,
        64: 32,
        128: 16,
        256: 8,
        512: 4}

    kwargs = EasyDict(train)
    kwargs.update(
        G_args=G,
        D_args=D,
        G_opt_args=G_opt,
        D_opt_args=D_opt,
        G_loss_args=G_loss,
        D_loss_args=D_loss,
        resume_run_id=resume_from,
        network_snapshot_ticks=save_ticks)
    kwargs.update(
        dataset_args=dataset,
        sched_args=sched,
        grid_args=grid,
        tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(
        result_dir)
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)
