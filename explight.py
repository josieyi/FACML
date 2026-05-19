import os
import re
import sys
import time
import json
import torch
import random
import logging
import numpy as np
from datetime import timedelta


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


def initialize_exp(params):
    """
    Initialize the experiment:
    - dump parameters
    - create a logger
    """
    # dump parameters
    exp_folder = get_dump_path(params)
    json.dump(vars(params), open(os.path.join(exp_folder, 'params.pkl'), 'w'), indent=4)

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(exp_folder, 'train.log'), rank=getattr(params, 'global_rank', 0))
    print('Runing command:', command)
    return logger, exp_folder


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0
    assert not params.dump_path in ('', None), \
        'Please choose your favorite destination for dump.'
    dump_path = params.dump_path
    prename = params.dataset + '_' + str(params.test_dataset) + '_' + str(params.n_support)
    result_path = os.path.join(dump_path, prename)
    os.makedirs(result_path, exist_ok=True)
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = result_path + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    exp_folder = path_to_results

    if os.path.exists(exp_folder):
        pass
    else:
        os.makedirs(exp_folder)

    return exp_folder


def describe_model(model, path, name='model'):
    file_path = os.path.join(path, f'{name}.describe')
    with open(file_path, 'w') as fout:
        print(model, file=fout)


def set_seed(seed):
    """
    Freeze every seed for reproducibility.
    torch.cuda.manual_seed_all is useful when using random generation on GPUs.
    e.g. torch.cuda.FloatTensor(100).uniform_()
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(model, save_dir, epoch=None, model_name='model'):
    model_to_save = model.module if hasattr(model, "module") else model
    if epoch is None:
        save_path = os.path.join(save_dir, f'{model_name}.pkl')
    else:
        save_path = os.path.join(save_dir, f'{model_name}-{epoch}.pkl')
    torch.save(model_to_save.state_dict(), save_path)


def load_model(path, map_location):
    return torch.load(path, map_location=map_location)
