
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import logging
from args_parser import args_parser
from explight import initialize_exp, set_seed, describe_model
from meta_learner import MetaLearner

logger = logging.getLogger()


class Runner:
    def __init__(self, args, logger_path):
        self.args = args
        self.meta_learner = MetaLearner(args)
        describe_model(self.meta_learner.maml.module, logger_path, name='model')
        self.logger_path = logger_path


    def run(self):

        best_score = float('-inf')
        pbar = tqdm(range(1, self.args.episode + 1))
        cost_time_ls = []

        if getattr(self.args, 'test_dataset', None) is not None:
            logger.info(f'Running transferable/cross-domain few-shot setting: {self.args.dataset} -> {self.args.test_dataset}')
        else:
            logger.info(f'Running within-dataset few-shot setting on {self.args.dataset}')

        for epoch in pbar:
            start = time.time()
            loss_cls = self.meta_learner.train_step(epoch)
            cost_time = time.time() - start
            cost_time_ls.append(cost_time)

            pbar.set_description(f'loss={loss_cls:.4f}')

            if epoch % self.args.eval_step == 0:
                score, task_auc = self.meta_learner.test_step()
                if not np.isnan(score) and score > best_score:
                    best_score = score

                logger.info(
                    f'{epoch} | '
                    f'current_score: {score:.5f}, '
                    f'best_score: {best_score:.5f}'
                )

        logger.info(f'best_score: {best_score:.5f}')
        logger.info(f'time cost: {np.mean(cost_time_ls):.5f}s')

        return {
            'dataset': self.args.dataset,
            'test_dataset': self.args.test_dataset,
            'best_score': float(best_score),
            'mean_time_per_epoch': float(np.mean(cost_time_ls)) if len(cost_time_ls) > 0 else float('nan'),
        }


def main():
    args = args_parser()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    set_seed(args.random_seed)
    logger, logger_path = initialize_exp(args)
    runner = Runner(args, logger_path)
    result = runner.run()


if __name__ == '__main__':
    main()
