import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

import distributed
from models.reporter import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optims,loss):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    grad_accum_count = args.accum_count
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0
    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path
    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)
    trainer = Trainer(args, model, optims, loss, grad_accum_count, n_gpu, gpu_rank, report_manager)

    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):

    def __init__(self, args, model, optims, loss,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        # Basic attributes.
        self.args = args
        self.model = model
        self.optims = optims
        self.loss = loss
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.grad_accum_count = grad_accum_count
        self.save_checkpoint_steps = args.save_checkpoint_steps

        assert grad_accum_count > 0
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):

        logger.info('Start training...')
        step =  self.optims[0]._step + 1

        true_batchs = []
        accum = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        self._gradient_accumulation(true_batchs, total_stats, report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optims[0].learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                if self.args.mode == 'validate':
                    src = batch.src
                    tgt = batch.tgt

                    pmt_msk = batch.pmt_msk
                    states = batch.states
                    ex_idx = batch.ex_idx
                    tgt_idx = batch.tgt_idx

                    mask_src = batch.mask_src
                    mask_tgt = batch.mask_tgt

                    outputs, _ = self.model(src, tgt, mask_src, pmt_msk, ex_idx)
                    init_logps, trans_logps = self.model.trans_logprobs()
                    ext_logps = self.model.external_logprobs()
                    batch_stats = self.loss.mono_compute_loss(batch, outputs, states,
                                                            ex_idx, tgt_idx, mask_tgt,
                                                            init_logps, trans_logps,
                                                            ext_logps)
                else:
                    src = batch.src
                    tgt = batch.tgt
                    segs = batch.segs
                    mask_src = batch.mask_src
                    mask_tgt = batch.mask_tgt

                    outputs, _ = self.model(src, tgt, segs, mask_src, mask_tgt)
                    batch_stats = self.loss.monolithic_compute_loss(batch, outputs)

                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)

            return stats


    def _gradient_accumulation(self, true_batchs, total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            if self.args.mode == 'train':
                src = batch.src
                tgt = batch.tgt
                pmt_msk = batch.pmt_msk
                states = batch.states
                ex_idx = batch.ex_idx
                tgt_idx = batch.tgt_idx

                mask_src = batch.mask_src
                mask_tgt = batch.mask_tgt

                outputs, _ = self.model(src, tgt, mask_src, pmt_msk, ex_idx)
                init_logps, trans_logps = self.model.trans_logprobs()
                ext_logps = self.model.external_logprobs()
                batch_stats = self.loss.compute_loss(batch, outputs, states,
                                                    ex_idx, tgt_idx, mask_tgt,
                                                    init_logps, trans_logps,
                                                    ext_logps)
            else:
                src = batch.src
                tgt = batch.tgt
                segs = batch.segs
                mask_src = batch.mask_src
                mask_tgt = batch.mask_tgt

                outputs, scores = self.model(src, tgt, segs, mask_src, mask_tgt)
                batch_stats = self.loss.sharded_compute_loss(batch, outputs, self.args.generator_shard_size)

            batch_stats.n_docs = int(src.size(0))
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))

                for o in self.optims:
                    o.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            for o in self.optims:
                o.step()


    def _save(self, step):
        real_model = self.model

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optims': self.optims,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

