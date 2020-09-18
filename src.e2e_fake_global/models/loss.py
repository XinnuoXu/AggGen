"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.reporter import Statistics


def hmm_loss(generator, pad_id, relation_path, vocab_size, device, train=True, label_smoothing=0.0):
    compute = HMMLoss(generator, pad_id, relation_path, vocab_size, device, label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute

class HMMLoss(nn.Module):
    def __init__(self, generator, pad_id, relation_path, vocab_size, device, label_smoothing=0.0):
        super(HMMLoss, self).__init__()
        self.generator = generator
        self.padding_idx = pad_id
        self.device = device

        with open(relation_path) as f:
            line = f.read().strip()
        relation_dict = json.loads(line)
        self.hsmm_emis_id =relation_dict['hsmm_emission']
        self.hsmm_sid = relation_dict['<s>']
    
    def compute_loss(self, batch, output, states, 
                      ex_idx, tgt_idx, mask_tgt, 
                      init_logps, trans_logps, ext_logps):

        target = batch.tgt[:,1:]
        fwd_obs_logps = self._obs_logprobs(output, target, tgt_idx, mask_tgt)
        loss = - self._compute_bwd(fwd_obs_logps, init_logps, trans_logps, ext_logps, ex_idx, states)

        #normalization = self._get_normalization(tgt_idx)
        normalization = batch.tgt_len
        batch_stats = Statistics(loss.clone().item(), normalization)
        loss.div(float(normalization)).backward()
        return batch_stats

    def mono_compute_loss(self, batch, output, states,
                           ex_idx, tgt_idx, mask_tgt,
                           init_logps, trans_logps, ext_logps):
        target = batch.tgt[:,1:]
        fwd_obs_logps = self._obs_logprobs(output, target, tgt_idx, mask_tgt)
        loss = - self._compute_bwd(fwd_obs_logps, init_logps, trans_logps, ext_logps, ex_idx, states)
        normalization = batch.tgt_len
        batch_stats = Statistics(loss.clone().item(), normalization)
        return batch_stats

    def _get_normalization(self, tgt_idx):
        normalization = 0.0
        for item in tgt_idx:
            b_p = item[0]
            e_p = item[1]+1
            normalization += (e_p-b_p)
        return normalization

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def _obs_logprobs(self, output, target, tgt_idx, mask_tgt):
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output)
        ph_size, tgt_len = target.size()
        gtruth = target.contiguous().view(-1, 1)
        psk = scores.gather(1, gtruth).view(ph_size, tgt_len)
        lls_k = []
        for i in range(ph_size):
            b_p = tgt_idx[i][0]
            e_p = tgt_idx[i][1]
            psk_i = psk[i][b_p:e_p+1]
            #lls_i = psk_i.sum(0).log()
            lls_i = psk_i.log().sum(0)
            lls_k.append(lls_i)
        return torch.stack(lls_k)

    def _permut_logps(self, init_logps, trans_logps, states):
        pmt_logps = []; init_pmt_logps = []; pre_s = -1
        for state in states:
            init_pmt_score = []
            pmt_score = []
            for i in range(len(state)):
                s = state[i]
                if i == 0:
                    init_pmt_score.append(init_logps[s])
                    #pmt_score.append(trans_logps[self.hsmm_emis_id][s])
                else:
                    init_pmt_score.append(trans_logps[pre_s][s])
                    pmt_score.append(trans_logps[pre_s][s])
                pre_s = s
            init_pmt_score = torch.stack(init_pmt_score)
            init_pmt_logps.append(init_pmt_score.sum(0))

            if len(pmt_score) > 0:
                pmt_score = torch.stack(pmt_score)
                pmt_logps.append(pmt_score.sum(0))
            else:
                pmt_logps.append(torch.tensor(0.0, device=self.device))

        return torch.stack(init_pmt_logps), torch.stack(pmt_logps)

    def _logsumexp0(self, X):
        axis = 0
        X2d = X.view(X.size(0), -1)
        maxes, _ = torch.max(X2d, axis, True)
        lse = maxes + torch.log(torch.sum(torch.exp(X2d - maxes.expand_as(X2d)), axis, True))
        return lse.squeeze()

    def _example_bwd(self, ex, fwd_obs_logps, 
                      init_pmt_logps, pmt_logps, 
                      states, trans_matrix, 
                      ext_matrix):

        # Fake a init node
        trans_logps = [[0]]
        obs_logps = [[0]]
        step_states = [[]]
        seqlen = len(ex) + 1

        for t in range(len(ex)):
            # For each time step in permutated states transition
            b_idx = ex[t][0]
            e_idx = ex[t][1]
            t_logps = []; o_lops = []; s_states = []
            for i in range(b_idx, e_idx):
                if t == 0:
                    t_logps.append(init_pmt_logps[i])
                else:
                    t_logps.append(pmt_logps[i])
                o_lops.append(fwd_obs_logps[i])
                s_states.append(states[i])
            trans_logps.append(t_logps)
            obs_logps.append(o_lops)
            step_states.append(s_states)

        beta = [None]*(seqlen)
        beta[seqlen-1] = len(trans_logps[seqlen-1])*[0]
        ctrans_id = 0
        for t in range(seqlen-2, -1, -1):
            # beta_{t}(i) = sum_j beta_{t+1}(j) p(x_{t+1}|z_{t+1}(j)) p(z_{t+1}(j)|z_{t}(i))
            beta[t] = []
            z_t = trans_logps[t]
            z_next = trans_logps[t+1]
            o_next = obs_logps[t+1]
            s_t = step_states[t]
            s_next = step_states[t+1]
            beta_next = beta[t+1]
            for i in range(len(z_t)):
                sum_j = []
                for j in range(len(z_next)):
                    # log[beta_{t+1}(j)] + log[p(x_{t+1}|z_{t+1}(j))] + log[p(z_{t+1}(j)|z_{t}(i))]
                    if t > 0:
                        s = s_next[j][0]
                        pre_s = s_t[i][-1]
                        inter_transition = trans_matrix[pre_s][s]
                        exter_transition = ext_matrix[pre_s][s]
                        if len(z_next) > 1 and len(z_t) > 1 and len(set(s_t[i])&set(s_next[j])) > 0:
                            continue
                        sum_j.append(beta_next[j] + o_next[j] + (z_next[j] + z_t[i] + inter_transition + exter_transition))
                    else:
                        s = s_next[j][0]
                        pre_s = self.hsmm_sid
                        exter_transition = ext_matrix[pre_s][s]
                        sum_j.append(beta_next[j] + o_next[j] + (z_next[j] + z_t[i]) + exter_transition)
                if len(sum_j) == 0:
                    sum_j = -float("inf")
                else:
                    sum_j = self._logsumexp0(torch.stack(sum_j))
                beta[t].append(sum_j)
        return beta

    def _compute_bwd(self, fwd_obs_logps, init_logps, trans_logps, ext_logps, ex_idx, states):
        init_pmt_logps, pmt_logps = self._permut_logps(init_logps, trans_logps, states)
        loss = []
        for ex in ex_idx:
            # For each example in a batch
            beta = self._example_bwd(ex, fwd_obs_logps,
                                      init_pmt_logps, 
                                      pmt_logps, states, 
                                      trans_logps, ext_logps)
            log_marg = self._logsumexp0(torch.stack(beta[0]))
            loss.append(log_marg)
        loss = torch.stack(loss).sum()
        return loss



def abs_loss(generator, symbols, vocab_size, device, train=True, label_smoothing=0.0):
    compute = NMTLossCompute(
        generator, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute


class LossComputeBase(nn.Module):
    def __init__(self, generator, pad_id):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = pad_id

    def _make_shard_state(self, batch, output,  attns=None):
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output):
        shard_state = self._make_shard_state(batch, output)
        _, batch_stats = self._compute_loss(batch, **shard_state)
        return batch_stats

    def sharded_compute_loss(self, batch, output, shard_size):
        batch_stats = Statistics()
        shard_state = self._make_shard_state(batch, output)
        normalization = batch.tgt[:, 1:].ne(self.padding_idx).sum().item()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return batch_stats

    def _stats(self, loss, scores, target):
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    def __init__(self, generator, symbols, vocab_size, label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, symbols['PAD'])
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output):
        return {
            "output": output,
            "target": batch.tgt[:,1:],
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output)
        gtruth = target.contiguous().view(-1)
        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)
        return loss, stats


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    if eval_only:
        yield filter_shard_state(state)
    else:
        non_none = dict(filter_shard_state(state, shard_size))
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
