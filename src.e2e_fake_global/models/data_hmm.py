import bisect
import gc
import glob
import random
import torch
from others.logging import logger


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False, autogressive=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.autogressive = autogressive
            pre_src = [x[0] for x in data]
            pre_src_mask = [x[1] for x in data]
            pre_state = [x[2] for x in data]
            pre_tgt = [x[3] for x in data]
            pre_auto = [x[4] for x in data]
            pre_len = [x[5] for x in data]
            relations = [x[6] for x in data]

            if not is_test:
                ex_idx, tgt_idx, src, pmt_msk, states, tgt, mask_src, mask_tgt = \
                    self._process(pre_src, pre_src_mask, pre_state, pre_tgt, pre_auto)

                setattr(self, 'src', src.to(device))
                setattr(self, 'tgt', tgt.to(device))
                setattr(self, 'pmt_msk', pmt_msk.to(device))

                setattr(self, 'states', states)
                setattr(self, 'ex_idx', ex_idx)
                setattr(self, 'tgt_idx', tgt_idx)
                setattr(self, 'tgt_len', sum(pre_len))

                setattr(self, 'mask_src', mask_src.to(device))
                setattr(self, 'mask_tgt', mask_tgt.to(device))

            else:
                ex_idx, src, pmt_msk, states, mask_src = \
                    self._process_test(pre_src, pre_src_mask, pre_state)

                setattr(self, 'src', src.to(device))
                setattr(self, 'pmt_msk', pmt_msk.to(device))

                setattr(self, 'states', states)
                setattr(self, 'ex_idx', ex_idx)
                setattr(self, 'relations', relations)

                setattr(self, 'mask_src', mask_src.to(device))

                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)


    def _process_test(self, pre_src, pre_mask, pre_state):
        ex_idx = []; tgt_idx = []
        src = []; pmt_msk = []; states = []
        for i in range(len(pre_src)):
            src_ex = pre_src[i]
            mask_ex = pre_mask[i]
            state_ex = pre_state[i]
            step_info = []; s_idx = len(pmt_msk)
            for step in range(len(mask_ex)):
                step_info.append((s_idx, s_idx+len(mask_ex[step])))
                s_idx += len(mask_ex[step])
                pmt_msk.extend(mask_ex[step])
                states.extend(state_ex[step])
            ex_idx.append(step_info)
            src.append(src_ex)
        src = torch.tensor(self._pad(src, 0))
        pmt_msk = torch.tensor(self._pad(pmt_msk, True))
        mask_src = ~(src == 0)
        return ex_idx, src, pmt_msk, states, mask_src


    def _process(self, pre_src, pre_mask, pre_state, pre_tgt, pre_auto):
        ex_idx = []; tgt_idx = []
        src = []; pmt_msk = []; states = []; tgt = []
        b_tok = pre_auto[0][0][0]
        for i in range(len(pre_src)):
            src_ex = pre_src[i]
            mask_ex = pre_mask[i]
            state_ex = pre_state[i]
            tgt_ex = pre_tgt[i]
            auto_ex = pre_auto[i]
            step_info = []; s_idx = len(pmt_msk)
            for step in range(len(mask_ex)):
                step_info.append((s_idx, s_idx+len(mask_ex[step])))
                s_idx += len(mask_ex[step])
                pmt_msk.extend(mask_ex[step])
                states.extend(state_ex[step])
                if self.autogressive:
                    t = auto_ex[step] + tgt_ex[step]
                    t_idx = (len(auto_ex[step])-1, len(t)-1)
                else:
                    t = [b_tok] + tgt_ex[step]
                    t_idx = (1-1, len(t)-1)
                tgt.extend([t for i in range(len(mask_ex[step]))])
                tgt_idx.extend([t_idx for i in range(len(mask_ex[step]))])
            ex_idx.append(step_info)
            src.append(src_ex)
            #print ([len(item) for item in src_ex])

        src = torch.tensor(self._pad(src, 0))
        tgt = torch.tensor(self._pad(tgt, 0))
        pmt_msk = torch.tensor(self._pad(pmt_msk, True))

        mask_src = ~(src == 0)
        mask_tgt = ~(tgt == 0)

        return ex_idx, tgt_idx, src, pmt_msk, states, tgt, mask_src, mask_tgt


    def __len__(self):
        return self.batch_size



def load_dataset(args, corpus_type, shuffle):
    assert corpus_type in ["train", "dev", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        self.batch_size_fn = abs_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        src_mask = ex['src_mask']
        relations = ex['relations']
        comb_rels = ex["comb_rels"]
        tgt = ex['tgt']
        tgt_atg = ex['tgt_atg']
        tgt_len = ex['tgt_len']

        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        if(is_test):
            return src, src_mask, comb_rels, tgt, tgt_atg, tgt_len, relations, src_txt, tgt_txt
        else:
            return src, src_mask, comb_rels, tgt, tgt_atg, tgt_len, relations

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            p_batch = sorted(buffer, key=lambda x: len(x[2]))
            p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            p_batch = self.batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test, self.args.autogressive)

                yield batch
            return


