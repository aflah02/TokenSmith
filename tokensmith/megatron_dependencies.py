"""
This file contains functions directly lifted from the gpt-neox/megatron codebase.
The aim is to remove dependencies on the megatron codebase, so that we can use the
tokensmith library without needing to install megatron.
"""
import logging
import numpy as np
import os
import time

logger = logging.getLogger(__name__)


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset.
    
    Taken from: https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/data/gpt2_dataset.py
    """
    return np.sum(sizes[documents])

def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence length, calculate how many
    epochs will be needed.
    
    Taken from: https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/data/gpt2_dataset.py
    """
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs

def _build_doc_idx(documents, num_epochs, np_rng):
    """Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document.
    
    Taken from: https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/data/gpt2_dataset.py#L416C1-L424C19
    """
    doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)
    return doc_idx

def _build_shuffle_idx(size, np_rng):
    """Build the range [0, size) and shuffle.
    
    Taken from: https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/data/gpt2_dataset.py#L475C1-L482C23
    """
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx

def build_index_mappings(
    name,
    data_prefix,
    documents,
    sizes,
    label_dataset,
    num_samples,
    num_epochs,
    seq_length,
    seed,
    packing_impl,
    allow_chopped=True,
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.

    Modified from: https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/data/gpt2_dataset.py#L201
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    if not num_epochs:
        num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    
    if num_epochs > 1:
        logger.warning(
            " > WARNING: num_epochs is set to {}. Be warned that this may overwrite the same document multiple times across epochs.".format(
                num_epochs
            )
        )

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    _filename += "_{}pi".format(packing_impl)
    if allow_chopped:
        _filename += "_ac"
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    # Build the indexed mapping if not exist.
    if (
        (not os.path.isfile(doc_idx_filename))
        or (not os.path.isfile(sample_idx_filename))
        or (not os.path.isfile(shuffle_idx_filename))
    ):
        logger.warning(
            " > WARNING: could not find index map files, building "
            "the indices on rank 0 ..."
        )
        # doc-idx.
        start_time = time.time()
        if packing_impl == "packed":
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            logger.info(
                " > elapsed time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            from megatron.data import helpers

            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32

            num_samples = (num_epochs * tokens_per_epoch - 1) / seq_length
            if 2 * (num_samples + 1) < np.iinfo(np.int32).max:
                sample_idx = helpers.build_sample_idx_int32(
                    sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
                )
            else:
                sample_idx = helpers.build_sample_idx_int64(
                    sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
                )
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            logger.info(
                " > elapsed time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retrieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            shuffle_idx = _build_shuffle_idx(sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            logger.info(
                " > elapsed time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time)
            )
        elif packing_impl == "pack_until_overflow":
            # Naively pack data until it overflows, then roll it over to a new one instead.
            shuffle_idx = np.arange(num_samples)  # Shuffle index around epochs
            np_rng.shuffle(shuffle_idx)
            sample_idx = []
            doc_idx = []
            # Iterate over files until we have enough samples.
            temp_shuffle_idx = np.arange(len(documents))
            np_rng.shuffle(temp_shuffle_idx)
            running_length = 0
            curr_shuffle_idx = 0
            while len(sample_idx) < num_samples:
                if not allow_chopped:
                    # +1 since we shift left/right by 1
                    if sizes[temp_shuffle_idx[curr_shuffle_idx]] > seq_length + 1:
                        curr_shuffle_idx += 1
                        continue
                # First, check if we need to skip this item...
                if label_dataset is not None:
                    if np.all(
                        label_dataset.get(temp_shuffle_idx[curr_shuffle_idx])[
                            : seq_length + 1
                        ]
                        == -100
                    ):
                        curr_shuffle_idx += 1
                        continue
                doc_length = sizes[temp_shuffle_idx[curr_shuffle_idx]]
                if running_length == 0:
                    sample_idx.append(np.array([len(doc_idx), 0]))
                    doc_idx.append(temp_shuffle_idx[curr_shuffle_idx])
                    running_length += doc_length
                else:
                    if running_length + doc_length > (seq_length + 1):
                        running_length = doc_length
                        sample_idx.append(np.array([len(doc_idx), 0]))
                    else:
                        running_length += doc_length
                    doc_idx.append(temp_shuffle_idx[curr_shuffle_idx])
                curr_shuffle_idx += 1
                if curr_shuffle_idx == len(documents):
                    curr_shuffle_idx = 0
                    np_rng.shuffle(temp_shuffle_idx)
            sample_idx.append(np.array([len(doc_idx), 0]))
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
        elif packing_impl == "unpacked":
            # Unpacked data, one sample per document.
            shuffle_idx = np.arange(num_samples)  # Shuffle index around epochs
            np_rng.shuffle(shuffle_idx)
            sample_idx = np.zeros((num_samples + 1, 2), dtype=np.int64)
            sample_idx[:, 0] = np.array([i for i in range(num_samples + 1)])
            sample_idx[:, 1] = 0
            doc_idx = list()
            doc_i = 0
            while len(doc_idx) <= num_samples:
                if not allow_chopped:
                    # +1 since we shift left/right by 1
                    if sizes[doc_i] > seq_length + 1:
                        doc_i = (doc_i + 1) % len(documents)
                        continue
                # Just in case we have bad data in the loop...
                if np.all(label_dataset.get(doc_i)[:seq_length] == -100):
                    doc_i = (doc_i + 1) % len(documents)
                    continue
                doc_idx.append(doc_i)
                doc_i = (doc_i + 1) % len(documents)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)

def get_train_valid_test_split_(splits_string, size):
    """Get dataset splits from comma or '/' separated string list.
    
    Taken from: https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/data/data_utils.py#L242
    """
    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index
