import logging
import itertools
import numpy as np


def batchfy_by_bin(sorted_data, batch_bins, num_batches=0, min_batch_size=1, shortest_first=False,
                   ikey="input", okey="output"):
    if batch_bins <= 0:
        raise ValueError(f"invalid batch_bins: {batch_bins}")
    length = len(sorted_data)
    idim = int(sorted_data[0][1][ikey][0]["shape"][1])
    odim = int(sorted_data[0][1][okey][0]["shape"][1])
    logging.info(f"#utts: {length}")

    minibatches = []
    start = 0
    n = 0
    while True:
        b = 0
        next_size = 0
        max_olen = 0
        while next_size < batch_bins and (start + b) < length:
            ilen = int(sorted_data[start + b][1][ikey][0]["shape"][0]) * idim
            olen = int(sorted_data[start + b][1][okey][0]["shape"][0]) * odim
            if olen > max_olen:
                max_olen = olen
            next_size = (max_olen + ilen) * (b + 1)
            if next_size <= batch_bins:
                b += 1
            elif next_size == 0:
                raise ValueError(f"Can't fit one sample in batch_bins ({batch_bins}): Please increase the value")
        end = min(length, start + max(min_batch_size, b))
        batch = sorted_data[start:end]
        if shortest_first:
            batch.reverse()
        minibatches.append(batch)

        i = -1
        while len(minibatches[i]) < min_batch_size:
            missing = min_batch_size - len(minibatches[i])
            if -i == len(minibatches):
                minibatches[i + 1].extend(minibatches[i])
                minibatches = minibatches[1:]
                break
            else:
                minibatches[i].extend(minibatches[i - 1][:missing])
                minibatches[i - 1] = minibatches[i - 1][missing:]
                i -= 1
        if end == length:
            break
        start = end
        n += 1
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
    lengths = [len(x) for x in minibatches]
    logging.info(f"{len(minibatches)} batches containing "
                 f"from {min(lengths)} to {max(lengths)} samples (avg {np.mean(lengths)} samples)")
    return minibatches


BATCH_COUNT_CHOICES = ["auto", "seq", "bin", "frame"]
BATCH_SORT_KEY_CHOICES = ["input", "output", "shuffle"]


def make_batchset(data, num_batches=0, min_batch_size=1, shortest_first=False, batch_sort_key="input",
                  swap_io=True, count="auto", batch_bins=0):
    if count not in BATCH_COUNT_CHOICES:
        raise ValueError(f"arg 'count' ({count}) should be one of {BATCH_COUNT_CHOICES}")
    if batch_sort_key not in BATCH_SORT_KEY_CHOICES:
        raise ValueError(f"arg 'batch_sort_key' ({batch_sort_key}) should be one of {BATCH_SORT_KEY_CHOICES}")

    batch_sort_axis = 0
    if swap_io:
        ikey = "output"
        okey = "input"
        if batch_sort_key == "input":
            batch_sort_key = "output"
        elif batch_sort_key == "output":
            batch_sort_key = "input"
    else:
        raise NotImplementedError("swap_io=False not implemented")

    if count == "auto":
        if batch_bins != 0:
            count = "bin"
        else:
            raise NotImplementedError("count other than bin is not implemented")
        logging.info(f"count is auto detected as {count}")

    if count != "seq" and batch_sort_key == "shuffle":
        raise ValueError(f"batch_sort_key=shuffle is only available if batch_count=seq")

    category2data = {}
    for k, v in data.items():
        category2data.setdefault(v.get("category"), {})[k] = v
    batches_list = []
    for d in category2data.values():
        if batch_sort_key == "shuffle":
            raise NotImplementedError("batch_sort_key: shuffle not implemented")
        sorted_data = sorted(d.items(),
                             key=lambda data: int(data[1][batch_sort_key][batch_sort_axis]["shape"][0]),
                             reverse=not shortest_first)
        logging.info(f"#utts: {len(sorted_data)}")
        if count == "bin":
            batches = batchfy_by_bin(sorted_data,
                                     batch_bins=batch_bins,
                                     min_batch_size=min_batch_size,
                                     shortest_first=shortest_first,
                                     ikey=ikey, okey=okey)
        else:
            raise NotImplementedError("count other than bin is not implemented")
        batches_list.append(batches)

    if len(batches_list) == 1:
        batches = batches_list[0]
    else:
        batches = list(itertools.chain(*batches_list))

    if num_batches > 0:
        batches = batches[:num_batches]
    logging.info(f"#minibatches: {len(batches)}")

    return batches
