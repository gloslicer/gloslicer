import torch
import pickle
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from tqdm import tqdm
import os

def make_dataloader(
    args,
    examples,
    tokenizer,
    logger,
    stage,
    path_to_dataloader=None,
    force_encode=False
):
    if path_to_dataloader:
        abs_pkl_path = os.path.abspath(path_to_dataloader)
        pkl_dir = os.path.dirname(abs_pkl_path)
        os.makedirs(pkl_dir, exist_ok=True)
        logger.info(f"[PKL] Current DataLoader cache path: {abs_pkl_path}")
    else:
        abs_pkl_path = None

    if abs_pkl_path and os.path.exists(abs_pkl_path) and not force_encode:
        try:
            if os.path.getsize(abs_pkl_path) == 0:
                logger.warning(f"[Cache] File {abs_pkl_path} is empty, will re-encode.")
            else:
                logger.info(f"[Cache] Trying to load DataLoader from {abs_pkl_path} ...")
                with open(abs_pkl_path, 'rb') as handler:
                    dataloader = pickle.load(handler)
                logger.info(f"[Cache] Successfully loaded DataLoader from {abs_pkl_path}.")
                return dataloader
        except Exception as e:
            logger.warning(f"[Cache] Failed to load {abs_pkl_path}: {e}, will re-encode.")

    if stage == 'train':
        batch_size = args.train_batch_size
    else:
        batch_size = args.eval_batch_size

    (all_input_ids, all_input_masks, all_statement_ids, all_variable_ids,
     all_variable_line_numbers, all_slice_labels) = [], [], [], [], [], []

    for ex_index, example in enumerate(tqdm(examples, desc=f"{stage} dataloader encoding", ncols=80)):
        variable = example.variable
        newline_idx = tokenizer.encode("\n")[1]
        variable_start, variable_end = example.variable_loc
        input_ids, statement_ids, variable_ids = [], [], []

        for line_id, line in enumerate(example.code.split('\n')):
            tokens_ids = []
            if line_id != example.line_number:
                tokens = tokenizer.tokenize(line)
                tokens_ids = tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
                input_ids += tokens_ids
            else:
                pre, post = line[:variable_start], line[variable_end:]
                tokens_pre = tokenizer.tokenize(pre)
                tokens_var = tokenizer.tokenize(variable)
                tokens_post = tokenizer.tokenize(post)
                tokens = tokens_pre + tokens_var + tokens_post
                variable_ids = list(range(
                    len(input_ids)+len(tokens_pre),
                    len(input_ids)+len(tokens_pre)+len(tokens_var)
                ))
                tokens_ids = tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
                input_ids += tokens_ids
            statement_ids += [line_id] * len(tokens_ids)

        if len(input_ids) > args.max_tokens:
            continue
        num_statements = max(statement_ids)
        variable_ids = variable_ids + [-1 for _ in range(args.max_tokens - len(variable_ids))]
        slice_labels = [0 for _ in range(num_statements + 1)]
        out_of_window = 0
        for idx in example.backward_slice:
            if 0 <= idx <= num_statements:
                slice_labels[idx] = 1
            else:
                out_of_window += 1
        for idx in example.forward_slice:
            if 0 <= idx <= num_statements:
                slice_labels[idx] = 1
            else:
                out_of_window += 1
        if out_of_window > 0:
            logger.warning(f"[Truncation] {example.eid} {out_of_window} slice idx out of window [{num_statements}]")
        slice_labels = slice_labels + [-1 for _ in range(args.max_tokens - len(slice_labels))]
        pad_len = args.max_tokens - len(input_ids)
        input_masks = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        statement_ids = statement_ids + [args.max_tokens-1] * (args.max_tokens - len(statement_ids))

        feature_lens = list(map(len, [input_ids, input_masks, statement_ids, variable_ids, slice_labels]))
        if not all(l == args.max_tokens for l in feature_lens):
            logger.warning(f"[BadSample] {example.eid} feature lens: {feature_lens}, skip.")
            continue

        all_input_ids.append(input_ids)
        all_input_masks.append(input_masks)
        all_statement_ids.append(statement_ids)
        all_variable_ids.append(variable_ids)
        all_variable_line_numbers.append(example.line_number)
        all_slice_labels.append(slice_labels)

    logger.info(
        f"input_ids: {len(all_input_ids)}, input_masks: {len(all_input_masks)}, "
        f"statement_ids: {len(all_statement_ids)}, variable_ids: {len(all_variable_ids)}, "
        f"variable_line_numbers: {len(all_variable_line_numbers)}, slice_labels: {len(all_slice_labels)}"
    )

    dataset = TensorDataset(
        torch.tensor(all_input_ids, dtype=torch.long),
        torch.tensor(all_input_masks, dtype=torch.long),
        torch.tensor(all_statement_ids, dtype=torch.long),
        torch.tensor(all_variable_ids, dtype=torch.long),
        torch.tensor(all_variable_line_numbers, dtype=torch.long),
        torch.tensor(all_slice_labels, dtype=torch.float),
    )
    sampler = RandomSampler(dataset) if stage == 'train' else SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    # -- 保存PKL --
    if abs_pkl_path:
        logger.info(f"[Save] Saving DataLoader to {abs_pkl_path} ...")
        logger.info(f"[{stage}] Number of samples after encoding: {len(all_input_ids)}")
        with open(abs_pkl_path, 'wb') as handler:
            pickle.dump(dataloader, handler)
        logger.info(f"[Save] DataLoader successfully saved to {abs_pkl_path}")

    return dataloader


class SlidingWindowSplitter:
    def __init__(self, window_size, stride=None):
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2

    def split(self, sequence, required_indices=None):
        if required_indices is not None and not isinstance(required_indices, list):
            required_indices = [required_indices]
        result = []
        n = len(sequence)
        for win_start in range(0, max(1, n - self.window_size + 1), self.stride):
            win_end = min(win_start + self.window_size, n)
            if required_indices is not None:
                if not any(idx >= win_start and idx < win_end for idx in required_indices):
                    continue
            window_seq = sequence[win_start:win_end]
            def global2local(idx, ws=win_start):
                return idx - ws
            result.append((window_seq, win_start, win_end, global2local))
        return result

class SlidingWindowDataLoaderBuilder:
    def __init__(self, args, tokenizer, logger):
        self.args = args
        self.tokenizer = tokenizer
        self.logger = logger
        stride = getattr(args, 'stride', None)
        if stride is None:
            stride = args.max_tokens // 2
        self.splitter = SlidingWindowSplitter(window_size=args.max_tokens, stride=stride)
        self.logger.info(f"[SlidingWindow] window_size={args.max_tokens}, stride={stride}")

    def encode_example(self, example):
        variable = example.variable
        newline_idx = self.tokenizer.encode("\n")[1]
        variable_start, variable_end = example.variable_loc
        code_lines = example.code.split('\n')
        line_token_ids = []
        line_token_starts = []
        current_pos = 0
        for i, line in enumerate(code_lines):
            if i != example.line_number:
                tokens = self.tokenizer.tokenize(line)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
            else:
                pre, post = line[:variable_start], line[variable_end:]
                tokens_pre = self.tokenizer.tokenize(pre)
                tokens_var = self.tokenizer.tokenize(variable)
                tokens_post = self.tokenizer.tokenize(post)
                tokens = tokens_pre + tokens_var + tokens_post
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
            line_token_ids.append(token_ids)
            line_token_starts.append(current_pos)
            current_pos += len(token_ids)
        total_tokens = [tid for tids in line_token_ids for tid in tids]
        total_statement_ids = [i for i, tids in enumerate(line_token_ids) for _ in tids]
        total_len = len(total_tokens)

        variable_line = example.line_number
        if variable_line >= len(line_token_ids):
            return []
        var_start_in_tokens = line_token_starts[variable_line]
        pre = code_lines[variable_line][:variable_start]
        tokens_pre = self.tokenizer.tokenize(pre)
        var_token_offset = len(tokens_pre)
        tokens_var = self.tokenizer.tokenize(variable)
        variable_token_indices = list(range(
            var_start_in_tokens + var_token_offset,
            var_start_in_tokens + var_token_offset + len(tokens_var)
        ))
        if len(variable_token_indices) == 0:
            return []

        windowed_inputs = []
        windows = self.splitter.split(total_tokens, required_indices=variable_token_indices)
        for window_tokens, win_start, win_end, global2local in windows:
            window_statement_ids = total_statement_ids[win_start:win_end]
            min_statement_id = min(window_statement_ids)
            rel_statement_ids = [sid - min_statement_id for sid in window_statement_ids]
            num_statements = max(rel_statement_ids)
            variable_ids_in_window = [global2local(idx) for idx in variable_token_indices if win_start <= idx < win_end]
            if len(variable_ids_in_window) == 0:
                continue
            variable_line_in_window = rel_statement_ids[variable_ids_in_window[0]]

            slice_labels = [0 for _ in range(num_statements + 1)]
            out_of_window = 0
            for idx in getattr(example, "backward_slice", []):
                rel_idx = idx - min_statement_id
                if 0 <= rel_idx <= num_statements:
                    slice_labels[rel_idx] = 1
                else:
                    out_of_window += 1
            for idx in getattr(example, "forward_slice", []):
                rel_idx = idx - min_statement_id
                if 0 <= rel_idx <= num_statements:
                    slice_labels[rel_idx] = 1
                else:
                    out_of_window += 1
            if out_of_window > 0:
                self.logger.warning(f"[Truncation] {getattr(example, 'eid', 'noid')} {out_of_window} slice idx out of window [{num_statements}]")
            pad_len = self.args.max_tokens - (win_end - win_start)
            input_ids = window_tokens + [self.tokenizer.pad_token_id] * pad_len
            input_masks = [1] * (win_end - win_start) + [0] * pad_len
            statement_ids = rel_statement_ids + [self.args.max_tokens - 1] * pad_len
            variable_ids = variable_ids_in_window + [-1 for _ in range(self.args.max_tokens - len(variable_ids_in_window))]
            slice_labels = slice_labels + [-1 for _ in range(self.args.max_tokens - len(slice_labels))]
            feature_lens = list(map(len, [input_ids, input_masks, statement_ids, variable_ids, slice_labels]))
            if not all(l == self.args.max_tokens for l in feature_lens):
                self.logger.warning(f"[BadSample] {getattr(example, 'eid', 'noid')} feature lens: {feature_lens}, skip.")
                continue

            windowed_inputs.append(dict(
                input_ids=input_ids,
                input_masks=input_masks,
                statement_ids=statement_ids,
                variable_ids=variable_ids,
                variable_line_number=variable_line_in_window,
                slice_labels=slice_labels,
                num_slices_truncated=out_of_window
            ))
        return windowed_inputs

    def build_dataset(self, examples):
        all_input_ids, all_input_masks, all_statement_ids, all_variable_ids, \
        all_variable_line_numbers, all_slice_labels = [], [], [], [], [], []

        num_examples = len(examples)
        num_windows = 0
        num_slices_truncated = 0

        for example in tqdm(examples, desc="SlidingWindow encoding"):
            example_slices_truncated = 0
            windows = self.encode_example(example)
            num_windows += len(windows)
            for win in windows:
                num_slices_in_window = sum([1 for s in win["slice_labels"] if s >= 0])
                example_slices_truncated += win.get("num_slices_truncated", 0)
                all_input_ids.append(win["input_ids"])
                all_input_masks.append(win["input_masks"])
                all_statement_ids.append(win["statement_ids"])
                all_variable_ids.append(win["variable_ids"])
                all_variable_line_numbers.append(win["variable_line_number"])
                all_slice_labels.append(win["slice_labels"])
            num_slices_truncated += example_slices_truncated

        self.logger.info(
            f"[Stats] Original example count in split: {num_examples}，Total window samples: {num_windows}，Total slices truncated: {num_slices_truncated}"
        )

        return TensorDataset(
            torch.tensor(all_input_ids, dtype=torch.long),
            torch.tensor(all_input_masks, dtype=torch.long),
            torch.tensor(all_statement_ids, dtype=torch.long),
            torch.tensor(all_variable_ids, dtype=torch.long),
            torch.tensor(all_variable_line_numbers, dtype=torch.long),
            torch.tensor(all_slice_labels, dtype=torch.float),
        )

    def make_dataloader(self, examples, stage="train", path_to_dataloader=None, force_encode=False):
        self.logger.info(f"[DEBUG] builder.make_dataloader called, path_to_dataloader={path_to_dataloader}, force_encode={force_encode}")
        abs_pkl_path = None
        if path_to_dataloader:
            abs_pkl_path = os.path.abspath(path_to_dataloader)
            pkl_dir = os.path.dirname(abs_pkl_path)
            os.makedirs(pkl_dir, exist_ok=True)
            self.logger.info(f"[PKL] Current SlidingWindow DataLoader cache: {abs_pkl_path}")

        if abs_pkl_path and os.path.exists(abs_pkl_path) and not force_encode:
            try:
                if os.path.getsize(abs_pkl_path) == 0:
                    self.logger.warning(f"[Cache] File {abs_pkl_path} is empty, will re-encode.")
                else:
                    self.logger.info(f"[Cache] Trying to load DataLoader from {abs_pkl_path} ...")
                    with open(abs_pkl_path, 'rb') as handler:
                        dataloader = pickle.load(handler)
                    self.logger.info(f"[Cache] Successfully loaded DataLoader from {abs_pkl_path}.")
                    return dataloader
            except Exception as e:
                self.logger.warning(f"[Cache] Failed to load {abs_pkl_path}: {e}, will re-encode.")

        dataset = self.build_dataset(examples)
        if stage == 'train':
            sampler = RandomSampler(dataset)
            batch_size = self.args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = self.args.eval_batch_size
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=batch_size,
            drop_last=True if stage == 'train' else False
        )

        if abs_pkl_path:
            self.logger.info(f"[Save] Saving SlidingWindow DataLoader to {abs_pkl_path} ...")
            self.logger.info(f"[{stage}] Number of samples after encoding: {len(dataset)}")
            with open(abs_pkl_path, 'wb') as handler:
                pickle.dump(dataloader, handler)
            self.logger.info(f"[Save] DataLoader successfully saved to {abs_pkl_path}")

        return dataloader