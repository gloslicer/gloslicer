# dataprocess_ns_slicer.py

import os
import json
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import random
import numpy as np


from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

def collect_all_slices(json_dir, filename='slices.json', split_keys=('train', 'val', 'test')):
    import glob
    pattern = os.path.join(json_dir, '**', filename)
    all_files = glob.glob(pattern, recursive=True)
    if len(all_files) == 0:
        logger.warning(f"Cannot find any files matching pattern: {pattern}")
        return {k: [] for k in split_keys}

    logger.info(f"Collecting all slice files: {pattern}")
    all_items = []
    for fpath in all_files:
        with open(fpath, 'r') as fin:
            items = json.load(fin)
            all_items.extend(items)
    logger.info(f"Collected {len(all_files)} files, merged into {len(all_items)} samples.")

    split_items = {k: [] for k in split_keys}
    for item in all_items:
        split = item.get('split', None)
        if split is None:
            split = 'train'
        if split in split_items:
            split_items[split].append(item)
        else:
            split_items['train'].append(item)

    return split_items

class InputExample:
    def __init__(
            self, eid, code, variable, variable_loc, line_number,
            backward_slice=None, forward_slice=None
    ):
        self.eid = eid
        self.code = code
        self.variable = variable
        self.variable_loc = variable_loc
        self.line_number = line_number
        self.backward_slice = backward_slice
        self.forward_slice = forward_slice

class BaseDataProcessor(ABC):
    def __init__(self, json_dir=None, filename='slices.json', split_keys=('train', 'val', 'test')):
        self.json_dir = json_dir
        self.filename = filename
        self.split_keys = split_keys
        self._split_examples = None

    def collect_split_examples(self):
        if self._split_examples is None:
            self._split_examples = collect_all_slices(
                self.json_dir, filename=self.filename, split_keys=self.split_keys
            )
        return self._split_examples

    def get_train_examples(self):
        split_examples = self.collect_split_examples()
        return self.build_examples(split_examples['train'])

    def get_val_examples(self):
        split_examples = self.collect_split_examples()
        return self.build_examples(split_examples['val'])

    def get_test_examples(self):
        split_examples = self.collect_split_examples()
        return self.build_examples(split_examples['test'])

    @abstractmethod
    def build_examples(self, items):
        pass

    def load_examples(self, path_to_file):
        with open(str(path_to_file), 'rb') as handler:
            examples = pickle.load(handler)
        return examples

    def save(self, examples, path_to_file):
        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(examples, handler)


class CompleteDataProcessorFromScratch(BaseDataProcessor):
    def __init__(self, data_dir='../data'):
        self.path_to_dataset = Path(data_dir)

    def create_examples(self, stage):

        projects = sorted(os.listdir(str(self.path_to_dataset)))
        projects = [project for project in projects \
                    if Path(self.path_to_dataset / project).is_dir()]
        num_train = int(0.8 * len(projects))
        num_val = int(0.1 * len(projects))

        if stage == 'train':
            stage_projects = projects[: num_train]
        elif stage == 'val':
            stage_projects = projects[num_train: num_train + num_val]
        elif stage == 'test':
            stage_projects = projects[num_train + num_val: ]

        examples = []
        for project in tqdm(stage_projects):
            json_files = os.listdir(str(self.path_to_dataset / project))
            json_files = [x for x in json_files if x.endswith(".json")]

            for json_file in json_files:
                path_to_file = self.path_to_dataset / project / json_file
                with open(str(path_to_file), 'r') as f:
                    contents = json.load(f)

                file_source_lines = contents['fileSource'].split('\n')
                for method in contents['methods']:
                    try:
                        start_line, end_line = int(method['methodStartLine']), int(method['methodEndLine'])
                        code_lines = file_source_lines[start_line: end_line + 1]
                        code = '\n'.join(code_lines)
                        for slice in method['slices']:
                            variable = slice['variableIdentifier']
                            line_number = int(slice['lineNumber']) - start_line - 1
                            previous_offset = len('\n'.join(file_source_lines[:int(slice['lineNumber']) - 1]))
                            variable_start = int(slice['variableStart']) - previous_offset - 1
                            variable_end = int(slice['variableEnd']) - previous_offset - 1
                            extracted_variable = code_lines[line_number][variable_start: variable_end]
                            if variable != extracted_variable:
                                continue
                            eid = f"{stage}-{project}-{Path(json_file).stem}-{variable}-{line_number}"
                            slice_lines = [x.strip() for x in slice['sliceSource'][2:].split('\n') \
                                           if x.strip() != ""]
                            slice_lines = [line for line in slice_lines if not line.startswith('import')]
                            if 'class' in slice_lines[0]:
                                slice_lines = slice_lines[1:-1]

                            slice_line_numbers = []
                            line_ctr = 0
                            for slice_line in slice_lines:
                                for _line_number, line in enumerate(code_lines[line_ctr:]):
                                    slice_line_without_space = slice_line.replace(' ', '')
                                    line_without_space = line.replace(' ', '')
                                    if slice_line_without_space == line_without_space:
                                        slice_line_numbers.append(line_ctr + _line_number)
                                        line_ctr = line_ctr + _line_number + 1
                                        break

                            if len(slice_line_numbers) != len(slice_lines):
                                continue

                            backward_slice = [x for x in slice_line_numbers if x < line_number]
                            forward_slice = [x for x in slice_line_numbers if x > line_number]

                            backward_is_balanced, forward_is_balanced = False, False
                            backward_ratio = len(backward_slice) / (line_number + 1)
                            forward_ratio = len(forward_slice) / (len(code_lines) - line_number)
                            if 0.3 < backward_ratio < 0.7:
                                backward_is_balanced = True

                            if 0.3 < forward_ratio < 0.7:
                                forward_is_balanced = True

                            if not (backward_is_balanced and forward_is_balanced):
                                continue

                            if len(backward_slice) <= 1 or len(forward_slice) <= 1:
                                continue

                            examples.append(
                                InputExample(
                                    eid=eid,
                                    code=code,
                                    variable=variable,
                                    variable_loc=(variable_start, variable_end),
                                    line_number=line_number,
                                    backward_slice=backward_slice,
                                    forward_slice=forward_slice,
                                )
                            )
                    except Exception: pass

        print(f'Number of examples: {len(examples)}')
        path_to_file = self.path_to_dataset / f"examples_{stage}.pkl"
        self.save(examples, path_to_file)
        return examples

    def load_examples(self, path_to_file):
        '''Load cached examples.

        Arguments:
            stage (str): One of 'train', 'val', 'test'.
            path_to_file (pathlib.Path): Destination to cache examples.
        '''
        with open(str(path_to_file), 'rb') as handler:
            examples = pickle.load(handler)
        return examples

    def save(self, examples, path_to_file):
        '''Cache examples.

        Arguments:
            examples (list): List of ``InputExample`` objects.
            path_to_file (pathlib.Path): Destination to cache examples.
        '''
        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(examples, handler)


class CompleteDataProcessor:
    def __init__(self, json_dir, filename='slices.json', split_keys=('train', 'val', 'test')):
        self.json_dir = json_dir
        self.filename = filename
        self.split_keys = split_keys
        self._cached = {}

    def _collect_split_items(self, split):
        import glob
        split_dir = os.path.join(self.json_dir, split)
        pattern = os.path.join(split_dir, '**', self.filename)
        all_files = glob.glob(pattern, recursive=True)
        logger.info(f"Collecting {split} slice files: {pattern}, total {len(all_files)}")
        all_items = []
        for fpath in all_files:
            with open(fpath, 'r') as fin:
                items = json.load(fin)
                all_items.extend(items)
        logger.info(f"{split}: Merged to obtain {len(all_items)} samples")
        return all_items

    def build_examples(self, items):
        examples = []
        for ex in tqdm(items, desc="Building InputExamples"):
            try:
                variable_loc = tuple(ex['variable_loc']) if 'variable_loc' in ex else (0, 0)
                examples.append(
                    InputExample(
                        eid=ex['eid'],
                        code=ex['code'],
                        variable=ex['variable'],
                        variable_loc=variable_loc,
                        line_number=ex['line_number'],
                        backward_slice=ex.get('backward_slice', ex.get('backward_slices', [])),
                        forward_slice=ex.get('forward_slice', ex.get('forward_slices', [])),
                    )
                )
            except Exception as e:
                logger.warning(f"Skipped={ex.get('eid', 'noid')} - {e}")
        logger.info(f"Generated {len(examples)} InputExamples")
        return examples

    def get_train_examples(self, pct):
        if 'train' not in self._cached or self._cached['train_pct'] != pct:
            items = self._collect_split_items('train')
            if pct < 1.0:
                random.shuffle(items)
                n = int(len(items) * pct)
                items = items[:n]
            self._cached['train'] = self.build_examples(items)
            self._cached['train_pct'] = pct
        return self._cached['train']

    def get_val_examples(self, pct):
        if 'val' not in self._cached or self._cached['val_pct'] != pct:
            items = self._collect_split_items('val')
            if pct < 1.0:
                random.shuffle(items)
                n = int(len(items) * pct)
                items = items[:n]
            self._cached['val'] = self.build_examples(items)
            self._cached['val_pct'] = pct
        return self._cached['val']

    def get_test_examples(self, pct):
        if 'test' not in self._cached or self._cached['test_pct'] != pct:
            items = self._collect_split_items('test')
            if pct < 1.0:
                random.shuffle(items)
                n = int(len(items) * pct)
                items = items[:n]
            self._cached['test'] = self.build_examples(items)
            self._cached['test_pct'] = pct
        return self._cached['test']

class PartialDataProcessor:
    def __init__(self, pct: float, json_dir: str, filename: str = 'slices.json', split_keys=('val', 'test')):
        self.pct = pct
        self.json_dir = json_dir
        self.filename = filename
        self.split_keys = split_keys
        self._complete_proc = CompleteDataProcessor(json_dir, filename, split_keys)
        self._cached = {}

    def build_examples(self, items: List[dict]) -> List[InputExample]:
        examples = []
        for ex in tqdm(items, desc=f"Building Partial(pct={self.pct:.2f}) InputExamples"):
            try:
                code_lines = ex['code'].split('\n')
                num_lines_to_remove = int(self.pct * len(code_lines))
                if num_lines_to_remove == 0:
                    partial_code_lines = code_lines
                else:
                    partial_code_lines = code_lines[num_lines_to_remove: -num_lines_to_remove]
                if not partial_code_lines:
                    continue
                partial_code = '\n'.join(partial_code_lines)
                line_number = ex['line_number'] - num_lines_to_remove
                if line_number < 0 or line_number >= len(partial_code_lines):
                    continue
                variable_start, variable_end = ex['variable_loc']
                if variable_start < 0 or variable_end > len(partial_code_lines[line_number]):
                    continue
                backward_slice = [
                    x - num_lines_to_remove for x in ex.get('backward_slice', ex.get('backward_slices', []))
                    if 0 <= x - num_lines_to_remove < len(partial_code_lines)
                ]
                forward_slice = [
                    x - num_lines_to_remove for x in ex.get('forward_slice', ex.get('forward_slices', []))
                    if 0 <= x - num_lines_to_remove < len(partial_code_lines)
                ]
                if len(backward_slice) <= 1 or len(forward_slice) <= 1:
                    continue
                examples.append(
                    InputExample(
                        eid=ex['eid'],
                        code=partial_code,
                        variable=ex['variable'],
                        variable_loc=(variable_start, variable_end),
                        line_number=line_number,
                        backward_slice=backward_slice,
                        forward_slice=forward_slice,
                    )
                )
            except Exception as e:
                logger.warning(f"【Partial data error】eid={ex.get('eid', 'noid')} - {e}")
        logger.info(f"Partial: Generated {len(examples)} InputExamples")
        return examples

    def get_train_examples(self):
        raise ValueError("PartialDataProcessor does not support train split. Use CompleteDataProcessor instead.")

    def get_val_examples(self, pct=1.0):
        if 'val' not in self._cached or self._cached.get('val_pct', None) != pct:
            complete = self._complete_proc.get_val_examples(pct)
            self._cached['val'] = self.build_examples([ex.__dict__ for ex in complete])
            self._cached['val_pct'] = pct
            logger.info(f"Partial: Cached {len(self._cached['val'])} val examples")
        return self._cached['val']

    def get_test_examples(self, pct=1.0):
        if 'test' not in self._cached or self._cached.get('test_pct', None) != pct:
            complete = self._complete_proc.get_test_examples(pct)
            self._cached['test'] = self.build_examples([ex.__dict__ for ex in complete])
            self._cached['test_pct'] = pct
        return self._cached['test']