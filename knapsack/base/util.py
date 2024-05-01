import numpy as np
import re
import time
import typing as t
from os import getenv
from os.path import expanduser
from pathlib import Path

import pandas as pd
import tomllib


class Timer(object):
    def __init__(self) -> None:
        self.timers = {}
        self._time_starts = {}
        self._time_ends = {}

    def start(self, key: str) -> None:
        if key not in self.timers:
            self.timers[key] = 0
        self._time_starts[key] = time.perf_counter()

    def end(self, key: str) -> None:
        end = time.perf_counter()
        if key in self._time_starts:
            self.timers[key] += end - self._time_starts[key]
            del self._time_starts[key]

    def get(self, key: str) -> float:
        return self.timers[key]

    def __str__(self) -> str:
        return str(self.timers)

"""
class ResourceTracker(object):
    def __init__(self) -> None:
        self.process_mem = {} 
        self.peak_mem = {}
        self.vmem = {} 
        self.cpu = {} 
        self.cpu_avg = {} 
        # self.gpu = {}
        # self.gpu_temp = {}

    def record(self) -> None:
        t = time.perf_counter()
        process = psutil.Process()
        self.process_mem[t] = process.memory_info().rss
        self.vmem[t] = psutil.virtual_memory()
        self.peak_mem[t] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.cpu[t] = psutil.cpu_percent(4, percpu=False)
        self.cpu_avg[t] = psutil.getloadavg()

    def _pprint_ntuple(self, nt):
        for name in nt._fields:
            value = getattr(nt, name)
            if name != 'percent':
                value = bytes2human(value)
            print('%-10s : %7s' % (name.capitalize(), value))

    def print(self):
        # TODO: output to file? 
        printer = pp.PrettyPrinter(indent=2, width=100)

        print("PROCESS MEMORY\n---------")
        # self._pprint_ntuple(self.process_mem)
        printer.pprint(self.process_mem)
        print("\n")

        print("VIRTUAL MEMORY\n---------")
        # self._pprint_ntuple(self.vmem)
        printer.pprint(self.vmem)
        print("\n")

        print("PEAK MEMORY\n---------")
        # self._pprint_ntuple(self.vmem)
        printer.pprint(self.peak_mem)
        print("\n")

        print("CPU\n---------")
        printer.pprint(self.cpu)
        print("\n")

        print("CPU AVG\n---------")
        printer.pprint(self.cpu_avg)
        print("\n")

    def __dict__(self) -> dict:
        return {
            "proc_mem": self.process_mem,
            "virtual_mem": self.vmem,
            "peak_mem": self.peak_mem,
            "cpu": self.cpu,
            "cpu_avgload": self.cpu_avg,
        }
"""
    

def df_merge(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    align_column: str,
    cols: list[str | None],
):
    if len(cols) == 0:
        return df1
    merged_df = pd.merge(df1, df2[[align_column] + cols], on=align_column, suffixes=('', '_df2'))
    print(f"merged cols: {merged_df.columns}")
    for col in cols:
        if col not in merged_df:
            merged_df[col] = merged_df[f'{col}_df2']
    merged_df.drop(columns=[f'{col}_df2' for col in cols if f'{col}_df2' in merged_df], inplace=True)
    return merged_df


def df_diff(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    col: str
):
    """
    Compare two dataframes on a specific column, 'col', and return rows from the second dataframe
    that differ in that column from the first dataframe.
    
    :return: DataFrame with rows from df2 that differ in column 'col'from df1
    """
    if col not in df1.columns or col not in df2.columns:
        raise ValueError(f"Comparison column '{col}' not found in both dataframes.")
    diff_values = set(df2[col]) - set(df1[col])
    return df2[df2[col].isin(diff_values)]


def hash_row(row, cols, timer: Timer):

    # timer.start('256')
    # hash256 = hashlib.sha256(str(row.values, ).encode(), usedforsecurity=False).hexdigest()
    # timer.end('256')

    # timer.start('1')
    # hash1 = hashlib.sha1(str(row.values).encode(), usedforsecurity=False).hexdigest()
    # timer.end('1')

    vals = ""
    for col in cols:
        vals += str(row[col])
    # timer.start('builtin')
    hash_builtin = str(hash(str(vals)))
    # timer.end('builtin')
    
    return hash_builtin


def _validate_model_cfg(model_cfg: dict[str, t.Any]):
    if 'sentence_transformers' in model_cfg:
        st = model_cfg['sentence_transformers']
        assert 'type' in st
        assert isinstance(st['type'], str)
        assert 'model_id' in st
        assert isinstance(st['model_id'], str)
    if 'llama.cpp' in model_cfg:
        cpp = model_cfg['llama.cpp']
        assert 'n_gpu_layers' in cpp
        assert isinstance(cpp['n_gpu_layers'], int)


def _validate_cluster_cfg(cluster_cfg: dict[str, t.Any]):
    pass 


def procure_cfg() -> dict[str, t.Any]:
    env_ks_toml = getenv("KNAPSACK_CONFIG", None)
    default_ks_toml = Path(expanduser("~/.knapsack.toml"))
    
    toml_location = env_ks_toml
    if env_ks_toml is None and default_ks_toml.exists():
        toml_location = default_ks_toml
    else:
        raise ValueError("Knapsack cannot find init.toml")

    with open(toml_location) as fileObj:
       content = fileObj.read()
       cfg = tomllib.loads(content)
       _validate_model_cfg(cfg.get('model', {}))
       return cfg

def is_valid_uuid(uuid: str) -> bool:
    simple_regex = r"^[0-9a-fA-F]{32}$"
    hyphenated_regex = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    urn_regex = r"^urn:uuid:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

    if (re.match(simple_regex, uuid) or 
        re.match(hyphenated_regex, uuid) or 
        re.match(urn_regex, uuid)):
        return True
    else:
        return False


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)
