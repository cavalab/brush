"""
Brush
copyright 2022 William La Cava and Joseph D. Romano
license: GNU/GPL v3

file: io.py

I/O routines and data structures for Brush, especially in loading OMOP-formatted
observational data.
"""

from dataclasses import dataclass
from typing import List
import datetime as dt
from glob import glob
import os

import numpy as np
import pandas as pd

import ipdb

@dataclass
class BrushData:
    """Class for storing input data to Brush GP."""
    dset_name:      str
    dset_type:      str
    data:           np.array
    created_at:     dt.datetime = dt.datetime.now()

@dataclass
class BrushOmopData(BrushData):
    """A Brush dataset from observational data following the OMOP CDM."""
    cdm_version:        str = 'v5.0'
    time_values:        str = 'timestamp'
    string_features:    str = 'onehot'

@dataclass
class Feature():
    """Information about a single feature present in an OMOP dataset."""
    concept_id: str
    dtype: str

class OmopFeatureMap():
    """Keeps track of features present in an OMOP dataset."""
    PERSON:                 List[Feature]
    DRUG_EXPOSURE:          List[Feature]
    MEASUREMENT:            List[Feature]
    PROCEDURE_OCCURRENCE:   List[Feature]


good_filenames = [
    'person.csv',
    'drug_exposure.csv',
    'measurement.csv',
    'procedure_occurrence.csv'
]

def bare_filename(full_path):
    return os.path.basename(full_path).split('.')[0].lower()


def load_omop_from_directory(dir):
    files_in_dir = glob(dir+"/*")
    
    try:
        assert len(files_in_dir) > 0
    except:
        raise AssertionError("Error - the directory provided was empty")

    # We'll only consider CSV files for now
    valid_files = [x for x in files_in_dir if x.endswith(".csv")]

    try:
        assert len(valid_files)
    except:
        raise AssertionError("Error - no CSV files are present in the directory")

    # Create a spot for each table and fill them in if they exist
    PERSON = None
    DRUG_EXPOSURE = None
    MEASUREMENT = None
    PROCEDURE_OCCURRENCE = None

    for f in valid_files:
        if bare_filename(f) == 'person':
            PERSON = pd.read_csv(f, sep="\t")
        elif bare_filename(f) == 'drug_exposure':
            DRUG_EXPOSURE = pd.read_csv(f)
        elif bare_filename(f) == 'measurement':
            MEASUREMENT = pd.read_csv(f)
        elif bare_filename(f) == 'procedure_occurrence':
            PROCEDURE_OCCURRENCE = pd.read_csv(f)

    ipdb.set_trace()


def load_omop_from_json(path):
    raise NotImplementedError

def load_omop_from_sql(hostname, auth):
    raise NotImplementedError

if __name__ == "__main__":
    example_dir = os.path.join("..", "..", "data", "OMOP")
    # print(glob(example_dir+"/*"))

    load_omop_from_directory(example_dir)