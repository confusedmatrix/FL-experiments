import argparse
import os
import subprocess
import sys
from urllib.parse import urlparse

from src.config import DATA_CACHE_DIR, EXPERIMENT_SETTINGS_FILE_NAME, ROUND_RESULTS_FILE_NAME, FINAL_RESULTS_FILE_NAME
from src.ExperimentSettings import ExperimentSettings, SETTINGS_DEFINITIONS
from src.Experiment import Experiment, CFLExperiment, FCFLExperiment

parser = argparse.ArgumentParser()

# Setup parser rule for each setting definition
for setting in SETTINGS_DEFINITIONS.values():
    default = setting['default'] if 'default' in setting else None
    choices = setting['choices'] if 'choices' in setting else None
    # parser.add_argument(setting['command'][0], setting['command']
    #                     [1], default=default, choices=choices, type=setting['type'])
    parser.add_argument(
        *setting['command'], default=default, choices=choices, type=setting['type'])

args = parser.parse_args()

# Check target accuracy or number of comm rounds is set
assert args.target_accuracy is not None or args.n_rounds is not None or args.weight_delta_norm_threshold is not None, "You must specify target_accuracy, n_rounds or global weight_delta_norm_threshold"

settings = ExperimentSettings(**vars(args))

# If running on GCP, load data from GCP storage
if settings.job_dir is not None:
    parsed = urlparse(settings.job_dir)
    job_dir_parent = f'{parsed.scheme}://{parsed.netloc}'
    os.mkdir(DATA_CACHE_DIR)
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', os.path.join(
        job_dir_parent, "data/*"), DATA_CACHE_DIR], stderr=sys.stdout)

# Select the correct experiment type
if settings.algorithm == 'CFL':
    experiment = CFLExperiment(settings)
elif settings.algorithm == 'FCFL':
    experiment = FCFLExperiment(settings)
else:
    experiment = Experiment(settings)

experiment.run()

# If running on GCP, save results files to GCP bucket
if settings.job_dir is not None:
    subprocess.check_call(['gsutil', 'cp', EXPERIMENT_SETTINGS_FILE_NAME, os.path.join(
        settings.job_dir, EXPERIMENT_SETTINGS_FILE_NAME)], stderr=sys.stdout)
    subprocess.check_call(['gsutil', 'cp', ROUND_RESULTS_FILE_NAME, os.path.join(
        settings.job_dir, ROUND_RESULTS_FILE_NAME)], stderr=sys.stdout)
    subprocess.check_call(['gsutil', 'cp', FINAL_RESULTS_FILE_NAME, os.path.join(
        settings.job_dir, FINAL_RESULTS_FILE_NAME)], stderr=sys.stdout)
