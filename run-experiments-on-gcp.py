import csv
from datetime import datetime
import subprocess

from src.config import EXPERIMENT_SUITE_FILE_NAME
from src.ExperimentSettings import ExperimentSettings

EXP_SETTINGS_FILE = EXPERIMENT_SUITE_FILE_NAME
EXP_SUITE_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Run an experiment for each row in the CSV file
with open(EXP_SETTINGS_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    for k, row in enumerate(reader):
        if k == 0:
            headers = row
        else:
            # Skip empty rows
            if len(row) == 0:
                continue

            row = (None if val == '' else val for val in row) # convert empty string values to None type
            settings_dict = dict(zip(headers, row))
            exp_settings = ExperimentSettings(**settings_dict)
            args = exp_settings.get_settings_as_command_line_arg_list()
            
             
            REGION = 'us-west1'
            PROJECT_ID = 'phd-experiments-248213'
            IMAGE_REPO_NAME = 'federated_learning'
            IMAGE_TAG = 'pytorch_gpu'
            IMAGE_URI = f'gcr.io/{PROJECT_ID}/{IMAGE_REPO_NAME}:{IMAGE_TAG}'
            BUCKET_NAME=f'{PROJECT_ID}-storage2'
            JOB_NAME = f'suite_{EXP_SUITE_TIMESTAMP}_{exp_settings.id}'
            JOB_DIR = f'suite_{EXP_SUITE_TIMESTAMP}/{exp_settings.id}'
            subprocess.run(['gcloud', 'ai-platform', 'jobs', 'submit', 'training', JOB_NAME,
	                        '--scale-tier', 'BASIC_GPU',
	                        '--region', REGION,
	                        '--master-image-uri', IMAGE_URI,
	                        f'--job-dir=gs://{BUCKET_NAME}/experiments/{JOB_DIR}',
	                        '--'] + args, check=True)