import csv
import subprocess

from src.ExperimentSettings import ExperimentSettings

EXP_SETTINGS_FILE = 'experiments.csv'

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
            
            # Run experiment in a subprocess to avoid state leakage
            print(f"\nRunning experiment with settings id #{exp_settings.id} as subprocess") # TODO add ID as first key in settings
            subprocess.run(['python', 'run-experiment.py'] + args, check=True)