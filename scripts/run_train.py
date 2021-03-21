import argparse
import sys
import subprocess

from azureml.core import Run, Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='forams_128')

    args = parser.parse_args()

    # Trappings needed to reuse an Azure ML workspace dataset
    with Run.get_context() as run:
        try:
            workspace = run.experiment.workspace
        except AttributeError:
            sys.exit()

        datastore = workspace.get_default_datastore()
        dataset = Dataset.File.from_files(path=(datastore, args.dataset))
        ds_mount = dataset.as_mount(args.dataset)
        # May or may not work
        subprocess.run(["stylegan2-pytorch",  "--data", ds_mount, "--aug-prob", 0.25])
