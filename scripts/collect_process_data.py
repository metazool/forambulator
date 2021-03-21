import argparse
import sys

from azureml.core import Run, Dataset

from forambulator.images import list_image_filenames, crop_foram, NoForamFound
from forambulator.download import download_capsules


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='./outputs/data')
    parser.add_argument('--processed', default='./outputs/processed')
    parser.add_argument('--dataset', default='forams_128')
    parser.add_argument('--limit', default=0, type=int)

    args = parser.parse_args()
    download_capsules(directory=args.directory, limit=args.limit)

    images = list_image_filenames(args.directory)
    for image in images:
        try:
            crop_foram(image, directory=args.processed)
        except NoForamFound:
            pass

    # Trappings needed to upload to an Azure ML workspace if within one
    with Run.get_context() as run:
        try:
            workspace = run.experiment.workspace
        except:
            sys.exit()
        # get the datastore to upload prepared data
        datastore = workspace.get_default_datastore()

        # upload the local file from src_dir to the target_path in datastore
        datastore.upload(src_dir=args.processed, target_path='forams_128')

        datastore_paths = [(datastore, args.dataset)]
        forams_ds = Dataset.File.from_files(path=datastore_paths)
        forams_ds.register(workspace=workspace,
                           name=args.dataset)
