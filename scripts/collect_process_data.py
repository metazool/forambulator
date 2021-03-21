import argparse
from forambulator.images import list_image_filenames, crop_foram, NoForamFound
from forambulator.download import download_capsules


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='./outputs/data')
    parser.add_argument('--processed', default='./outputs/processed')
    parser.add_argument('--limit', default=0, type=int)

    args = parser.parse_args()
    download_capsules(directory=args.directory, limit=args.limit)

    images = list_image_filenames(args.directory)
    for image in images:
        try:
            crop_foram(image, directory=args.processed)
        except NoForamFound:
            pass
