import argparse
from forambulator.download import download_capsules

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='./outputs/data')
    args = parser.parse_args()
    download_capsules(directory=args.directory)
