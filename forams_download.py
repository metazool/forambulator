import json
import os
import requests, zipfile, io 

START = 'http://endlessforams.org/summary'
DATA = 'http://endlessforams.org/randomizer/download/{0}/{1}?download=capsule.zip'


def download_capsules():
    summary = requests.get(START).json()
    for taxon in summary['results']:
        download_data(taxon['sci_name'], taxon['amount_images'])

def download_data(name, number):
    url = DATA.format(name, number)
    # "results": [{"sci_name": "Beella digitata", "amount_images": 40},
    dir_name = os.path.join(os.getcwd(), 'data', name.replace(' ','_'))
    os.makedirs(dir_name)
    r = requests.get(url, stream=True)

    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dir_name)

if __name__ == '__main__':
    download_capsules()
