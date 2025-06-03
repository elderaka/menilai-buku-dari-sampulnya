#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import codecs
import pandas as pd
from argparse import ArgumentParser
from urllib import request
from tqdm import trange
from joblib import Parallel, delayed

parser = ArgumentParser()
parser.add_argument(
    'output_dirpath',
    type=str,
    help='output directory path'
)
parser.add_argument(
    'csv_filepath',
    type=str,
    help='csv filepath'
)
args = parser.parse_args()

header_names = ['Amazon ID (ASIN)', 'Filename', 'Image URL', 'Title', 'Author', 'Category ID',
                'Category']
with codecs.open(args.csv_filepath, mode='r', encoding='utf-8', errors='ignore') as f:
    csv = pd.read_csv(f, delimiter=",", header=None, names=header_names)

if not os.path.isdir(args.output_dirpath):
    os.makedirs(args.output_dirpath)

print('[Download images into "{}"]'.format(args.output_dirpath))

def needs_download(i):
    filename = csv.iloc[i]['Filename']
    category = csv.iloc[i]['Category']
    inner_output_dirpath = os.path.join(args.output_dirpath, category)
    output_filepath = os.path.join(inner_output_dirpath, filename)
    return not os.path.isfile(output_filepath)

indices_to_download = [i for i in range(len(csv)) if needs_download(i)]

print(f"Found {len(indices_to_download)} images to download (out of {len(csv)})")

def download_image(i):
    try:
        filename = csv.iloc[i]['Filename']
        category = csv.iloc[i]['Category']
        inner_output_dirpath = os.path.join(args.output_dirpath, category)
        os.makedirs(inner_output_dirpath, exist_ok=True)
        output_filepath = os.path.join(inner_output_dirpath, filename)

        url = csv.iloc[i]['Image URL']
        downloaded_img = request.urlopen(url)
        with open(output_filepath, mode='wb') as f:
            f.write(downloaded_img.read())
        downloaded_img.close()
    except Exception as e:
        print(f"Failed to download index {i}: {e}")

Parallel(n_jobs=4)(delayed(download_image)(i) for i in trange(len(indices_to_download)))