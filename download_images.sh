set -eu

pip3 install -U pandas tqdm joblib

OUTPUT_DIRPATH="images"
CSV_FILEPATH="book32-listing.csv"
python3 download_images.py ${OUTPUT_DIRPATH} ${CSV_FILEPATH}