cd ../lib
python create_dataset.py
cd ../src
python train-segmentation.py
python test-all-segmetnation.py
python detect_watershed.py
