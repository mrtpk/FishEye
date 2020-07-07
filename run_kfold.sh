echo "Running K-Fold"

python /home/tpk/workspace/visum_project/tmp/convert_gt.py --kfold-idx 0
python train.py

python /home/tpk/workspace/visum_project/tmp/convert_gt.py --kfold-idx 1
python train.py
