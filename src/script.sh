source activate dl
python train.py 2>&1 | tee ../model/log.txt
python train_binary.py 2>&1 | tee ../model/log_binary.tx
