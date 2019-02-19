source activate dl
python train.py 2>&1 | tee ../model/log_2.txt
python train_binary.py 2>&1 | tee ../model/log_binary_1.txt
