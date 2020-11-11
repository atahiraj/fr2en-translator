
##### 0)
# Baseline
sed -i -e "s/TEST = [^\n#]*/TEST = False/g" \
       -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_10e_500h_4l.pt\"~g" \
       -e "s/EMB = [^\n#]*/EMB = \"w2v\"/g" \
       -e "s/N_EPOCHS = [^\n#]*/N_EPOCHS = 10/g" \
       -e "s/N_LAYERS = [^\n#]*/N_LAYERS = 4/g" \
       -e "s/HID_DIM = [^\n#]*/HID_DIM = 512/g" \
       -e "s/ENC_DROPOUT = [^\n#]*/ENC_DROPOUT = 0/g" \
       -e "s/DEC_DROPOUT = [^\n#]*/DEC_DROPOUT = 0/g" \
       config.py

python3 main.py | tee w2v_10e_500h_4l.txt

# baseline 50 epochs
sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_50e_500h_4l.pt\"~g" \
       -e "s/N_EPOCHS = [^\n#]*/N_EPOCHS = 50/g" \
       config.py

python3 main.py | tee w2v_50e_500h_4l.txt

##### 1
# Glove 50 epochs
sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./glove_50e_500h_4l.pt\"~g" \
       -e "s/EMB = [^\n#]*/EMB = \"glove\"/g" \
       config.py

python3 main.py | tee glove_50e_500h_4l.txt

# Fasttext
sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./fasttext_50e_500h_4l.pt\"~g" \
       -e "s/EMB = [^\n#]*/EMB = \"fasttext\"/g" \
       config.py

python3 main.py | tee fasttext_50e_500h_4l.txt

##### 2
sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_64h.pt\"~g" \
       -e "s/EMB = [^\n#]*/EMB = \"w2v\"/g" \
       -e "s/HID_DIM = [^\n#]*/HID_DIM = 64/g" \
       config.py

python3 main.py | tee w2v_64h.txt

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_128h.pt\"~g" \
       -e "s/HID_DIM = [^\n#]*/HID_DIM = 128/g" \
       config.py

python3 main.py | tee w2v_128h.txt

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_256h.pt\"~g" \
       -e "s/HID_DIM = [^\n#]*/HID_DIM = 256/g" \
       config.py

python3 main.py | tee w2v_256h.txt

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_1024h.pt\"~g" \
       -e "s/HID_DIM = [^\n#]*/HID_DIM = 1024/g" \
       config.py

python3 main.py | tee w2v_1024h.txt

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_2048h.pt\"~g" \
       -e "s/HID_DIM = [^\n#]*/HID_DIM = 2048/g" \
       config.py

python3 main.py | tee w2v_2048h.txt


##### 3
sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_1l.pt\"~g" \
       -e "s/HID_DIM = [^\n#]*/HID_DIM = 512/g" \
       -e "s/N_LAYERS = [^\n#]*/N_LAYERS = 1/g" \
       config.py

python3 main.py | tee w2v_1l.txt

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_2l.pt\"~g" \
       -e "s/N_LAYERS = [^\n#]*/N_LAYERS = 2/g" \
       config.py

python3 main.py | tee w2v_2l.txt

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_8l.pt\"~g" \
       -e "s/N_LAYERS = [^\n#]*/N_LAYERS = 8/g" \
       config.py

python3 main.py | tee w2v_8l.txt

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_16l.pt\"~g" \
       -e "s/N_LAYERS = [^\n#]*/N_LAYERS = 16/g" \
       config.py

python3 main.py | tee w2v_16l.txt


##### 5 Dropout

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_02d.pt\"~g" \
       -e "s/N_LAYERS = [^\n#]*/N_LAYERS = 4/g" \
       -e "s/ENC_DROPOUT = [^\n#]*/ENC_DROPOUT = 0.2/g" \
       -e "s/DEC_DROPOUT = [^\n#]*/DEC_DROPOUT = 0.2/g" \
       config.py

python3 main.py | tee w2v_02d.txt

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_03d.pt\"~g" \
       -e "s/ENC_DROPOUT = [^\n#]*/ENC_DROPOUT = 0.33333/g" \
       -e "s/DEC_DROPOUT = [^\n#]*/DEC_DROPOUT = 0.33333/g" \
       config.py

python3 main.py | tee w2v_03d.txt

sed -i -e "s~SAVE = [^\n#]*~SAVE = \"\./w2v_05d.pt\"~g" \
       -e "s/ENC_DROPOUT = [^\n#]*/ENC_DROPOUT = 0.5/g" \
       -e "s/DEC_DROPOUT = [^\n#]*/DEC_DROPOUT = 0.5/g" \
       config.py

python3 main.py | tee w2v_05d.txt
