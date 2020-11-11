# Mode, test or train
TEST = False

# Paths
SAVE = "./test.pt"
LOAD = "./xxx.pt"  # Weights to load for testing

# Embedding type
EMB = "w2v"
EMB_SIZE = 200 # For GloVe

# Training config
BATCH_SIZE = 128
N_EPOCHS = 10

# Archtecture config
N_LAYERS = 4
HID_DIM = 512
ENC_DROPOUT = 0
DEC_DROPOUT = 0
