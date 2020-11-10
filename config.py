# Mode
TEST = False

# Paths
SAVE = "./weights.pt" # Where to save weights for training
LOAD = "./weights.pt" # Weights to load for testing

# Training config
BATCH_SIZE = 128
N_EPOCHS = 10

# Archtecture config
ATTN = True # include attention mechanism or not
ENC_NUM_LAYERS = 2
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ATTN_DIM = 64
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
