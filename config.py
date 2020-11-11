"""
INFO2049-1: Practical project: Machine translation 1
Authors: Folon Nora, Tahiraj Arian
Parts of the code are inspired from:
    - Title: Torchtext translation tutorial
      Authors: Seith Weidman
      Availability: https://github.com/pytorch/tutorials/blob/master/beginner_source/torchtext_translation_tutorial.py
    - Title: Pytorch Seq2Seq
      Authors: Ben Trevett
      Availability: https://github.com/bentrevett/pytorch-seq2seq
"""

# Mode, test or train
TEST = False

# Paths
SAVE = "./weights.pt"
LOAD = "./weights.pt"  # Weights to load for testing

# Embedding type
EMB = "fasttext"  # glove, fasttext or word2vec (case insensitive)

# Training config
BATCH_SIZE = 128
N_EPOCHS = 50

# Archtecture config
N_LAYERS = 2
HID_DIM = 256
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

# Freeze embedding layers
FREEZE = False
