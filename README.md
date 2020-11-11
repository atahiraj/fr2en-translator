# French-to-English Translator

## About

This repository contains a sequence-to-sequence neural machine translation
project for translating French text into English. It uses an LSTM
encoder-decoder model, the French-English
[Multi30k dataset](https://github.com/multi30k/dataset), and pretrained
FastText, GloVe, or Word2Vec embeddings. Evaluation reports loss, perplexity,
and BLEU score.

The project was developed for the University of Liège INFO2049-1 course and
targets the 2020-era PyTorch and TorchText APIs captured in `environment.yml`.

## Setup

Create the pinned Conda environment:

```sh
conda env create -f environment.yml
conda activate text-translation
```

Install the English and French models expected by the pinned spaCy release:

```sh
python -m spacy download en
python -m spacy download fr
```

Multi30k and the selected FastText or GloVe vectors are downloaded by
TorchText when the program first runs. Word2Vec mode instead expects these
files in `.vector_cache/`:

```text
.vector_cache/enwiki_20180420_300d.txt
.vector_cache/frwiki_20180420_300d.txt
```

## Usage

Set the experiment options in `config.py`:

- `TEST` selects training (`False`) or evaluation (`True`).
- `SAVE` is the checkpoint written during training.
- `LOAD` is the checkpoint loaded during evaluation.
- `EMB` selects `fasttext`, `glove`, or `word2vec`.
- `BATCH_SIZE`, `N_EPOCHS`, `N_LAYERS`, and `HID_DIM` configure training and
  the model architecture.
- `ENC_DROPOUT`, `DEC_DROPOUT`, and `FREEZE` configure regularization and
  embedding updates.

Run the configured experiment with:

```sh
python main.py
```

The program automatically uses CUDA when it is available and otherwise runs
on the CPU.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the
complete terms.
