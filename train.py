import math
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from model import Seq2Seq
import config


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    SRC = Field(tokenize = "spacy",
                tokenizer_language="fr",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    TRG = Field(tokenize = "spacy",
                tokenizer_language="en",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.fr', '.en'),
                                                        fields = (SRC, TRG))

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = config.BATCH_SIZE,
        device = device)



    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    model = Seq2Seq(
        input_dim = INPUT_DIM,
        enc_emb_dim = config.ENC_EMB_DIM,
        enc_hid_dim = config.ENC_HID_DIM,
        enc_dropout = config.ENC_DROPOUT,
        attn_dim = config.ATTN_DIM,
        output_dim = OUTPUT_DIM,
        dec_emb_dim = config.DEC_EMB_DIM,
        dec_hid_dim = config.DEC_HID_DIM,
        dec_dropout = config.DEC_DROPOUT,
        device=device
    ).to(device)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())


    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'The model has {count_parameters(model):,} trainable parameters')

    PAD_IDX = TRG.vocab.stoi['<pad>']

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def train(model: nn.Module,
            iterator: BucketIterator,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            clip: float):

        model.train()

        epoch_loss = 0

        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            output = model(src, trg)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)


    def evaluate(model: nn.Module,
                iterator: BucketIterator,
                criterion: nn.Module):

        model.eval()

        epoch_loss = 0

        with torch.no_grad():

            for _, batch in enumerate(iterator):

                src = batch.src
                trg = batch.trg

                output = model(src, trg, 0) #turn off teacher forcing

                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)


    def epoch_time(start_time: int,
                end_time: int):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(config.N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
