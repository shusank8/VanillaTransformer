from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):

    def __init__(self, df, src_tokenizer, tgt_tokenizer, src, tgt, seq_len):
        super().__init__()

        self.df = df
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src = src
        self.tgt = tgt
        self.seq_len = seq_len
        # SOS TOKEN
        self.sos_token = self.src_tokenizer.encode("[SOS]").ids
        # EOS TOKEN
        self.eos_token = self.src_tokenizer.encode('[EOS]').ids
        # PAD TOKEN
        self.pad_token = self.src_tokenizer.encode('[PAD]').ids


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        # getting single instance
        src_tgt_pair = self.df.iloc[index]
        # gettin src text
        src_text = src_tgt_pair[self.src]
        # getting tgt text
        tgt_text = src_tgt_pair[self.tgt]

        # use tokenizer to tokenize src and tgt
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        tgt_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        # calculate paddings
        # subtracting 2 beacuse later we will add SOS AND EOS
        enc_padding_len = self.seq_len - len(enc_input_tokens) - 2

        dec_padding_len = self.seq_len - len(tgt_input_tokens) - 1

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens),
                self.eos_token,
                torch.tensor(self.pad_token * enc_padding_len)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_input_tokens),
                torch.tensor(self.pad_token, dec_padding_len)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(tgt_input_tokens),
                self.eos_token,
                torch.tensor(self.pad_token * dec_padding_len)
            ]
        )

        encoder_mask = (encoder_input!=self.pad_token).int()
        decoder_mask = (decoder_input!=self.pad_token).int()
        decoder_mask = decoder_mask & causal_mask(decoder_input.shape[0])

        return {
            "encoder_input":encoder_input,
            "decoder_input":decoder_input, 
            "label":label,
            "encoder_mask":encoder_mask,
            "decoder_mask":decoder_mask
        }
    
    def causal_mask(seqlen):
        x = torch.ones(seqlen, seqlen)
        x = torch.tril(x)
        return x
        



