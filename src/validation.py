import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def validation_run(model, val_dataset, example, batch_size, device, loss_fun, neptokenizer, engtokenizer, epoch):
    model.eval()
    df_val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
    losses = torch.zeros(example)
    df_val_dataloader = iter(df_val_dataloader)

    for _ in range(example):
        data = next(df_val_dataloader)
        encoder_input = data['encoder_input'].to(device)
        decoder_input = data["decoder_input"].to(device)
        label = data['label'].to(device)
        encoder_mask = data['encoder_mask'].to(device)
        decoder_mask = data['decoder_mask'].to(device)
        encoder_output = model.encoder_fun(encoder_input, encoder_mask)
        decoder_output = model.decoder_fun(decoder_input,encoder_output, encoder_mask, decoder_mask )
        logits = model.projection(decoder_output)
        B,T,C = logits.shape
        loss = loss_fun(logits.view(-1, C), label.view(-1))
        losses[_] = loss.item()
    
    # do inference on one example:
    x = next(df_val_dataloader)
    
    # [B, T, C]
    # getting one single batch and adding batch dim to it
    data = {
        "encoder_input" : x["encoder_input"][0].unsqueeze(0),
        "decoder_input" : x["decoder_input"][0].unsqueeze(0),
        "encoder_mask" : x["encoder_mask"][0].unsqueeze(0),
        "decoder_mask" : x["decoder_mask"][0].unsqueeze(0),
        "label" : x["label"][0].unsqueeze(0)
        }

    # generate response
    out = inference(model, data, device, 200, neptokenizer )
    given_english_sentence = engtokenizer.decode(data['encoder_input'][0].tolist())
    expected_output = neptokenizer.decode(data['decoder_input'][0].tolist())
    with open(f"./train_info/epoch", "a") as file:
        file.write(f"validation loss=> {losses.mean().item()}\n")
        file.write(f"Given English Sentence=>  {given_english_sentence}\n" )
        file.write(f"Expected Nepali Sentence=> {expected_output}\n")
        file.write(f"Generated Nepali Output=> {out}\n")
        file.write("---------------------------------------------------------------------")
        
        
    return losses.mean().item()


def inference(model, data, device, maxlength, tgttokenizer):

    # data to be a single example
    model.eval()
    encoder_input = data['encoder_input'].to(device)
    encoder_mask = data['encoder_mask'].to(device)

    encoder_output = model.encoder_fun(encoder_input, encoder_mask)
    eos = tgttokenizer.encode("[EOS]").ids
    generating = torch.tensor(tgttokenizer.encode('[SOS]').ids).unsqueeze(0)
    while maxlength>generating.shape[-1]:
        decoder_output = model.decoder_fun(generating, encoder_output, encoder_mask, None)
        logits = model.projection(decoder_output)
        logits = logits.softmax(dim=-1)
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        nextid = torch.multinomial(logits[-1, :], num_samples=1)
        if nextid.item() == eos:
            break
      
        generating = torch.cat(
            [
                generating,
                nextid.unsqueeze(0)
            ],dim=-1
        )
    
    output = generating[0].tolist()
   
    return tgttokenizer.decode(output)

