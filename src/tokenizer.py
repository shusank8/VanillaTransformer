from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
import pandas as pd
import os

dataset_path =  "/Users/shusanketbasyal/.cache/kagglehub/datasets/jigarpanjiyar/english-to-manipuri-dataset/versions/1"+"//english-nepali.xlsx"

# read the dataset

dfmain = pd.read_excel(dataset_path)

# making sure no nan values:

dfmain['english_sent']  = dfmain["english_sent"].apply(lambda x: str(x))
dfmain['nepali_sent']  = dfmain["nepali_sent"].apply(lambda x: str(x))

# extracting eng and nep and preprocessing 
# as each row is one sentence and less than at most 30 words
# so mergin multiple rows

eng = dfmain['english_sent']
nep = dfmain['nepali_sent']


# a simple iterator to combine multiple rows.
# and make sure the length is less than 

eng_com = []
nep_com = []
eng_lens = []
nep_lens = []
step = 8

for x in range(0, len(eng), step):
    engobj = " ".join(eng[x : x + step])
    nepobj = " ".join(nep[x : x + step])

    englen = len(engobj.split(" "))
    neplen = len(nepobj.split(" "))

    # we will have seq len of 256 for the transformer so making sure the 
    # the length is less than 200
    # later in tokenizer one word will be converted to two. 


    if englen>200 or neplen>200:
        continue

    eng_com.append(engobj)
    eng_lens.append(englen)

    nep_com.append(nepobj)
    nep_lens.append(neplen)

# can visualize the len of each input from eng_lens or nep_lens
    


# build or load tokenizer if exists
    
def build_load_tokenizer(language_name, data):
    if language_name=="eng":
        filename = "engtokenizer.json"
    elif language_name == "nep":
        filename = "neptokenizer.json"
    else:
        return "Language not recognized. {language_name}. Can either be 'eng' or 'nep'"
    

    # does file exists?
    if os.path.isfile(filename):
        # if exists load and return 
        tokenizer = Tokenizer.from_file(filename)
        return tokenizer

    else:
        # file does not exists, build one
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        trainer = WordLevelTrainer(
            special_tokens = [
                "[UNK]",
                '[SOS]',
                '[PAD]',
                '[MASK]',
                '[EOS]'
            ]
        )

        tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
        if language_name=="eng":
            tokenizer.normalizer = Lowercase()
        tokenizer.train_from_iterator(data, trainer=trainer)
        tokenizer.save(filename)
        return tokenizer
