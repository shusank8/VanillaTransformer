from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
import pandas as pd
import os


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


def get_raw_data(filepath):

    dfmain = pd.read_excel(filepath)

    # making sure no nan values:

    dfmain['english_sent']  = dfmain["english_sent"].apply(lambda x: str(x))
    dfmain['nepali_sent']  = dfmain["nepali_sent"].apply(lambda x: str(x))
    return [dfmain['english_sent'].tolist(), dfmain['nepali_sent'].tolist()]


def get_train_data(filepath, engtokenizer, neptokenizer, split=False):
     # read the dataset

    dfmain = pd.read_excel(filepath)

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


        if len(engtokenizer.encode(engobj).ids)>240 or len(neptokenizer.encode(nepobj).ids)>200:
            continue

        eng_com.append(engobj)
        eng_lens.append(englen)

        nep_com.append(nepobj)
        nep_lens.append(neplen)
    if split is False:
        return [eng_com, nep_com]
    else:
        trainlen = int(0.9*len(eng_com))
        eng_com_train = eng_com[0:trainlen]
        nep_com_train = nep_com[0:trainlen]

        eng_com_test = eng_com[trainlen:]
        nep_com_test = nep_com[trainlen:]

        train_data = {
            "eng":eng_com_train, "nep":nep_com_train
        }

        test_data = {
            "eng":eng_com_test, "nep":nep_com_test
        }
        df_train = pd.DataFrame(train_data)
        df_test = pd.DataFrame(test_data)
        return [df_train, df_test]

def get_tokenizer(filepath):
        eng_com, nep_com = get_raw_data(filepath)
        engtokenizer = build_load_tokenizer("eng", eng_com)
        neptokenizer = build_load_tokenizer("nep", nep_com)
        return [engtokenizer, neptokenizer]
    
        

        
