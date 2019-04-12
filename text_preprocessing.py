#loading the data 

train=pd.read_csv("~/train.csv")
test=pd.read_csv("~/input/test.csv")

def build_vocab(sentences,verbose=True):
    vocab={}
    
    for sentence in tqdm(sentences,disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

#spliting the text using white space token
sentences = train["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})


def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x
    

train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
test["question_text"] = test["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)


import re
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

train['question_text'] = train['question_text'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))
test['question_text'] = test['question_text'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))
mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
   
   
   
train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
test["question_text"] = test["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)



def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
train['question_text'] = train['question_text'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))
test['question_text'] = test['question_text'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
