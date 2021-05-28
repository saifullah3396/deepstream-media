from utils import CharacterTable, transform
from utils import restore_model, decode_sequences
from utils import read_text, tokenize
from math import log

def load_books():
    path = './data'
    books = ['nietzsche.txt', 'pride_and_prejudice.txt', 'shakespeare.txt', 'war_and_peace.txt']
    text  = read_text(path, books)
    vocab = tokenize(text)
    vocab = list(filter(None, set(vocab)))

    return vocab

def load_words():

    path = './data'
    books = ['nietzsche.txt', 'pride_and_prejudice.txt', 'shakespeare.txt', 'war_and_peace.txt']
    text  = read_text(path, books)
    vocab = tokenize(text)
    words = list(filter(None, set(vocab)))
    wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
    maxword = max(len(x) for x in words)

    return wordcost, maxword

def character_tables(vocab):

    error_rate = 0.6

    maxlen = max([len(token) for token in vocab]) + 2
    train_encoder, train_decoder, _ = transform(
        vocab, maxlen, error_rate=error_rate, shuffle=False)
    
    input_chars = set(' '.join(train_encoder))
    target_chars = set(' '.join(train_decoder))
    input_ctable = CharacterTable(input_chars)
    target_ctable = CharacterTable(target_chars)

    return input_ctable, target_ctable, maxlen

def encoder_decoder():
    model_path = 'Models/seq2seq.h5'
    hidden_size = 512

    encoder_model, decoder_model = restore_model(model_path, hidden_size)

    return encoder_model, decoder_model

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).

    wordcost, maxword = load_words()
   

    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))

def spell_check(test_text, encoder_model, decoder_model, input_ctable, target_ctable, maxlen):

    error_rate = 0.6

    tokens = tokenize(test_text)
    tokens = list(filter(None, tokens))
    nb_tokens = len(tokens)
    misspelled_tokens, _, target_tokens = transform(
        tokens, maxlen, error_rate=error_rate, shuffle=False)
    
    _, target_tokens, _ = decode_sequences(
        inputs=misspelled_tokens, 
        targets=target_tokens, 
        input_ctable=input_ctable, 
        target_ctable=target_ctable,
        maxlen=maxlen, 
        reverse=True, 
        encoder_model=encoder_model, 
        decoder_model=decoder_model, 
        nb_examples=nb_tokens,
        sample_mode='argmax', random=False)
    
    return target_tokens