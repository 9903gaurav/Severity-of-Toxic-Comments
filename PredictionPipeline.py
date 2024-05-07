import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import re

file = open('tokenizer.pickle', 'rb')
tokenizer = pickle.load(file)
model = load_model('fast_text_model')


def predict(text):
    # clean text
    comments_text = []

    for comments in text:
        comment_new = clean_text(comments)
        comments_text.append(comment_new)

    # get predictions
    predictions = []
    
    for comment in comments_text:
        # index = comments_text.index(comment)
        sequence = tokenizer.texts_to_sequences([comment])
        padded = keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=170)
        prediction = model.predict(padded)
        append_string = 'Postive' if prediction[0][0] > 0.5 else 'Negative'
        predictions.append({
            "processed_comment": comment,
            "prediction": append_string})
    return predictions
    # return Response(json.dumps(predictions), mimetype='application/json')


def clean_text(text, remove_repeat_text=True, remove_patterns_text=True, is_lower=True):

    # extract hashtags
    # hashtags = re.findall(r'#\w+', text)

    if is_lower:
        text = text.lower()

    if remove_patterns_text:
        for target, patterns in RE_PATTERNS.items():
            for pat in patterns:
                text = str(text).replace(pat, target)

    if remove_repeat_text:
        text = re.sub(r'(.)\1{2,}', r'\1', text)

    text = re.sub(r'#\w+', '', text)
    text = re.sub('@[^\s]+', '', text)
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
    text = str(text).replace("\n", " ")
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub('[0-9]', "", text)
    text = re.sub(" +", " ", text)
    text = re.sub("([^\x00-\x7F])+", " ", text)

    return text


# Preprocess patterns:
RE_PATTERNS = {

    " are not ": ["aren't"],
    " cannot ": ["can't"],
    " cannot have ": ["can't've"],
    " because ": ["cause"],
    " could have ": ["could've"],
    " could not ": ["couldn't"],
    " could not have ": ["couldn't've"],
    " did not ": ["didn't"],
    " does not ": ["doesn't"],
    " do not ": ["don't"],
    " had not ": ["hadn't"],
    " had not have ": ["hadn't've"],
    " has not ": ["hasn't"],
    " have not ": ["haven't"],
    " he would ": ["he'd"],
    " he would have ": ["he'd've"],
    " he will ": ["he'll"],
    " he is ": ["he's"],
    " how did ": ["how'd"],
    " how will ": ["how'll"],
    " how is ": ["how's"],
    " i would ": ["i'd"],
    " i will ": ["i'll"],
    " i am ": ["i'm"],
    " i have ": ["i've"],
    " is not ": ["isn't"],
    " it would ": ["it'd"],
    " it will ": ["it'll"],
    " it is ": ["it's"],
    " let us ": ["let's"],
    " madam ": ["ma'am"],
    " may not ": ["mayn't"],
    " might have ": ["might've"],
    " might not ": ["mightn't"],
    " must have ": ["must've"],
    " must not ": ["mustn't"],
    " need not ": ["needn't"],
    " ought not ": ["oughtn't"],
    " shall not ": ["shan't","sha'n't"],
    " she would ": ["she'd"],
    " she will ": ["she'll"],
    " she is ": ["she's"],
    " should have ": ["should've"],
    " should not ": ["shouldn't"],
    " that would ": ["that'd"],
    " that is ": ["that's"],
    " there had ": ["there'd"],
    " there is ": ["there's"],
    " they would ": ["they'd"],
    " they will ": ["they'll"],
    " they are ": ["they're"],
    " they have ": ["they've"],
    " was not ": ["wasn't"],
    " we would ": ["we'd"],
    " we will ": ["we'll"],
    " we are ": ["we're"],
    " we have ": ["we've"],
    " were not ": ["weren't"],
    " what will ": ["what'll"],
    " what are ": ["what're"],
    " what is ": ["what's"],
    " what have ": ["what've"],
    " where did ": ["where'd"],
    " where is ": ["where's"],
    " who will ": ["who'll"],
    " who is ": ["who's"],
    " will not ": ["won't"],
    " would not ": ["wouldn't"],
    " you would ": ["you'd"],
    " you will ": ["you'll"],
    " you are ": ["you're"],
    ' american ':
        [
            'amerikan'
    ],

    ' adolf ':
        [
            'adolf'
    ],


    ' hitler ':
        [
            'hitler'
    ],

    ' fuck':
        [
            '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
            ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',
            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',
            'feck ', ' fux ', 'f\*\*',
            'f\-ing', 'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck', 'fuk', 'wtf', 'fucck', 'f cking'
    ],

    ' ass ':
        [
            '[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\$\$'
                                                           '[^a-z]anus', ' a\*s\*s', '[^a-z]ass[^a-z ]',
            'a[@#\$%\^&\*][@#\$%\^&\*]', '[^a-z]anal ', 'a s s'
    ],

    ' asshole ':
        [
            ' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole', 'ass hole'
    ],

    ' bitch ':
        [
            'b[w]*i[t]*ch', 'b!tch',
            'bi\+ch', 'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',
            'biatch', 'bi\*\*h', 'bytch', 'b i t c h', 'beetch'
    ],

    ' bastard ':
        [
            'ba[s|z]+t[e|a]+rd'
    ],

    ' transgender':
        [
            'transgender', 'trans gender'
    ],

    ' gay ':
        [
            'gay'
    ],

    ' cock ':
        [
            '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
            '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'
    ],

    ' dick ':
        [
            ' dick[^aeiou]', 'deek', 'd i c k', 'diick '
    ],

    ' suck ':
        [
            'sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'
    ],

    ' cunt ':
        [
            'cunt', 'c u n t'
    ],

    ' bullshit ':
        [
            'bullsh\*t', 'bull\$hit', 'bs'
    ],

    ' homosexual':
        [
            'homo sexual', 'homosex'
    ],

    ' jerk ':
        [
            'jerk'
    ],

    ' idiot ':
        [
            'i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots', 'i d i o t'
    ],

    ' dumb ':
        [
            '(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'
    ],

    ' shit ':
        [
            'shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\$hit', 's h i t'
    ],

    ' shithole ':
        [
            'shythole', 'shit hole'
    ],

    ' retard ':
        [
            'returd', 'retad', 'retard', 'wiktard', 'wikitud'
    ],

    ' rape ':
        [
            ' raped'
    ],

    ' dumbass':
        [
            'dumb ass', 'dubass'
    ],

    ' asshead':
        [
            'butthead', 'ass head'
    ],

    ' sex ':
        [
            's3x', 'sexuality',
    ],


    ' nigger ':
        [
            'nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'
    ],

    ' shut the fuck up':
        [
            'stfu'
    ],

    ' pussy ':
        [
            'pussy[^c]', 'pusy', 'pussi[^l]', 'pusses'
    ],

    ' faggot ':
        [
            'faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
            '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot',
    ],

    ' motherfucker':
        [
            ' motha ', ' motha f', ' mother f', 'motherucker', 'mother fucker'
    ],

    ' whore ':
        [
            'wh\*\*\*', 'w h o r e'
    ],
    ' though ': ['tho'],
    # ' picture ': ['pic', 'pics'],
    ' soo ': ['so'],
    ' should ': ['shoulda'],
    " aint ": ["am not"],
}
