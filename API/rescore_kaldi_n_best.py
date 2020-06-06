from API.sentence_wer import get_sentence_wer
from blist import sorteddict

def get_rescore_kaldi_n_best(kaldi_n_best_sentence, kaldi_n_best_acoustics, session, m, fasttext_model, max_word_length, char_vocab, rnn_state):
    rescore_kaldi_n_best = {}

    for sentence, acoustics in zip(kaldi_n_best_sentence, kaldi_n_best_acoustics):
        sentence_wer = get_sentence_wer(sentence, acoustics, session, m, fasttext_model, max_word_length, char_vocab, rnn_state)
        rescore_kaldi_n_best[sentence] = sentence_wer

    return sorteddict(rescore_kaldi_n_best).keys()


