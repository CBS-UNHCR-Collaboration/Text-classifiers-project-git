from os import path


def LoadStopwords(filepath):
    """
    This functions returns a set of stopwords if the file path exits.
    Otherwise, it returns an empty set.
    :param filepath:
    :return:
    """

    if not path.exists(filepath):
        return set()

    stop_words_reader = open(filepath, 'rU',
                             encoding='utf8', errors="ignore", newline=None)

    stop_word_list = stop_words_reader.readlines()

    words = [word.strip().lower() for word in stop_word_list]

    return set(words)