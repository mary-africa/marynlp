from typing import Iterator


def summarize_ner_data(ner_data_gen: Iterator[tuple]):
    """
    This function is supposed to create a summary object (pd.DataFrame)
    containing summary information about the data

    This accounted for:
    - **corpus_data_summary
    - number of [TAG] in BIO format and normal format
    """
    # contain summary info about tag
    tag_info = {}

    # contain general summary info
    summary_info = {}

    # store the words as sentences
    texts = []

    # initiate sentence
    text = []

    for word_tag_val in ner_data_gen:
        if word_tag_val is None:
            # add the new word
            texts.append(text)

            # reset the text
            text = []
        else:
            # deal as word tag
            word, tag = word_tag_val

            # add to text set
            text.append(word)
            if tag not in tag_info:
                # init it
                tag_info[tag] = 0

            tag_info[tag] = tag_info[tag] + 1

    # determine the max number of lines
    line_count = len(texts)
    word_counts_per_text = list(map(len, texts))

    max_word_count = max(word_counts_per_text)
    min_word_count = min(word_counts_per_text)
    mean_word_count = sum(word_counts_per_text) // line_count

    summary_info.update({"tag_info": tag_info,
                         "counts": {
                             "lines": line_count,
                             "avg_word_count": mean_word_count,
                             "min_word_count": min_word_count,
                             "max_word_count": max_word_count
                         }})

    return summary_info