"""
Categorical placement of the models related data.
Such data can be model weights, text / yaml files. Any thing.. (all of which are in a ZIP file)

TODO: Make sure the code is NOT case INsensitive
"""

# Embeddings related models
embeddings_gcp = {
    'sw-clean-fasttext': 'models/embeddings/sw-clean-ft-dim100-minn2-maxn5-epoch50-lr0.1.zip',    
}

# Language models
lm_gcp = {
    
}

# All flair related data
#  This is to accomodate on the fact that flair has a unique way of storing their weights
#  so for any related models that use
flair_gcp = {
    'sw-exp-sent_analy-small': 'flair/classifier/sw-exp-sent_analy-small.zip',
    'early-wpc': 'flair/classifier/exp-wpc-small.zip',
    'early-sentiment-hasf': 'flair/classifier/sw-ft100-ffw-bilstm-exp-sent_analy-small-h256-noreproj.zip',
    'early-alpha-tag-ner': 'flair/taggers/sw-ner-gen1f-base.zip',
    'early-alpha-tag-pos': 'flair/taggers/sw-pos-early-h256.zip',
}

voice_gcp = {
    'mnm-pre-early-beta-15k-ep20-bc128': 'voice/mnm-pre-early-beta-15k-ep20-bc128.zip'
}

# Miscellaneous models.
#  Models that don't have a proper catergory to fit them
misc_models_gcp = {
    'early-tfidf-lgb-afhs': 'models/early-tfidf-lgb-afhs.zip'
}


_PRETRAINED_MODELS = {
    'embeddings': embeddings_gcp,
    'lm': lm_gcp,
    'flair': flair_gcp,
    'voice': voice_gcp,
    'misc': misc_models_gcp
}
