"""
Categorical placement of the models related data.
Such data can be model weights, text / yaml files. Any thing.. (all of which are in a ZIP file)

TODO: Make sure the code is NOT case INsensitive
"""

# Embeddings related models
embeddings_gcp = {
    
}

# Language models
lm_gcp = {
    
}

# All flair related data
#  This is to accomodate on the fact that flair has a unique way of storing their weights
#  so for any related models that use
flair_gcp = {
    'sw-exp-sent_analy-small': 'flair/classifier/sw-exp-sent_analy-small.zip',
    'early-wpc': 'flair/classifier/exp-wpc-small.zip'
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
