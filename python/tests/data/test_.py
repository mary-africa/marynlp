from marynlp.data.transformers import SwahiliTextTransformer

def test_validate_text():
    """
    Checks if the `SwahiliTextTransformer` is doing a good job in
    transforming a text into the right format
    """
    text_to_validate = 'Mimi ni Mwanafunzi MZURI sana. Nina ng\'ombe wawili.'

    transform = SwahiliTextTransformer()
    assert transform(text_to_validate) == 'mimi ni mwanafunzi mzuri sana . nina ng\'ombe wawili .'
