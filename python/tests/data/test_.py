from marynlp.data.transformers import SwahiliTextTransformer, DataTextTransformer

def test_validate_text():
    """
    Checks if the `SwahiliTextTransformer` is doing a good job in
    transforming a text into the right format
    """
    text_to_validate = 'Mimi ni Mwanafunzi MZURI sana. Nina ng\'ombe wawili.'

    transform = SwahiliTextTransformer()
    assert transform(text_to_validate) == 'mimi ni mwanafunzi mzuri sana . nina ng\'ombe wawili .'


class CountCharacterDT(DataTextTransformer):
    def transform(self, text: str):
        return len(text)

def test_something_good():
    text_to_count = 'kevin-nelson-people'

    counter = CountCharacterDT()
    assert counter(text_to_count) == 19, "There should be 19"