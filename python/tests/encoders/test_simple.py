from marynlp.encoders.label import LabelEncoder

lb_encoder = LabelEncoder(['a', 'e', 'i', 'o', 'u'])

def test_label_encoder():
    """
    Making sure the label encode is mapping appropriately
    """
    assert lb_encoder.encode('e') == 1, "Invalid mapping. 'e' must map to 1"
