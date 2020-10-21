class LabelEncoder:
    def __init__(self, items: list, sorted = False):
        self.items = items if not sorted else sorted(items)

    def encode(self, item):
        return float(self.items.index(item))

    def __call__(self, input):
        return self.encode(input)

    def decode(self, ix):
        return self.items[int(ix)]