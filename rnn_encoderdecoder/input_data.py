"""
Processing for the dataset into memory.
"""

dataset = "data/training/"

class europarl:
    
    def __init__(self):
        self.english = process_file('europarl-v7.fr-en.en')
        self.french = process_file('europarl-v7.fr-en.fr')

    def process_file(self, filename):
        sentences = load_file(filename)
        lengths = [len(s.split()) for s in sentences]
        max_len = max(lengths)
        
        return (clean_data(sentences), max_len)

    def load_file(self, filename):
        file = open(dataset + filename, mode='rt', encoding='utf-8')
        text = file.read()
        file.close()

        return text.strip().split('\n')
    
    def clean_data(self, sentences):
        cleaned = []

        for line in sentences:
            line = line.lower().encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            
            line = line.split()

            cleaned.append(' '.join(line))

        return cleaned


