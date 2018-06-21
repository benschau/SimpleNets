"""
Processing for the dataset into memory.
"""

class LanguageData:
    dataset = "data/training/"
    
    def __init__(self):
        self.english = self.process_file('europarl-v7.fr-en.en')
        self.french = self.process_file('europarl-v7.fr-en.fr')
    
    """
    process_file
    
    Clean data within the specified file for NN usage.

        @args filename file to preprocess 
        @return cleaned data and the maximum length of a sentence.
    """
    def process_file(self, filename):
        sentences = self.load_file(filename)
        lengths = [len(s.split()) for s in sentences]
        max_len = max(lengths)
        
        return (self.clean_data(sentences), max_len)

    """
    load_file
    
    Empty the contents of the file into a list.

        @args filename file to read
        @return a list of strings from the file, stripped of whitespace and split by newline.
    """
    def load_file(self, filename):
        file = open(self.dataset + filename, mode='rt', encoding='utf-8')
        text = file.read()
        file.close()

        return text.strip().split('\n')
    
    """
    clean_data

    Clean the list of sentences by:
    1) ensuring that each sentence has the same encoding, and decoding if some of the sentence is not
    2) remove extra spaces within sentences.

        @args sentences list of sentences from the read file
        @return cleaned, a list of cleaned sentences.
    """
    def clean_data(self, sentences):
        cleaned = []

        for line in sentences:
            line = line.lower().encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            
            line = line.split()

            cleaned.append(' '.join(line))

        return cleaned


