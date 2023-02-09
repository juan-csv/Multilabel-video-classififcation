class Map_index2label():
    def __init__(self, PATH_VOCABULARY):
        # load vocabulary
        self.vocabulary_df = pd.read_csv(PATH_VOCABULARY)
        self.label_mapping = self.vocabulary_df[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name']

    def __call__(self, index):
        if index in list( self.label_mapping.keys() ):
            return self.label_mapping[index]
        else:
            return ''

label_mapping = Map_index2label(PATH_VOCABULARY)