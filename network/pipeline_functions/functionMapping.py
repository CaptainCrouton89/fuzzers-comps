"""A collection of common function mappings for transforming columns
of a pandas dataframe. 
"""

# df: dataframe object
# column: dataframe column name
def example_func(df, column: str, learn=True):
    # perform normalization or other transformation on column here
    return df[column]

class MapFunc():
    # This isn't being used for anything, but I'm keeping it anyways in case we want to come back.
    # It would be the parent  class to map functions.
    def __init__(self, df, column):
        self.df = df
        self.column = column
        self.data = None

    def load_from_file(self, path):
        pass

    def learn_apply(self):
        pass

    def apply(self):
        pass