import csv
import numpy as np
from Environment.core import DataGenerator
import pandas as pd
from stockstats import StockDataFrame as Sdf
from sklearn import preprocessing

class TAStreamer(DataGenerator):
    """Data generator from csv file.
    The csv file should no index columns.

    Args:
        filename (str): Filepath to a csv file.
        header (bool): True if the file has got a header, False otherwise
    """
    @staticmethod
    def _generator(filename, header=False, split=0.8, mode='train',spread=.005):
        df = pd.read_csv(filename)
        if "Name" in df:
            df.drop('Name',axis=1,inplace=True)

        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(df[['open', 'high','low','close','volume']])
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized.columns = ['open','high','low','close','volume']

        split_len=int(split*len(df_normalized))

        if(mode=='train'):
            raw_data = df_normalized[['open','high','low','close','volume']].iloc[:split_len,:]
        else:
            raw_data = df_normalized[['open','high','low','close','volume']].iloc[split_len:,:]

        for index, row in raw_data.iterrows():
            yield row.to_numpy()


    def _iterator_end(self):
        """Rewinds if end of data reached.
        """
        # print("End of data reached, rewinding.")
        super(self.__class__, self).rewind()

    def rewind(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        self._iterator_end()
