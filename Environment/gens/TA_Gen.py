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
        _stock = Sdf.retype(df.copy())
        _stock.get('cci_14')
        _stock.get('rsi_14')
        _stock.get('dx_14')
        _stock = _stock.dropna(how='any')

        # Fit minmax scaler on training data only to prevent look-forward
        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        fit_sample = _stock[['rsi_14', 'cci_14','dx_14','volume']]
        fit_split_len = int((split-0.1)*len(fit_sample)) # deduct 0.1 for validation_split from dueling_dqn observe()
        min_max_scaler = min_max_scaler.fit(fit_sample.iloc[:fit_split_len,:])
        np_scaled = min_max_scaler.transform(fit_sample)
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized.columns = ['rsi_14', 'cci_14','dx_14','volume']

        split_len=int(split*len(df_normalized))

        if(mode=='train'):
            raw_data = df_normalized[['rsi_14','cci_14','dx_14','volume']].iloc[:split_len,:]
        else:
            raw_data = df_normalized[['rsi_14', 'cci_14','dx_14','volume']].iloc[split_len:,:]

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