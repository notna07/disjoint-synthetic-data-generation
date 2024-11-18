# Description: Script for generating synthetic datasets in a variety of ways
# Author: Anton D. Lautrup
# Date: 18-11-2024

import os

import pandas as pd

from typing import List
from pandas import DataFrame
from abc import ABC, abstractmethod

def _load_data(train_data_name):
    df_train = pd.read_csv(train_data_name +'_train.csv').dropna()
    return df_train

class DataGeneratorAdapter(ABC):
    @abstractmethod
    def generate(self, train_data_name: str) -> DataFrame:
        """ Generate synthetic data based on the training data.

        Args:
            train_data_name (str): The name of the training data file.

        Returns:
            DataFrame: The generated synthetic data.
        """
        pass

class SynthCityAdapter(DataGeneratorAdapter):
    """ SynthCity Adapter for generating synthetic data.

    Attributes:
        gen_model (str): The generative model to use.
    """
    def __init__(self, gen_model):
        self.gen_model = gen_model

    def generate(self, train_data_name: str) -> DataFrame:
        """ Generate synthetic data using SynthCity.

        Reference:
            Qian, Z., Cebere, B.-C., & van der Schaar, M. (2023). Synthcity: facilitating innovative 
            use cases of synthetic data in different data modalities. http://arxiv.org/abs/2301.07573

        Args:
            train_data_name (str): The name of the training data file.

        Returns:
            DataFrame: The generated synthetic data.

        Example:
            >>> adapter = SynthCityAdapter('privbayes')
            >>> df_syn = adapter.generate('tests/dummy') # doctest: +ELLIPSIS
            -etc-
            >>> isinstance(df_syn, pd.DataFrame)
            True
        """
        from synthcity.plugins import Plugins
        df_train = _load_data(train_data_name)

        syn_model = Plugins().get(self.gen_model)
        syn_model.fit(df_train)

        df_syn = syn_model.generate(count=len(df_train)).dataframe()
        return df_syn

class SynthPopAdapter(DataGeneratorAdapter):
    def generate(self, train_data_name: str) -> DataFrame:
        """ Generate synthetic data using SynthPop.

        Reference:
            Nowok, B., Raab, G. M., & Dibben, C. (2016). synthpop: Bespoke Creation of Synthetic Data in R. 
            Journal of Statistical Software, 74(11), 1--26. https://doi.org/10.18637/jss.v074.i11

        Args:
            train_data_name (str): The name of the training data file.

        Returns:
            DataFrame: The generated synthetic data.

        Example:
            >>> adapter = SynthPopAdapter()
            >>> df_syn = adapter.generate('tests/dummy')
            >>> isinstance(df_syn, pd.DataFrame)
            True
        """
        import os, subprocess
        df_train = _load_data(train_data_name)

        info_dir = 'synthesis_info_' + train_data_name.split('/')[0]
        if not os.path.exists(info_dir):
            os.makedirs(info_dir)

        command = [
                    "Rscript",
                    "disjoint_generative_model/utils/subprocess/synthpop_subprocess.R",
                    train_data_name +"_train.csv",
                    train_data_name + "_synthpop"
                ]
        subprocess.run(command, check=True)

        df_syn = pd.read_csv(train_data_name + '_synthpop.csv')
        df_syn.columns = [col for col in df_train.columns]

        os.remove(train_data_name + '_synthpop.csv')
        os.remove('synthesis_info_' + train_data_name + '_synthpop.txt')
        os.removedirs(info_dir)
        return df_syn

class DataSynthesizerAdapter(DataGeneratorAdapter):
    def generate(self, train_data_name: str) -> DataFrame:
        """ Generate synthetic data using DataSynthesizer.

        Reference:
            Ping, H., Stoyanovich, J., & Howe, B. 2017. DataSynthesizer: Privacy-Preserving Synthetic Datasets. 
            In Proceedings of the 29th International Conference on Scientific and Statistical Database Management (SSDBM '17). 
            Association for Computing Machinery, New York, NY, USA, Article 42, 1--5. https://doi.org/10.1145/3085504.3091117

        Args:
            train_data_name (str): The name of the training data file.

        Returns:
            DataFrame: The generated synthetic data.

        Example:
            >>> adapter = DataSynthesizerAdapter()
            >>> df_syn = adapter.generate('tests/dummy') # doctest: +ELLIPSIS

            >>> isinstance(df_syn, pd.DataFrame)
            True
        """
        from DataSynthesizer.DataDescriber import DataDescriber
        from DataSynthesizer.DataGenerator import DataGenerator
        df_train = _load_data(train_data_name)

        input_data = train_data_name +'_train.csv'
        description_file = train_data_name + "_datasynthesizer_info.json"

        describer = DataDescriber(category_threshold=10)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                                epsilon=0, 
                                                                k=2,
                                                                attribute_to_is_categorical={})
        describer.save_dataset_description_to_file(description_file)

        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(len(df_train), description_file)

        df_syn = generator.synthetic_dataset

        os.remove(description_file)
        return df_syn

class DebugAdapter(DataGeneratorAdapter):
    def generate(self, train_data_name: str) -> DataFrame:
        return _load_data(train_data_name)

def generate_synthetic_data(train_data_name: str, gen_model: str) -> DataFrame:
    """ Generate synthetic data using the specified generative model.

    Args:
        train_data_name (str): The name of the training data file.
        gen_model (str): The name of the generative model.

    Returns:
        DataFrame: The generated synthetic data.

    Example:
        >>> df_syn = generate_synthetic_data('tests/dummy', 'synthpop')
        >>> isinstance(df_syn, pd.DataFrame)
        True
    """
    adapters = {
        'ctgan': SynthCityAdapter(gen_model),
        'adsgan': SynthCityAdapter(gen_model),
        'tvae': SynthCityAdapter(gen_model),
        'nflow': SynthCityAdapter(gen_model),
        'ddpm': SynthCityAdapter(gen_model),
        'dpgan': SynthCityAdapter(gen_model),
        'privbayes': SynthCityAdapter(gen_model),
        'synthpop': SynthPopAdapter(),
        'datasynthesizer': DataSynthesizerAdapter(),
        'debug': DebugAdapter()
    }

    if gen_model in adapters:
        adapter = adapters[gen_model]
        df_syn = adapter.generate(train_data_name)
    else:
        raise NotImplementedError("The chosen generative model could not be run!")

    df_syn.to_csv(train_data_name + '_' + gen_model + '.csv', index=False)
    return df_syn


if __name__ == "__main__":
    import doctest
    doctest.ELLIPSIS_MARKER = '-etc-'
    doctest.testmod()