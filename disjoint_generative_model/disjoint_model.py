# Description: Disjoint Generative Model Manager Class
# Date: 14-11-2024
# Version: 0.1
# Author : Anton D. Lautrup

from utils.dataset_manager import DataManager
from utils.generative_model_adapters import generate_synthetic_data

from utils.joining_strategies import JoiningModule, RandomJoining, UsingJoiningValidator

class DisjointGenerativeModels:

    def __init__(self,
                 training_data,
                 generative_models,
                 ):
        
        self.training_data = training_data
        self.generative_models = generative_models

        self.join_validator_model = None
        pass
    
    def _setup(self):

        pass

    def _prepare_data(self):
        pass

    def _evaluate_splits(self):
        pass

    def fit(self):




        pass

    def generate(self):
        

        


        pass

    def conduct_joining(self):
        

        JoiningModule(RandomJoining()).perform_joining()

        # JoiningModule(UsingJoiningValidator()).perform_joining()


        pass

        

        