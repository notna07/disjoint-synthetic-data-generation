# Description: Disjoint Generative Model Manager Class
# Date: 14-11-2024
# Version: 0.1
# Author : Anton D. Lautrup


class DisjointGenerativeModels:

    def __init__(self,
                 training_data,
                 generative_models,
                 ):
        
        self.training_data = training_data
        self.generative_models = generative_models
