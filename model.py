import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st

class ModelHandler:
    def __init__(self, model_path='cnn_model.h5'):
        """
        Initialize the model handler by loading the model
        
        Parameters:
        -----------
        model_path : str
            Path to the pre-trained model file
        """
        self.model_path = model_path
        self.model = self._load_model()
        
    @st.cache_resource
    def _load_model(self):
        """
        Load the pre-trained model or create a new one if it doesn't exist
        
        Returns:
        --------
        keras.Model
            The loaded or newly created model
        """
        if os.path.exists(self.model_path):
        np.save(filename, image)
