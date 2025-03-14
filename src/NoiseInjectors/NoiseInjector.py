"""
Script contains a parent class NoiseInjector.

This abstact class is used as a outline for the methods
and functionality of LLM noise injectors which inherit
from this class.
@author: Stephen Krol

This file contains the following class:
    - NoiseInjector
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, List

class NoiseInjector(ABC):
    """
    Abstract class for a LLM noise injector.
    @author: Stephen Krol

    This class contains the following
    abstract methods:

        - init : intialising the object with a model
            and tokenizer object.
    """
    @abstractmethod
    def __init__(self, model, tokenizer) -> None:
        """
        Abstract intialiser, noise injectors should be
        initial with a model and tokenizer.

        Args:
            model: a model object.
            tokenizer: a tokenizer for the model.
        """
        
        pass

    @abstractmethod
    def generate_with_noise(self,
                          prompt: str,
                          noise_config: Dict[str, Union[float, Dict]],
                          max_length: int = 100,
                          **kwargs) -> str:
        """
        Absract method for generation with noise injection,
        @author: Stephen Krol

        Args:
            prompt (str): input prompt to generate from.
            noise_config (dict): config controlling the type of
                noise being injected.
            max_length (int): max length of output.
        
        Returns:
            str: generated output text
        """
        pass


