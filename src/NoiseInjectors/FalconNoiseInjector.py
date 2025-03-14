"""
Script contains the FalconNoiseInjector class.

This class injects noise into various layers
of the Falcon LLM.
@author: Stephen Krol
@author: CLAUDE
@author: Tace McNamara

This file contains the following class:
    - FalconNoiseInjector
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from NoiseInjector import NoiseInjector

class FalconNoiseInjector(NoiseInjector):
    """
    Class for the FalconNoiseInjector which
    injects noise into the falcon model
    during generation.
    @author: Stephen Krol
    @author: CLAUDE
    @author: Tace McNamara
    """
    def __init__(self, model, tokenizer):
        """
        Class constructor.

        Args:
            model: the model object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.layer_mapping = self._create_layer_mapping()
        self.layer_noise_stats = {}

        print(self.device)

    
    def generate_with_noise(self, prompt, noise_config, max_length = 100, **kwargs):
        
        return


if __name__ == "__main__":

    model_name = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True,
                                                device_map="auto")