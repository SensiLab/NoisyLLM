"""
Script contains the FalconNoiseInjector class.

This class injects noise into various layers
of the Falcon LLM.
@author: Stephen Krol

This file contains the following class:
    - FalconNoiseInjector
"""

from typing import Union, Dict, List

import torch
from NoiseInjector import NoiseInjector
from transformers import AutoTokenizer, AutoModelForCausalLM

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
            model: the model object loaded in the transformers
                library.
            tokenizer: tokenizer for the falcon model.
        
        Returns:
            None
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.layer_noise_stats = {}


    def generate_with_noise(self,
                            prompt: str,
                            noise_config: Dict[str, Union[float, Dict]],
                            max_length: int = 100,
                            **kwargs) -> str:
        """
        TODO: wtf is this doing
        """
        
        self.noise_config = self.__validate_config(noise_config)

        self.layer_noise_stats = {}  # Reset noise statistics
        
        input_ids, attention_mask = self.__prepare_inputs(prompt) # TODO: wtf is this
        # hooks = [] # TODO: wtf is a hook?

        # # Register hooks for each attention layer
        # for name, module in self.model.named_modules():
        #     if "attention" in name.lower():
        #         pre_hook = module.register_forward_pre_hook(self.attention_forward_pre_hook)
        #         hooks.append(pre_hook)
        #         post_hook = module.register_forward_hook(self.attention_forward_hook)
        #         hooks.append(post_hook)

        # try:
        #     # Set up generation parameters
        #     generation_config = {
        #         'max_length': max_length,
        #         'do_sample': True,
        #         'pad_token_id': self.tokenizer.pad_token_id,
        #         'eos_token_id': self.tokenizer.eos_token_id,
        #         'top_k': noise_config.get('top_k', 50),
        #         'top_p': noise_config.get('top_p', 1.0),
        #         'temperature': noise_config.get('temperature', 1.0),
        #     }

        #     # Add any additional kwargs
        #     generation_config.update(kwargs)

        #     # Generate with noise injection
        #     outputs = self.model.generate(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         **generation_config
        #     )
           
            
        #     # Print noise statistics if requested
        #     if noise_config.get('verbose', False):
        #         print("\nNoise Statistics per Layer:")
        #         for layer_idx, stats in sorted(self.layer_noise_stats.items()):
        #             print(f"Layer {layer_idx}:")
        #             print(f"  Noise std: {stats['noise_std']:.4f}")
        #             print(f"  Tensor std: {stats['tensor_std']:.4f}")
        #             print(f"  Noise/Tensor ratio: {stats['ratio']:.4f}")

        #     # Decode the output
        #     return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # finally:
        #     # Clean up hooks
        #     for hook in hooks:
        #         hook.remove()

    def __validate_config(self, config: dict) -> dict:
        """
        Private method verfies the values of the config are valid.
        Specifically, it verifies the validatity of the temperature,
        top_p and key noise parameters.
        @author: CLAUDE
        @author: Tace McNamara
        @modified: Stephen Krol
        # TODO: Add validation for all arguments.

        Args:
            config (dict): config arguments for noise injection.
        Returns:
            dict: validated config dictionary.
        """
        
        # Validate temperature
        if config.get('temperature', 1.0) <= 0:
            print("Warning: Temperature must be positive. Setting to 1.0")
            config['temperature'] = 1.0
            
        # Validate top_p
        if config.get('top_p', 1.0) <= 0 or config.get('top_p', 1.0) > 1:
            print("Warning: top_p must be between 0 and 1. Setting to 0.9")
            config['top_p'] = 0.9
            
        # Validate noise values
        noise_keys = ['query_noise', 'key_noise', 'value_noise', 'output_noise']
        for key in noise_keys:
            if key in config and config[key] < 0:
                print(f"Warning: {key} cannot be negative. Setting to 0")
                config[key] = 0.0

        return config 

    def __prepare_inputs(self, prompt: str):
        """
        Private method for preparing model inputs.
        Tokenizers input, sends them to the relevant
        device and generates attention mask.
        @author: CLAUDE
        @author: Tace McNamara

        Args:
            prompt (str): Input prompt for LLM.
        Returns:
            torch.tensor: input tokens
            torch.tensor: attention mask
        """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        return input_ids, attention_mask

if __name__ == "__main__":

    model_name = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    falcon = AutoModelForCausalLM.from_pretrained(model_name, 
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True,
                                                device_map="auto")

    falcon_injector = FalconNoiseInjector(falcon, tokenizer)

    noise_config = {
        'query_noise':0.0,
        'key_noise': 0.0,
        'value_noise': 0.0,
        'output_noise': 0.0, 
        'noise_strategy': 'uniform', # 'exp_increase' 'uniform', 'exp_decay', 'first_layer_only', 'last_layer_only'
        'exp_rate': 1.0,  # Controls rate of exponential change in noise 
        
        'temperature': 0.1,  #temperature must be higher than 0.0
        'top_k': 0,  #
        'top_p': 1.0, # Disable top-p sampling = 1.0
        'verbose': False,
        'test' : 0
    }


    falcon_injector.generate_with_noise("test", noise_config)


