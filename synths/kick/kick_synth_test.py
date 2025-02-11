# main/synths/kick/kick_synth_tests.py

import torch

# Define a dictionary of test cases for different kick sounds.
# Each test case is a dictionary mapping parameter names (as expected by the synthesizer)
# to values (in human-readable units). These values will later be converted to tensors.
TEST_CASES = {
  "classic_kick": {
    "f_start": 100.0,          
    "pitch_decay": 10.0,       
    "detune_amount": 0.2,       
    "sub_osc_mix": 0.0,        
    "attack_time": 0.005,       
    "decay_time": 0.02,        
    "sustain_level": 0.0,       
    "release_time": 0.02,       
    "distortion_drive": 1.5,    
    "shell_filter_type": 0.0,   
    "shell_filter_cutoff": 120.0,  
    "shell_filter_res": 0.7,    
    "shell_decay": 0.3,         
    "shell_mix": 0.4,           
    "lfo_rate": 0.0,
    "lfo_amount": 0.0,
    "transient_decay": 1200.0,   
    "transient_level": 2.0,     
    "transient_hp_cutoff": 1040.0,
    "use_modal": 0.0,           
    "modal_freq": 200.0,
    "modal_decay": 0.3,
    "modal_amp": 0.0,
},
    "heavy_hard_style": {
        "f_start": 100.0,
        "pitch_decay": 10.0,
        "detune_amount": 0.0,
        "sub_osc_mix": 0.0,
        "attack_time": 0.005,
        "decay_time": 0.1,
        "sustain_level": 0.0,
        "release_time": 0.05,
        "distortion_drive": 3.0,
        "shell_filter_type": 0.0,
        "shell_filter_cutoff": 120.0,
        "shell_filter_res": 0.7,
        "shell_decay": 0.3,
        "shell_mix": 0.4,
        "lfo_rate": 0.0,
        "lfo_amount": 0.0,
        "transient_decay": 400.0,
        "transient_level": 2.0,
        "transient_hp_cutoff": 4000.0,
        "use_modal": 0.0,
        "modal_freq": 200.0,
        "modal_decay": 0.2,
        "modal_amp": 0.0,
    },
    "electro_kick": {
        "f_start": 900.0,
        "pitch_decay": 50.0,
        "detune_amount": 0.1,
        "sub_osc_mix": 0.0,
        "attack_time": 0.005,
        "decay_time": 0.08,
        "sustain_level": 0.1,
        "release_time": 0.1,
        "distortion_drive": 2.0,
        "shell_filter_type": 0.0,
        "shell_filter_cutoff": 150.0,
        "shell_filter_res": 0.9,
        "shell_decay": 0.4,
        "shell_mix": 0.6,
        "lfo_rate": 0.0,
        "lfo_amount": 0.0,
        "transient_decay": 300.0,
        "transient_level": 1.0,
        "transient_hp_cutoff": 5000.0,
        "use_modal": 1.0,       
        "modal_freq": 250.0,
        "modal_decay": 0.05,     
        "modal_amp": 0.1,        
    }
}

def get_test_parameters(test_case: dict, device=torch.device("cpu")) -> dict:
    """
    Converts a test case (a dictionary of parameter values) into a dictionary
    of PyTorch tensors (each of shape [1]) on the specified device.
    """
    # Convert every value to a tensor of shape [1] (float32).
    params = {k: torch.tensor([v], dtype=torch.float32, device=device) for k, v in test_case.items()}
    return params
