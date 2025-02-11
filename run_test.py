import os
import torch
import torch.nn.functional as F
import soundfile as sf
from synths.kick.kick_synthesizer import DiffKickSynth
from synths.kick.kick_synth_test import TEST_CASES, get_test_parameters

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create output folder (in the current working directory) called "output_test_wavs".
output_dir = "output_test_wavs"
os.makedirs(output_dir, exist_ok=True)

# Instantiate the synthesizer.
synth = DiffKickSynth(sample_rate=44100).to(device)

# Define synthesis parameters.
duration = 1.0       # 1 second
desired_length = 44100  # samples (44100 Hz)

# Iterate over all test cases.
for test_name, param_dict in TEST_CASES.items():
    # Convert the test case parameters to PyTorch tensors.
    params = get_test_parameters(param_dict, device=device)
    
    # Call the synthesizer with the parameters.
    # The synthesizer expects parameters in the following order:
    # [f_start, pitch_decay, detune_amount, sub_osc_mix,
    #  attack_time, decay_time, sustain_level, release_time,
    #  distortion_drive,
    #  shell_filter_type, shell_filter_cutoff, shell_filter_res, shell_decay,
    #  shell_mix, lfo_rate, lfo_amount,
    #  transient_decay, transient_level, transient_hp_cutoff,
    #  use_modal, modal_freq, modal_decay, modal_amp]
    kick = synth(
        params["f_start"], params["pitch_decay"], params["detune_amount"], params["sub_osc_mix"],
        params["attack_time"], params["decay_time"], params["sustain_level"], params["release_time"],
        params["distortion_drive"],
        params["shell_filter_type"], params["shell_filter_cutoff"], params["shell_filter_res"],
        params["shell_decay"], params["shell_mix"], params["lfo_rate"], params["lfo_amount"],
        params["transient_decay"], params["transient_level"], params["transient_hp_cutoff"],
        params["use_modal"], params["modal_freq"], params["modal_decay"], params["modal_amp"],
        duration, use_fir=True
    )  # Output shape: [1, num_samples]
    
    # If the output is not the desired length, interpolate.
    if kick.shape[1] != desired_length:
        kick = F.interpolate(kick.unsqueeze(1), size=desired_length, mode='linear', align_corners=False).squeeze(1)
    
    # Remove the batch dimension and convert to a NumPy array.
    kick_np = kick.squeeze(0).detach().cpu().numpy()
    # Normalize the waveform.
    kick_np = kick_np / kick_np.max()
    
    # Save the output WAV file.
    output_path = os.path.join(output_dir, f"{test_name}.wav")
    sf.write(output_path, kick_np, 44100)
    print(f"Saved '{test_name}' kick to {output_path}")
