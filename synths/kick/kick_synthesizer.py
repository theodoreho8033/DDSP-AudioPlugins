import torch
import torch.nn as nn
import torch.nn.functional as F
import math




def scan(step_fn, init_state, xs):
        state = init_state
        outputs = []
        for x in xs:
            state, out = step_fn(state, x)
            outputs.append(out)
        return state, torch.stack(outputs)

class DiffKickSynth(nn.Module):
    def __init__(self, sample_rate: int = 44100, fir_kernel_size: int = 101) -> None:
        """
        Args:
            sample_rate: Sampling rate in Hz.
            fir_kernel_size: Length of the FIR filter kernels (should be odd).
        """
        super(DiffKickSynth, self).__init__()
        self.sample_rate: int = sample_rate
        self.fir_kernel_size: int = fir_kernel_size

    def adsr_envelope(self, t: torch.Tensor, attack: torch.Tensor, decay: torch.Tensor,
                       sustain: torch.Tensor, release: torch.Tensor, duration: float) -> torch.Tensor:
        """
        Computes a batched piecewise ADSR envelope.
        Args:
            t: Time vector, shape [B, T]
            attack, decay, sustain, release: Tensors of shape [B, 1]
            duration: Scalar float (same for all batch elements)
        Returns:
            Envelope of shape [B, T]
        """
        release_start = duration - release  # shape [B, 1]
        envelope = torch.zeros_like(t)
        # Attack phase:
        attack_phase = t < attack
        envelope = torch.where(attack_phase, t / attack, envelope)
        # Decay phase:
        decay_phase = (t >= attack) & (t < (attack + decay))
        decay_env = 1 - ((t - attack) / decay) * (1 - sustain)
        envelope = torch.where(decay_phase, decay_env, envelope)
        # Sustain phase:
        sustain_phase = (t >= (attack + decay)) & (t < release_start)
        envelope = torch.where(sustain_phase, sustain * torch.ones_like(t), envelope)
        # Release phase:
        release_phase = t >= release_start
        release_env = sustain * (1 - (t - release_start) / release)
        envelope = torch.where(release_phase, release_env, envelope)
        return envelope

    # -------------- FIR Filter Implementations --------------
    def fir_lowpass_kernel(self, cutoff: torch.Tensor) -> torch.Tensor:
        """
        Computes a batched FIR lowpass filter kernel.
        Args:
            cutoff: Tensor of shape [B, 1] in Hz.
        Returns:
            FIR kernel tensor of shape [B, 1, kernel_size]
        """
        # Ensure cutoff is 2D: [B, 1]
        if cutoff.dim() == 1:
            cutoff = cutoff.unsqueeze(1)
        B = cutoff.shape[0]
        K = self.fir_kernel_size
        # Create time indices centered at zero.
        n = torch.arange(K, device=cutoff.device, dtype=torch.float32) - (K - 1) / 2.0  # shape [K]
        # Normalize cutoff as a fraction of the sampling rate.
        fc = cutoff / self.sample_rate  # shape [B, 1]
        # Ideal impulse response: h(n) = 2 * fc * sinc(2 * fc * n)
        h = 2 * fc * torch.sinc(2 * fc * n)  # broadcast to shape [B, K]
        h = h.unsqueeze(1)  # shape [B, 1, K]
        # Apply a Hann window.
        window = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(K, device=cutoff.device, dtype=torch.float32) / (K - 1))
        window = window.unsqueeze(0).unsqueeze(0)  # shape [1, 1, K]
        h = h * window
        # Normalize each kernel so that its sum equals 1.
        h = h / h.sum(dim=-1, keepdim=True)
        return h

    def fir_highpass_kernel(self, cutoff: torch.Tensor) -> torch.Tensor:
        """
        Computes a batched FIR highpass filter kernel via spectral inversion.
        Args:
            cutoff: Tensor of shape [B, 1] in Hz.
        Returns:
            FIR highpass kernel of shape [B, 1, kernel_size]
        """
        lowpass = self.fir_lowpass_kernel(cutoff)  # shape [B, 1, K]
        B, _, K = lowpass.shape
        delta = torch.zeros(1, 1, K, device=lowpass.device, dtype=lowpass.dtype)
        delta[..., K // 2] = 1.0
        delta = delta.expand(B, -1, -1)
        highpass = delta - lowpass
        return highpass

    def fir_bandpass_kernel(self, low_cutoff: torch.Tensor, high_cutoff: torch.Tensor) -> torch.Tensor:
        """
        Computes a batched FIR bandpass filter kernel by subtracting two lowpass kernels.
        Args:
            low_cutoff, high_cutoff: Tensors of shape [B, 1] in Hz.
        Returns:
            FIR bandpass kernel of shape [B, 1, kernel_size]
        """
        # Ensure inputs are 2D.
        if low_cutoff.dim() == 1:
            low_cutoff = low_cutoff.unsqueeze(1)
        if high_cutoff.dim() == 1:
            high_cutoff = high_cutoff.unsqueeze(1)
        lowpass_high = self.fir_lowpass_kernel(high_cutoff)
        lowpass_low = self.fir_lowpass_kernel(low_cutoff)
        bandpass = lowpass_high - lowpass_low
        return bandpass

    def apply_fir_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Applies an FIR filter using convolution.
        Instead of using group convolution directly (which requires the input channel count
        to match the groups), we reshape the batch dimension into the channel dimension.
        Args:
            x: Input tensor of shape [B, T]
            kernel: FIR kernel of shape [B, 1, K]
        Returns:
            Filtered output of shape [B, T]
        """
        B, T = x.shape
        # Reshape: merge batch and channel dimensions.
        x_reshaped = x.view(1, B, T)  # shape: [1, B, T]
        kernel_reshaped = kernel.view(B, 1, self.fir_kernel_size)  # shape: [B, 1, K]
        # Perform convolution with groups = B.
        filtered = F.conv1d(x_reshaped, kernel_reshaped, groups=B, padding=self.fir_kernel_size // 2)
        # Reshape back to [B, T]
        filtered = filtered.view(B, T)
        return filtered

    # -------------- End FIR Filter Implementations --------------

    def forward(self, 
                # All parameter tensors are expected to be batched with shape [B, 1]
                f_start: torch.Tensor, pitch_decay: torch.Tensor, detune_amount: torch.Tensor, sub_osc_mix: torch.Tensor,
                attack_time: torch.Tensor, decay_time: torch.Tensor, sustain_level: torch.Tensor, release_time: torch.Tensor,
                distortion_drive: torch.Tensor,
                shell_filter_type: torch.Tensor, shell_filter_cutoff: torch.Tensor, shell_filter_res: torch.Tensor,
                shell_decay: torch.Tensor, shell_mix: torch.Tensor, lfo_rate: torch.Tensor, lfo_amount: torch.Tensor,
                transient_decay: torch.Tensor, transient_level: torch.Tensor, transient_hp_cutoff: torch.Tensor,
                use_modal: torch.Tensor, modal_freq: torch.Tensor, modal_decay: torch.Tensor, modal_amp: torch.Tensor,
                duration: float = 1.0, use_fir: bool = True) -> torch.Tensor:
        """
        Synthesizes a batch of kick sounds given the parameters.
        All parameter tensors should have shape [B, 1] (or be broadcastable to that shape).
        The output is a tensor of shape [B, num_samples], where num_samples = sample_rate * duration.
        If use_fir is True, FIR-based filtering is used.
        """
        B = f_start.shape[0]
        num_samples = int(self.sample_rate * duration)
        # Create time vector of shape [1, T] and expand to [B, T]
        t = torch.linspace(0, duration, steps=num_samples, device=f_start.device, dtype=torch.float32)
        t = t.unsqueeze(0).expand(B, num_samples)  # shape [B, T]
        
        # --- 1. Main Oscillator with Pitch Envelope ---
        f_env = f_start * torch.exp(-pitch_decay * t)  # [B, T]
        f_env = torch.clamp(f_env, min=20, max=5000)
        phase = 2 * math.pi * torch.cumsum(f_env / self.sample_rate, dim=1)  # [B, T]
        main_osc = torch.sin(phase)  # [B, T]
        
        # --- 2. Detuned Oscillator ---
        detuned_weight = torch.clamp(detune_amount, 0.0, 1.0)  # [B, 1]
        f_env_detuned = f_start * torch.exp(-pitch_decay * t) * (1.0 + detune_amount)  # [B, T]
        phase_detuned = 2 * math.pi * torch.cumsum(f_env_detuned / self.sample_rate, dim=1)
        osc_detuned = torch.sin(phase_detuned)  # [B, T]
        main_osc = (1 - detuned_weight) * main_osc + detuned_weight * osc_detuned  # [B, T]
        
        # --- 3. Sub Oscillator (One Octave Below) ---
        f_sub = f_env / 2.0  # [B, T]
        phase_sub = 2 * math.pi * torch.cumsum(f_sub / self.sample_rate, dim=1)
        sub_osc = torch.sin(phase_sub)  # [B, T]
        
        # Blend main and sub oscillators.
        osc_mixed = (1 - sub_osc_mix) * main_osc + sub_osc_mix * sub_osc  # [B, T]
        
        # --- 4. Direct Body Envelope (ADSR) ---
        direct_env = self.adsr_envelope(t, attack_time, decay_time, sustain_level, release_time, duration)  # [B, T]
        direct_body = osc_mixed * direct_env  # [B, T]
        
        # --- 5. Shell Resonance ---
        if use_fir:
            if (shell_filter_type == 0).all():
                kernel = self.fir_lowpass_kernel(shell_filter_cutoff)  # [B, 1, K]
                shell_filtered = self.apply_fir_filter(main_osc, kernel)
            elif (shell_filter_type == 1).all():
                low_cutoff = torch.clamp(shell_filter_cutoff - shell_filter_res * 20, min=20.0)
                high_cutoff = shell_filter_cutoff + shell_filter_res * 20
                kernel = self.fir_bandpass_kernel(low_cutoff, high_cutoff)  # [B, 1, K]
                shell_filtered = self.apply_fir_filter(main_osc, kernel)
            elif (shell_filter_type == 2).all():
                kernel = self.fir_highpass_kernel(shell_filter_cutoff)
                shell_filtered = self.apply_fir_filter(main_osc, kernel)
            else:
                kernel = self.fir_lowpass_kernel(shell_filter_cutoff)
                shell_filtered = self.apply_fir_filter(main_osc, kernel)
        else:
            kernel = self.fir_lowpass_kernel(shell_filter_cutoff)
            shell_filtered = self.apply_fir_filter(main_osc, kernel)
        
        shell_env = torch.exp(-t / shell_decay)  # [B, T]
        shell_resonance = shell_filtered * shell_env  # [B, T]
        combined_body = direct_body + shell_mix * shell_resonance  # [B, T]
        
        # --- 6. Transient (Pop/Crack) Component ---
        raw_transient = torch.randn(B, num_samples, device=t.device, dtype=t.dtype)  # [B, T]
        transient_env = torch.exp(-t * transient_decay)  # [B, T]
        transient_component = raw_transient * transient_env * transient_level  # [B, T]
        transient_hp_kernel = self.fir_highpass_kernel(transient_hp_cutoff)  # [B, 1, K]
        transient_hp = self.apply_fir_filter(transient_component, transient_hp_kernel)  # [B, T]
        transient_processed = torch.tanh(transient_hp * 2.0)  # [B, T]
        
        # --- 7. Modal Synthesis (Optional) ---
        modal_component = use_modal * (modal_amp * torch.sin(2 * math.pi * modal_freq * t) * torch.exp(-t / modal_decay))
        # modal_component: [B, T]
        
        # --- 8. Combine and Final Soft Clipping ---
        kick = combined_body + transient_processed + modal_component  # [B, T]
        kick = torch.atan(kick * distortion_drive) * 13.0  # [B, T]
        
        return kick


