# DDSP-AudioPlugins

**Fully Differentiable, Deterministic, and GPU-Optimized High-Quality Audio Plugins**

DDSP-AudioPlugins is a collection of fully differentiable audio plugins built in PyTorch. These plugins enable gradient-based optimization and analysis-by-synthesis for music production, audio inversion, and machine learning research.  
Currently, the repository features a kick synthesizer that demonstrates a differentiable approach to sound synthesis. In the future, additional plugins will be added to expand the toolkit.

**The plugin is not perfect and I'm still working on it. However, it still generates some decent sounding kicks.

## Overview

This differentiable kick synthesizer is designed with a DSP architecture for precise control over every aspect of the sound. Key components include:

- **Differentiable DSP Components:**  
  Every element—from the oscillators and ADSR envelopes to FIR-based filtering and transient generation—is fully differentiable. This enables gradient-based optimization, making the plugin ideal for inversion tasks and analysis-by-synthesis workflows.

- **Modular Design:**  
  The synthesizer includes several DSP modules:
  - **Oscillator and Pitch Envelope:** Generates a dynamic pitch contour starting at a high frequency and decaying rapidly.
  - **ADSR Envelopes:** Customizable amplitude shaping ensures a percussive attack and controlled decay.
  - **FIR-based Filters:** Fully vectorized lowpass, highpass, and bandpass filters that shape the resonant “body” of the kick.
  - **Transient Generation:** A noise-based transient component with adjustable decay, level, and filtering to create the initial click.
  - **Modal Synthesis (Optional):** An additional resonant module that can be toggled on or off to emulate physical drum resonances.

- **GPU Optimization and Determinism:**  
  The entire synthesizer is implemented in a batched, vectorized manner and optimized for GPU execution, ensuring efficient performance even for complex synthesis tasks.

- **Inversion and Analysis-by-Synthesis:**  
  Because the plugin is fully differentiable, it can be used as a loss function in machine learning pipelines. Instead of training a model to directly generate audio, you can train it to produce synthesis parameters that, when passed through the plugin, yield the desired sound. This is particularly useful for tasks like kick inversion.




