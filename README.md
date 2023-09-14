# Toolbox Burst Detection (TBD)

This a toolbox of Pyhton functions for the detection and analysis of bursts in neural signals.

## Acknowledgments

- [MNE Python](https://mne.tools/stable/index.html)

- [Spectral Events toolbox](https://github.com/jonescompneurolab/SpectralEvents)

- [hmmlearn] (hmmlearn)

- Please, let me know if there are functions, code, or methods that are not properly acknowledged and I will amend it.


## How to use the TBD

You first need to clone the repository using ´git clone https ...´

Then you can import the ´ToolboxBurstsDetection.py´ module by setting the path to the ´ToolboxBurstsDetection´ folder.

Example:

```python
import sys
sys.path.append('/YourPath/ToolboxBurstsDetection')
import ToolboxBurstsDetection as TBD
```

## Data structure of the neural signals

- One neural signals: 'numpy.ndarray' of shape (Nsamples,)
- Nepochs of signals: 'numpy.ndarray' of shape (Nepochs,Nsamples)