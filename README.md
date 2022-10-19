# crop-type
A collection of notebooks describing a repeatable workflow for predicting crop types using the Digital Earth Africa platform

## Generate Sampling Strategy
These notebooks provide a method to use unsupervised clustering to identify different groups in spectral features, which may correspond to different crop types. It uses the DE Africa crop map to only cluster points from within classified crop areas. These clusters are then used to generate suggested sampling locations for field surveys.

## Prepare Samples for ML
Clean data collected through ECAAS ODK-Toolkit. This workflow is customised to match the specific ECAAS output, but many of the steps apply to any crop type data, particularly unification of crop type labels.

