# Analyzing analytical methods

This repository contains the instructions and code to help you reproduce the results in the following paper:

Grzegorz Chrupała, Bertrand Higy and Afra Alishahi (2020). Analyzing analytical methods: The case of phonology in neural models of spoken language. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.

## RNN-VGS experiments
- Download and unpack the  model and data from [TODO]. 
- Create the input data for analyses:
```
python -c 'import prepare as P; P.prepare_rnn_vgs()'
```
- [TODO]


## ASR-VGS experiments
[TODO]

## Transformer-ASR experiments

The transformer-ASR model is a transformer model trained with
[ESPnet](https://github.com/espnet/espnet) on
[Librispeech](http://www.openslr.org/12/). The forked version of the code we
used to extract the activations is available from https://github.com/bhigy/espnet/tree/phoneme-repr. The
[README](https://github.com/bhigy/espnet/blob/phoneme-repr/README.md) at the
root of the repository provides instructions for installation, while details about
activations extraction can be found under
[egs/librispeech/asr1/README.md](https://github.com/bhigy/espnet/blob/phoneme-repr/egs/librispeech/asr1/README.md).