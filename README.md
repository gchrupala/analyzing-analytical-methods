# Analyzing analytical methods

This repository contains the instructions and code to help you reproduce the results in the following paper:

Grzegorz Chrupała, Bertrand Higy and Afra Alishahi (2020). Analyzing analytical methods: The case of phonology in neural models of spoken language. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.
https://arxiv.org/abs/2004.07070

## Prerequisites

For all experiments:
- Download and unpack the  model and data from https://surfdrive.surf.nl/files/index.php/s/XIEKFGIB8TwoKLd.
- Install requirements:
```
pip install -r requirements.txt
```

## RNN-VGS experiments

- Create the input data for analyses:
```
python -c 'import prepare; prepare.prepare_rnn_vgs()'
```
- Run analyses:
```
python -c 'import analyze; analyze.analyze_rnn_vgs()'
```
- Plot main results (Figure 2)
```
python -c 'import analyze; analyze.plot_rnn_vgs()'
```
- Plot Figure 4:
```
python -c 'import analyze; analyze.plot_r2_partial()'

```
- Plot Figure 5
```
python -c 'import analyze; analyze.plot_pooled_feature_std()
```

## RNN-ASR experiments
- Create the input data for analyses:
```
python -c 'import prepare; prepare.prepare_rnn_asr()'
```
- Run analyses:
```
python -c 'import analyze; analyze.analyze_rnn_asr()'
```
- Plot main results (Figure 3)
```
python -c 'import analyze; analyze.plot_rnn_asr()'
```

## Transformer-ASR experiments

The transformer-ASR model is a transformer model trained with
[ESPnet](https://github.com/espnet/espnet) on
[Librispeech](http://www.openslr.org/12/). The forked version of the code we
used to extract the activations is available from https://github.com/bhigy/espnet/tree/phoneme-repr. The
[README](https://github.com/bhigy/espnet/blob/phoneme-repr/README.md) at the
root of the repository provides instructions for installation, while details about
activations extraction can be found under
[egs/librispeech/asr1/README.md](https://github.com/bhigy/espnet/blob/phoneme-repr/egs/librispeech/asr1/README.md). Alternativaly, the activations are also provided as part of the data you downloaded in the prerequisites, under `data/activations/transformer-asr`.

- Copy the activations you extracted or the ones we provide under `data/out/transformer-asr`.
- Create the input data for analyses:
```
python -c 'import prepare; prepare.prepare_transformer_asr()'
```
- Run analyses:
```
python -c 'import analyze; analyze.analyze_transformer_asr()'
```
- Plot main results (Figure 1)
```
python -c 'import analyze; analyze.plot_transformer_asr()'
