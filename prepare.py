SEED=666
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import pickle
import logging
import platalea.asr as asr
import platalea.basic as basic
import platalea.dataset as dataset
import json
import os
from pathlib import Path


def prepare_rnn_vgs():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading pytorch models")
    net_rand = basic.SpeechImage(basic.DEFAULT_CONFIG).cuda()
    net_rand.eval()
    net_train = basic.SpeechImage(basic.DEFAULT_CONFIG)
    net_train.load_state_dict(torch.load("models/rnn-vgs/net.20.pt").state_dict())
    net_train.cuda()
    net_train.eval()
    nets = [('trained', net_train), ('random', net_rand)]
    with torch.no_grad():
        save_data(nets, directory='data/out/rnn-vgs', batch_size=32)


def prepare_rnn_asr():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading pytorch models")
    conf = pickle.load(open('models/rnn-asr/config.pkl', 'rb'))
    fd = dataset.Flickr8KData
    fd.le = conf['label_encoder']
    config = dict(
        SpeechEncoder=dict(
            conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                      padding=0, bias=False),
            rnn=dict(input_size=64, hidden_size=1024, num_layers=4,
                     bidirectional=False, dropout=0.0),
            rnn_layer_type=torch.nn.GRU),
        TextDecoder=dict(
            emb=dict(num_embeddings=fd.vocabulary_size(),
                     embedding_dim=1024),
            drop=dict(p=0.0),
            att=dict(in_size_enc=1024, in_size_state=1024,
                     hidden_size=1024),
            rnn=dict(input_size=1024 * 2, hidden_size=1024,
                     num_layers=1, dropout=0.0),
            out=dict(in_features=1024 * 2,
                     out_features=fd.vocabulary_size()),
            rnn_layer_type=torch.nn.GRU,
            max_output_length=400,  # max length for flickr annotations is 199
            sos_id=fd.get_token_id(fd.sos),
            eos_id=fd.get_token_id(fd.eos),
            pad_id=fd.get_token_id(fd.pad)),
        inverse_transform_fn=fd.get_label_encoder().inverse_transform)
    net_rand = asr.SpeechTranscriber(config).cuda()
    net_rand.eval()
    net_train = asr.SpeechTranscriber(config)
    net_train.load_state_dict(torch.load("models/rnn-asr/net.13.pt"))
    net_train.cuda()
    net_train.eval()
    nets = [('trained', net_train), ('random', net_rand)]
    with torch.no_grad():
        save_data(nets, directory='data/out/rnn-asr', batch_size=32)


def prepare_transformer_asr():
    logging.getLogger().setLevel('INFO')
    ds_fpath = "data/activations/transformer-asr/downsampling_factors.pkl"
    factors = pickle.load(open(ds_fpath, "rb"))
    #layers = factors.keys()
    logging.info("Loading input")
    global_input_in_fpath = "data/activations/transformer-asr/global_input.pkl"
    data = pickle.load(open(global_input_in_fpath, "rb"))
    #logging.info("Fixing IDs")
    #data['audio_id'] = np.array([ i + '.wav' for i in data['audio_id']])
    logging.info("Adding IPA")
    alignment = load_alignment("data/datasets/librispeech/fa.json")
    data['ipa'] = np.array([align2ipa(alignment[i]) for i in data['audio_id']])
    logging.info("Saving input")
    global_input_out_fpath = "data/out/transformer-asr/global_input.pkl"
    pickle.dump(data, open(global_input_out_fpath, "wb"), protocol=4)
    #for mode in ['trained', 'random']:
    #    for layer in layers:
    #        logging.info("Loading data for {} {}".format(mode, layer))
    #        fname = "global_{}.{}.pkl".format(mode, layer)
    #        fpath = "data/{}/transormer-asr/" + fname
    #        data = pickle.load(open(fpath.format('activations'), "rb"))
    #        data = {layer: data}
    #        logging.info("Saving data for {} {}".format(mode, layer))
    #        pickle.dump(data, open(fpath.format('out'), "wb"), protocol=4)


def vgs_factors():
    return {'conv': {'pad': 0, 'ksize': 6, 'stride': 2 },
                   'rnn0': None,
                   'rnn1': None,
                   'rnn2': None,
                   'rnn3': None }


def save_data(nets, directory, batch_size=32):
    Path(directory).mkdir(parents=True, exist_ok=True)
    save_global_data(nets, directory=directory, batch_size=batch_size) # FIXME adapt this per directory too
    save_local_data(directory=directory)

def save_global_data(nets, directory='.', batch_size=32):
    """Generate data for training a phoneme decoding model."""
    logging.info("Loading alignments")
    data = load_alignment("data/datasets/flickr8k/fa.json")
    logging.info("Loading audio features")
    val = dataset.Flickr8KData(root='data/datasets/flickr8k/', split='val',
                               feature_fname='mfcc_features.pt' )
    # Vocabulary should be initialized even if we are not going to use text data
    if dataset.Flickr8KData.le is None:
        dataset.Flickr8KData.init_vocabulary(val)
    #
    alignments = [ data[sent['audio_id']] for sent in val ]
    # Only consider cases where alignement does not fail
    alignments = [item for item in alignments if good_alignment(item) ]
    sentids = set(item['audio_id'] for item in alignments)
    audio = [ sent['audio'] for sent in val if sent['audio_id'] in sentids ]
    audio_np = [ a.numpy() for a in audio]

    ## Global data

    global_input = dict(audio_id = np.array([datum['audio_id'] for datum in alignments]),
                       ipa =      np.array([align2ipa(datum)  for datum in alignments]),
                       text =     np.array([datum['transcript'] for datum in alignments]),
                       audio = np.array(audio_np))
    global_input_path = Path(directory) / 'global_input.pkl'
    pickle.dump(global_input, open(global_input_path, "wb"), protocol=4)

    for mode, net in nets:
        global_act = collect_activations(net, audio, batch_size=batch_size)
        for layer in global_act:
            logging.info("Saving global data in {}/global_{}_{}.pkl".format(directory, mode, layer))
            pickle.dump({layer: global_act[layer]}, open("{}/global_{}_{}.pkl".format(directory, mode, layer), "wb"), protocol=4)

def filter_global_data(directory):
    """Remove sentences with OOV items from data."""
    logging.getLogger().setLevel('INFO')
    logging.info("Loading raw data")
    global_input =  pickle.load(open("{}/global_input_raw.pkl".format(directory), "rb"))
    logging.info("Loading alignments")
    adata = load_alignment("data/datasets/flickr8k/fa.json".format(directory))

    logging.info("Filtering out failed alignments and OOV")
    alignments = [ adata.get(i, adata.get(i+'.wav')) for i in global_input['audio_id'] ]
    # Only consider cases where alignement does not fail
    # Only consider cases with no OOV items
    alignments = [item for item in alignments if good_alignment(item) ]
    sentids = set(item['audio_id'] for item in alignments)
    ## Global data

    include = np.array([ i in sentids for i in global_input['audio_id'] ])

    #filtered = { key: np.array(value)[include] for key, value in global_input.items() }
    # Hack to fix broken format
    filtered = {}
    for key, value in global_input.items():
        if key == "audio":
            value = [ v.numpy() for v in value ]
        filtered[key] = np.array(value)[include]

    logging.info("Saving filtered data")
    pickle.dump(filtered, open("{}/global_input.pkl".format(directory), "wb"), protocol=4)

    for name in ['trained', 'random']:
        global_act = pickle.load(open("{}/global_{}_raw.pkl".format(directory, name), "rb"))
        filtered = { key: np.array(value)[include] for key, value in global_act.items() }
        logging.info("Saving filtered global data in global_{}.pkl".format(name))
        pickle.dump(filtered, open("{}/global_{}.pkl".format(directory, name), "wb"), protocol=4)

def good_alignment(item):
    for word in item['words']:
        if word['case'] != 'success' or word['alignedWord'] == '<unk>':
            return False
    return True

def make_indexer(factors, layer):
     def inout(pad, ksize, stride, L):
         return ((L + 2*pad - 1*(ksize-1) - 1) // stride + 1)
     def index(ms):
         t = ms//10
         for l in factors:
             if factors[l] is None:
                 pass
             else:
                 pad = factors[l]['pad']
                 ksize = factors[l]['ksize']
                 stride = factors[l]['stride']
                 t = inout(pad, ksize, stride, t)
             if l == layer:
                 break
         return t
     return index

def save_local_data(directory):
    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    global_input =  pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    adata = load_alignment("data/datasets/flickr8k/fa.json".format(directory))
    alignments = [ adata.get(i, adata.get(i+'.wav')) for i in global_input['audio_id'] ]
    ## Local data
    local_data = {}
    logging.info("Computing local data for MFCC")
    y, X = phoneme_activations(global_input['audio'], alignments, index=lambda ms: ms//10, framewise=True)
    local_input = check_nan(features=X, labels=y)
    pickle.dump(local_input, open("{}/local_input.pkl".format(directory), "wb"), protocol=4)
    try:
        factors = pickle.load(open("{}/downsampling_factors.pkl".format(directory), "rb"))
    except FileNotFoundError:
        # Default VGS settings
        factors = vgs_factors()
    for mode in ['trained', 'random']:
        for layer in factors.keys():
            if layer == "conv1":
                pass # This data is too big
            else:
                global_act = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
                local_act = {}
                index = make_indexer(factors, layer)
                logging.info("Computing local data for {}, {}".format(mode, layer))
                y, X = phoneme_activations(global_act[layer], alignments, index=index, framewise=True)
                local_act[layer] = check_nan(features=X, labels=y)
                logging.info("Saving local data in local_{}_{}.pkl".format(mode, layer))
                pickle.dump(local_act, open("{}/local_{}_{}.pkl".format(directory, mode, layer), "wb"), protocol=4)

def load_alignment(path):
    data = {}
    for line in open(path):
        item = json.loads(line)
        item['audio_id'] = os.path.basename(item['audiopath'])
        data[item['audio_id']] = item
    return data

def phoneme_activations(activations, alignments, index=lambda ms: ms//10, framewise=True):
    """Return array of phoneme labels and array of corresponding mean-pooled activation states."""
    labels = []
    states = []
    for activation, alignment in zip(activations, alignments):
        # extract phoneme labels and activations for current utterance
        if framewise:
            fr = list(frames(alignment, activation, index=index))
        else:
            fr = list(slices(alignment, activation, index=index))
        if len(fr) > 0:
            y, X = zip(*fr)
            y = np.array(y)
            X = np.stack(X)
            labels.append(y)
            states.append(X)
    return np.concatenate(labels), np.concatenate(states)

def align2ipa(datum):
    """Extract IPA transcription from alignment information for a sentence."""
    from ipa import arpa2ipa
    result = []
    for word in datum['words']:
        for phoneme in word['phones']:
            result.append(arpa2ipa(phoneme['phone'].split('_')[0], '_'))
    return ''.join(result)

def slices(utt, rep, index, aggregate=lambda x: x.mean(axis=0)):
    """Return sequence of slices associated with phoneme labels, given an
       alignment object `utt`, a representation array `rep`, and
       indexing function `index`, and an aggregating function\
       `aggregate`.
    """
    for phoneme in phones(utt):
        phone, start, end = phoneme
        assert index(start)<index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
        yield (phone, aggregate(rep[index(start):index(end)+1]))

def frames(utt, rep, index):
     """Return pair sequence of (phoneme label, frame), given an
        alignment object `utt`, a representation array `rep`, and
       indexing function `index`.
     """
     for phoneme in phones(utt):
         phone, start, end = phoneme
         assert index(start)<index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
         for j in range(index(start), index(end)+1):
             if j < rep.shape[0]:
                 yield (phone, rep[j])
             else:
                 logging.warning("Index out of bounds: {} {}".format(j, rep.shape))

def phones(utt):
    """Return sequence of phoneme labels associated with start and end
     time corresponding to the alignment JSON object `utt`.

    """
    for word in utt['words']:
        pos = word['start']
        for phone in word['phones']:
            start = pos
            end = pos + phone['duration']
            pos = end
            label = phone['phone'].split('_')[0]
            if label != 'oov':
                yield (label, int(start*1000), int(end*1000))

def check_nan(labels, features):
    # Get rid of NaNs
    ix = np.isnan(features.sum(axis=1))
    logging.info("Found {} NaNs".format(sum(ix)))
    X = features[~ix]
    y = labels[~ix]
    return dict(features=X, labels=y)

def collect_activations(net, audio, batch_size=32):
    data = torch.utils.data.DataLoader(dataset=audio,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=dataset.batch_audio)
    out = {}
    for au, l in data:
        act = net.SpeechEncoder.introspect(au.cuda(), l.cuda())
        for k in act:
            if k not in out:
                out[k] = []
            out[k]  += [ item.detach().cpu().numpy() for item in act[k] ]
    return { k: np.array(v) for k,v in out.items() }


def spec(d):
    if type(d) != type(dict()):
        return type(d)
    else:
        return { key:spec(val) for key, val in d.items() }
