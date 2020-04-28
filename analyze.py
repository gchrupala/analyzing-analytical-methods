SEED=666
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn
import platalea.attention
from pathlib import Path
import logging
logging.getLogger().setLevel('INFO')
import json

from plotnine import *
import pandas as pd
import ursa.similarity as S
import ursa.util as U
import pickle


def analyze_rnn_vgs():
    layers = ['conv'] + ['rnn{}'.format(i) for i in range(4)]
    analyze('data/out/rnn-vgs', layers)

    logging.info("Mean pooling; global RSA partial")
    config = dict(directory=' data/out/rnn-vgs',
                  output='data/out/rnn-vgs/mean',
                  epochs=60,
                  test_size=1/2,
                  layers=['conv'] + ['rnn{}'.format(i) for i in range(4)],
                  device='cpu'
                  )
    global_rsa_partial(config)


def analyze_rnn_asr():
    layers = ['conv'] + ['rnn{}'.format(i) for i in range(4)]
    analyze('data/out/rnn-asr', layers)


def analyze(output_root_dir, layers):
    output_dir = Path(output_root_dir) / 'attn'
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Attention pooling; global diagnostic")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  attention='linear',
                  standardize=True,
                  hidden_size=None,
                  attention_hidden_size=None,
                  epochs=500,
                  test_size=1/2,
                  layers=layers,
                  device='cuda',
                  runs=3)
    global_diagnostic(config)

    logging.info("Attention pooling; global RSA")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  attention='linear',
                  standardize=True,
                  attention_hidden_size=None,
                  epochs=60,
                  test_size=1/2,
                  layers=layers,
                  device='cuda',
                  runs=3)
    global_rsa(config)


    output_dir = Path(output_root_dir) / 'mean'
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Mean pooling; global diagnostic")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  attention='mean',
                  hidden_size=None,
                  attention_hidden_size=None,
                  epochs=500,
                  test_size=1/2,
                  layers=layers,
                  device='cuda',
                  runs=3)
    global_diagnostic(config)

    logging.info("Mean pooling; global RSA")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  attention='mean',
                  epochs=60,
                  test_size=1/2,
                  layers=layers,
                  device='cpu',
                  runs=1)
    global_rsa(config)


    output_dir = Path(output_root_dir) / 'local'
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Local diagnostic")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  hidden=None,
                  epochs=40,
                  layers=layers,
                  runs=3)
    local_diagnostic(config)

    logging.info("Local RSA")
    config = dict(directory=output_root_dir,
                  output=output_dir,
                  size=793964 // 2,
                  layers=layers,
                  matrix=False,
                  runs=1)
    local_rsa(config)


def analyze_transformer_asr():
    layers = ['conv'] + ['rnn{}'.format(i) for i in range(4)]
    analyze('data/out/transformer-asr', layers)


## Models
### Local

def local_diagnostic(config):
    directory = config['directory']
    out = config["output"] / "local_diagnostic.json"
    del config['output']
    runs = range(1, config['runs']+1)
    del config['runs']
    output = []
    for run in runs:
        logging.info("Starting run {}".format(run))
        data_mfcc = pickle.load(open('{}/local_input.pkl'.format(directory), 'rb'))
        for mode in ['random', 'trained']:
            logging.info("Fitting local classifier for mfcc")
            result  = local_classifier(data_mfcc['features'], data_mfcc['labels'], epochs=config['epochs'], device='cuda:0', hidden=config['hidden'])
            logging.info("Result for {}, {} = {}".format(mode, 'mfcc', result['acc']))
            result['model'] = mode
            result['layer'] = 'mfcc'
            result['run'] = run
            output.append(result)
            for layer in config['layers']:
                data = pickle.load(open('{}/local_{}_{}.pkl'.format(directory, mode, layer), 'rb'))
                logging.info("Fitting local classifier for {}, {}".format(mode, layer))
                result = local_classifier(data[layer]['features'], data[layer]['labels'], epochs=config['epochs'], device='cuda:0', hidden=config['hidden'])
                logging.info("Result for {}, {} = {}".format(mode, layer, result['acc']))
                result['model'] = mode
                result['layer'] = layer
                result['run'] = run
                output.append(result)
    logging.info("Writing {}".format(out))
    json.dump(output, open(out, "w"), indent=True)

def local_rsa(config):
    out = config['output'] / "local_rsa.json"
    del config['output']
    del config['runs']
    if config['matrix']:
        raise NotImplementedError
        #result = framewise_RSA_matrix(directory, layers=config['layers'], size=config['size'])
    else:
        del config['matrix']
        result = framewise_RSA(**config)
    logging.info("Writing {}".format(out))
    json.dump(result, open(out, 'w'), indent=2)

### Global

def global_rsa(config):
    out = config['output'] / 'global_rsa.json'
    del config['output']
    result = []
    runs = range(1, config['runs']+1)
    del config['runs']
    for run in runs:
        result += inject(weighted_average_RSA(**config), {'run': run})
    logging.info("Writing {}".format(out))
    json.dump(result, open(out, 'w'), indent=2)

def global_rsa_partial(config):
    out = config['output'] / 'global_rsa_partial.json'
    del config['output']
    del config['runs']
    result = weighted_average_RSA_partial(**config)
    json.dump(result, open(out, 'w'), indent=2)

def global_diagnostic(config):
    out = config['output'] / 'global_diagnostic.json'
    del config['output']
    result = []
    runs = range(1, config['runs']+1)
    del config['runs']
    for run in runs:
        logging.info("Starting run {}".format(run))
        result += inject(weighted_average_diagnostic(**config), {'run':run})
    logging.info("Writing {}".format(out))
    json.dump(result, open(out, 'w') , indent=2)

def plots():
    local_diagnostic_plot()
    global_diagnostic_plot()
    local_rsa_plot()
    global_rsa_plot()

## Plotting
def local_diagnostic_plot(directory='.'):
    data = pd.read_json("{}/local_diagnostic.json".format(directory), orient='records')
    order = list(data['layer'].unique())
    data['layer_id'] = [ order.index(x) for x in data['layer'] ]
    data['rer'] = rer(data['acc'], data['baseline'])
    g = ggplot(data, aes(x='layer_id', y='rer', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0, 1) + ggtitle("Local diagnostic")
    ggsave(g, 'local_diagnostic.png')

def global_diagnostic_plot(directory='.'):
    data = pd.read_json("{}/global_diagnostic.json".format(directory), orient='records')
    order = list(data['layer'].unique())
    data['layer_id'] = [ order.index(x) for x in data['layer'] ]
    data['rer'] = rer(data['acc'], data['baseline'])
    g = ggplot(data, aes(x='layer_id', y='rer', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(-0.1, 1) + ggtitle("Global diagnostic")
    ggsave(g, "global_diagnostic.png")

def local_rsa_plot(directory='.'):
    data = pd.read_json("{}/local_rsa.json".format(directory), orient='records')
    order = list(data['layer'].unique())
    data['layer_id'] = [ order.index(x) for x in data['layer'] ]
    g = ggplot(data, aes(x='layer_id', y='cor', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0, 1) + ggtitle("Local RSA")
    ggsave(g, 'local_rsa.png')

def global_rsa_plot(directory='.'):
    data = pd.read_json("{}/global_rsa.json".format(directory), orient='records')
    order = list(data['layer'].unique())
    data['layer_id'] = [ order.index(x) for x in data['layer'] ]
    g = ggplot(data, aes(x='layer_id', y='cor', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(-0.1, 1) + ggtitle("Global RSA")
    ggsave(g, "global_rsa.png")

def plot_pooled_feature_std():
    path = 'data/out/rnn-vgs/'
    layers=['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data = pd.DataFrame()
    for model in ['trained', 'random']:
        layer = 'mfcc'
        act = pickle.load(open("{}/global_input.pkl".format(path), "rb"))['audio']
        act_avg = np.stack([x.mean(axis=0) for x in act])
        rows=pd.DataFrame(data=dict(std=act_avg.std(axis=1), model=np.repeat(model, len(act_avg)), layer=np.repeat(layer, len(act_avg))))
        data = data.append(rows)
        for layer in layers:
            act = pickle.load(open("{}/global_{}_{}.pkl".format(path, model, layer), "rb"))[layer]
            act_avg = np.stack([x.mean(axis=0) for x in act])
            rows=pd.DataFrame(data=dict(std=act_avg.std(axis=1), model=np.repeat(model, len(act_avg)), layer=np.repeat(layer, len(act_avg))))
            data = data.append(rows)
    order = list(data['layer'].unique())
    data['layer_id'] = [ order.index(x) for x in data['layer'] ]
    # Only plot RNN layers
    data = data[data['layer'].str.startswith('rnn')]
    z = data.groupby(['layer_id', 'model']).mean().reset_index()
    g = ggplot(data, aes(x='layer_id', y='std', color='model')) + \
                            geom_point(alpha=0.1, size=2, position='jitter', fill='white') +  \
                            geom_line(data=z, size=2, alpha=1.0) + \
                            ylab("Standard deviation") +\
                            xlab("layer id")
    ggsave(g, 'fig/rnn-vgs/pooled_feature_std.png')

## Tables
def learning_effect():
    def le(trained, random):
        return 1 - (random/trained)
    ld = pd.read_json("local_diagnostic.json", orient="records");    ld['scope'] = 'local';   ld['method'] = 'diagnostic'
    lr = pd.read_json("local_rsa.json", orient="records");           lr['scope'] = 'local';   lr['method'] = 'rsa'
    gd = pd.read_json("global_diagnostic.json", orient="records");   gd['scope'] = 'global';  gd['method'] = 'diagnostic'
    gr = pd.read_json("global_rsa.json", orient="records");          gr['scope'] = 'global';  gr['method'] = 'rsa'
    data = pd.concat([ld, lr, gd, gr], sort=False)
    data['rer'] = rer(data['acc'], data['baseline'])
    data['score'] = data['rer'].fillna(0.0) + data['cor'].fillna(0.0)
    trained = data.loc[data['model']=='trained']
    random  = data.loc[data['model']=='random']
    trained['learning_effect'] = le(trained['score'].values, random['score'].values)
    data = trained[['epoch', 'layer', 'method', 'scope', 'score', 'learning_effect']].reset_index()
    order = list(data['layer'].unique())
    data['layer_id'] = [ order.index(x) for x in data['layer'] ]
    json.dump(data.to_dict(orient='records'), open('learning_effect.json', 'w'))
    g = ggplot(data, aes(x='layer_id', y='learning_effect', color='method', linetype='scope')) + geom_point(size=2) + geom_line(size=1)
    ggsave(g, "learning_effect.png")

def majority_binary(y):
    return (y.mean(dim=0) >= 0.5).float()

def majority_multiclass(y):
    labels, counts = np.unique(y, return_counts=True)
    return labels[counts.argmax()]

def local_classifier(features, labels, test_size=1/2, epochs=1, device='cpu', hidden=None, weight_decay=0.0):
    """Fit classifier on part of features and labels and return accuracy on the other part."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split

    splitseed = random.randint(0, 1024)

    X, X_val, y, y_val = train_test_split(features, labels, test_size=test_size, random_state=splitseed)

    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(y)).long()
    y_val = torch.tensor(le.transform(y_val)).long()

    scaler = StandardScaler()
    X = torch.tensor(scaler.fit_transform(X)).float()
    X_val  = torch.tensor(scaler.transform(X_val)).float()
    logging.info("Setting up model on {}".format(device))
    if hidden is None:
        model = SoftmaxClassifier(X.size(1), y.max().item()+1, weight_decay=weight_decay).to(device)
    else:
        model = MLP(X.size(1), y.max().item()+1, hidden_size=hidden).to(device)
    result = train_classifier(model, X, y, X_val, y_val, epochs=epochs, majority=majority_multiclass)
    return result


def weight_variance():
    kinds = ['rnn0', 'rnn1', 'rnn2', 'rnn3']
    w = []
    layer = []
    trained = []
    for kind in kinds:
        rand = np.load("logreg_w_random_{}.npy".format(kind)).flatten()
        train = np.load("logreg_w_trained_{}.npy".format(kind)).flatten()
        w.append(rand)
        w.append(train)
        for _ in rand:
            layer.append(kind)
            trained.append('random')
        for _ in train:
            layer.append(kind)
            trained.append('trained')
        print(kind, "random", np.var(rand))
        print(kind, "trained", np.var(train))
    data = pd.DataFrame(dict(w = np.concatenate(w), layer=np.array(layer), trained=np.array(trained)))
    #g = ggplot(data, aes(y='w', x='layer')) + geom_violin() + facet_wrap('~trained', nrow=2, scales="free_y")
    g = ggplot(data, aes(y='w', x='layer')) + geom_point(position='jitter', alpha=0.1) + facet_wrap('~trained', nrow=2, scales="free_y")
    ggsave(g, 'weight_variance.png')


def framewise_RSA_matrix(directory, layers, size=70000):

    from sklearn.model_selection import train_test_split
    splitseed = random.randint(0, 1024)
    result = []
    mfcc_done = False
    data = pickle.load(open("{}/local_input.pkl".format(directory), "rb"))
    for mode in ["trained", "random"]:
        mfcc_cor = [ item['cor']  for item in result if item['layer'] == 'mfcc']
        if len(mfcc_cor) > 0:
            logging.info("Result for MFCC computed previously")
            result.append(dict(model=mode, layer='mfcc', cor=mfcc_cor[0]))
        else:

            X, X_val, y, y_val = train_test_split(data['features'], data['labels'], test_size=size, random_state=splitseed)
            logging.info("Computing label identity matrix for {} datapoints".format(len(y_val)))
            y_val_sim = torch.tensor(y_val.reshape((-1, 1)) == y_val).float()
            logging.info("Computing activation similarities for {} datapoints".format(len(X_val)))
            X_val = torch.tensor(X_val).float()
            X_val_sim = S.cosine_matrix(X_val, X_val)
            cor = S.pearson_r(S.triu(y_val_sim), S.triu(X_val_sim)).item()
            logging.info("Point biserial correlation for {}, mfcc: {}".format(mode, cor))
            result.append(dict(model=mode, layer='mfcc', cor=cor))
        for layer in layers:
            logging.info("Loading phoneme data for {} {}".format(mode, layer))
            data = pickle.load(open("{}/local_{}_{}.pkl".format(directory, mode, layer), "rb"))
            X, X_val, y, y_val = train_test_split(data[layer]['features'], data[layer]['labels'], test_size=size, random_state=splitseed)
            logging.info("Computing label identity matrix for {} datapoints".format(len(y_val)))
            y_val_sim = torch.tensor(y_val.reshape((-1, 1)) == y_val).float()
            logging.info("Computing activation similarities for {} datapoints".format(len(X_val)))
            X_val = torch.tensor(X_val).float()
            X_val_sim = S.cosine_matrix(X_val, X_val)
            cor = S.pearson_r(S.triu(y_val_sim), S.triu(X_val_sim)).item()
            logging.info("Point biserial correlation for {}, {}: {}".format(mode, layer, cor))
            result.append(dict(model=mode, layer=layer, cor=cor))
    return result

def framewise_RSA(directory='.', layers=[], size=70000):
    result = []
    mfcc_done = False
    data = pickle.load(open("{}/local_input.pkl".format(directory), "rb"))
    for mode in ["trained", "random"]:
        mfcc_cor = [ item['cor']  for item in result if item['layer'] == 'mfcc']
        if len(mfcc_cor) > 0:
            logging.info("Result for MFCC computed previously")
            result.append(dict(model=mode, layer='mfcc', cor=mfcc_cor[0]))
        else:
            cor = correlation_score(data['features'], data['labels'], size=size)
            logging.info("Point biserial correlation for {}, mfcc: {}".format(mode, cor))
            result.append(dict(model=mode, layer='mfcc', cor=cor))
        for layer in layers:
            logging.info("Loading phoneme data for {} {}".format(mode, layer))
            data = pickle.load(open("{}/local_{}_{}.pkl".format(directory, mode, layer), "rb"))
            cor = correlation_score(data[layer]['features'], data[layer]['labels'], size=size)
            logging.info("Point biserial correlation for {}, {}: {}".format(mode, layer, cor))
            result.append(dict(model=mode, layer=layer, cor=cor))
    return result

def correlation_score(features, labels, size):
    from sklearn.metrics.pairwise import paired_cosine_distances
    from scipy.stats import pearsonr
    logging.info("Sampling 2x{} stimuli from a total of {}".format(size, len(labels)))
    indices = np.array(random.sample(range(len(labels)), size*2))
    y = labels[indices]
    x = features[indices]
    y_sim = y[: size] == y[size :]
    x_sim = 1 - paired_cosine_distances(x[: size], x[size :])
    return pearsonr(x_sim, y_sim)[0]


def weighted_average_RSA(directory='.', layers=[], attention='linear', test_size=1/2,  attention_hidden_size=None, standardize=False, epochs=1, device='cpu'):
    from sklearn.model_selection import train_test_split
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = random.randint(0, 1024)
    result = []
    logging.info("Loading transcription data")
    data = pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    trans = data['ipa']
    act = [ torch.tensor([item[:, :]]).float().to(device) for item in data['audio'] ]

    trans, trans_val, act, act_val = train_test_split(trans, act, test_size=test_size, random_state=splitseed)
    if standardize:
        logging.info("Standardizing data")
        act, act_val = normalize(act, act_val)
    logging.info("Computing edit distances")
    edit_sim = torch.tensor(U.pairwise(S.stringsim, trans)).float().to(device)
    edit_sim_val = torch.tensor(U.pairwise(S.stringsim, trans_val)).float().to(device)
    logging.info("Training for input features")
    this = train_wa(edit_sim, edit_sim_val, act, act_val, attention=attention, attention_hidden_size=None, epochs=epochs, device=device)
    result.append({**this, 'model': 'random', 'layer': 'mfcc'})
    result.append({**this, 'model': 'trained', 'layer': 'mfcc'})
    del act, act_val
    logging.info("Maximum correlation on val: {} at epoch {}".format(result[-1]['cor'], result[-1]['epoch']))
    for mode in ["trained", "random"]:
        for layer in layers:
            logging.info("Loading activations for {} {}".format(mode, layer))
            data = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
            logging.info("Training for {} {}".format(mode, layer))
            act = [ torch.tensor([item[:, :]]).float().to(device) for item in data[layer] ]
            act, act_val = train_test_split(act, test_size=test_size, random_state=splitseed)
            if standardize:
                logging.info("Standardizing data")
                act, act_val = normalize(act, act_val)
            this = train_wa(edit_sim, edit_sim_val, act, act_val, attention=attention, attention_hidden_size=None, epochs=epochs, device=device)
            result.append({**this, 'model': mode, 'layer': layer})
            del act, act_val
            logging.info("Maximum correlation on val: {} at epoch {}".format(result[-1]['cor'], result[-1]['epoch']))
    return result


def train_wa(edit_sim, edit_sim_val, stack, stack_val, attention='scalar', attention_hidden_size=None, epochs=1, device='cpu'):
    if attention == 'scalar':
        wa = platalea.attention.ScalarAttention(stack[0].size(2), hidden_size).to(device)
    elif attention == 'linear':
        wa = platalea.attention.LinearAttention(stack[0].size(2)).to(device)
    elif attention == 'mean':
        wa = platalea.attention.MeanPool().to(device)
        avg_pool_val = torch.cat([ wa(item) for item in stack_val])
        avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
        cor_val = S.pearson_r(S.triu(avg_pool_sim_val), S.triu(edit_sim_val))
        return {'epoch': None, 'cor': cor_val.item() }

    else:
        wa = platalea.attention.Attention(stack[0].size(2), attention_hidden_size).to(device)

    optim = torch.optim.Adam(wa.parameters())
    minloss = 0; minepoch = None
    logging.info("Optimizing for {} epochs".format(epochs))
    for epoch in range(1, 1+epochs):
        avg_pool = torch.cat([ wa(item) for item in stack])
        avg_pool_sim = S.cosine_matrix(avg_pool, avg_pool)
        loss = -S.pearson_r(S.triu(avg_pool_sim), S.triu(edit_sim))
        with torch.no_grad():
            avg_pool_val = torch.cat([ wa(item) for item in stack_val])
            avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
            loss_val = -S.pearson_r(S.triu(avg_pool_sim_val), S.triu(edit_sim_val))
        logging.info("{} {} {}".format(epoch, -loss.item(), -loss_val.item()))
        if loss_val.item() <= minloss:
            minloss = loss_val.item()
            minepoch = epoch
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Release CUDA-allocated tensors
        del loss, loss_val,  avg_pool, avg_pool_sim, avg_pool_val, avg_pool_sim_val
    del wa, optim
    return {'epoch': minepoch, 'cor': -minloss}

def weighted_average_RSA_partial(directory='.', layers=[], test_size=1/2,  standardize=False, epochs=1, device='cpu'):
    from sklearn.model_selection import train_test_split
    from platalea.dataset import Flickr8KData
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = random.randint(0, 1024)
    result = []
    logging.info("Loading transcription data")
    data = pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    trans = data['ipa']
    act = [ torch.tensor([item[:, :]]).float().to(device) for item in data['audio'] ]
    val = Flickr8KData(root='/roaming/gchrupal/datasets/flickr8k/', split='val')
    image_map = { item['audio_id']: item['image'] for item in val }
    image = np.stack([ image_map[item] for item in data['audio_id'] ])

    trans, trans_val, act, act_val, image, image_val = train_test_split(trans, act, image, test_size=test_size, random_state=splitseed)
    if standardize:
        logging.info("Standardizing data")
        act, act_val = normalize(act, act_val)
    logging.info("Computing edit distances")
    edit_sim = torch.tensor(U.pairwise(S.stringsim, trans)).float().to(device)
    edit_sim_val = torch.tensor(U.pairwise(S.stringsim, trans_val)).float().to(device)
    logging.info("Computing image similarities")
    image = torch.tensor(image).float()
    image_val = torch.tensor(image_val).float()
    sim_image = S.cosine_matrix(image, image)
    sim_image_val = S.cosine_matrix(image_val, image_val)

    logging.info("Computing partial correlation for input features (mean pooling)")
    wa = platalea.attention.MeanPool().to(device)
    avg_pool = torch.cat([ wa(item) for item in act])
    avg_pool_sim = S.cosine_matrix(avg_pool, avg_pool)
    avg_pool_val = torch.cat([ wa(item) for item in act_val])
    avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
    # Training data
    #  Edit ~ Act + Image
    Edit  = S.triu(edit_sim).cpu().numpy()
    Image = S.triu(sim_image).cpu().numpy()
    Act   = S.triu(avg_pool_sim).cpu().numpy()
    # Val data
    Edit_val  = S.triu(edit_sim_val).cpu().numpy()
    Image_val = S.triu(sim_image_val).cpu().numpy()
    Act_val   = S.triu(avg_pool_sim_val).cpu().numpy()
    e_full, e_base, e_mean = partial_r2(Edit, Act, Image, Edit_val, Act_val, Image_val)
    logging.info("Full, base, mean error: {} {}".format(e_full, e_base, e_mean))
    r2 =  (e_base - e_full)/e_base
    this =  {'epoch': None, 'error': e_full, 'baseline': e_base, 'error_mean': e_mean, 'r2': r2  }

    #this = train_wa(edit_sim, edit_sim_val, act, act_val, attention=attention, attention_hidden_size=None, epochs=epochs, device=device)
    result.append({**this, 'model': 'random', 'layer': 'mfcc'})
    result.append({**this, 'model': 'trained', 'layer': 'mfcc'})
    del act, act_val
    logging.info("Partial r2 on val: {} at epoch {}".format(result[-1]['r2'], result[-1]['epoch']))
    for mode in ["trained", "random"]:
        for layer in layers:
            logging.info("Loading activations for {} {}".format(mode, layer))
            data = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
            logging.info("Training for {} {}".format(mode, layer))
            act = [ torch.tensor([item[:, :]]).float().to(device) for item in data[layer] ]
            act, act_val = train_test_split(act, test_size=test_size, random_state=splitseed)
            if standardize:
                logging.info("Standardizing data")
                act, act_val = normalize(act, act_val)
            avg_pool = torch.cat([ wa(item) for item in act])
            avg_pool_sim = S.cosine_matrix(avg_pool, avg_pool)
            avg_pool_val = torch.cat([ wa(item) for item in act_val])
            avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
            Act   = S.triu(avg_pool_sim).cpu().numpy()
            Act_val   = S.triu(avg_pool_sim_val).cpu().numpy()
            e_full, e_base, e_mean = partial_r2(Edit, Act, Image, Edit_val, Act_val, Image_val)
            logging.info("Full, base, mean error: {} {}".format(e_full, e_base, e_mean))
            r2 =  (e_base - e_full)/e_base
            this =  {'epoch': None, 'error': e_full, 'baseline': e_base, 'error_mean': e_mean, 'r2': r2  }
            pickle.dump(dict(Edit=Edit, Act=Act, Image=Image, Edit_val=Edit_val, Act_val=Act_val, Image_val=Image_val), open("fufi_{}_{}.pkl".format(mode, layer), "wb"), protocol=4)
            result.append({**this, 'model': mode, 'layer': layer})
            del act, act_val
            logging.info("Partial R2 on val: {} at epoch {}".format(result[-1]['r2'], result[-1]['epoch']))
    return result

def partial_r2(Y, X, Z, y, x, z):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    full = LinearRegression()
    full.fit(np.stack([X, Z], axis=1), Y)
    base = LinearRegression()
    base.fit(Z.reshape((-1, 1)), Y)
    y_full = full.predict(np.stack([x, z], axis=1))
    y_base = base.predict(z.reshape((-1, 1)))
    e_full = mean_squared_error(y, y_full)
    e_base = mean_squared_error(y, y_base)
    e_mean = mean_squared_error(y, np.repeat(Y.mean().item(), len(y)))
    r_full = pearsonr(y, y_full)[0]
    r_base = pearsonr(y, y_base)[0]
    logging.info("Pearson's r full : {}".format(r_full))
    logging.info("Pearson's r base : {}".format(r_base))
    logging.info("Pearson's partial: {}".format(rer(r_full, r_base)))
    return e_full.item(), e_base.item(), e_mean.item()

def normalize(X, X_val):
    device = X[0].device
    X = [x.cpu() for x in X ]
    X_val = [x.cpu() for x in X_val]
    d = X[0].shape[-1]
    flat = torch.cat([ x.view(-1, d) for x in X])
    mu = flat.mean(dim=0)
    sigma = flat.std(dim=0)
    X_norm = [ (item - mu) / sigma for item in X]
    X_val_norm = [ (item - mu) /sigma for item in X_val ]
    return [x.to(device) for x in X_norm], [x.to(device) for x in X_val_norm ]


def weighted_average_diagnostic(directory='.', layers=[], attention='scalar', test_size=1/2, attention_hidden_size=None, hidden_size=None, standardize=False, epochs=1, factor=0.1, device='cpu'):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = random.randint(0, 1024)
    result = []
    logging.info("Loading transcription data")
    data = pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    trans = data['ipa']
    act = [ torch.tensor(item[:, :]).float().to(device) for item in data['audio'] ]

    trans, trans_val, X, X_val = train_test_split(trans, act, test_size=test_size, random_state=splitseed)
    if standardize:
        logging.info("Standardizing data")
        X, X_val = normalize(X, X_val)
    logging.info("Computing targets")
    vec = CountVectorizer(lowercase=False, analyzer='char')
    # Binary instead of counts
    y = torch.tensor(vec.fit_transform(trans).toarray()).float().clamp(min=0, max=1)
    y_val = torch.tensor(vec.transform(trans_val).toarray()).float().clamp(min=0, max=1)
    logging.info("Training for input features")
    model = PooledClassifier(input_size=X[0].shape[1],  output_size=y[0].shape[0],
                             hidden_size=hidden_size, attention_hidden_size=attention_hidden_size, attention=attention).to(device)
    this = train_classifier(model, X, y, X_val, y_val, epochs=epochs, factor=factor)
    result.append({**this, 'model': 'random', 'layer': 'mfcc'})
    result.append({**this, 'model': 'trained', 'layer': 'mfcc'})
    del X, X_val
    logging.info("Maximum accuracy on val: {} at epoch {}".format(result[-1]['acc'], result[-1]['epoch']))
    for mode in ["trained", "random"]:
        for layer in layers:
            logging.info("Loading activations for {} {}".format(mode, layer))
            data = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
            logging.info("Training for {} {}".format(mode, layer))
            act = [ torch.tensor(item[:, :]).float() for item in data[layer] ]
            X, X_val = train_test_split(act, test_size=test_size, random_state=splitseed)
            if standardize:
                logging.info("Standardizing data")
                X, X_val = normalize(X, X_val)
            model = PooledClassifier(input_size=X[0].shape[1], output_size=y[0].shape[0],
                                     hidden_size=hidden_size, attention_hidden_size=attention_hidden_size, attention=attention).to(device)
            this = train_classifier(model, X, y, X_val, y_val, epochs=epochs, factor=factor)
            result.append({**this, 'model': mode, 'layer': layer})
            del X, X_val
            logging.info("Maximum accuracy on val: {} at epoch {}".format(result[-1]['acc'], result[-1]['epoch']))
    return result

class PooledClassifier(nn.Module):

    def __init__(self, input_size, output_size, attention_hidden_size=1024, hidden_size=None, attention='scalar', weight_decay=0.0):
        super(PooledClassifier, self).__init__()
        if attention == 'scalar':
            self.wa = platalea.attention.ScalarAttention(input_size, attention_hidden_size)
        elif attention == 'linear':
            self.wa = platalea.attention.LinearAttention(input_size)
        elif attention == 'mean':
            self.wa = platalea.attention.MeanPool()
        else:
            self.wa = platalea.attention.Attention(input_size, attention_hidden_size)
        if hidden_size is None:
            self.project = nn.Linear(in_features=input_size, out_features=output_size)
        else:
            self.project = MLP(input_size, output_size, hidden_size=hidden_size)
        self.loss = nn.functional.binary_cross_entropy_with_logits
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.project(self.wa(x))

    def predict(self, x):
        logit = self.project(self.wa(x))
        return (logit >= 0.0).float()


class SoftmaxClassifier(nn.Module):

    def __init__(self, input_size, output_size, weight_decay=0.0):
        super(SoftmaxClassifier, self).__init__()
        self.project = nn.Linear(in_features=input_size, out_features=output_size)
        self.loss = nn.functional.cross_entropy
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.project(x)

    def predict(self, x):
        return self.project(x).argmax(dim=1)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=500, weight_decay=0.0):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.Dropout = nn.Dropout(p=0.5)
        self.h2o = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.loss = nn.functional.cross_entropy
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.h2o(torch.relu(self.Dropout(self.i2h(x))))

    def predict(self, x):
        return self.forward(x).argmax(dim=1)


def collate(items):
    x, y = zip(*items)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x, y

def tuple_stack(xy):
    x, y = zip(*xy)
    return torch.stack(x), torch.stack(y)

def rer(hi, lo):
    return ((1-lo) - (1-hi))/(1-lo)

def train_classifier(model, X, y, X_val, y_val, epochs=1, patience=50, factor=0.1, majority=majority_binary):
    device = list(model.parameters())[0].device
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=model.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=factor, patience=10)
    data = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=64, shuffle=True, collate_fn=collate)
    data_val = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=64, shuffle=False, collate_fn=collate)
    logging.info("Optimizing for {} epochs".format(epochs))
    scores = []
    with torch.no_grad():
        model.eval()
        maj = majority(y)
        baseline = np.mean([ (maj == y_i).cpu().numpy() for y_i in y_val ])
        logging.info("Baseline accuracy: {}".format(baseline))
    for epoch in range(1, 1+epochs):
        model.train()
        epoch_loss = []
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = model.loss(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())
        with torch.no_grad():
            model.eval()
            loss_val = np.mean( [model.loss(model(x.to(device)), y.to(device)).item() for x,y in data_val])
            accuracy_val = np.concatenate([ model.predict(x.to(device)).cpu().numpy() == y.cpu().numpy() for x, y in data_val]).mean()
            #scheduler.step(loss_val)
            scheduler.step(1-accuracy_val)
            logging.info("{} {} {} {} {}".format(epoch, optim.state_dict()['param_groups'][0]['lr'], np.mean(epoch_loss), loss_val, accuracy_val))
        scores.append(dict(epoch=epoch, train_loss=np.mean(epoch_loss), acc=accuracy_val, loss=loss_val, baseline=baseline))
        minepoch = max(scores, key=lambda a: a['acc'])['epoch']
        if epoch - minepoch >= patience:
            logging.info("No improvement for {} epochs, stopping.".format(patience))
            break
        # Release CUDA-allocated tensors
        del x, y, loss, loss_val,  y_pred
    del model, optim
    return max(scores, key=lambda a: a['acc'])

# PLOTTING
import pandas as pd
from plotnine import *
from plotnine.options import figure_size


def plot_rnn_vgs():
    Path('fig/rnn-vgs').mkdir(parents=True, exist_ok=True)
    plot(path="data/out/rnn-vgs", output="fig/rnn-vgs")


def plot_rnn_asr():
    Path('fig/rnn-asr').mkdir(parents=True, exist_ok=True)
    plot(path="data/out/rnn-asr", output="fig/rnn-asr")


def plot_transformer_asr():
    Path('fig/transformer-asr').mkdir(parents=True, exist_ok=True)
    plot(path="data/out/transformer-asr", output="fig/transformer-asr")


def plot(path, output):
    ld = pd.read_json("{}/local/local_diagnostic.json".format(path), orient="records");   ld['scope'] = 'local';      ld['method'] = 'diagnostic'
    lr = pd.read_json("{}/local/local_rsa.json".format(path), orient="records");          lr['scope'] = 'local';      lr['method'] = 'rsa'
    gd = pd.read_json("{}/mean/global_diagnostic.json".format(path), orient="records");   gd['scope'] = 'mean pool';  gd['method'] = 'diagnostic'
    gr = pd.read_json("{}/mean/global_rsa.json".format(path), orient="records");          gr['scope'] = 'mean pool';  gr['method'] = 'rsa'
    ad = pd.read_json("{}/attn/global_diagnostic.json".format(path), orient="records");   ad['scope'] = 'attn pool';  ad['method'] = 'diagnostic'
    ar = pd.read_json("{}/attn/global_rsa.json".format(path), orient="records");          ar['scope'] = 'attn pool';  ar['method'] = 'rsa'
    data = pd.concat([ld, lr, gd, gr, ad, ar], sort=False)

    data['rer'] = rer(data['acc'], data['baseline'])
    data['score'] = data['rer'].fillna(0.0) + data['cor'].fillna(0.0)

    order = list(data['layer'].unique())
    data['layer id'] = [ order.index(x) for x in data['layer'] ]
    # Reorder scope
    data['scope'] = pd.Categorical(data['scope'], categories=['local', 'mean pool', 'attn pool'])
    # Reorder model
    data['model'] = pd.Categorical(data['model'], categories=['trained', 'random'])
    # Make variable to group model x run interaction for plotting multiple runs.
    data['modelxrun'] = data.apply(lambda x: "{} {}".format(x['model'], x['run']), axis=1)
    # g = ggplot(data, aes(x='layer id', y='score', color='model', linetype='model', shape='model')) + geom_point() + geom_line() + \
    g = ggplot(data, aes(x='layer id', y='score', color='model', linetype='model', shape='model')) + geom_point() +  geom_line(aes(group='modelxrun')) + \
                            facet_wrap('~ method + scope') + \
                            theme(figure_size=(figure_size[0]*1.5, figure_size[1]*1.5))
    ggsave(g, '{}/plot.png'.format(output))

def rer(hi, lo):
    return ((1-lo) - (1-hi))/(1-lo)

def plot_r2_partial():
    path = 'data/out/rnn-vgs/mean/'
    data = pd.read_json("{}/global_rsa_partial.json".format(path), orient="records")
    order = list(data['layer'].unique())
    data['layer id'] = [ order.index(x) for x in data['layer'] ]
    data['partial R²'] = (data['baseline'] - data['error']) / data['baseline']
    data['cor'] = abs(data['partial R²'])**0.5
    # Reorder model
    data['model'] = pd.Categorical(data['model'], categories=['trained', 'random'])
    g = ggplot(data, aes(x='layer id', y='cor', color='model', linetype='model', shape='model')) + geom_point() + geom_line() +\
                            ylab("√R² (partial)")
    ggsave(g, 'fig/rnn-vgs/r2_partial.png')

def partialing(path='.'):
    data = pd.read_json("{}/global_rsa_partial.json".format(path), orient="records")
    order = list(data['layer'].unique())
    data['layer id'] = [ order.index(x) for x in data['layer'] ]
    data['partial R²'] = (data['baseline'] - data['error']) / data['baseline']
    # Reorder model
    data['model'] = pd.Categorical(data['model'], categories=['trained', 'random'])
    #
    pass

    ggsave(g, 'partialing.png')


def inject(x, e):
    return [ {**xi, **e} for xi in x ]

