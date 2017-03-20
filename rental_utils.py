import itertools
from collections import Counter


def get_features(n_features, train):
    fl = [[w.lower() for w in l] for l in train['features'].values]
    totals = Counter(i for i in list(itertools.chain.from_iterable(fl)))

    features = [x[0] for x in totals.most_common(n_features)]
    feature_names = [x.replace(' ', '_') for x in features]

    return [features, feature_names]


def add_features(df, features, feature_names):
    feature_list = [[w.lower() for w in l] for l in df['features'].values]
    for i in range(len(features)):
        df[feature_names[i]] = [features[i] in f for f in feature_list]
    return(df)


def add_variables(df):
    df['description_length'] = [len(x) for x in df['description'].values]
    df['n_features'] = [len(x) for x in df['features'].values]
    df['n_photos'] = [len(x) for x in df['photos'].values]
    df['month'] = [x[5:7] for x in df['created'].values]

    return(df)
