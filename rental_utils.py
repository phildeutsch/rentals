def add_features(df, features, feature_names):
    feature_list = [[w.lower() for w in l] for l in df['features'].values]
    for i in range(len(features)):
        df[feature_names[i]] = [features[i] in f for f in feature_list]
    return(df)
