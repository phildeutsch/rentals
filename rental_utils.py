from collections import Counter
import geopandas as gpd
import itertools
import pandas as pd
from shapely.geometry import Point

from sklearn.feature_extraction import DictVectorizer


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


def add_variables(df, train_raw):
    df['description_length'] = [len(x) for x in df['description'].values]
    df['n_features'] = [len(x) for x in df['features'].values]
    df['n_photos'] = [len(x) for x in df['photos'].values]
    df['month'] = [x[5:7] for x in df['created'].values]

    return(df)


def vectorizer(varname, train):
    dv = DictVectorizer(sparse=False)

    df_in = pd.DataFrame(train[['County']])
    dv.fit(df_in.to_dict(orient='records'))

    return(dv)


def add_region(df):
    filename = "Data/ZillowNeighborhoods-NY/ZillowNeighborhoods-NY.shp"
    ny = gpd.read_file(filename)
    nyc = ny[ny['City'] == 'New York']
    nyc = nyc[['County', 'Name', 'RegionID', 'geometry']]

    geometry = [Point(xy) for xy in zip(df.longitude,
                                        df.latitude)]
    df = df.drop(['longitude', 'latitude'], axis=1)
    crs = nyc.crs
    geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    df_region = gpd.tools.sjoin(geo_df, nyc, how="left")
    df_region = pd.DataFrame(df_region)
    df_region.drop_duplicates('listing_id', inplace=True)
    del df_region['index_right']

    df_region.loc[pd.isnull(df_region['County']), 'County'] = 'None'

    return(df_region)
