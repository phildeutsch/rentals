from collections import Counter
import geopandas as gpd
import itertools
import pandas as pd
from shapely.geometry import Point
from sklearn.feature_extraction import DictVectorizer
import datetime


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


def clean(df):
    df['rooms'] = df['bedrooms'] + df['bathrooms']
    df.loc[df['rooms'] == 0, 'rooms'] = 1
    df.loc[df['price'] > 10000, 'price'] = 10000

    df['features'] = ([[w.lower().replace('-', '') for w in l]
                      for l in df['features'].values])

    return(df)


def add_variables(df, train_raw):
    df['description_length'] = (df["description"].
                                apply(lambda x: len(x.split(' '))))
    df['n_features'] = [len(x) for x in df['features'].values]
    df['n_photos'] = [len(x) for x in df['photos'].values]

    df['price_per_room'] = df['price']/df['rooms']

    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour
    df["created_weekday"] = df["created"].dt.weekday

    return(df)


def vectorizer(varname, train):
    dv = DictVectorizer(sparse=False)

    df_in = pd.DataFrame(train[[varname]])
    dv.fit(df_in.to_dict(orient='records'))

    return(dv)


def one_hot_encode(vectorizer, df, colname):
    counties = pd.DataFrame(
        vectorizer.transform(pd.DataFrame(df[[colname]]).
                             to_dict(orient='records')),
        columns=vectorizer.feature_names_)
    df = pd.concat([df.reset_index(drop=True), counties], axis=1)
    del df[colname]

    return(df)


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


def predict(model, X, cutoffs=[1, 1, 1]):
    probs = model.predict_proba(X)
    preds = model.predict(X)

    probs = pd.DataFrame(probs, columns=model.classes_)

    low_accuracy, med_accuracy, high_accuracy = cutoffs

    probs.loc[probs['high'] > high_accuracy, 'high'] = high_accuracy
    probs.loc[probs['high'] < 1 - high_accuracy, 'high'] = 1 - high_accuracy
    probs.loc[probs['medium'] > med_accuracy, 'medium'] = med_accuracy
    probs.loc[probs['medium'] < 1 - med_accuracy, 'medium'] = 1 - med_accuracy
    probs.loc[probs['low'] > low_accuracy, 'low'] = low_accuracy
    probs.loc[probs['low'] < 1 - low_accuracy, 'low'] = 1 - low_accuracy

    return preds, probs


def prepare_submission(model, test, independent):
    submission = test[['listing_id']]
    preds, probs = predict(model, test[independent])
    submission = pd.concat([submission.reset_index(drop=True),
                            pd.DataFrame(probs, columns=model.classes_)],
                           axis=1)
    submission = submission[['listing_id', 'high', 'medium', 'low']]

    timestamp = str(datetime.datetime.now())[:16]
    submission_name = 'Submissions/submission ' + timestamp + '.csv'
    submission_name = submission_name.replace(' ', '_').replace(':', '')
    submission.to_csv(submission_name, index=False)

    print('Written to file ' + submission_name)
    print(submission.head())

    return 0
