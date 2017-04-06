from collections import Counter
import geopandas as gpd
import itertools
import pandas as pd
from shapely.geometry import Point
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import datetime
import re
import nltk
from bs4 import BeautifulSoup
import progressbar
from nltk.corpus import stopwords

nltk.download('stopwords')


def load_data():
    print('Loading raw data')
    train_raw = clean(pd.read_json('Data/train.json'))
    test_raw = clean(pd.read_json('Data/test.json'))

    print('Adding features')
    [features, feature_names] = get_features(25, train_raw)
    train = add_features(train_raw, features, feature_names)
    test = add_features(test_raw, features, feature_names)

    print('Adding regions')
    train = add_region(train)
    test = add_region(test)

    print('Adding variables')
    train = add_variables(train, train)
    test = add_variables(test, train)

    print('Dummyfying')
    dv_county = vectorizer('County', train)
    train = one_hot_encode(dv_county, train, 'County')
    test = one_hot_encode(dv_county, test, 'County')

    dv_name = vectorizer('Name', train)
    train = one_hot_encode(dv_name, train, 'Name')
    test = one_hot_encode(dv_name, test, 'Name')

    dv_region = vectorizer('RegionID', train)
    train = one_hot_encode(dv_region, train, 'RegionID')
    test = one_hot_encode(dv_region, test, 'RegionID')

    independent = (['bathrooms', 'bedrooms', 'rooms', 'price'] +
                   ['description_length', 'n_features', 'n_photos'] +
                   ['price_per_room', 'created_hour'] +
                   ['created_year', 'created_month', 'created_weekday'] +
                   ['n_listings', 'manager_interest_low'] +
                   ['manager_interest_high'] +
                   [x for x in train.columns.values if 'County' in x] +
                   [x for x in train.columns.values if 'Name' in x] +
                   [x for x in train.columns.values if 'Region' in x] +
                   feature_names
                   )

    print('Splitting data')
    data = train_test_split(train[independent], train['interest_level'],
                            test_size=0.33, random_state=1)

    return data, independent, test


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

    # Manager variables
    manager_grouped = train_raw.groupby('manager_id')

    manager_listings = pd.DataFrame(manager_grouped['listing_id'].agg(len))
    manager_listings.reset_index(inplace=True)
    manager_listings.columns = ['manager_id', 'n_listings']
    df = pd.merge(df, manager_listings, 'left', on='manager_id')
    df.loc[pd.isnull(df['n_listings']), 'n_listings'] = 0

    manager_high = pd.DataFrame(manager_grouped['interest_level'].
                                agg(lambda x: sum(x == 'high')/len(x)))
    manager_high.reset_index(inplace=True)
    manager_high.columns = ['manager_id', 'manager_interest_high']
    df = pd.merge(df, manager_high, 'left', on='manager_id')
    df.loc[pd.isnull(df['manager_interest_high']), 'manager_interest_high'] = 0

    manager_low = pd.DataFrame(manager_grouped['interest_level'].
                               agg(lambda x: sum(x == 'low')/len(x)))
    manager_low.reset_index(inplace=True)
    manager_low.columns = ['manager_id', 'manager_interest_low']
    df = pd.merge(df, manager_low, 'left', on='manager_id')
    df.loc[pd.isnull(df['manager_interest_low']), 'manager_interest_low'] = 0

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

    return None


def description_to_words(raw_description):
    review_text = BeautifulSoup(raw_description, "html5lib").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]
    return(" ".join(meaningful_words))


def convert_to_words(data):
    num_descriptions = data["description"].values.size
    clean_descriptions = []
    with progressbar.ProgressBar(max_value=num_descriptions) as bar:
        for i in xrange(0, num_descriptions):
            clean_descriptions.append(
                description_to_words(data["description"].values[i]))
            bar.update(i)
    return(clean_descriptions)
