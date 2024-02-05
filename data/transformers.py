import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


class TimeLeftTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['time_left'] = np.where(X['time_left'] == 'SHORT', 0.5, X['time_left'])
        X['time_left'] = np.where(X['time_left'] == 'MEDIUM', 2, X['time_left'])
        X['time_left'] = np.where(X['time_left'] == 'LONG', 12, X['time_left'])
        X['time_left'] = np.where(X['time_left'] == 'VERY_LONG', 48, X['time_left'])


        return X


class MedianCompetitorPriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['median_buyout_price'] = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['buyout_in_gold'].transform('median')
        X['median_bid_price'] = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['bid_in_gold'].transform('median')
        X['median_unit_price'] = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].transform('median')

        X['rank_buyout_price'] = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['buyout_in_gold'].rank(ascending=True)
        X['rank_bid_price'] = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['bid_in_gold'].rank(ascending=True)
        X['rank_unit_price'] = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].rank(ascending=True)

        return X


class AvgCompetitorPriceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X.head())
         
        avg_competitor_price = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].mean().reset_index(name='avg_competitor_price')
        std_competitor_price = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].std().reset_index(name='std_competitor_price')

        X = pd.merge(X, avg_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
        X['avg_competitor_price'] = X['avg_competitor_price'].fillna(0)

        X = pd.merge(X, std_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
        X['std_competitor_price'] = X['std_competitor_price'].fillna(0)

        return X


class CompetitorCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        competitor_count = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].count().reset_index(name='competitor_count')
        X = pd.merge(X, competitor_count, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
        X['competitor_count'] = X['competitor_count'].fillna(0)
        return X


class MinCompetitorPriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        minimum_competitor_price = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].quantile(0.15).reset_index(name='lowest_competitor_price')
        X = pd.merge(X, minimum_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
        X['lowest_competitor_price'] = X['lowest_competitor_price'].fillna(0)
        return X


class TopCompetitorPriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        top_competitor_price = X.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].quantile(0.80).reset_index(name='top_competitor_price')
        X = pd.merge(X, top_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
        X['top_competitor_price'] = X['top_competitor_price'].fillna(0)
        return X


class RelativeDifferencesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['relative_price_difference'] = (X['unit_price'] - X['median_unit_price']) / (X['median_unit_price'] + 1e-6)
        X['relative_price_difference'] = X['relative_price_difference'].fillna(0)

        X['relative_buyout_difference'] = (X['buyout_in_gold'] - X['median_buyout_price']) / (X['median_buyout_price'] + 1e-6)
        X['relative_buyout_difference'] = X['relative_buyout_difference'].fillna(0)

        X['relative_bid_difference'] = (X['bid_in_gold'] - X['median_bid_price']) / (X['median_bid_price'] + 1e-6)
        X['relative_bid_difference'] = X['relative_bid_difference'].fillna(0)

        X['relative_price_to_lowest_competitor'] = (X['unit_price'] - X['lowest_competitor_price']) / (X['lowest_competitor_price'] + 1e-6)
        X['relative_price_to_lowest_competitor'] = X['relative_price_to_lowest_competitor'].fillna(0)

        X['relative_price_to_top_competitor'] = (X['unit_price'] - X['top_competitor_price']) / (X['top_competitor_price'] + 1e-6)
        X['relative_price_to_top_competitor'] = X['relative_price_to_top_competitor'].fillna(0)
        
        X['relative_avg_price_difference'] = (X['unit_price'] - X['avg_competitor_price']) / (X['std_competitor_price'] + 1e-6)
        X['relative_avg_price_difference'] = X['relative_avg_price_difference'].fillna(0)


        return X


def transform_data(df):
    numerical_columns = [
        'quantity',
        'unit_price',
        'bid_in_gold',
        'buyout_in_gold',
        'time_left',
        'median_buyout_price',
        'median_bid_price',
        'median_unit_price',
        'lowest_competitor_price',
        'avg_competitor_price',
        'std_competitor_price',
        'top_competitor_price',
        'competitor_count',
        'rank_buyout_price',
        'rank_bid_price',
        'rank_unit_price',
        'relative_price_difference',
        'relative_avg_price_difference',
        'relative_buyout_difference',
        'relative_bid_difference',
        'relative_price_to_lowest_competitor',
        'relative_price_to_top_competitor',
        'purchase_price_gold',
        'sell_price_gold',
        'required_level',
        'item_level'
    ]

    categorical_columns_ordinal = [
        'item_id',
        'quality',
        'item_class',
        'item_subclass'
    ]

    categorical_columns_onehot = [
    'is_stackable'
    ]

    column_transformer = make_column_transformer(
        (OrdinalEncoder(), categorical_columns_ordinal),
        (OneHotEncoder(sparse_output=False), categorical_columns_onehot),
        remainder='passthrough'
    )

    preprocessing_pipeline = Pipeline(steps=[
        ('time_left', TimeLeftTransformer()),
        ('median_competitor_price', MedianCompetitorPriceTransformer()),
        ('avg_competitor_price', AvgCompetitorPriceTransformer()),
        ('competitor_count', CompetitorCountTransformer()),
        ('min_competitor_price', MinCompetitorPriceTransformer()),
        ('top_competitor_price', TopCompetitorPriceTransformer()),
        ('relative_differences', RelativeDifferencesTransformer()),
    ])

    df_transformed = preprocessing_pipeline.fit_transform(df)
    df_transformed = df_transformed[numerical_columns + categorical_columns_ordinal + categorical_columns_onehot]

    X = column_transformer.fit_transform(df_transformed)
    y = df['hours_on_sale']

    return X, y
