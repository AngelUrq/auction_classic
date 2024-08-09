import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

class AuctionPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = []

    def fit(self, df):
        numerical_columns = [
          'quantity',
          'unit_price',
          'time_left',
          'median_unit_price',
          'lowest_competitor_price',
          'top_competitor_price',
          'competitor_count',
          'rank_unit_price',
          'relative_price_difference',
          'relative_price_to_lowest_competitor',
          'relative_price_to_top_competitor',
        ]

        categorical_columns_ordinal = [
        ]

        categorical_columns_onehot = [
          'quality',
          'item_class',
          'item_subclass',
          'is_stackable'
        ]

        self.feature_names = numerical_columns + categorical_columns_ordinal + categorical_columns_onehot

        onehot_transformer = OneHotEncoder(sparse_output=False)

        self.column_transformer = make_column_transformer(
            (onehot_transformer, categorical_columns_onehot),
            remainder='passthrough'
        )

        X = df[self.feature_names]
        self.column_transformer.fit(X)

        return self

    def transform_time_left(self, df):
        time_left_mapping = {
            'SHORT': 0.5,
            'MEDIUM': 2,
            'LONG': 12,
            'VERY_LONG': 48
        }
        df['time_left'] = df['time_left'].map(time_left_mapping)
        return df

    def compute_median_competitor_price(self, df):
        group = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day', 'first_appearance_hour'])
        df['median_buyout_price'] = group['buyout_in_gold'].transform('median')
        df['median_bid_price'] = group['bid_in_gold'].transform('median')
        df['median_unit_price'] = group['unit_price'].transform('median')

        df['rank_buyout_price'] = group['buyout_in_gold'].rank(ascending=True)
        df['rank_bid_price'] = group['bid_in_gold'].rank(ascending=True)
        df['rank_unit_price'] = group['unit_price'].rank(ascending=True)

        return df

    def compute_avg_competitor_price(self, df):
        group = df.groupby(['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day', 'first_appearance_hour'])
        df['avg_competitor_price'] = group['unit_price'].transform('mean')
        df['std_competitor_price'] = group['unit_price'].transform('std')

        df['avg_competitor_price'] = df['avg_competitor_price'].fillna(0)
        df['std_competitor_price'] = df['std_competitor_price'].fillna(0)

        return df

    def compute_competitor_count(self, df):
        df['competitor_count'] = df.groupby(['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day', 'first_appearance_hour'])['unit_price'].transform('count')
        df['competitor_count'] = df['competitor_count'].fillna(0)

        return df

    def compute_minimum_competitor_price(self, df):
        minimum_competitor_price = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].quantile(0.15).reset_index(name='lowest_competitor_price')

        df_merged = pd.merge(df, minimum_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
        df_merged['lowest_competitor_price'] = df_merged['lowest_competitor_price'].fillna(0)

        return df_merged

    def compute_top_competitor_price(self, df):
        top_competitor_price = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].quantile(0.80).reset_index(name='top_competitor_price')

        df_merged = pd.merge(df, top_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
        df_merged['top_competitor_price'] = df_merged['top_competitor_price'].fillna(0)

        return df_merged

    def compute_relative_differences(self, df):
        relative_differences = {
            'relative_price_difference': ('unit_price', 'median_unit_price'),
            'relative_avg_price_difference': ('unit_price', 'avg_competitor_price'),
            'relative_buyout_difference': ('buyout_in_gold', 'median_buyout_price'),
            'relative_bid_difference': ('bid_in_gold', 'median_bid_price'),
            'relative_price_to_lowest_competitor': ('unit_price', 'lowest_competitor_price'),
            'relative_price_to_top_competitor': ('unit_price', 'top_competitor_price')
        }

        for col_name, (col1, col2) in relative_differences.items():
            df[col_name] = df[col1] / (df[col2] + 1e-6)
            df[col_name] = df[col_name].fillna(0)

        return df

    def add_features(self, df):
        df = self.transform_time_left(df)
        df = self.compute_median_competitor_price(df)
        df = self.compute_avg_competitor_price(df)
        df = self.compute_competitor_count(df)
        df = self.compute_minimum_competitor_price(df)
        df = self.compute_top_competitor_price(df)
        df = self.compute_relative_differences(df)
        return df

    def transform(self, df):
        X = df[self.feature_names].copy()
        y = df['hours_on_sale']

        X = self.column_transformer.transform(X)
        y = np.array(y)

        return X, y
    