import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

feature_names = []

def transform_time_left(df):
  df['time_left'] = np.where(df['time_left'] == 'SHORT', 2, df['time_left'])
  df['time_left'] = np.where(df['time_left'] == 'MEDIUM', 12, df['time_left'])
  df['time_left'] = np.where(df['time_left'] == 'LONG', 24, df['time_left'])
  df['time_left'] = np.where(df['time_left'] == 'VERY_LONG', 48, df['time_left'])

  return df


def compute_median_competitor_price(df):
    group = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day', 'first_appearance_hour'])
    df['median_buyout_price'] = group['buyout_in_gold'].transform('median')
    df['median_bid_price'] = group['bid_in_gold'].transform('median')
    df['median_unit_price'] = group['unit_price'].transform('median')

    df['rank_buyout_price'] = group['buyout_in_gold'].rank(ascending=True)
    df['rank_bid_price'] = group['bid_in_gold'].rank(ascending=True)
    df['rank_unit_price'] = group['unit_price'].rank(ascending=True)

    return df


def compute_avg_competitor_price(df):
    df['avg_competitor_price'] = df.groupby(['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day', 'first_appearance_hour'])['unit_price'].transform('mean')
    df['std_competitor_price'] = df.groupby(['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day', 'first_appearance_hour'])['unit_price'].transform('std')

    df['avg_competitor_price'].fillna(0, inplace=True)
    df['std_competitor_price'].fillna(0, inplace=True)

    return df


def compute_competitor_count(df):
    df['competitor_count'] = df.groupby(['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day', 'first_appearance_hour'])['unit_price'].transform('count')
    df['competitor_count'].fillna(0, inplace=True)

    return df


def compute_minimum_competitor_price(df):
    minimum_competitor_price = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].quantile(0.15).reset_index(name='lowest_competitor_price')

    df_merged = pd.merge(df, minimum_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
    df_merged['lowest_competitor_price'] = df_merged['lowest_competitor_price'].fillna(0)

    return df_merged


def compute_top_competitor_price(df):
    top_competitor_price = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].quantile(0.80).reset_index(name='top_competitor_price')

    df_merged = pd.merge(df, top_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
    df_merged['top_competitor_price'] = df_merged['top_competitor_price'].fillna(0)

    return df_merged


def compute_relative_differences(df):
  df['relative_price_difference'] = np.log(df['unit_price']+ 1e-6) - np.log(df['median_unit_price'] + 1e-6)
  df['relative_price_difference'] = df['relative_price_difference'].fillna(0)

  df['relative_avg_price_difference'] = np.log(df['unit_price']+ 1e-6) - np.log(df['avg_competitor_price'] + 1e-6)
  df['relative_avg_price_difference'] = df['relative_avg_price_difference'].fillna(0)

  df['relative_buyout_difference'] = np.log(df['buyout_in_gold']+ 1e-6) - np.log(df['median_buyout_price'] + 1e-6)
  df['relative_buyout_difference'] = df['relative_buyout_difference'].fillna(0)

  df['relative_bid_difference'] = np.log(df['bid_in_gold']+ 1e-6) - np.log(df['median_bid_price'] + 1e-6)
  df['relative_bid_difference'] = df['relative_bid_difference'].fillna(0)

  df['relative_price_to_lowest_competitor'] = np.log(df['unit_price']+ 1e-6) - np.log(df['lowest_competitor_price'] + 1e-6)
  df['relative_price_to_lowest_competitor'] = df['relative_price_to_lowest_competitor'].fillna(0)

  df['relative_price_to_top_competitor'] = np.log(df['unit_price']+ 1e-6) - np.log(df['top_competitor_price'] + 1e-6)
  df['relative_price_to_top_competitor'] = df['relative_price_to_top_competitor'].fillna(0)

  return df


def add_features(df):
  df = transform_time_left(df)
  df = compute_median_competitor_price(df)
  df = compute_avg_competitor_price(df)
  df = compute_competitor_count(df)
  df = compute_minimum_competitor_price(df)
  df = compute_top_competitor_price(df)
  df = compute_relative_differences(df)

  return df


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
    'item_level',
    'item_id',
  ]

  categorical_columns_ordinal = [
    'quality',
    'item_class',
    'item_subclass',
    'is_stackable'
  ]

  assert set(numerical_columns + categorical_columns_ordinal).issubset(df.columns)

  X = df[numerical_columns + categorical_columns_ordinal]
  y = df['hours_on_sale']

  num_transformer = StandardScaler()
  ordinal_transformer = OrdinalEncoder()
  onehot_transformer = OneHotEncoder(sparse_output=False)

  column_transformer = make_column_transformer(
      #(num_transformer, numerical_columns),
      (ordinal_transformer, categorical_columns_ordinal),
      remainder='passthrough'
  )

  X = column_transformer.fit_transform(X)
  y = np.array(y)

  return X, y
