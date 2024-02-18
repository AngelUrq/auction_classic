import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


def transform_time_left(df):
  df['time_left'] = np.where(df['time_left'] == 'SHORT', 2, df['time_left'])
  df['time_left'] = np.where(df['time_left'] == 'MEDIUM', 12, df['time_left'])
  df['time_left'] = np.where(df['time_left'] == 'LONG', 24, df['time_left'])
  df['time_left'] = np.where(df['time_left'] == 'VERY_LONG', 48, df['time_left'])

  return df


def compute_median_competitor_price(df):
    df['median_buyout_price'] = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['buyout_in_gold'].transform('median')
    df['median_bid_price'] = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['bid_in_gold'].transform('median')
    df['median_unit_price'] = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].transform('median')

    df['rank_buyout_price'] = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['buyout_in_gold'].rank(ascending=True)
    df['rank_bid_price'] = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['bid_in_gold'].rank(ascending=True)
    df['rank_unit_price'] = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].rank(ascending=True)

    return df


def compute_avg_competitor_price(df):
    avg_competitor_price = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].mean().reset_index(name='avg_competitor_price')
    std_competitor_price = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].std().reset_index(name='std_competitor_price')

    df_merged = pd.merge(df, avg_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
    df_merged['avg_competitor_price'] = df_merged['avg_competitor_price'].fillna(0)

    df_merged = pd.merge(df_merged, std_competitor_price, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
    df_merged['std_competitor_price'] = df_merged['std_competitor_price'].fillna(0)

    return df_merged


def compute_competitor_count(df):
    competitor_count = df.groupby(by=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'])['unit_price'].count().reset_index(name='competitor_count')

    df_merged = pd.merge(df, competitor_count, on=['item_id', 'first_appearance_year', 'first_appearance_month', 'first_appearance_day'], how='left')
    df_merged['competitor_count'] = df_merged['competitor_count'].fillna(0)

    return df_merged


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
  df['relative_price_difference'] = (df['unit_price'] - df['median_unit_price']) / (df['median_unit_price'] + 1e-6)
  df['relative_price_difference'] = df['relative_price_difference'].fillna(0)

  df['relative_avg_price_difference'] = (df['unit_price'] - df['avg_competitor_price']) / (df['std_competitor_price'] + 1e-6)
  df['relative_avg_price_difference'] = df['relative_avg_price_difference'].fillna(0)

  df['relative_buyout_difference'] = (df['buyout_in_gold'] - df['median_buyout_price']) / (df['median_buyout_price'] + 1e-6)
  df['relative_buyout_difference'] = df['relative_buyout_difference'].fillna(0)

  df['relative_bid_difference'] = (df['bid_in_gold'] - df['median_bid_price']) / (df['median_bid_price'] + 1e-6)
  df['relative_bid_difference'] = df['relative_bid_difference'].fillna(0)

  df['relative_price_to_lowest_competitor'] = (df['unit_price'] - df['lowest_competitor_price']) / (df['lowest_competitor_price'] + 1e-6)
  df['relative_price_to_lowest_competitor'] = df['relative_price_to_lowest_competitor'].fillna(0)

  df['relative_price_to_top_competitor'] = (df['unit_price'] - df['top_competitor_price']) / (df['top_competitor_price'] + 1e-6)
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

  X = df[numerical_columns + categorical_columns_ordinal + categorical_columns_onehot]
  y = df['hours_on_sale']

  num_transformer = StandardScaler()
  ordinal_transformer = OrdinalEncoder()
  onehot_transformer = OneHotEncoder(sparse_output=False)

  column_transformer = make_column_transformer(
      #(num_transformer, numerical_columns),
      (ordinal_transformer, categorical_columns_ordinal),
      (onehot_transformer, categorical_columns_onehot),
      remainder='passthrough'
  )

  X = column_transformer.fit_transform(X)
  y = np.array(y)

  ordinal_feature_names = column_transformer.named_transformers_['ordinalencoder'].get_feature_names_out(categorical_columns_ordinal)

  # Get feature names for one-hot encoded categorical columns
  onehot_feature_names = column_transformer.named_transformers_['onehotencoder'].get_feature_names_out(categorical_columns_onehot)

  # Combine feature names
  feature_names = np.concatenate([ordinal_feature_names, onehot_feature_names, numerical_columns])

  return X, y
