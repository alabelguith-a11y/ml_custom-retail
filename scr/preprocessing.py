"""
Retail Customer Preprocessing & Feature Engineering Pipeline.

This module encapsulates the complete data preprocessing workflow:
1. Data loading & cleaning
2. Feature engineering (date parsing, IP analysis)
3. Encoding (ordinal, one-hot, target)
4. Feature selection (variance, correlation, importance)
5. Imputation (median, KNN, iterative)
6. Normalization (StandardScaler)
7. Dimensionality reduction (PCA)
8. Class balancing (SMOTE)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import joblib
import os
from pathlib import Path

from utils import (
    ip_is_private, 
    ip_first_octet,
    detect_missing_values,
    get_high_correlation_pairs,
    create_engineered_features,
    print_preprocessing_summary
)


class RetailPreprocessor:
    """
    Complete preprocessing pipeline for retail customer data.
    """
    
    def __init__(self, raw_data_path: str, output_dir: str, models_dir: str):
        """
        Initialize preprocessor with data paths.
        
        Args:
            raw_data_path: Path to raw CSV file
            output_dir: Directory for processed data output
            models_dir: Directory for model persistence
        """
        self.raw_data_path = raw_data_path
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories if missing
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.X_full = None
        self.X_selected = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        
        # Preprocessing objects
        self.scaler = None
        self.knn_imputer = None
        self.iter_imputer = None
        self.pca_2d = None
        self.pca_10 = None
        self.pca_90 = None
    
    def load_data(self) -> None:
        """Load raw data from CSV."""
        self.df = pd.read_csv(self.raw_data_path)
        print(f'✅ Data loaded: {self.df.shape}')
    
    def step_1_cleaning(self) -> None:
        """Step 1: Initial data cleaning and feature engineering."""
        # 1.2: Drop constant variance features
        self.df.drop(columns=['NewsletterSubscribed'], inplace=True)
        print('✅ Dropped NewsletterSubscribed (variance = 0)')
        
        # 1.3: Parse RegistrationDate → temporal features
        self.df['RegistrationDate'] = pd.to_datetime(
            self.df['RegistrationDate'],
            dayfirst=True,
            errors='coerce'
        )
        
        self.df['RegYear'] = self.df['RegistrationDate'].dt.year
        self.df['RegMonth'] = self.df['RegistrationDate'].dt.month
        self.df['RegDay'] = self.df['RegistrationDate'].dt.day
        self.df['RegWeekday'] = self.df['RegistrationDate'].dt.weekday
        
        self.df.drop(columns=['RegistrationDate'], inplace=True)
        print('✅ RegistrationDate → RegYear, RegMonth, RegDay, RegWeekday')
        
        # 1.4: IP features
        self.df['IP_IsPrivate'] = self.df['LastLoginIP'].apply(ip_is_private)
        self.df['IP_FirstOctet'] = self.df['LastLoginIP'].apply(ip_first_octet)
        self.df.drop(columns=['LastLoginIP'], inplace=True)
        print('✅ LastLoginIP → IP_IsPrivate, IP_FirstOctet')
        
        # 1.5: Correct aberrant values
        self.df['SupportTicketsCount'] = self.df['SupportTicketsCount'].replace([-1, 999], np.nan)
        self.df['SatisfactionScore'] = self.df['SatisfactionScore'].replace([-1, 0, 99], np.nan)
        print('✅ Aberrant values → NaN')
        
        # 1.9: Drop CustomerID
        self.df.drop(columns=['CustomerID'], inplace=True)
        print('✅ Dropped CustomerID')
    
    def step_2_encoding(self) -> None:
        """Step 2: Ordinal, one-hot, and target encoding."""
        # 1.6: Ordinal encoding
        ordinal_maps = {
            'AgeCategory': {
                'Inconnu': 0, '18-24': 1, '25-34': 2,
                '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6
            },
            'SpendingCategory': {'Low': 1, 'Medium': 2, 'High': 3, 'VIP': 4},
            'LoyaltyLevel': {'Nouveau': 1, 'Jeune': 2, 'Établi': 3, 'Ancien': 4},
            'ChurnRiskCategory': {'Faible': 1, 'Moyen': 2, 'Élevé': 3, 'Critique': 4},
            'BasketSizeCategory': {'Inconnu': 0, 'Petit': 1, 'Moyen': 2, 'Grand': 3},
            'PreferredTimeOfDay': {'Nuit': 0, 'Matin': 1, 'Midi': 2, 'Après-midi': 3, 'Soir': 4},
        }
        
        for col, mapping in ordinal_maps.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)
        
        print(f'✅ Ordinal encoding: {len(ordinal_maps)} features')
        
        # 1.7: One-Hot Encoding
        ohe_features = [
            'RFMSegment', 'CustomerType', 'FavoriteSeason',
            'WeekendPreference', 'ProductDiversity', 'Gender',
            'AccountStatus', 'Region'
        ]
        
        self.df = pd.get_dummies(
            self.df, columns=ohe_features,
            prefix=ohe_features,
            drop_first=True,
            dtype=int
        )
        
        n_ohe = sum(1 for c in self.df.columns 
                    if any(c.startswith(f+'_') for f in ohe_features))
        print(f'✅ One-Hot Encoding: {len(ohe_features)} features → {n_ohe} columns')
        
        # 1.8: Target Encoding (Country)
        if 'Country' in self.df.columns:
            country_churn_map = self.df.groupby('Country')['Churn'].mean()
            self.df['Country_TargetEnc'] = self.df['Country'].map(country_churn_map)
            self.df.drop(columns=['Country'], inplace=True)
            print('✅ Country → Country_TargetEnc (target encoding)')
    
    def step_3_feature_engineering(self) -> None:
        """Step 3: Create derived features."""
        self.df = create_engineered_features(self.df)
        print('✅ 10 engineered features created')
        print(f'   Total shape: {self.df.shape}')
    
    def step_4_feature_selection(self) -> None:
        """Step 4: Feature selection via multiple criteria."""
        # Separate features and target
        self.X_full = self.df.drop(columns=['Churn'])
        y = self.df['Churn'].copy()
        
        # Criterion 1: Variance Threshold
        vt = VarianceThreshold(threshold=0.0)
        X_temp = self.X_full.fillna(self.X_full.median(numeric_only=True))
        vt.fit(X_temp)
        zero_var_cols = self.X_full.columns[~vt.get_support()].tolist()
        
        # Criterion 2: Missing values > 50%
        missing_pct = self.X_full.isnull().mean() * 100
        high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
        
        # Criterion 3: High correlation (multicollinearity)
        corr_matrix = X_temp.corr()
        high_corr_pairs = get_high_correlation_pairs(corr_matrix, threshold=0.85)
        
        cols_to_drop_corr = [
            'NegativeQuantityCount', 'UniqueInvoices', 'RegYear',
            'UniqueDescriptions', 'MonetaryMin', 'MinQuantity',
            'MaxQuantity', 'AvgLinesPerInvoice', 'AvgProductsPerTransaction',
            'LoyaltyLevel', 'MaxOrderShare', 'WeekendPurchaseRatio',
            'IsNewCustomer', 'TotalTransactions', 'MonetaryPerDay',
        ]
        cols_to_drop_corr = [c for c in cols_to_drop_corr if c in self.X_full.columns]
        
        # Criterion 4: Random Forest importance
        X_for_rf = X_temp.drop(columns=cols_to_drop_corr, errors='ignore')
        
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_for_rf, y)
        
        importance_series = pd.Series(rf.feature_importances_, index=X_for_rf.columns)
        zero_importance_cols = importance_series[importance_series == 0.0].index.tolist()
        
        # Apply all filters
        all_cols_to_drop = list(set(
            zero_var_cols + high_missing_cols + cols_to_drop_corr + zero_importance_cols
        ))
        all_cols_to_drop = [c for c in all_cols_to_drop if c in self.X_full.columns]
        
        self.X_selected = self.X_full.drop(columns=all_cols_to_drop)
        
        print('=' * 60)
        print('FEATURE SELECTION SUMMARY')
        print('=' * 60)
        print(f'  Initial features        : {self.X_full.shape[1]}')
        print(f'  Zero variance           : -{len(zero_var_cols)}')
        print(f'  Missing > 50%           : -{len(high_missing_cols)}')
        print(f'  High correlation        : -{len([c for c in cols_to_drop_corr if c in self.X_full.columns])}')
        print(f'  Zero RF importance      : -{len([c for c in zero_importance_cols if c in self.X_full.columns])}')
        print(f'  Final features          : {self.X_selected.shape[1]}')
        print('=' * 60)
    
    def step_5_train_test_split(self) -> None:
        """Step 5: Split data (before imputation to avoid data leakage)."""
        y = self.df['Churn']
        
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            self.X_selected, y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
        
        print(f'✅ Train/Test split:')
        print(f'   Train: {self.X_train_raw.shape}  |  Test: {self.X_test_raw.shape}')
        print(f'   Churn rate — Train: {self.y_train.mean():.3f}  |  Test: {self.y_test.mean():.3f}')
    
    def step_6_imputation(self) -> None:
        """Step 6: Advanced imputation strategy."""
        self.X_train_scaled = self.X_train_raw.copy()
        self.X_test_scaled = self.X_test_raw.copy()
        
        # 4-A: Median imputation
        median_cols = ['AvgDaysBetweenPurchases', 'SupportTicketsCount', 'IP_FirstOctet', 'SupportBurden']
        median_cols = [c for c in median_cols if c in self.X_train_scaled.columns]
        
        if median_cols:
            medians = self.X_train_scaled[median_cols].median()
            self.X_train_scaled[median_cols] = self.X_train_scaled[median_cols].fillna(medians)
            self.X_test_scaled[median_cols] = self.X_test_scaled[median_cols].fillna(medians)
            print(f'✅ Median imputation: {median_cols}')
        
        # 4-B: Iterative imputation for SatisfactionScore
        if 'SatisfactionScore' in self.X_train_scaled.columns and \
           self.X_train_scaled['SatisfactionScore'].isna().any():
            
            context_cols = ['SatisfactionScore', 'Recency', 'Frequency',
                           'CustomerTenureDays', 'SpendingCategory']
            context_cols = [c for c in context_cols if c in self.X_train_scaled.columns]
            
            self.iter_imputer = IterativeImputer(max_iter=10, random_state=42)
            
            X_train_iter = self.iter_imputer.fit_transform(self.X_train_scaled[context_cols])
            X_test_iter = self.iter_imputer.transform(self.X_test_scaled[context_cols])
            
            idx = context_cols.index('SatisfactionScore')
            self.X_train_scaled['SatisfactionScore'] = X_train_iter[:, idx].clip(1, 5)
            self.X_test_scaled['SatisfactionScore'] = X_test_iter[:, idx].clip(1, 5)
            
            joblib.dump(self.iter_imputer, 
                       self.models_dir / 'iterative_imputer_satisfaction.joblib')
            print('✅ Iterative Imputation: SatisfactionScore')
        
        # 4-C: KNN imputation for Age
        knn_context = ['Age', 'Frequency', 'MonetaryTotal', 'CustomerTenureDays',
                      'Recency', 'AgeCategory', 'SpendingCategory']
        knn_context = [c for c in knn_context if c in self.X_train_scaled.columns]
        
        if 'Age' in self.X_train_scaled.columns and self.X_train_scaled['Age'].isna().any():
            self.knn_imputer = KNNImputer(n_neighbors=5)
            
            knn_result_train = self.knn_imputer.fit_transform(self.X_train_scaled[knn_context])
            knn_result_test = self.knn_imputer.transform(self.X_test_scaled[knn_context])
            
            age_idx = knn_context.index('Age')
            self.X_train_scaled['Age'] = knn_result_train[:, age_idx].clip(18, 81)
            self.X_test_scaled['Age'] = knn_result_test[:, age_idx].clip(18, 81)
            
            joblib.dump(self.knn_imputer, self.models_dir / 'knn_imputer_age.joblib')
            print('✅ KNN Imputation: Age')
        
        total_nan_train = self.X_train_scaled.isnull().sum().sum()
        total_nan_test = self.X_test_scaled.isnull().sum().sum()
        print(f'✅ NaN remaining — Train: {total_nan_train}  |  Test: {total_nan_test}')
    
    def step_7_scaling(self) -> None:
        """Step 7: StandardScaler normalization."""
        # Identify binary columns (don't scale)
        binary_cols = [
            c for c in self.X_train_scaled.columns
            if self.X_train_scaled[c].nunique() <= 2 and 
            set(self.X_train_scaled[c].dropna().unique()).issubset({0, 1})
        ]
        cols_to_scale = [c for c in self.X_train_scaled.columns if c not in binary_cols]
        
        self.scaler = StandardScaler()
        self.X_train_scaled[cols_to_scale] = self.scaler.fit_transform(
            self.X_train_scaled[cols_to_scale]
        )
        self.X_test_scaled[cols_to_scale] = self.scaler.transform(
            self.X_test_scaled[cols_to_scale]
        )
        
        joblib.dump(self.scaler, self.models_dir / 'standard_scaler_v2.joblib')
        print(f'✅ StandardScaler: {len(cols_to_scale)} features scaled')
    
    def step_8_pca(self, n_components_2d=2, n_components_10=10) -> None:
        """Step 8: Principal Component Analysis."""
        # Full PCA for analysis
        pca_full = PCA(random_state=42)
        pca_full.fit(self.X_train_scaled)
        
        explained_var = pca_full.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        n_90 = np.argmax(cumulative_var >= 0.90) + 1
        
        # PCA 2D
        self.pca_2d = PCA(n_components=n_components_2d, random_state=42)
        self.X_train_pca2 = self.pca_2d.fit_transform(self.X_train_scaled)
        self.X_test_pca2 = self.pca_2d.transform(self.X_test_scaled)
        
        var_2d = self.pca_2d.explained_variance_ratio_.sum() * 100
        
        # PCA 10 components
        self.pca_10 = PCA(n_components=n_components_10, random_state=42)
        self.X_train_pca10 = self.pca_10.fit_transform(self.X_train_scaled)
        self.X_test_pca10 = self.pca_10.transform(self.X_test_scaled)
        
        var_10 = self.pca_10.explained_variance_ratio_.sum() * 100
        
        # PCA 90% variance
        self.pca_90 = PCA(n_components=n_90, random_state=42)
        self.X_train_pca90 = self.pca_90.fit_transform(self.X_train_scaled)
        self.X_test_pca90 = self.pca_90.transform(self.X_test_scaled)
        
        var_90 = self.pca_90.explained_variance_ratio_.sum() * 100
        
        # Save models
        joblib.dump(self.pca_2d, self.models_dir / 'pca_2d.joblib')
        joblib.dump(self.pca_10, self.models_dir / 'pca_10.joblib')
        joblib.dump(self.pca_90, self.models_dir / 'pca_90.joblib')
        
        print(f'✅ PCA models trained:')
        print(f'   PCA 2D  : {var_2d:.1f}% variance')
        print(f'   PCA 10  : {var_10:.1f}% variance')
        print(f'   PCA {n_90:2d}  : {var_90:.1f}% variance')
        
        self.n_components_10 = n_components_10
        self.n_components_90 = n_90
    
    def step_9_smote_export(self) -> None:
        """Step 9: Apply SMOTE and export all datasets."""
        # Create DataFrames with proper column names
        pca10_cols = [f'PC{i+1}' for i in range(self.n_components_10)]
        pca90_cols = [f'PC{i+1}' for i in range(self.n_components_90)]
        
        X_train_pca10_df = pd.DataFrame(self.X_train_pca10, columns=pca10_cols)
        X_test_pca10_df = pd.DataFrame(self.X_test_pca10, columns=pca10_cols)
        X_train_pca90_df = pd.DataFrame(self.X_train_pca90, columns=pca90_cols)
        X_test_pca90_df = pd.DataFrame(self.X_test_pca90, columns=pca90_cols)
        
        # Apply SMOTE
        print('Applying SMOTE...')
        print(f'  Before: {pd.Series(self.y_train).value_counts().to_dict()}')
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        
        X_train_smote, y_train_smote = smote.fit_resample(
            self.X_train_scaled, self.y_train
        )
        X_train_pca10_smote, y_train_pca10_smote = smote.fit_resample(
            X_train_pca10_df, self.y_train
        )
        
        print(f'  After:  {pd.Series(y_train_smote).value_counts().to_dict()}')
        
        # Export datasets
        export_files = {
            self.output_dir / 'X_train_scaled.csv': self.X_train_scaled,
            self.output_dir / 'X_test_scaled.csv': self.X_test_scaled,
            self.output_dir / 'X_train_scaled_smote.csv': pd.DataFrame(X_train_smote, 
                                                                       columns=self.X_train_scaled.columns),
            self.output_dir / 'X_train_pca10.csv': X_train_pca10_df,
            self.output_dir / 'X_test_pca10.csv': X_test_pca10_df,
            self.output_dir / 'X_train_pca10_smote.csv': pd.DataFrame(X_train_pca10_smote, 
                                                                       columns=pca10_cols),
            self.output_dir / 'X_train_pca90.csv': X_train_pca90_df,
            self.output_dir / 'X_test_pca90.csv': X_test_pca90_df,
        }
        
        # Export target variables
        pd.Series(self.y_train, name='Churn').to_csv(
            self.output_dir / 'y_train.csv', index=False
        )
        pd.Series(self.y_test, name='Churn').to_csv(
            self.output_dir / 'y_test.csv', index=False
        )
        pd.Series(y_train_smote, name='Churn').to_csv(
            self.output_dir / 'y_train_smote.csv', index=False
        )
        pd.Series(y_train_pca10_smote, name='Churn').to_csv(
            self.output_dir / 'y_train_pca10_smote.csv', index=False
        )
        
        for path, df in export_files.items():
            df.to_csv(path, index=False)
        
        print(f'\n✅ {len(export_files)} datasets exported to {self.output_dir}')
        print(f'✅ Target variables exported')
    
    def run_pipeline(self) -> None:
        """Execute complete preprocessing pipeline."""
        print('\n' + '='*70)
        print('RETAIL CUSTOMER PREPROCESSING PIPELINE')
        print('='*70 + '\n')
        
        self.load_data()
        self.step_1_cleaning()
        self.step_2_encoding()
        self.step_3_feature_engineering()
        self.step_4_feature_selection()
        self.step_5_train_test_split()
        self.step_6_imputation()
        self.step_7_scaling()
        self.step_8_pca()
        self.step_9_smote_export()
        
        print('\n' + '='*70)
        print('✅ PIPELINE COMPLETE')
        print('='*70 + '\n')


def main():
    """Main entry point."""
    # Configure paths
    raw_data_path = r'data\raw\retail_customers_COMPLETE_CATEGORICAL.csv'
    output_dir = r'data\train_test'
    models_dir = r'models'
    
    # Run pipeline
    preprocessor = RetailPreprocessor(raw_data_path, output_dir, models_dir)
    preprocessor.run_pipeline()


if __name__ == '__main__':
    main()
