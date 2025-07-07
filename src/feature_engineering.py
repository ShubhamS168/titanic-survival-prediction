"""
Advanced feature engineering for high accuracy - COMPLETE VERSION
This should contain ALL features used in train_high_performance.py
"""

import pandas as pd
import numpy as np
import re
from typing import List, Optional
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

def extract_title(name: str) -> str:
    """Extract title from passenger name with enhanced patterns"""
    if pd.isna(name):
        return "Unknown"
    
    # Enhanced title extraction patterns
    title_search = re.search(r' ([A-Za-z]+)\.', str(name))
    if title_search:
        return title_search.group(1)
    
    # Alternative patterns for edge cases
    alt_search = re.search(r'^([A-Za-z]+)\.', str(name))
    if alt_search:
        return alt_search.group(1)
    
    return "Unknown"

def clean_title(title: str) -> str:
    """Enhanced title cleaning with comprehensive grouping"""
    if pd.isna(title):
        return "Unknown"
    
    title = str(title)
    
    # Enhanced rare title grouping
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
                   'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Ms', 'Mlle']
    
    if title in rare_titles:
        return 'Rare'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Mme':
        return 'Mrs'
    elif title in ['Master']:
        return 'Master'
    else:
        return title

def create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced family feature engineering"""
    df = df.copy()
    
    # Basic family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Enhanced family categories
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['IsSmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
    df['IsLargeFamily'] = (df['FamilySize'] > 4).astype(int)
    
    # Family size groups with optimal binning
    df['FamilySizeGroup'] = pd.cut(df['FamilySize'], 
                                   bins=[0, 1, 4, 7, float('inf')], 
                                   labels=['Alone', 'Small', 'Medium', 'Large'])
    
    # Parent-child relationships
    df['HasParents'] = (df['Parch'] > 0).astype(int)
    df['HasChildren'] = (df['Parch'] > 0).astype(int)
    df['HasSiblings'] = (df['SibSp'] > 0).astype(int)
    
    return df

def create_fare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced fare feature engineering"""
    df = df.copy()
    
    # Handle missing fares with class-based imputation
    if 'Fare' in df.columns:
        # Fill missing fares with median by class
        for pclass in df['Pclass'].unique():
            class_median = df[df['Pclass'] == pclass]['Fare'].median()
            mask = (df['Pclass'] == pclass) & df['Fare'].isnull()
            df.loc[mask, 'Fare'] = class_median
        
        # Fill any remaining with overall median
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
        # Enhanced ticket sharing analysis
        if 'Ticket' in df.columns:
            ticket_counts = df['Ticket'].value_counts()
            df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
            df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
            
            # Shared ticket indicator
            df['SharedTicket'] = (df['TicketGroupSize'] > 1).astype(int)
        else:
            df['FarePerPerson'] = df['Fare']
            df['TicketGroupSize'] = 1
            df['SharedTicket'] = 0
        
        # Enhanced fare categorization
        if df["Fare"].nunique() >= 4:
            df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
        else:
            df['FareGroup'] = 'Unknown'
        
        # Log transformation for better distribution
        df['LogFare'] = np.log1p(df['Fare'])
        df['LogFarePerPerson'] = np.log1p(df['FarePerPerson'])
        
        # Fare relative to class (normalized fare)
        df['FareRelativeToClass'] = df.groupby('Pclass')['Fare'].transform(lambda x: (x - x.mean()) / x.std())
        
        # High/low fare indicators
        df['HighFare'] = (df['Fare'] > df['Fare'].quantile(0.75)).astype(int)
        df['LowFare'] = (df['Fare'] < df['Fare'].quantile(0.25)).astype(int)
    
    return df

def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced age feature engineering with advanced imputation"""
    df = df.copy()
    
    if 'Age' in df.columns:
        # Store original age for comparison
        df['AgeOriginal'] = df['Age'].copy()
        
        # Enhanced age imputation using multiple features
        if 'Title' in df.columns and 'Pclass' in df.columns:
            # First try title + class imputation
            for title in df['Title'].unique():
                for pclass in df['Pclass'].unique():
                    mask = (df['Title'] == title) & (df['Pclass'] == pclass) & df['Age'].isnull()
                    if mask.any():
                        age_group = df[(df['Title'] == title) & (df['Pclass'] == pclass) & df['Age'].notna()]
                        if len(age_group) > 0:
                            median_age = age_group['Age'].median()
                            df.loc[mask, 'Age'] = median_age
        
        # Fill any remaining with median
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Enhanced age categories
        df['AgeGroup'] = pd.cut(df['Age'], 
                                bins=[0, 12, 18, 25, 35, 50, 65, float('inf')], 
                                labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior', 'Elderly'])
        
        # Age indicators
        df['IsChild'] = (df['Age'] < 16).astype(int)
        df['IsElderly'] = (df['Age'] > 60).astype(int)
        df['IsAdult'] = ((df['Age'] >= 18) & (df['Age'] <= 60)).astype(int)
        
        # Age transformations - THESE ARE THE MISSING FEATURES
        df['AgeSquared'] = df['Age'] ** 2
        df['AgeCubed'] = df['Age'] ** 3
        df['SqrtAge'] = np.sqrt(df['Age'])
        
        # Age-class interaction - THIS WAS MISSING
        df['AgeClass_Ratio'] = df['Age'] / (df['Pclass'] + 1)
    
    return df

def create_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced cabin feature engineering"""
    df = df.copy()
    
    if 'Cabin' in df.columns:
        # Convert to string for processing
        df['Cabin'] = df['Cabin'].astype(str)
        
        # Cabin deck (letter)
        df['CabinLetter'] = df['Cabin'].str[0]
        df['CabinLetter'] = df['CabinLetter'].replace('nan', 'U')  # Unknown
        
        # Cabin number
        df['CabinNumber'] = df['Cabin'].str.extract(r'(\d+)')[0]
        df['CabinNumber'] = pd.to_numeric(df['CabinNumber'], errors='coerce')
        
        # Has cabin information
        df['HasCabin'] = (~df['Cabin'].str.contains('nan')).astype(int)
        
        # Number of cabins (some passengers had multiple)
        df['CabinCount'] = df['Cabin'].str.count(' ') + 1
        df['CabinCount'] = df['CabinCount'].fillna(0)
        
        # Cabin side (odd/even numbers often indicate ship sides)
        df['CabinSide'] = df['CabinNumber'].fillna(0) % 2
        
        # High-value cabin indicators
        high_value_decks = ['A', 'B', 'C']
        df['HighValueCabin'] = df['CabinLetter'].isin(high_value_decks).astype(int)
    
    return df

def create_ticket_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced ticket feature engineering"""
    df = df.copy()
    
    if 'Ticket' in df.columns:
        # Convert to string
        df['Ticket'] = df['Ticket'].astype(str)
        
        # Ticket prefix (letters)
        df['TicketPrefix'] = df['Ticket'].str.extract('([A-Za-z]+)')[0]
        df['TicketPrefix'] = df['TicketPrefix'].fillna('None')
        
        # Ticket number
        df['TicketNumber'] = df['Ticket'].str.extract(r'(\d+)')[0]
        df['TicketNumber'] = pd.to_numeric(df['TicketNumber'], errors='coerce')
        
        # Ticket characteristics
        df['TicketIsNumeric'] = df['Ticket'].str.isdigit().astype(int)
        df['TicketLength'] = df['Ticket'].str.len()
        
        # Special ticket types
        special_prefixes = ['PC', 'STON', 'CA', 'SOTON']
        df['SpecialTicket'] = df['TicketPrefix'].isin(special_prefixes).astype(int)
    
    return df

def create_name_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract additional features from passenger names"""
    df = df.copy()
    
    if 'Name' in df.columns:
        # Name length
        df['NameLength'] = df['Name'].astype(str).str.len()
        
        # Number of words in name
        df['NameWordCount'] = df['Name'].astype(str).str.split().str.len()
        
        # Has parentheses (often indicates maiden name)
        df['HasParentheses'] = df['Name'].astype(str).str.contains(r'\(').astype(int)
        
        # Extract potential nickname
        df['HasNickname'] = df['Name'].astype(str).str.contains('"').astype(int)
    
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced interaction features between variables"""
    df = df.copy()
    
    # Age and class interactions
    if 'Age' in df.columns and 'Pclass' in df.columns:
        df['Age_Pclass'] = df['Age'] * df['Pclass']
        df['AgeClass_Ratio'] = df['Age'] / (df['Pclass'] + 1)  # Avoid division by zero
    
    # Sex and class interactions
    if 'Sex' in df.columns and 'Pclass' in df.columns:
        df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
        df['FemaleFirstClass'] = ((df['Sex'] == 'female') & (df['Pclass'] == 1)).astype(int)
        df['MaleThirdClass'] = ((df['Sex'] == 'male') & (df['Pclass'] == 3)).astype(int)
    
    # Family and fare interactions
    if 'FamilySize' in df.columns and 'Fare' in df.columns:
        df['FamilySize_Fare'] = df['FamilySize'] * df['Fare']
        df['FarePerFamily'] = df['Fare'] / df['FamilySize']
    
    # Advanced combinations - THESE WERE MISSING
    if all(col in df.columns for col in ['Sex', 'Age', 'Pclass']):
        df['WomanChild_FirstClass'] = ((df['Sex'] == 'female') & (df['Age'] < 18) & (df['Pclass'] == 1)).astype(int)
        df['Adult_Male_ThirdClass'] = ((df['Sex'] == 'male') & (df['Age'] >= 18) & (df['Pclass'] == 3)).astype(int)
    
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to build all enhanced features for deployment compatibility"""
    df = df.copy()
    
    print("Starting enhanced feature engineering...")
    
    # Extract and clean titles first (needed for other features)
    if 'Name' in df.columns:
        df['Title'] = df['Name'].apply(extract_title)
        df['Title'] = df['Title'].apply(clean_title)
        print("✓ Title extraction completed")
    
    # Create all feature groups
    df = create_family_features(df)
    print("✓ Family features created")
    
    df = create_fare_features(df)
    print("✓ Fare features created")
    
    df = create_age_features(df)
    print("✓ Age features created")
    
    df = create_cabin_features(df)
    print("✓ Cabin features created")
    
    df = create_ticket_features(df)
    print("✓ Ticket features created")
    
    df = create_name_features(df)
    print("✓ Name features created")
    
    df = create_interaction_features(df)
    print("✓ Interaction features created")
    
    # Handle missing values in categorical columns
    categorical_columns = ['Embarked', 'CabinLetter', 'TicketPrefix', 'FamilySizeGroup', 'AgeGroup', 'FareGroup']
    for col in categorical_columns:
        if col in df.columns:
            mode_value = df[col].mode()
            fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_value)
    
    # Convert data types for compatibility
    print("✓ Converting data types for compatibility...")
    
    # Fix object columns that might cause issues
    for col in df.columns:
        if df[col].dtype == 'object':
            # Ensure all object columns are proper strings
            df[col] = df[col].astype(str)
        elif hasattr(df[col].dtype, 'name') and df[col].dtype.name.startswith('Int'):
            # Convert nullable integer types to standard int64
            df[col] = df[col].astype('int64')
        elif hasattr(df[col].dtype, 'name') and df[col].dtype.name.startswith('Float'):
            # Convert nullable float types to standard float64
            df[col] = df[col].astype('float64')
    
    # Ensure categorical columns are properly encoded
    categorical_cols = ['FamilySizeGroup', 'AgeGroup', 'FareGroup']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    print(f"✓ Enhanced feature engineering completed! New shape: {df.shape}")
    return df
