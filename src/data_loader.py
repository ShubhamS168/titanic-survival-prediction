"""
Enhanced data loading utilities for high-accuracy Titanic prediction
Prioritizes local data over downloads
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
import requests
from pathlib import Path

def download_titanic_data():
    """Load Titanic dataset with priority for local data/train.csv"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # PRIORITY 1: Check for existing local train.csv
    train_path = data_dir / "train.csv"
    if train_path.exists():
        try:
            st.success("✅ Using local dataset: data/train.csv")
            return pd.read_csv(train_path)
        except Exception as e:
            st.error(f"Error reading local train.csv: {e}")
            # Don't fallback to download - let user fix the local file
            st.stop()
    
    # PRIORITY 2: Only download if local file doesn't exist
    st.warning("⚠️ Local data/train.csv not found!")
    
    # Ask user preference before downloading
    if st.button("📥 Download Titanic dataset from remote source"):
        try:
            st.info("Downloading Titanic dataset...")
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            
            with st.spinner("Downloading dataset..."):
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(train_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            
            st.success("✅ Titanic dataset downloaded successfully")
            return pd.read_csv(train_path)
            
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.info("Please add your own data/train.csv file to the project.")
            st.stop()
    
    # PRIORITY 3: Show instructions if no data available
    st.error("""
    ❌ **No dataset available**
    
    **To use this application:**
    1. Add your `train.csv` file to the `data/` folder
    2. OR click the download button above to get the standard Titanic dataset
    3. OR create the data/train.csv file manually
    
    **Expected file location:** `data/train.csv`
    """)
    st.stop()

def create_high_quality_synthetic_data():
    """Create high-quality synthetic Titanic data for 85%+ accuracy"""
    st.info("🔄 Creating synthetic dataset as fallback...")
    
    np.random.seed(42)
    n_samples = 891
    
    data = []
    for i in range(n_samples):
        # Realistic class distribution
        pclass = np.random.choice([1, 2, 3], p=[0.24, 0.21, 0.55])
        
        # Gender distribution
        sex = np.random.choice(['male', 'female'], p=[0.65, 0.35])
        
        # Age distribution based on class and gender
        if pclass == 1:
            age_mean = 38 if sex == 'male' else 35
        elif pclass == 2:
            age_mean = 30 if sex == 'male' else 28
        else:
            age_mean = 26 if sex == 'male' else 22
        
        age = max(0.5, min(80, np.random.normal(age_mean, 12)))
        
        # Add realistic missing age pattern (20%)
        if np.random.random() < 0.2:
            age = np.nan
        
        # Family size patterns
        if age < 18:  # Children
            sibsp = np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])
            parch = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        elif sex == 'female' and age > 25:  # Likely married women
            sibsp = np.random.choice([0, 1], p=[0.3, 0.7])
            parch = np.random.poisson(1.5)
        else:
            sibsp = np.random.poisson(0.4)
            parch = np.random.poisson(0.3)
        
        # Fare based on class with realistic distribution
        if pclass == 1:
            fare = np.random.lognormal(4.3, 0.6)  # High-end fares
        elif pclass == 2:
            fare = np.random.lognormal(3.2, 0.4)  # Mid-range fares
        else:
            fare = np.random.lognormal(2.1, 0.5)  # Economy fares
        
        # Embarked port distribution
        embarked = np.random.choice(['S', 'C', 'Q'], p=[0.72, 0.19, 0.09])
        
        # Enhanced survival probability for realistic patterns
        survival_base = 0.15
        
        # Gender effect (strongest predictor)
        if sex == 'female':
            survival_base += 0.55
        
        # Age effects
        if age < 16:
            survival_base += 0.35
        elif age > 60:
            survival_base -= 0.15
        
        # Class effects
        survival_base += {1: 0.35, 2: 0.15, 3: 0}[pclass]
        
        # Family size effects
        family_size = sibsp + parch + 1
        if 2 <= family_size <= 4:
            survival_base += 0.15
        elif family_size > 4:
            survival_base -= 0.25
        
        # Fare effect
        if fare > np.percentile([50, 25, 8], pclass-1):
            survival_base += 0.1
        
        survived = 1 if np.random.random() < np.clip(survival_base, 0.05, 0.95) else 0
        
        # Generate cabin with realistic patterns
        if pclass == 1 and np.random.random() < 0.6:
            cabin_letter = np.random.choice(['A', 'B', 'C'], p=[0.2, 0.4, 0.4])
            cabin = f"{cabin_letter}{np.random.randint(1, 100)}"
        elif pclass == 2 and np.random.random() < 0.4:
            cabin_letter = np.random.choice(['D', 'E'], p=[0.5, 0.5])
            cabin = f"{cabin_letter}{np.random.randint(1, 100)}"
        elif pclass == 3 and np.random.random() < 0.1:
            cabin_letter = np.random.choice(['F', 'G'], p=[0.6, 0.4])
            cabin = f"{cabin_letter}{np.random.randint(1, 100)}"
        else:
            cabin = np.nan
        
        # Generate ticket
        if np.random.random() < 0.3:
            ticket_prefix = np.random.choice(['PC', 'STON/O', 'CA', 'A/5', 'SOTON/O.Q.'])
            ticket_number = np.random.randint(1000, 99999)
            ticket = f"{ticket_prefix} {ticket_number}"
        else:
            ticket = str(np.random.randint(100000, 999999))
        
        # Generate realistic names with titles
        if sex == 'female':
            if age < 18:
                name = f"Smith, Miss. Emma {i+1}"
            elif sibsp > 0 or age > 25:
                name = f"Johnson, Mrs. Mary {i+1}"
            else:
                name = f"Williams, Miss. Sarah {i+1}"
        else:
            if age < 16:
                name = f"Brown, Master. James {i+1}"
            else:
                name = f"Jones, Mr. William {i+1}"
        
        data.append({
            'PassengerId': i + 1,
            'Survived': survived,
            'Pclass': pclass,
            'Name': name,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Ticket': ticket,
            'Fare': fare,
            'Cabin': cabin,
            'Embarked': embarked
        })
    
    df = pd.DataFrame(data)
    
    # Save the synthetic data as train.csv for future use
    df.to_csv('data/train.csv', index=False)
    st.success("✅ Synthetic dataset created and saved as data/train.csv")
    
    return df

@st.cache_data
def load_titanic_data():
    """Load Titanic data with caching - prioritizes local data/train.csv"""
    try:
        return download_titanic_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def validate_data(df):
    """Validate the loaded dataset"""
    if df is None:
        return False
        
    required_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def get_data_summary(df):
    """Get comprehensive data summary"""
    if df is None or 'Survived' not in df.columns:
        return {}
        
    return {
        'total_passengers': len(df),
        'survivors': df['Survived'].sum(),
        'survival_rate': df['Survived'].mean(),
        'missing_age': df['Age'].isnull().sum(),
        'missing_cabin': df['Cabin'].isnull().sum() if 'Cabin' in df.columns else 0,
        'class_distribution': df['Pclass'].value_counts().to_dict(),
        'gender_distribution': df['Sex'].value_counts().to_dict()
    }

def check_data_availability():
    """Check if local data/train.csv is available"""
    train_path = Path("data/train.csv")
    return train_path.exists()

def get_data_info():
    """Get information about the current data source"""
    train_path = Path("data/train.csv")
    
    if train_path.exists():
        try:
            # Get file info
            stat = train_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            
            # Get row count
            df = pd.read_csv(train_path)
            row_count = len(df)
            
            return {
                'source': 'Local file',
                'path': 'data/train.csv',
                'size_mb': round(size_mb, 2),
                'rows': row_count,
                'available': True
            }
        except Exception as e:
            return {
                'source': 'Local file (error)',
                'path': 'data/train.csv',
                'error': str(e),
                'available': False
            }
    else:
        return {
            'source': 'Not found',
            'path': 'data/train.csv',
            'available': False
        }


# """
# Enhanced data loading utilities for high-accuracy Titanic prediction
# """

# import pandas as pd
# import numpy as np
# import streamlit as st
# import os
# import requests
# from pathlib import Path

# def download_titanic_data():
#     """Download the real Titanic dataset if not available"""
#     data_dir = Path("data")
#     data_dir.mkdir(exist_ok=True)
    
#     train_path = data_dir / "train.csv"
    
#     if not train_path.exists():
#         try:
#             st.info("Downloading Titanic dataset...")
#             url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
#             response = requests.get(url, timeout=30)
#             response.raise_for_status()
            
#             with open(train_path, 'w', encoding='utf-8') as f:
#                 f.write(response.text)
            
#             st.success("✅ Titanic dataset downloaded successfully")
#             return pd.read_csv(train_path)
            
#         except Exception as e:
#             st.warning(f"Download failed: {e}. Using high-quality synthetic data.")
#             return create_high_quality_synthetic_data()
    
#     return pd.read_csv(train_path)

# def create_high_quality_synthetic_data():
#     """Create high-quality synthetic Titanic data for 85%+ accuracy"""
#     np.random.seed(42)
#     n_samples = 891
    
#     data = []
#     for i in range(n_samples):
#         # Realistic class distribution
#         pclass = np.random.choice([1, 2, 3], p=[0.24, 0.21, 0.55])
        
#         # Gender distribution
#         sex = np.random.choice(['male', 'female'], p=[0.65, 0.35])
        
#         # Age distribution based on class and gender
#         if pclass == 1:
#             age_mean = 38 if sex == 'male' else 35
#         elif pclass == 2:
#             age_mean = 30 if sex == 'male' else 28
#         else:
#             age_mean = 26 if sex == 'male' else 22
        
#         age = max(0.5, min(80, np.random.normal(age_mean, 12)))
        
#         # Add realistic missing age pattern (20%)
#         if np.random.random() < 0.2:
#             age = np.nan
        
#         # Family size patterns
#         if age < 18:  # Children
#             sibsp = np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])
#             parch = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
#         elif sex == 'female' and age > 25:  # Likely married women
#             sibsp = np.random.choice([0, 1], p=[0.3, 0.7])
#             parch = np.random.poisson(1.5)
#         else:
#             sibsp = np.random.poisson(0.4)
#             parch = np.random.poisson(0.3)
        
#         # Fare based on class with realistic distribution
#         if pclass == 1:
#             fare = np.random.lognormal(4.3, 0.6)  # High-end fares
#         elif pclass == 2:
#             fare = np.random.lognormal(3.2, 0.4)  # Mid-range fares
#         else:
#             fare = np.random.lognormal(2.1, 0.5)  # Economy fares
        
#         # Embarked port distribution
#         embarked = np.random.choice(['S', 'C', 'Q'], p=[0.72, 0.19, 0.09])
        
#         # Enhanced survival probability for realistic patterns
#         survival_base = 0.15
        
#         # Gender effect (strongest predictor)
#         if sex == 'female':
#             survival_base += 0.55
        
#         # Age effects
#         if age < 16:
#             survival_base += 0.35
#         elif age > 60:
#             survival_base -= 0.15
        
#         # Class effects
#         survival_base += {1: 0.35, 2: 0.15, 3: 0}[pclass]
        
#         # Family size effects
#         family_size = sibsp + parch + 1
#         if 2 <= family_size <= 4:
#             survival_base += 0.15
#         elif family_size > 4:
#             survival_base -= 0.25
        
#         # Fare effect
#         if fare > np.percentile([50, 25, 8], pclass-1):
#             survival_base += 0.1
        
#         survived = 1 if np.random.random() < np.clip(survival_base, 0.05, 0.95) else 0
        
#         # Generate cabin with realistic patterns
#         if pclass == 1 and np.random.random() < 0.6:
#             cabin_letter = np.random.choice(['A', 'B', 'C'], p=[0.2, 0.4, 0.4])
#             cabin = f"{cabin_letter}{np.random.randint(1, 100)}"
#         elif pclass == 2 and np.random.random() < 0.4:
#             cabin_letter = np.random.choice(['D', 'E'], p=[0.5, 0.5])
#             cabin = f"{cabin_letter}{np.random.randint(1, 100)}"
#         elif pclass == 3 and np.random.random() < 0.1:
#             cabin_letter = np.random.choice(['F', 'G'], p=[0.6, 0.4])
#             cabin = f"{cabin_letter}{np.random.randint(1, 100)}"
#         else:
#             cabin = np.nan
        
#         # Generate ticket
#         if np.random.random() < 0.3:
#             ticket_prefix = np.random.choice(['PC', 'STON/O', 'CA', 'A/5', 'SOTON/O.Q.'])
#             ticket_number = np.random.randint(1000, 99999)
#             ticket = f"{ticket_prefix} {ticket_number}"
#         else:
#             ticket = str(np.random.randint(100000, 999999))
        
#         # Generate realistic names with titles
#         if sex == 'female':
#             if age < 18:
#                 name = f"Smith, Miss. Emma {i+1}"
#             elif sibsp > 0 or age > 25:
#                 name = f"Johnson, Mrs. Mary {i+1}"
#             else:
#                 name = f"Williams, Miss. Sarah {i+1}"
#         else:
#             if age < 16:
#                 name = f"Brown, Master. James {i+1}"
#             else:
#                 name = f"Jones, Mr. William {i+1}"
        
#         data.append({
#             'PassengerId': i + 1,
#             'Survived': survived,
#             'Pclass': pclass,
#             'Name': name,
#             'Sex': sex,
#             'Age': age,
#             'SibSp': sibsp,
#             'Parch': parch,
#             'Ticket': ticket,
#             'Fare': fare,
#             'Cabin': cabin,
#             'Embarked': embarked
#         })
    
#     df = pd.DataFrame(data)
    
#     # Save for future use
#     os.makedirs('data', exist_ok=True)
#     df.to_csv('data/train.csv', index=False)
    
#     return df

# @st.cache_data
# def load_titanic_data():
#     """Load Titanic data with caching"""
#     try:
#         return download_titanic_data()
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return None

# def validate_data(df):
#     """Validate the loaded dataset"""
#     required_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    
#     missing_columns = [col for col in required_columns if col not in df.columns]
    
#     if missing_columns:
#         st.error(f"Missing required columns: {missing_columns}")
#         return False
    
#     return True

# def get_data_summary(df):
#     """Get comprehensive data summary"""
#     if 'Survived' in df.columns:
#         return {
#             'total_passengers': len(df),
#             'survivors': df['Survived'].sum(),
#             'survival_rate': df['Survived'].mean(),
#             'missing_age': df['Age'].isnull().sum(),
#             'missing_cabin': df['Cabin'].isnull().sum() if 'Cabin' in df.columns else 0,
#             'class_distribution': df['Pclass'].value_counts().to_dict(),
#             'gender_distribution': df['Sex'].value_counts().to_dict()
#         }
#     return {}
