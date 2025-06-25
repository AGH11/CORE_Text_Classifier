import warnings
import pandas as pd
from sklearn.utils import resample

# Suppress future warnings to avoid clutter
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')

# ----------------------------
# Load datasets
# ----------------------------
train_file = "train.tsv.gz"
test_file = "test.tsv.gz"
dev_file = "dev.tsv.gz"  # Optional use

# Load training and testing data
train_data = pd.read_csv(train_file, sep='\t', compression='gzip')
test_data = pd.read_csv(test_file, sep='\t', compression='gzip')

# Rename columns for consistency
train_data.columns = ['category_code', 'document_id', 'content']
test_data.columns = ['category_code', 'document_id', 'content']

# ----------------------------
# Analyze label distribution (train set)
# ----------------------------
# Explode multi-label entries (if any) in category_code
all_train_labels = train_data['category_code'].dropna().str.split().explode()

# Count label frequencies
train_label_counts = all_train_labels.value_counts()
print("Train label distribution:\n", train_label_counts)

# ----------------------------
# Balance both train and test datasets
# ----------------------------

# Define major and minor classes
major_classes = ['NA', 'IN', 'OP']
minor_classes = ['HI', 'IP', 'LY', 'SP', 'OTHER', 'ID']

# ----------------------------
# Balance Train Set
# ----------------------------
df_train_major = train_data[train_data['category_code'].isin(major_classes)]
df_train_minors = [train_data[train_data['category_code'] == cls] for cls in minor_classes]

upsampled_train_minors = [
    resample(df, replace=True, n_samples=700, random_state=42)
    for df in df_train_minors
]

balanced_train = pd.concat([df_train_major] + upsampled_train_minors)
balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------
# Balance Test Set
# ----------------------------
df_test_major = test_data[test_data['category_code'].isin(major_classes)]
df_test_minors = [test_data[test_data['category_code'] == cls] for cls in minor_classes]

upsampled_test_minors = [
    resample(df, replace=True, n_samples=700, random_state=42)
    for df in df_test_minors
]

balanced_test = pd.concat([df_test_major] + upsampled_test_minors)
balanced_test = balanced_test.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------
# Save to CSV
# ----------------------------
balanced_train.to_csv('balanced_train_data.csv', index=False)
balanced_test.to_csv('balanced_test_data.csv', index=False)

print("\nBalanced datasets saved successfully.")
