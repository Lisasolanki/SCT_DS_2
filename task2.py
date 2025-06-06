#  Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Settings
sns.set(style="whitegrid")
plt.style.use('ggplot')

#  Load the dataset
file_path = r'C:\Users\windows 10\Downloads\titanic\train.csv'  
df = pd.read_csv(file_path)

# Preview the data
print("First 5 rows:")
print(df.head())

#  Basic Info
print("\nInfo:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

#  Fill or drop missing data
df['Age'].fillna(df['Age'].median(), inplace=True)         # Fill Age with median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill Embarked with mode
df.drop(columns=['Cabin'], inplace=True)                   # Too many missing values

#  Convert categorical variables to category type
cat_cols = ['Sex', 'Embarked', 'Pclass']
for col in cat_cols:
    df[col] = df[col].astype('category')

#  Cleaned dataset preview
print("\nCleaned Data:")
print(df.head())

# -------------------------------------
#  EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------

#  1. Survival Count
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.xticks([0,1], ['Died', 'Survived'])
plt.show()

#  2. Survival by Gender
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival by Gender')
plt.xticks(rotation=0)
plt.legend(['Died', 'Survived'])
plt.show()

#  3. Survival by Passenger Class
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival by Passenger Class')
plt.legend(['Died', 'Survived'])
plt.show()

#  4. Age Distribution
sns.histplot(data=df, x='Age', bins=30, kde=True, hue='Survived', multiple='stack')
plt.title('Age Distribution by Survival')
plt.show()

#  5. Fare Distribution
sns.boxplot(data=df, x='Survived', y='Fare')
plt.title('Fare Paid by Survival Status')
plt.xticks([0,1], ['Died', 'Survived'])
plt.show()

#  6. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#  7. Survival by Embarked Location
sns.countplot(data=df, x='Embarked', hue='Survived')
plt.title('Survival by Embarked Port')
plt.legend(['Died', 'Survived'])
plt.show()
