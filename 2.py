import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path
data = pd.read_csv("/Users/ayushkumar/Desktop/Python practice/StudentsPerformance.csv")

# Define features and target
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
numerical_features = ['reading score', 'writing score']
target = 'math score'  # Replace 'your_target_column' with the actual target column name

# Separate numerical and categorical features
X_categorical = pd.get_dummies(data[categorical_features], drop_first=True)
X_numerical = data[numerical_features]
X = pd.concat([X_categorical, X_numerical], axis=1)

# Encode the target variable if it is categorical
# If it is numerical, you can skip this step
# y = pd.get_dummies(data[target], drop_first=True)

y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Create an explainer
explainer = RegressionExplainer(model, X_train, y_train, cats = X_categorical.columns.tolist())

# Run the dashboard
db = ExplainerDashboard(explainer)
db.run(port=8051)
