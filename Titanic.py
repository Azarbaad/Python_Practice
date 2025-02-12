# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Load the Titanic dataset
df = pd.read_csv('DataSet/titanic.csv')

# Display all columns of the first 5 rows
print(df.head().to_string())

# Get dataset information
print(df.info())

# Get statistical summary
print(df.describe())

#  Get number of rows and columns
print(df.shape)

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Fill missing 'age' with mean and round to nearest integer
df['age'] = df['age'].fillna(df['age'].mean()).round().astype(int)

# Drop irrelevant columns (like 'ticket', 'cabin', 'name', 'home.dest','body', 'boat', 'embarked')
df.drop(['ticket', 'cabin', 'name', 'home.dest', 'body', 'boat', 'embarked'], axis=1, inplace=True)

# Dropping rows with missing 'fare', 'pclass', 'survived', 'sex', 'sibsp' and 'parch' values
df.dropna(subset=['fare', 'pclass', 'survived', 'sex', 'sibsp', 'parch' ], inplace=True)

missing_values = df.isnull().sum()
print(missing_values)
print(df.shape)

# Check for duplicate rows
duplicates = df.duplicated()

# Display the number of duplicates
print(f"Number of duplicate rows: {duplicates.sum()}")

# Optionally, display the duplicated rows
if duplicates.sum() > 0:
    print(df[duplicates])

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Display the shape of the DataFrame after removing duplicates
print(f"Shape of the dataset after removing duplicates: {df.shape}")

# visualize the survival rate based on 'sex', 'age', 'pclass', 'fare', 'sibsp' and 'parch'
# List of columns and plot types
# plots_info = [
#
#     ('sex', 'bar'),
#     ('age', 'hist'),
#     ('pclass', 'bar'),
#     ('fare', 'hist'),
#     ('sibsp', 'bar'),
#     ('parch', 'bar')
# ]
#
# # Initialize the figure
# plt.figure(figsize=(10, 6))
#
# # Loop through the plots_info list and generate the plots dynamically
# for i, (column, plot_type) in enumerate(plots_info, 1):
#     plt.subplot(2, 3, i)
#
#     # Choose barplot or histplot based on the plot_type
#     if plot_type == 'bar':
#         sns.barplot(x=column, y='survived', data=df)
#     elif plot_type == 'hist':
#         sns.histplot(x=column, hue='survived', data=df, bins=10, kde=False)
#
#     plt.title(f'Survival rate by {column}')
#
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Display the plots
# plt.show()
#
# # Checking for outliers using a boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(df['fare'])
# plt.title('Boxplot for Fare (checking for outliers)')
# plt.show()
#
# # Display the plots
# plt.show()
#
# # Filter out rows where fare is greater than 300
# df = df[df['fare'] <= 300]
#
# # Display the number of rows after outlier removal
# print(df.shape)

# Now the 'fare' column no longer contains values above 300

# Convert categorical columns (like 'sex', 'embarked') to numerical
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Define target variable (y) and features (X)
X = df[['sex', 'age', 'pclass', 'fare', 'sibsp', 'parch']]  # Features
y = df['survived']  # Target

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List to store results of polynomial degrees and their accuracy
best_accuracy = 0
best_degree = 0

# Loop through polynomial degrees (from 1 to 4 for example)
for degree in range(1, 11):
    # Adding polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Now apply scaling to the polynomial features
    scaler = StandardScaler()
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)  # Fit and transform on training set
    X_test_poly_scaled = scaler.transform(X_test_poly)  # Transform test set

    # Train logistic regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_poly_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_poly_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Degree: {degree}, Accuracy: {accuracy}')

    # Find the best degree based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_degree = degree

print(f'Best Degree: {best_degree}, Best Accuracy: {best_accuracy}')

# Final model with the best degree
poly_best = PolynomialFeatures(degree=best_degree)
X_train_poly_best = poly_best.fit_transform(X_train)
X_test_poly_best = poly_best.transform(X_test)

# Scale the final polynomial features
scaler = StandardScaler()  # Reinitialize the scaler
X_train_poly_best_scaled = scaler.fit_transform(X_train_poly_best)  # Fit-transform on best polynomial train set
X_test_poly_best_scaled = scaler.transform(X_test_poly_best)  # Transform on test set

final_model = LogisticRegression(max_iter=10000)
final_model.fit(X_train_poly_best_scaled, y_train)
final_accuracy = accuracy_score(y_test, final_model.predict(X_test_poly_best_scaled))

print(f'Final Model Accuracy: {final_accuracy}')

# Function to predict survival
def predict_survival():
    # Collect user input
    age = float(entry_fields['Age :'].get())
    sex = gender_var.get()  # 0 for male, 1 for female
    pclass = int(entry_fields['Pclass :'].get())
    sibsp = int(entry_fields['Siblings/Spouse (SibSp) :'].get())
    parch = int(entry_fields['Parents/Children (Parch) :'].get())
    fare = float(entry_fields['Fare :'].get())

    # Prepare the input for the model (same order as training data)
    input_data = [[pclass, sex, age, sibsp, parch, fare]]

    # Apply the same scaler and polynomial transformation used during training
    input_data_scaled = scaler.transform(input_data)
    input_data_poly = poly_best.transform(input_data_scaled)

    survival_probability = final_model.predict_proba(input_data_poly)[0][1]  # Probability of surviving (class 1)

    # Convert to percentage
    survival_percentage = survival_probability * 100

    # Display the result in a message box
    messagebox.showinfo("Prediction", f"The model predicts that the person has a {final_accuracy * 100:.2f}% chance of survival.")
def add_placeholder(entry, placeholder):
    entry.insert(0, placeholder)
    entry.bind("<FocusIn>", lambda event: clear_placeholder(entry, placeholder))
    entry.bind("<FocusOut>", lambda event: restore_placeholder(entry, placeholder))

def clear_placeholder(entry, placeholder):
    if entry.get() == placeholder:
        entry.delete(0, 'end')
        entry.config(fg='black')

def restore_placeholder(entry, placeholder):
    if entry.get() == '':
        entry.insert(0, placeholder)
        entry.config(fg='grey')


def predict_survival():
    try:
        # Collect and validate user input
        age = float(entry_fields['Age :'].get())
        if age < 0 or age > 80:
            raise ValueError("Age must be between 0 and 80.")

        sex = int(gender_var.get())
        if sex not in [0, 1]:
            raise ValueError("Sex must be 0 (Male) or 1 (Female).")

        pclass = int(entry_fields['Pclass :'].get())
        if pclass not in [1, 2, 3]:
            raise ValueError("Pclass must be 1, 2, or 3.")

        sibsp = int(entry_fields['Siblings/Spouse (SibSp) :'].get())
        if sibsp < 0 or sibsp > 8:
            raise ValueError("Siblings/Spouse (SibSp) must be between 0 and 8.")

        parch = int(entry_fields['Parents/Children (Parch) :'].get())
        if parch < 0 or parch > 6:
            raise ValueError("Parents/Children (Parch) must be between 0 and 6.")

        fare = float(entry_fields['Fare :'].get())
        if fare < 0 or fare > 500:
            raise ValueError("Fare must be between 0 and 500.")

        # Prepare the input for the model (same order as training data)
        input_data = pd.DataFrame([[sex, age, pclass, fare, sibsp, parch]],
                                  columns=['sex', 'age', 'pclass', 'fare', 'sibsp', 'parch'])
        input_data_poly = poly_best.transform(input_data)
        input_data_poly_scaled = scaler.transform(input_data_poly)


        # Predict the survival probability using the best polynomial regression model
        survival_probability = final_model.predict_proba(input_data_poly_scaled)[0][1]
        survival_percentage = survival_probability * 100

        # Display the result in a message box
        messagebox.showinfo("Prediction",
                            f"The model predicts that the person has a {survival_percentage:.2f}% chance of survival.")

    except ValueError as e:
        messagebox.showerror("Invalid Input", str(e))
# Create labels and entry widgets in a loop with placeholders
placeholders = {
    'Age :': ' 0 - 80 ',
    'Pclass :': ' 1, 2 or 3',
    'Siblings/Spouse (SibSp) :': ' 0 - 8',
    'Parents/Children (Parch) :': ' 0 - 9',
    'Fare :': ' 0 - 300 '
}
# Function to create the radio button for gender selection
def create_gender_selection():
    gender_var = tk.IntVar()  # 0 for male, 1 for female
    gender_var.set(0)  # Default is male

    # Create radio buttons for male and female
    canvas.create_text(180, 50, text="Sex :", font=('Arial', 10, 'bold'), fill='black', anchor='e')
    male_radio = tk.Radiobutton(root, text='Male', variable=gender_var, value=0, font=('Arial', 10))
    female_radio = tk.Radiobutton(root, text='Female', variable=gender_var, value=1, font=('Arial', 10))

    canvas.create_window(230, 50, window=male_radio)
    canvas.create_window(300, 50, window=female_radio)

    return gender_var
# GUI Setup
root = tk.Tk()
root.title("Titanic Survival Predictor")
root.iconbitmap('Images/ship_boat_vessel_icon_183225.ico')
root.resizable(0, 0)
root.geometry('350x300')
right = int(root.winfo_screenwidth() / 2 - 350 / 2)
down = int(root.winfo_screenheight() / 2 - 300 / 2)
root.geometry('+{}+{}'.format(right, down))

# Load the background image
bg_image = Image.open('Images/ship_boat_vessel_icon_183225.png')  # Replace with your image path
bg_image = bg_image.resize((350, 300), Image.Resampling.LANCZOS)  # Resize to fit the window
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Canvas widget to hold the background image
canvas = tk.Canvas(root, width=350, height=300)
canvas.pack(fill='both', expand=True)

# Display the image on the Canvas
canvas.create_image(0, 0, image=bg_photo, anchor='nw')

# Create a semi-transparent overlay (use rgba where 'a' controls the opacity)
canvas.create_rectangle(0, 0, 350, 300, fill='#ffffff', stipple='gray50')  # Semi-transparent white

# Define the fields to loop over
fields = ['Age :', 'Pclass :', 'Siblings/Spouse (SibSp) :', 'Parents/Children (Parch) :', 'Fare :']
entry_fields = {}

gender_var = create_gender_selection()

# Create labels and entry widgets in a loop
for i, field in enumerate(fields):
    canvas.create_text(180, 80 + (i * 30), text=field, font=('Arial', 10, 'bold'), fill='black', anchor='e')  # No label widget

    entry = tk.Entry(root, fg='gray')
    add_placeholder(entry, placeholders[field])
    canvas.create_window(260, 80 + (i * 30), window=entry)
    entry_fields[field] = entry

# Predict button
predict_button = tk.Button(root, text='chance of survival', command=predict_survival, font=('Arial', 10, 'bold'),padx=10, pady=5)  # Make the button bigger with padding
canvas.create_window(175, 250, window=predict_button)

root.mainloop()
