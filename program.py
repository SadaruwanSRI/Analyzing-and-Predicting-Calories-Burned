import joblib
import tkinter as tk
from tkinter import messagebox

# Load the models
poly_features = joblib.load('poly_features.pkl')
poly_regressor_model = joblib.load('poly_regressor_model.pkl')

# Create the main window
root = tk.Tk()
root.title("Calories Burned Predictor")

# Create entries for user input
entries = []
for i,name in enumerate(['Gender', 'Age', 'Height', 'Weight', 'Heart_Rate',
       'Body_Temp']):
    tk.Label(root, text=name).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)


# Create checkboxes
checkboxes = []
for i in range(30):
    var = tk.IntVar()
    checkbox = tk.Checkbutton(root, text=f"Minutes {i+1}", variable=var)
    checkbox.grid(row=i//5, column=2 + i%5)
    checkboxes.append(var)

# Function to calculate and display the result
def calculate():
    try:
        # Get entry values
        entry_values = [float(entry.get()) for entry in entries]
        
        # Get checkbox values
        checkbox_values = [var.get() for var in checkboxes]
        
        # Combine entry and checkbox values
        input_values = entry_values + checkbox_values
        
        # Transform the input values using the polynomial features
        transformed_values = poly_features.transform([input_values])
        
        # Predict the result using the regressor model
        result = poly_regressor_model.predict(transformed_values)
        
        # Display the result
        messagebox.showinfo("Prediction", f"Predicted Calories Burned: {result[0]}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the calculate button
calculate_button = tk.Button(root, text="Calculate", command=calculate)
calculate_button.grid(row=7, column=0, columnspan=2)

# Run the main loop
root.mainloop()