# Import necessary libraries
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from io import BytesIO

# Machine learning libraries
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
data = pd.read_excel('dataset.xlsx')  # Replace 'dataset.xlsx' with your actual dataset file

# Define input and output variables
features = data.iloc[:, :-2]
targets = data.iloc[:, -2:]

X = features
y = targets

# Normalize the data using Min-Max scaling
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Train XGBoost Regressor
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_scaled, y)

# GUI Application
def predict_values():
    try:
        # Get input values
        inputs = []
        for var in input_vars:
            val = float(var.get())
            inputs.append(val)
        
        # Convert inputs to a numpy array and reshape
        inputs_array = np.array(inputs).reshape(1, -1)
        
        # Normalize inputs using the same scaler as training data
        inputs_scaled = scaler_X.transform(inputs_array)
        
        # Predict using the trained model
        y_pred = model.predict(inputs_scaled)
        
        # Extract predictions
        P_cu_pred = y_pred[0][0]
        epsilon_cc_pred = y_pred[0][1]
        
        # Display the predictions with units
        messagebox.showinfo("Prediction", f"Predicted P_{{cu}}: {P_cu_pred:.2f} kN\nPredicted ε_{{cc}}: {epsilon_cc_pred:.5f} mm/mm")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to create LaTeX labels using matplotlib and PIL
def create_latex_label(master, var_name, unit):
    latex_text = f"${var_name} \\; (\\mathrm{{{unit}}})$"
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)
    ax.axis('off')
    plt.text(0.5, 0.5, latex_text, fontsize=4, ha='center', va='center')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(master, image=photo)
    label.image = photo  # Keep a reference!
    return label

# Create the main window
root = tk.Tk()
root.title("Prediction of P_{cu} and ε_{cc} using DNN Model")

# Set window size and position
root.geometry("800x600")

# Create a frame to hold inputs and table
main_frame = tk.Frame(root)
main_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Create a frame for inputs
inputs_frame = tk.Frame(main_frame)
inputs_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)

# Input fields with LaTeX labels and units
input_vars = []
input_labels = [
    ('A_{c}', 'mm²', 'Area of concrete section'),
    ("f'_{c}", 'MPa', 'Concrete strength of unconfined concrete'),
    ('t_{f}', 'mm', 'Total thickness of FRP wraps'),
    ('E_{f}', 'MPa', 'Elastic modulus of FRP'),
    ('A_{s}', 'mm²', 'Area of steel tubes'),
    ('f_{y}', 'MPa', 'Yield strength of internal steel tubes')
]

for idx, (var_name, unit, desc) in enumerate(input_labels):
    frame = ttk.Frame(inputs_frame)
    frame.pack(pady=5, anchor='w')
    
    label = create_latex_label(frame, var_name, unit)
    label.pack(side=tk.LEFT, padx=5)
    
    var = tk.StringVar()
    entry = ttk.Entry(frame, textvariable=var, width=20)
    entry.pack(side=tk.LEFT)
    
    input_vars.append(var)

# Create Predict button with blue color
predict_button = tk.Button(inputs_frame, text="Predict", command=predict_values, bg='blue', fg='white', width=15)
predict_button.pack(pady=20)

# Create a frame for the table
table_frame = tk.Frame(main_frame)
table_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

# Create the Treeview for variable descriptions
columns = ('Variable', 'Unit', 'Description')
tree = ttk.Treeview(table_frame, columns=columns, show='headings')

# Define headings
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=200)

# Add data to the table
# Add data to the table
data_table = [
    ('A_c', 'mm²', 'Area of concrete section'),  # Using '_c' to represent 'c' subscript
    ("f'_c", 'MPa', 'Concrete strength of unconfined concrete'),  # Using 'f'_c' where 'c' is subscript and prime is directly added
    ('t_f', 'mm', 'Total thickness of FRP wraps'),  # 'f' as subscript
    ('E_f', 'MPa', 'Elastic modulus of FRP'),  # 'f' as subscript
    ('A_s', 'mm²', 'Area of steel tubes'),  # 's' as subscript
    ('f_y', 'MPa', 'Yield strength of internal steel tubes'),  # 'y' as subscript
    ('P_cu', 'kN', 'Load carrying capacity of the FRP-confined column'),  # 'cu' as subscript
    ('ε_cc', 'mm/mm', 'Ultimate strain capacity of FRP-confined column')  # 'cc' as subscript
]


for item in data_table:
    tree.insert('', tk.END, values=item)

tree.pack(fill=tk.BOTH, expand=True)

# Add a scrollbar to the table
scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Start the GUI event loop
root.mainloop()
