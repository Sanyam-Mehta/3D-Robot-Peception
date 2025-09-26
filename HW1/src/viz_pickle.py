import pickle
import matplotlib.pyplot as plt

with open('/Users/sanyam/Projects/UMich/3D-robot-perception/HW1/graph1.pkl', 'rb') as f:
    fig = pickle.load(f)
    
# Required for unpickled figures
plt.show()  # Works with GUI backends (Qt, Tk, etc.)

# wait for user input to close the window
input("Press Enter to continue...")
