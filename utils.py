import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
import random
from tensorflow.keras.datasets import mnist

def draw_example_digits():
    """
    Displays example digits from the MNIST dataset
    """
    try:
        # Load MNIST dataset
        (x_train, y_train), _ = mnist.load_data()
        
        # Create a grid of example digits (one for each class 0-9)
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        axes = axes.flatten()
        
        for i in range(10):
            # Find an example of each digit
            idx = np.where(y_train == i)[0][0]
            
            # Display the image
            axes[i].imshow(x_train[idx], cmap='gray')
            axes[i].set_title(f"Digit: {i}")
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading MNIST examples: {e}")
        
        # Fallback to display a message if MNIST can't be loaded
        st.info("MNIST examples couldn't be loaded. Make sure TensorFlow is properly installed.")

def load_misclassified_examples(max_examples=5):
    """
    Load examples of misclassified digits for display
    
    Parameters:
    -----------
    max_examples : int
        Maximum number of examples to display
        
    Returns:
    --------
    list
        List of tuples (image, true_label, predicted_label)
    """
    # Check if misclassified directory exists
    if not os.path.exists('misclassified'):
        return []
    
    # Get list of misclassified files
    files = [f for f in os.listdir('misclassified') if f.endswith('.npy')]
    
    examples = []
    for file in files[:max_examples]:
        try:
            # Extract information from filename
            parts = file.split('_')
            true_label = int(parts[1])
            pred_label = int(parts[3])
            
            # Load the image
            image = np.load(os.path.join('misclassified', file))
            
            examples.append((image, true_label, pred_label))
        except Exception as e:
            print(f"Error loading misclassified example {file}: {e}")
    
    return examples

def save_user_feedback(image, true_label, predicted_label, feedback):
    """
    Save user feedback on predictions for improving the model
    
    Parameters:
    -----------
    image : numpy.ndarray
        The image that user drew
    true_label : int
        The correct digit (as indicated by user)
    predicted_label : int
        The model's prediction
    feedback : str
        User feedback comments
    """
    # Create feedback directory if it doesn't exist
    os.makedirs('user_feedback', exist_ok=True)
    
    # Generate a unique filename
    timestamp = int(time.time())
    filename = f"user_feedback/digit_{true_label}_pred_{predicted_label}_{timestamp}"
    
    # Save the image
    np.save(f"{filename}.npy", image)
    
    # Save the feedback
    with open(f"{filename}.txt", 'w') as f:
        f.write(feedback)

def generate_sample_misclassified():
    """
    Generate sample misclassified examples for demonstration
    """
    # Create directory
    os.makedirs('misclassified', exist_ok=True)
    
    # Load MNIST
    (_, _), (x_test, y_test) = mnist.load_data()
    
    # Create some examples
    examples = [
        (8, 3),  # 8 misclassified as 3
        (7, 9),  # 7 misclassified as 9
        (4, 9),  # 4 misclassified as 9
        (5, 3),  # 5 misclassified as 3
        (2, 7)   # 2 misclassified as 7
    ]
    
    for true_label, pred_label in examples:
        # Find an example of this digit
        indices = np.where(y_test == true_label)[0]
        if len(indices) > 0:
            idx = random.choice(indices)
            image = x_test[idx]
            
            # Save it as a misclassified example
            filename = f"misclassified/digit_{true_label}_pred_{pred_label}_{idx}.npy"
            np.save(filename, image)
