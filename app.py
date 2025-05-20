import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import time
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from model import ModelHandler
from preprocessing import preprocess_image
from utils import draw_example_digits, load_misclassified_examples
import cv2

# Page configuration
st.set_page_config(
    page_title="Digit Recognition App",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='title'>Handwritten Digit Recognition</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])
)
