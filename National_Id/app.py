import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd

def perform_ocr(image):
    text = pytesseract.image_to_string(image)
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    data = {'Text': non_empty_lines}
    df = pd.DataFrame(data)
    return df.iloc[2:7]  # Select rows 2 to 5 (0-based indexing)

def main():
    st.title("Simple OCR Web App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Perform OCR
        image = Image.open(uploaded_file)
        df_text = perform_ocr(image)

        # Display the extracted text as DataFrame
        st.header("Extracted Text")
        st.dataframe(df_text)

if __name__ == "__main__":
    main()
