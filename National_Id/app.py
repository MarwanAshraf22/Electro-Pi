import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def perform_ocr(image):
    text = pytesseract.image_to_string(image)
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    data = {'Name': [], 'Value': []}
    
    for line in non_empty_lines:
        parts = line.split(':')
        if len(parts) == 2:
            name, value = parts[0].strip(), parts[1].strip()
            data['Name'].append(name)
            data['Value'].append(value)
            
    df = pd.DataFrame(data)
    return df

def main():
    st.title('Text extractor')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Extracting...")

        # Perform OCR
        image = Image.open(uploaded_file)
        df_text = perform_ocr(image)

        # Display the extracted text as DataFrame
        st.header("Extracted Text")
        st.dataframe(df_text)

if __name__ == "__main__":
    main()
