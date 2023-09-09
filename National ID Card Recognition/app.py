import cv2
import numpy as np
import pytesseract
import pandas as pd
import streamlit as st
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

with st.sidebar:
    st.title("National ID Card Recognition")
    st.info("This project is powered by electropi.ai using Tesseract OCR and cv2 to detect text on ID cards")

def upload_id_card():
    st.title("Upload Your ID card please in jpg or png format!")

    try:
        file = st.file_uploader("Upload Your Dataset", type=['jpg', 'png'])
        if file is not None:
            st.success("File uploaded successfully!")

        else:
            st.info("Please upload a file in jpg or png format.")
    except:
        st.error('Please upload jpg or png format only!')

    if st.button('Generate text') :
        st.image(file,'this is you uploaded id')
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        my_config = r'--psm 11 --oem 3'


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        df = pytesseract.image_to_data(gray, output_type=Output.DATAFRAME, config=my_config)
        df.dropna(inplace=True)
        df_text = pd.DataFrame(df['text'])
        st.write(df_text)

        boxes = pytesseract.image_to_boxes(img, config=my_config)
        for box in boxes.splitlines():
            box = box.split(' ')
            img_text = cv2.rectangle(img, (int(box[1]), height - int(box[2])),
                                     (int(box[3]), height - int(box[4])), (0, 255, 0), 2)

        st.image(img_text, 'This is detected text')

upload_id_card()
