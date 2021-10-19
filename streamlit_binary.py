import streamlit as st
import cv2
import io
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from PIL import Image, ImageEnhance

model_path = "D:/CellClass/resnet18_weights.pth"
@st.cache
def model_load():
    model = torch.load(model_path)
    return model

def main():
    st.set_page_config(layout="wide")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st.markdown("<h1 style='text-align: center; color: white;'>Binary Classification of Cell Image</h1>", unsafe_allow_html=True)
    menu = ['VEH00','AB30']
    st.sidebar.header('Input label')
    choice = st.sidebar.selectbox('What label?', menu)
    #st.image('banner.png',use_column_width = False
    #)
    

    sample_img = st.file_uploader('Upload your portrait here',type=['tif','jpg'])
    if sample_img is not None:
        col1, col2, col3 = st.columns(3)
        img = Image.open(sample_img).convert('L')

        enhancer = ImageEnhance.Contrast(img)
        enhancer1 = ImageEnhance.Brightness(img)
        factor = 0.001 #gives original image

        img_enhanced = enhancer.enhance(factor)
        img_enhanced = enhancer1.enhance(1)
        

        #Image = tf.image.decode_image(Image, channels=3).numpy()                  
        #Image = adjust_gamma(Image, gamma=gamma)
        with col1:
            col1.header('Input')
            st.image(img_enhanced)
        #input_image = loadtest(Image,cropornot=Autocrop)
        #prediction = comic_model(input_image, training=True)
        #prediction = tf.squeeze(prediction,0)
        #prediction = prediction* 0.5 + 0.5
        #prediction = tf.image.resize(prediction, 
                        #[outputsize, outputsize],
                        #method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #prediction=  prediction.numpy()
        with col2:
            tsfm_val = transforms.Compose([transforms.Resize(512),
                                transforms.CenterCrop(448),
                                 transforms.ToTensor(),
                                 #transforms.Normalize((.6662),(.2179)),
                                 transforms.ToPILImage()
                              ])
            img2 = tsfm_val(img)

            col2.header('Transformed Input')
            st.image(img2)    

        with col3:
            tsfm = transforms.Compose([transforms.Resize(512),
                    transforms.CenterCrop(448),
                        transforms.ToTensor(),
                        transforms.Normalize((.6662),(.2179)),
                    ])
            img3 = tsfm(img).unsqueeze(0).to(device)
            model = model_load()

            output = model(img3).to(device)
            _,output = torch.max(output.data,1)

            out = 'No-match'
            if output == 1:
                out = 'AB30'
            if output == 0:
                out = 'VEH00'
            
            col3.header('Output')


            st.markdown("Input label:"+ choice)
            #st.markdown(out, unsafe_allow_html=False)
            st.markdown("Predicted label:"+ out)
            #st.markdown(out, unsafe_allow_html=False)
         
        if choice == out:
                        st.markdown('## **Congrats! Prediction matched!**')
        else:
            st.markdown('## **Sorry, the model needs to be improved...**')

        
if __name__ == '__main__':
    main()