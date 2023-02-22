import streamlit as st
from torchvision import models
from helper_functions import *
from style_tranfer import get_stylized_img

stylization_done = False
vgg = models.vgg19(pretrained=True).features
# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

st.title('Style Transfer')

st.write(
    "Upload a content image and a style image. By pressing the **Get Stylized "
    "Image** button a new image is going to be generated similar to your "
    "conent "
    "image, but with the style of the style image. \n"
    "Images should be the same size so both of them have resized to the same "
    "dimensions. \n"
    "DISCLAIMER: Streamlit cloud doesn't offer GPUs, so this process runs on "
    "CPU and may take a while!"
)

st.sidebar.write('## Upload & Download')

content_image = st.sidebar.file_uploader("Upload content image", type=["png",
                                                                       "jpg",
                                                                       "jpeg"])
style_image = st.sidebar.file_uploader("Upload style image", type=["png",
                                                                   "jpg",
                                                                   "jpeg"])
col1, col2 = st.columns(2)

if content_image is not None:
    col1.write("Content Image :camera:")
    content = Image.open(content_image)
    new_content = content.resize((500, 500))
    new_content.save('content_test.jpg')
    col1.image(new_content)

if style_image is not None:
    col2.write("Style Image :art:")
    style = Image.open(style_image)
    new_style = style.resize((500, 500))
    new_style.save('style_test.jpg')
    col2.image(new_style)

if content_image is not None and style_image is not None:
    begin_stylization = st.button('Get Stylized Image')
    if begin_stylization:
        target = get_stylized_img(model = vgg,
                                  content_path = 'content_test.jpg',
                                  style_path = 'style_test.jpg',
                                  device = device)
        target = im_convert(target)
        stylization_done = True

if stylization_done:
    st.write('Stylized Image')
    st.image(target)
    btn = st.download_button(label = "Download Stylized Image",
                             data = target, file_name = "stylized_img.jpg",
                             mime = "image/jpg")
    # btn = st.sidebar.button('Download Stylized Image',
    #                         on_click = target.save("stylized_img.jpg"))
