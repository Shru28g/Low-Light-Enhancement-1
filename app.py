%%writefile app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2

from PIL import Image


class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)


class Encoder(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.conv = ConvBlock(in_c, out_c)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        s = self.conv(x)

        p = self.pool(s)

        return s, p


class Decoder(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_c,
            out_c,
            kernel_size=2,
            stride=2
        )

        self.conv = ConvBlock(out_c*2, out_c)

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([x, skip], axis=1)

        x = self.conv(x)

        return x


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.e1 = Encoder(3,64)
        self.e2 = Encoder(64,128)
        self.e3 = Encoder(128,256)

        self.b = ConvBlock(256,512)

        self.d1 = Decoder(512,256)
        self.d2 = Decoder(256,128)
        self.d3 = Decoder(128,64)

        self.out = nn.Conv2d(64,3,1)

    def forward(self, x):

        s1,p1 = self.e1(x)
        s2,p2 = self.e2(p1)
        s3,p3 = self.e3(p2)

        b = self.b(p3)

        d1 = self.d1(b,s3)
        d2 = self.d2(d1,s2)
        d3 = self.d3(d2,s1)

        out = self.out(d3)

        return torch.sigmoid(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)

model.load_state_dict(
    torch.load("low_light_model.pth", map_location=device)
)

model.eval()


st.title("Low Light Image Enhancement")


uploaded_file = st.file_uploader(
    "Upload Low-Light Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    original_size = image.size

    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

    output = output.squeeze(0).permute(1,2,0).cpu().numpy()

    output = np.clip(output, 0, 1)

    output = (output * 255).astype(np.uint8)

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    output = cv2.filter2D(output, -1, kernel)

    output_image = Image.fromarray(output)

    output_image = output_image.resize(original_size)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original")

    with col2:
        st.image(output_image, caption="Enhanced")
