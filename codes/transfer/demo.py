import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

import argparse
import os

from net.network import SelfAttention, vgg, vgg_reverse, SAVANet
from net.models import Transform, SAVA_test

import streamlit as st
import os 
import glob
import time

parser = argparse.ArgumentParser()
parser.add_argument('--transformer_model', type=str, default='../../models/sava_training_hard/transformer_iter_96000.pth',
                    help='Directory path to a batch of transformer model')
parser.add_argument('--decoder_model', type=str, default='../../models/sava_training_hard/decoder_iter_96000.pth',
                    help='Directory path to a batch of decoder model')

parser.add_argument('--vgg_model', type=str, default='../../models/vgg/vgg_normalised.pth')
parser.add_argument('--attn_model', type=str, default='../../models/attention_training/attention_kernel_iter_80000.pth')

parser.add_argument('--content_dir', type=str, default='../../testing_data/content/',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='../../testing_data/style/',
                    help='Directory path to a batch of style images')
parser.add_argument('--result_dir', type=str, default='../../testing_data/result/',
                    help='Directory path to a result images')

args = parser.parse_args('')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def test_transform(size = 512):
    transform = transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((_R_MEAN/255.0, _G_MEAN/255.0, _B_MEAN/255.0), (0.5, 0.5, 0.5))
    ])
    return transform

def test_transform_inv():
    transform = transforms.Compose([
        transforms.Normalize((0, 0, 0), (2, 2, 2)),
        transforms.Normalize((-_R_MEAN/255.0, -_G_MEAN/255.0, -_B_MEAN/255.0), (1, 1, 1))
    ])
    return transform

content_tf = test_transform(int(800))
style_tf = test_transform(int(512))
content_tf_inv = test_transform_inv()

def state_to_device(parameter, device):
    state_dict = parameter.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(device)
    parameter.cuda()

def get_image(data_dir):
    img = Image.open(data_dir)
    img = img.resize((512, 512), Image.ANTIALIAS)
    return img

def attn_transfer(attn_map):
    channel_num = attn_map.size()[1]
    mean_sal = torch.mean(attn_map, 1, False)
    mean_sal_np = mean_sal.cpu().detach().numpy()
    mean_sal_np = mean_sal_np - np.min(mean_sal_np)
    mean_sal_np = mean_sal_np * 1.0 / np.max(mean_sal_np)
    return mean_sal_np[0]

def save_attn_map(attn_maps, save_dir=None):
    cnt = 0
    for attn in attn_maps:
        cnt += 1
        attn = attn_transfer(attn)
        plt.imshow(attn, cmap=cm.get_cmap('rainbow', 1000))
        plt.savefig(save_dir + 'attn_' + str(cnt) + '.jpg' )
    del attn_maps

def imshow_recon(img, save_dir=None):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("Transfer Output")
    if save_dir != None:
        torchvision.utils.save_image(img, save_dir)

def save_output(output, save_dir=None):
    content_result = content_tf_inv(output.squeeze())
    imshow_recon(torchvision.utils.make_grid(content_result.cpu()), save_dir)

def show_diagram(attn_maps):
    f, axarr = plt.subplots(2,1) 
    
    x = np.arange(1,attn_maps[0].shape[0]+1)
    axarr[0].plot(x,attn_maps[0])
    axarr[0].set_title("Relative Frequency Histogram")
    
    x = np.arange(1,attn_maps[1].shape[0]+1)
    axarr[1].plot(x,attn_maps[1])
    st.pyplot(f)

st.markdown('# SAVA Style Transfer Demostration')
st.markdown('## **1.** Inputs')
content_paths = os.path.join(args.content_dir, '*jpg')
content_files = glob.glob(content_paths)
content_names = sorted([ os.path.basename(path) for path in content_files ])

style_paths = os.path.join(args.style_dir, '*jpg')
style_files = glob.glob(style_paths)
style_names = sorted([ os.path.basename(path) for path in style_files ])

content_name = st.sidebar.selectbox(
    "Select a content image.",
    content_names
)
style_name = st.sidebar.selectbox(
    "Select a style image.",
    style_names
)

content_image = get_image(args.content_dir + content_name)
style_image = get_image(args.style_dir + style_name)

col1, col2 = st.columns(2)
with col1:
    st.image(content_image, caption = 'content image')
with col2:
    st.image(style_image, caption = 'style image')

start_transfer = st.sidebar.button('Start Transfer')
vgg.load_state_dict(torch.load(args.vgg_model))
state_to_device(vgg, device)
self_attn = SelfAttention()
self_attn.load_state_dict(torch.load(args.attn_model))
state_to_device(self_attn, device)
transformer = Transform(in_channel = 512, self_attn=self_attn)
transformer.load_state_dict(torch.load(args.transformer_model))
state_to_device(transformer, device)
vgg_reverse.load_state_dict(torch.load(args.decoder_model))
state_to_device(vgg_reverse, device)
model = SAVA_test(transformer=transformer, encoder=vgg, decoder=vgg_reverse)
model = model.cuda()

if start_transfer:
    content_img = content_tf(content_image.convert('RGB'))
    style_img = style_tf(style_image.convert('RGB'))

    content = torch.stack([content_img], dim = 0)
    content = content.to(device)
    style = torch.stack([style_img], dim = 0)
    style = style.to(device)
    output, swapped_features, [content_attn4_1, style_attn4_1, content_attn5_1, style_attn5_1] = model.transfer(content, style)
    
    content_mask = transformer.savanet4_1.attn_mask(content_attn4_1)
    style_mask = transformer.savanet4_1.attn_mask(style_attn4_1)
    attns = [content_attn4_1, content_mask, style_attn4_1, style_mask]
    save_attn_map(attns, args.result_dir)
    attn_imgs = [Image.open(args.result_dir + 'attn_' + str(cnt) + '.jpg') for cnt in range (1, 5)]
    
    st.markdown('## **2.** Self-Attention Maps')
    col1, col2, col3= st.columns([1,1, 2])
    with col1:
        st.image([attn_imgs[0], attn_imgs[2]], width = 150, caption = ['content attn map', 'style attn map'])
    with col2:
        st.image([attn_imgs[1], attn_imgs[3]], width = 150, caption = ['content attn mask', 'style attn mask'])
    with col3:
        attn_frequency = [content_attn4_1[0].reshape((1, -1)).cpu().detach().numpy()[0], style_attn4_1[0].reshape((1, -1)).cpu().detach().numpy()[0]]
        show_diagram(attn_frequency)

      
    
    st.markdown('## **3.** Style Transfer Result')    
    save_output(output, args.result_dir + 'test.jpg')
    output_image = get_image(args.result_dir + 'test.jpg')

    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.write("")
    with col2:
        st.image(output_image, caption = 'output image')
    with col3:
        st.write("")

    del content, style, output, swapped_features
    del content_attn4_1, style_attn4_1, content_attn5_1, style_attn5_1
    del content_mask, style_mask, attns, attn_imgs


