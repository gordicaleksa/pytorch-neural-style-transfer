from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt


def load_image(filename, width=None, size=None, scale=None, return_pil=False):
    img = Image.open(filename)
    if width is not None:
        ratio = width / img.size[0]  # PIL size returns (w, h)
        height = int(img.size[1] * ratio)
        img = img.resize((width, height), Image.ANTIALIAS)
    elif size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img if return_pil else np.array(img)


def prepare_img(img_path, new_width, device):
    img = load_image(img_path, width=new_width, return_pil=True)

    transform_prenormalized = transforms.Compose([
        transforms.ToTensor(),
    ])

    # normalize using ImageNet's mean and std (VGG was trained on images normalized this way)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_prenormalized = transform_prenormalized(img).to(device)
    img_prenormalized = img_prenormalized.unsqueeze(0)

    img = transform(img).to(device)
    img = img.unsqueeze(0)

    return img_prenormalized, img


def save_display(optimizing_img, dump_path, img_id, img_format, should_save=True, should_display=False):
    out_img = optimizing_img.squeeze(axis=0).to('cpu').numpy()
    out_img = np.moveaxis(out_img, 0, 2)  # swap channel from 1st to 3rd position: ch, _, _ -> _, _, ch
    out_img -= np.min(out_img)
    out_img /= np.max(out_img)  # bring image into [0.0, 1.0] range
    out_img = np.uint8(out_img * 255)
    if should_save:
        out_img = Image.fromarray(out_img)
        out_img.save(os.path.join(dump_path, str(img_id).zfill(img_format[0]) + img_format[1]))
    if should_display:
        plt.imshow(out_img)
        plt.show()


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
