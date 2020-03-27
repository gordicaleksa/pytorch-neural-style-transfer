from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt


def load_image(filename, width=None, size=None, scale=None, is_pil=False):
    img = Image.open(filename)
    if width is not None:
        ratio = width / img.size[0]
        height = int(img.size[1] * ratio)
        img = img.resize((width, height), Image.ANTIALIAS)
    elif size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img if is_pil else np.array(img)


def prepare_img(img_path, new_width, device):
    img = load_image(img_path, width=new_width, is_pil=True)

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


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def save_image_from_vid(filename, img):
    # print(img.shape, np.min(img), np.max(img))
    img = np.clip(img, 0, 255)
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def save_display(optimizing_img, dump_path, img_id, should_save=True, should_display=False):
    disp_img = optimizing_img[0].to('cpu').numpy()
    out = np.moveaxis(disp_img, 0, 2)
    # print('info (shape, max, min):', out.shape, np.max(out), np.min(out))
    out -= np.min(out)
    out /= np.max(out)
    out *= 255
    out = np.uint8(out)
    if should_save:
        out = Image.fromarray(out)
        out.save(os.path.join(dump_path, str(img_id).zfill(4) + '.png'))
    if should_display:
        plt.imshow(out)
        plt.show()


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def denormalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    return batch*std + mean


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
