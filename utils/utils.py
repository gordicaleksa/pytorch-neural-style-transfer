from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt


from models.definitions.vgg_nets import Vgg16, Vgg19


def load_image(filename, width=None, size=None, scale=None, return_pil=False):
    img = Image.open(filename)
    if width is not None and width != -1:
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

    img_prenormalized = transform_prenormalized(img).to(device).unsqueeze(0)
    img = transform(img).to(device).unsqueeze(0)

    return img_prenormalized, img


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')


def save_image(img, img_path):
    img = Image.fromarray(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(img_path)


def save_and_maybe_display(optimizing_img, dump_path, img_format, img_id, num_of_iterations, saving_freq=-1, should_display=False):
    out_img = optimizing_img.squeeze(axis=0).to('cpu').numpy()
    out_img = np.moveaxis(out_img, 0, 2)  # swap channel from 1st to 3rd position: ch, _, _ -> _, _, chr

    # for saving_freq == -1 save only the final result (otherwise save with frequency saving_freq and save the last pic)
    if img_id == num_of_iterations-1 or (saving_freq > 0 and img_id % saving_freq == 0):
        # print(np.min(out_img), np.mean(out_img), np.max(out_img))
        # _ = plt.hist(out_img[:, :, 0], bins='auto')  # arguments are passed to np.histogram
        # plt.title("Histogram with 'auto' bins")
        # plt.show()
        np.save(os.path.join(dump_path, 'out.npy'), out_img)

        out_img = Image.fromarray(np.uint8(get_uint8_range(out_img)))
        out_img.save(os.path.join(dump_path, str(img_id).zfill(img_format[0]) + img_format[1]))
    if should_display:
        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()


# initially it takes some time for PyTorch to download the models into local cache
def prepare_model(model, device):
    # we are not tuning model weights -> we are only tuning optimizing_img's pixels! (that's why requires_grad=False)
    if model == 'vgg16':
        model = Vgg16(requires_grad=False, show_progress=True)
    elif model == 'vgg19':
        model = Vgg19(requires_grad=False, show_progress=True)
    else:
        raise ValueError(f'{model} not supported.')

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
