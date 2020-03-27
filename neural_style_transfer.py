import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results
from models.definitions.vgg_nets import Vgg16, Vgg19

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


cnt = 0


# initially it takes some time for PyTorch to download the models
def prepare_model(model, device):
    if model == 'vgg16':
        # we are not tuning model weights -> we are tuning optimizing_img's pixels! (that's why requires_grad=False)
        return Vgg16(requires_grad=False, show_progress=True).to(device).eval()
    elif model == 'vgg19':
        return Vgg19(requires_grad=False, show_progress=True).to(device).eval()
    else:
        raise ValueError(f'{model} not supported.')


def neural_style_transfer(config):
    content_img_path = os.path.join(content_images_dir, config['content_img_name'])
    style_img_path = os.path.join(style_images_dir, config['style_img_name'])

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img_prenorm, content_img = utils.prepare_img(content_img_path, config['width'], device)
    style_img_prenorm, style_img = utils.prepare_img(style_img_path, config['width'], device)

    if config['init_method'] == 'random':
        # todo: try other values other than 0.1
        # hacky way to set standard deviation to 0.1 <- no specific reason for 0.1, it just works fine
        init_img = torch.randn(content_img.shape, device=device) * 0.1
    elif config['init_method'] == 'content':
        init_img = content_img_prenorm
    else:
        init_img = style_img_prenorm
    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net = prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_features = neural_net(content_img)
    style_features = neural_net(style_img)

    content_representation = content_features.relu3_3.squeeze(axis=0)
    style_representation = [utils.gram_matrix(y) for y in style_features]

    loss_fn = torch.nn.MSELoss(reduction='mean')

    content_weight = config['content_weight']
    style_weight = config['style_weight']
    tv_weight = config['tv_weight']

    def closure():
        global cnt
        optimizer.zero_grad()

        current_features = neural_net(optimizing_img)
        content_loss = loss_fn(content_representation, current_features.relu3_3[0])
        # todo: first normalize and then do the MSE - that's how it was done for Gram

        style_loss = 0.0
        style_representation_prediction = [utils.gram_matrix(y) for y in current_features]
        for gram_gt, gram_hat in zip(style_representation, style_representation_prediction):
            cur_loss = loss_fn(gram_gt[0], gram_hat[0])
            style_loss += (1 / len(style_representation)) * cur_loss
            # print('loss', loss_fn(gram_gt[0], gram_hat[0]))

        total_loss = content_weight*content_loss + style_weight*style_loss + tv_weight*utils.total_variation(optimizing_img)

        total_loss.backward()
        with torch.no_grad():
            print(f'L-BFGS | iteration: {cnt:03}, current loss={total_loss.item()}')
            utils.save_display(optimizing_img, dump_path, cnt, config['image_format'], should_save=True, should_display=False)
            cnt += 1
        return total_loss

    # magic numbers in general are a big no no - as this is usually not a hyperparam we make an exception to the rule
    optimizer = LBFGS((optimizing_img,), max_iter=500)
    optimizer.step(closure)

    return dump_path


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.png')  # saves images in the format: %04d.png

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='lion.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='mosaic.jpg')
    parser.add_argument("--width", type=int, help="width of content and style images", default=256)
    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e9)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=5e8)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e-3)
    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='random')
    parser.add_argument("--model", type=str, choices=['vgg16'], default='vgg16')  # only supporting vgg16 for now
    args = parser.parse_args()

    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['image_format'] = img_format

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    results_path = neural_style_transfer(optimization_config)

    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)
