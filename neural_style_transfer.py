import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results
# from models.definitions.vgg_nets import Vgg16, Vgg19

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


cnt = 0


def neural_style_transfer(config):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

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

    neural_net, content_layer_index, style_layers_indices = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    content_representation = content_img_set_of_feature_maps[content_layer_index].squeeze(axis=0)
    norm_coeff = 1 / content_representation.numel()**2

    style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_layers_indices]
    norm_weights = [1 / (y.shape[1]**2 * (y.shape[2]*y.shape[3])**2) for y in style_img_set_of_feature_maps]

    loss_fn = torch.nn.MSELoss(reduction='mean')

    content_weight = config['content_weight']
    style_weight = config['style_weight']
    tv_weight = config['tv_weight']
    # magic numbers in general are a big no no - as this is usually not a hyperparam we make an exception to the rule
    num_of_iterations = {
        "lbfgs": 500,
        "Adam": 500
    }

    def closure():
        global cnt
        optimizer.zero_grad()

        #
        # main logic
        #
        current_features = neural_net(optimizing_img)

        content_representation_prediction = current_features[content_layer_index].squeeze(axis=0)
        content_loss = norm_coeff*torch.nn.MSELoss(reduction='sum')(content_representation, content_representation_prediction)

        style_loss = 0.0
        w_blend_style = (1 / len(style_representation))
        current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_features) if cnt in style_layers_indices]
        for gram_gt, gram_hat, norm_weight in zip(style_representation, current_style_representation, norm_weights):
            cur_loss = norm_weight * torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
            style_loss += w_blend_style * cur_loss
            # print('loss term', cur_loss)

        total_loss = content_weight*content_loss + style_weight*style_loss + tv_weight*utils.total_variation(optimizing_img)

        total_loss.backward()
        #
        # end of main logic
        #
        with torch.no_grad():
            print(f'L-BFGS | iteration: {cnt:03}, '
                  f'current loss={total_loss.item():12.4f},'
                  f' content_loss={content_weight*content_loss.item():12.4f}'
                  f' style loss={style_weight*style_loss.item():12.4f}')
            utils.save_and_maybe_display(
                optimizing_img,
                dump_path,
                config['img_format'],
                cnt,
                num_of_iterations['lbfgs'],
                saving_freq=config['saving_freq'],
                should_display=False)
            cnt += 1
        return total_loss

    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'])
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
    parser.add_argument("--style_img_name", type=str, help="style image name", default='candy.jpg')
    parser.add_argument("--width", type=int, help="width of content and style images", default=512)
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=10)
    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e4)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=1e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e-3)
    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='random')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    args = parser.parse_args()

    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    results_path = neural_style_transfer(optimization_config)

    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)
