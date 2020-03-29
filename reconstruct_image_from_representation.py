import utils.utils as utils

import os
import argparse
import torch
from torch.autograd import Variable
from torch.optim import Adam, LBFGS
import numpy as np
import matplotlib.pyplot as plt


def make_tuning_step(model, optimizer, target_representation, should_reconstruct_content, content_feature_maps_index, style_feature_maps_indices):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        # Finds the current representation
        set_of_feature_maps = model(optimizing_img)
        if should_reconstruct_content:
            current_representation = set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
        else:
            current_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(set_of_feature_maps) if i in style_feature_maps_indices]

        # Computes the loss between current and target representations
        loss = 0.0
        if should_reconstruct_content:
            loss = torch.nn.MSELoss(reduction='mean')(target_representation, current_representation)
        else:
            for gram_gt, gram_hat in zip(target_representation, current_representation):
                loss += (1 / len(target_representation)) * torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])

        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item(), current_representation

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def reconstruct_image_from_representation(config):
    should_reconstruct_content = config['should_reconstruct_content']
    should_visualize_representation = config['should_visualize_representation']
    dump_path = os.path.join(config['output_img_dir'], ('c' if should_reconstruct_content else 's') + '_reconstruction_' + config['optimizer'])
    os.makedirs(dump_path, exist_ok=True)

    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    img_path = content_img_path if should_reconstruct_content else style_img_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, img = utils.prepare_img(img_path, config['width'], device)

    optimizing_img = Variable(torch.randn(img.shape, device=device)*0.1, requires_grad=True)  # standard deviation = 0.1

    # indices pick relevant feature maps (say conv4_1, relu1_1, etc.)
    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    # don't want to expose everything that's not crucial so some things are hardcoded
    num_of_iterations = {'adam': 6000, 'lbfgs': 500}
    save_frequency = {'adam': 100, 'lbfgs': 10}

    set_of_feature_maps = neural_net(img)

    #
    # Visualize feature maps and Gram matrices (depending whether you're reconstructing content or style img)
    #
    if should_reconstruct_content:
        target_content_representation = set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
        if should_visualize_representation:
            num_of_feature_maps = target_content_representation.size()[0]
            print(f'Number of feature maps: {num_of_feature_maps}')
            for i in range(num_of_feature_maps):
                feature_map = target_content_representation[i].to('cpu').numpy()
                feature_map = np.uint8(utils.get_uint8_range(feature_map))
                plt.imshow(feature_map)
                plt.title(f'Feature map {i+1}/{num_of_feature_maps} from layer {content_feature_maps_index_name[1]} (model={config["model"]}) for {config["content_img_name"]} image.')
                plt.show()
                filename = 'fm_' + str(i).zfill(config['img_format'][0]) + config['img_format'][1]
                utils.save_image(feature_map, os.path.join(dump_path, filename))
    else:
        target_style_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(set_of_feature_maps) if i in style_feature_maps_indices_names[0]]
        if should_visualize_representation:
            num_of_gram_matrices = len(target_style_representation)
            print(f'Number of Gram matrices: {num_of_gram_matrices}')
            for i in range(num_of_gram_matrices):
                Gram_matrix = target_style_representation[i].squeeze(axis=0).to('cpu').numpy()
                Gram_matrix = np.uint8(utils.get_uint8_range(Gram_matrix))
                plt.imshow(Gram_matrix)
                plt.title(f'Gram matrix from layer {style_feature_maps_indices_names[1][i]} (model={config["model"]}) for {config["style_img_name"]} image.')
                plt.show()
                filename = 'gram_' + str(i).zfill(config['img_format'][0]) + config['img_format'][1]
                utils.save_image(Gram_matrix, os.path.join(dump_path, filename))

    #
    # Start of optimization procedure
    #
    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,))
        target_representation = target_content_representation if should_reconstruct_content else target_style_representation
        tuning_step = make_tuning_step(neural_net, optimizer, target_representation, should_reconstruct_content, content_feature_maps_index_name[0], style_feature_maps_indices_names[0])
        for it in range(num_of_iterations[config['optimizer']]):
            loss, _ = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Iteration: {it}, current {"content" if should_reconstruct_content else "style"} loss={loss:10.8f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config['img_format'], it, num_of_iterations[config['optimizer']], saving_freq=save_frequency[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        cnt = 0

        # closure is a function required by L-BFGS optimizer
        def closure():
            nonlocal cnt
            optimizer.zero_grad()
            loss = 0.0
            if should_reconstruct_content:
                loss = loss_fn(target_content_representation, neural_net(optimizing_img)[content_feature_maps_index_name[0]].squeeze(axis=0))
            else:
                current_set_of_feature_maps = neural_net(optimizing_img)
                current_style_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(current_set_of_feature_maps) if i in style_feature_maps_indices_names[0]]
                for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                    loss += (1 / len(target_style_representation)) * torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
            loss.backward()
            with torch.no_grad():
                print(f'Iteration: {cnt}, current {"content" if should_reconstruct_content else "style"} loss={loss.item()}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config['img_format'], cnt, num_of_iterations[config['optimizer']], saving_freq=save_frequency[config['optimizer']], should_display=False)
                cnt += 1
            return loss

        optimizer = torch.optim.LBFGS((optimizing_img,), max_iter=500)
        optimizer.step(closure)


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason (default img locations and img dump format)
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
    parser.add_argument("--should_reconstruct_content", type=bool, help="pick between content or style image reconstruction", default=True)
    parser.add_argument("--should_visualize_representation", type=bool, help="visualize feature maps or Gram matrices", default=False)

    parser.add_argument("--content_img_name", type=str, help="content image name", default='lion.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='starry_night.jpg')
    parser.add_argument("--width", type=int, help="width of content and style images (-1 keep original)", default=512)

    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg16')
    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    args = parser.parse_args()

    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    # reconstruct style or content image purely from their representation
    reconstruct_image_from_representation(optimization_config)
