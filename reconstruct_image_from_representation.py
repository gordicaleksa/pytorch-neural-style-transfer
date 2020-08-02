import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

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
    dump_path = os.path.join(dump_path, os.path.basename(config['content_img_name']).split('.')[0] if should_reconstruct_content else os.path.basename(config['style_img_name']).split('.')[0])
    os.makedirs(dump_path, exist_ok=True)

    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    img_path = content_img_path if should_reconstruct_content else style_img_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = utils.prepare_img(img_path, config['height'], device)

    gaussian_noise_img = np.random.normal(loc=0, scale=90., size=img.shape).astype(np.float32)
    white_noise_img = np.random.uniform(-90., 90., img.shape).astype(np.float32)
    init_img = torch.from_numpy(white_noise_img).float().to(device)
    optimizing_img = Variable(init_img, requires_grad=True)

    # indices pick relevant feature maps (say conv4_1, relu1_1, etc.)
    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)

    # don't want to expose everything that's not crucial so some things are hardcoded
    num_of_iterations = {'adam': 3000, 'lbfgs': 350}

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
                filename = f'fm_{config["model"]}_{content_feature_maps_index_name[1]}_{str(i).zfill(config["img_format"][0])}{config["img_format"][1]}'
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
                filename = f'gram_{config["model"]}_{style_feature_maps_indices_names[1][i]}_{str(i).zfill(config["img_format"][0])}{config["img_format"][1]}'
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
                utils.save_and_maybe_display(optimizing_img, dump_path, config, it, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        cnt = 0

        # closure is a function required by L-BFGS optimizer
        def closure():
            nonlocal cnt
            optimizer.zero_grad()
            loss = 0.0
            if should_reconstruct_content:
                loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, neural_net(optimizing_img)[content_feature_maps_index_name[0]].squeeze(axis=0))
            else:
                current_set_of_feature_maps = neural_net(optimizing_img)
                current_style_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(current_set_of_feature_maps) if i in style_feature_maps_indices_names[0]]
                for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                    loss += (1 / len(target_style_representation)) * torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
            loss.backward()
            with torch.no_grad():
                print(f'Iteration: {cnt}, current {"content" if should_reconstruct_content else "style"} loss={loss.item()}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
                cnt += 1
            return loss

        optimizer = torch.optim.LBFGS((optimizing_img,), max_iter=num_of_iterations[config['optimizer']], line_search_fn='strong_wolfe')
        optimizer.step(closure)

    return dump_path


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason (default img locations and img dump format)
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--should_reconstruct_content", type=bool, help="pick between content or style image reconstruction", default=True)
    parser.add_argument("--should_visualize_representation", type=bool, help="visualize feature maps or Gram matrices", default=False)

    parser.add_argument("--content_img_name", type=str, help="content image name", default='lion.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='ben_giles.jpg')
    parser.add_argument("--height", type=int, help="width of content and style images (-1 keep original)", default=500)

    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=1)
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--reconstruct_script", type=str, help='dummy param - used in saving func', default=True)
    args = parser.parse_args()

    # just wrapping settings into a dictionary
    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    # reconstruct style or content image purely from their representation
    results_path = reconstruct_image_from_representation(optimization_config)

    # create_video_from_intermediate_results(results_path, img_format)
