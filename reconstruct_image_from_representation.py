import utils.utils as utils
from models.definitions.vgg_nets import Vgg16, Vgg19

import os
import argparse
import torch
from torch.autograd import Variable
from torch.optim import Adam, LBFGS
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


cnt = 0


def make_train_step(model, loss_fn, optimizer, should_reconstruct_content):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.eval()
        # Makes predictions
        if should_reconstruct_content:
            yhat = model(x).relu2_2[0]
        else:
            out = model(x)
            yhat = [utils.gram_matrix(y) for y in out]

        # Computes loss
        loss = 0.
        if should_reconstruct_content:
            loss = loss_fn(y, yhat)
            loss *= 1e5
        else:
            for gram_gt, gram_hat in zip(y, yhat):
                loss += (1/len(y))*loss_fn(gram_gt[0], gram_hat[0])
            loss *= 1e10

        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item(), yhat

    # Returns the function that will be called inside the train loop
    return train_step


def reconstruct_image_from_representation(config):
    should_reconstruct_content = config['should_reconstruct_content']
    should_visualize_representation = config['should_visualize_representation']
    dump_path = os.path.join(config['output_img_dir'], 'c' if should_reconstruct_content else 's' + '_reconstruction_' + config['optimizer'])
    os.makedirs(dump_path, exist_ok=True)

    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    img_path = content_img_path if should_reconstruct_content else style_img_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, img = utils.prepare_img(img_path, config['width'], device)

    optimizing_img = Variable(torch.randn(img.shape, device=device)*0.1, requires_grad=True)

    neural_net, content_feature_maps_index, style_feature_maps_indices = utils.prepare_model(config['model'], device)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    num_of_iterations = {'adam': 6000, 'lbfgs': 500}
    save_frequency = {'adam': 10, 'lbfgs': 10}

    set_of_feature_maps = neural_net(img)

    #
    # Visualize feature maps and Gram matrices (depending whether you're reconstructing content or style img)
    #
    if should_reconstruct_content:
        content_representation = set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
        if should_visualize_representation:
            num_of_feature_maps = content_representation.size()[0]
            print(f'Number of feature maps: {num_of_feature_maps}')
            for i in range(num_of_feature_maps):
                feature_map = content_representation[i].to('cpu').numpy()
                feature_map = np.uint8(utils.get_uint8_range(feature_map))
                plt.imshow(feature_map)
                plt.title(f'Feature map from {config["content_img_name"]} image.')
                plt.show()
                filename = 'fm_' + str(i).zfill(config['img_format'][0]) + config['img_format'][1]
                utils.save_image(feature_map, os.path.join(dump_path, filename))
    else:
        style_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(set_of_feature_maps) if i in style_feature_maps_indices]
        if should_visualize_representation:
            ar_len = len(style_representation)
            print(f'Number of Gram matrices: {ar_len}')
            for i in range(ar_len):
                Gram_matrix = style_representation[i][0].to('cpu').numpy()
                Gram_matrix = np.uint8(utils.get_uint8_range(Gram_matrix))
                plt.imshow(Gram_matrix)
                plt.title(f'Gram matrix from {config["style_img_name"]}, shape={Gram_matrix.shape}')
                plt.show()
                filename = 'gram_' + str(i).zfill(config['img_format'][0]) + config['img_format'][1]
                utils.save_image(Gram_matrix, filename)

    #
    # Start of optimization procedure
    #
    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,))
        train_step = make_train_step(neural_net, loss_fn, optimizer, should_reconstruct_content)
        for it in range(num_of_iterations[optimizer_type]):
            loss, _ = train_step(optimizing_img, content_representation if should_reconstruct_content else style_representation)
            with torch.no_grad():
                print('current loss=', loss)
                utils.save_display(optimizing_img, dump_path, config['img_format'], it, num_of_iterations[optimizer_type], saving_freq=save_frequency[optimizer_type], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        def closure():
            global cnt
            optimizer.zero_grad()
            loss = 0.
            if should_reconstruct_content:
                loss = loss_fn(content_representation, neural_net(optimizing_img).relu2_2[0])
            else:
                out = neural_net(optimizing_img)
                style_representation_prediction = [utils.gram_matrix(y) for y in out]
                for gram_gt, gram_hat in zip(style_representation, style_representation_prediction):
                    loss += (1 / len(style_representation)) * loss_fn(gram_gt[0], gram_hat[0])
                    print('loss', loss_fn(gram_gt[0], gram_hat[0]))
                loss *= 1e4
            loss.backward()
            with torch.no_grad():
                print('current loss=', loss.item())
                utils.save_display(optimizing_img, dump_path, config['img_format'], cnt, num_of_iterations[optimizer_type], saving_freq=save_frequency[optimizer_type], should_display=False)
                cnt += 1
            return loss

        optimizer = torch.optim.LBFGS((optimizing_img,), max_iter=500)
        optimizer.step(closure)


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
    parser.add_argument("--should_reconstruct_content", type=bool, help="pick between content or style image", default=True)
    parser.add_argument("--should_visualize_representation", type=bool, help="visualize feature maps or Gram matrices", default=True)

    parser.add_argument("--content_img_name", type=str, help="content image name", default='lion.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='candy.jpg')
    parser.add_argument("--width", type=int, help="width of content and style images", default=256)

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