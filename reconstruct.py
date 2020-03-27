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


def reconstruct_image_from_representation(args, should_reconstruct_content=True, should_visualize_representation=True):
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    content_img_path = os.path.join(content_images_dir, args.content_img_name)
    style_img_path = os.path.join(style_images_dir, args.style_img_name)

    img_path = content_img_path if should_reconstruct_content else style_img_path
    img = utils.load_image(img_path, return_pil=True)
    print('Image to reconstruct shape=', np.array(img).shape)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = transform(img).to(device)
    img = img.unsqueeze(0)
    img = utils.normalize_batch(img, should_divide=False)

    optimizing_img = Variable(torch.randn(1, 3, 256, 256, device=device, requires_grad=True)*0.1, requires_grad=True)

    vgg = Vgg16(requires_grad=False).to(device).eval()

    img_output = vgg(img)
    if should_reconstruct_content:
        content_representation = img_output.relu2_2[0]
        if should_visualize_representation:
            feature_maps_len = content_representation.size()[0]
            print(f'Number of feature maps: {feature_maps_len}')
            for i in range(feature_maps_len):
                feature_map = content_representation[i].to('cpu').numpy()

                feature_map /= np.max(feature_map)
                feature_map *= 255
                feature_map = np.uint8(feature_map)

                print('fm shape, min, max -', feature_map.shape, np.min(feature_map), np.max(feature_map))
                plt.imshow(feature_map)
                plt.show()
                feature_map = Image.fromarray(feature_map)
                if feature_map.mode != 'RGB':
                    feature_map = feature_map.convert('RGB')
                feature_map.save('fm_' + str(i).zfill(4) + '.png')
    else:
        style_representation = [utils.gram_matrix(y) for y in img_output]
        if should_visualize_representation:
            ar_len = len(style_representation)
            print(f'Number of Gram matrices: {ar_len}')
            for i in range(ar_len):
                Gram = style_representation[i][0].to('cpu').numpy()

                Gram /= np.max(Gram)
                Gram *= 255
                Gram = np.uint8(Gram)
                print(Gram.shape, np.min(Gram), np.max(Gram))

                plt.imshow(Gram)
                plt.show()
                Gram = Image.fromarray(Gram)
                if Gram.mode != 'RGB':
                    Gram = Gram.convert('RGB')
                Gram.save('gram_' + str(i).zfill(4) + '.png')

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer_type = 'lbfgs'
    num_of_iterations = {'adam': 6000, 'lbfgs': 500}
    save_frequency = {'adam': 1, 'lbfgs': 1}

    dump_path = os.path.join(
        default_resource_dir,
        'content_reconstruction_' + optimizer_type if should_reconstruct_content else 'style_reconstruction2_' + optimizer_type)
    os.makedirs(dump_path, exist_ok=True)

    if optimizer_type == 'adam':
        optimizer = Adam((optimizing_img,))
        train_step = make_train_step(vgg, loss_fn, optimizer, should_reconstruct_content)
        for it in range(num_of_iterations[optimizer_type]):
            loss, _ = train_step(optimizing_img, content_representation if should_reconstruct_content else style_representation)
            if it % save_frequency[optimizer_type] == 0:
                with torch.no_grad():
                    print('current loss=', loss)
                    save_display(optimizing_img, dump_path, int(it/save_frequency[optimizer_type]), should_save=True, should_display=False)
    elif optimizer_type == 'lbfgs':
        def closure():
            global cnt
            optimizer.zero_grad()
            loss = 0.
            if should_reconstruct_content:
                loss = loss_fn(content_representation, vgg(optimizing_img).relu2_2[0])
            else:
                out = vgg(optimizing_img)
                style_representation_prediction = [utils.gram_matrix(y) for y in out]
                for gram_gt, gram_hat in zip(style_representation, style_representation_prediction):
                    loss += (1 / len(style_representation)) * loss_fn(gram_gt[0], gram_hat[0])
                    print('loss', loss_fn(gram_gt[0], gram_hat[0]))
                loss *= 1e4
            loss.backward()
            with torch.no_grad():
                print('current loss=', loss.item())
                save_display(optimizing_img, dump_path, cnt, should_save=True, should_display=False)
                cnt += 1
            return loss

        optimizer = torch.optim.LBFGS((optimizing_img,), max_iter=500)
        optimizer.step(closure)