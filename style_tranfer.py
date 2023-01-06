from helper_functions import *
from torch import optim


def get_stylized_img(model, content_path: str, style_path: str, device):
    content = img2tensor(content_path).to(device)
    style = img2tensor(style_path).to(device)

    # get content and style features only once before training
    content_features = get_features(content, model)
    style_features = get_features(style, model)

    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in
                   style_features}

    # create a third "target" image and prep it for change
    # it is a good idea to start off with the target as a copy of our *content* image
    # then iteratively change its style
    target = content.clone().requires_grad_(True).to(device)

    # weights for each style layer
    # weighting earlier layers more will result in *larger* style artifacts
    # notice we are excluding `conv4_2` our content representation
    style_weights = {'conv1_1': 1.0,
                     'conv2_1': 0.9,
                     'conv3_1': 0.7,
                     'conv4_1': 0.3,
                     'conv5_1': 0.1}

    content_weight = 1  # alpha
    style_weight = 1e7  # beta

    optimizer = optim.Adam([target], lr = 0.09)
    steps = 80  # decide how many iterations to update your image (5000)

    total_loss = 0

    for i in range(1, steps + 1):
        # get the features from your target image
        target_features = get_features(target, model)

        # the content loss
        content_loss = torch.mean(
            (target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # then add to it for each layer's gram matrix loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean(
                (target_gram - style_gram) ** 2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)

        # calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return target
