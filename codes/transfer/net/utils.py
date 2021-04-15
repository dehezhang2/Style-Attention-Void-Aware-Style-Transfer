import torch
import torch.nn as nn

_R_MEAN = 123.68 / 255.0
_G_MEAN = 116.78 / 255.0
_B_MEAN = 103.94 / 255.0

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adain_normalization(features):
    epsilon = 1e-5
    colorization_kernels, mean_features = torch.std_mean(features, (2, 3), keepdim=True)
    normalized_features = torch.div(features - mean_features, epsilon + colorization_kernels)
    return normalized_features, colorization_kernels, mean_features

def adain_colorization(normalized_features, colorization_kernels, mean_features):
    return colorization_kernels * normalized_features + mean_features

def zca(features):
    # [b, c, h, w]
    shape = features.shape

    # reshape the features to orderless feature vectors
    mean_features = torch.mean(features, dim=(2, 3), keepdims=True)
    unbiased_features = (features - mean_features).view(shape[0], shape[1], -1) # [b, c, h*w]

    # get the convariance matrix
    gram = torch.bmm(unbiased_features, unbiased_features.permute(0, 2, 1)) # [b, c, c]
    gram = gram / (shape[1] * shape[2] * shape[3])

    # converting the feature spaces
    u, s, v = torch.svd(gram.cpu(), compute_uv=True)
    # u: [b, c, c], s: [b, c], v: [b, c, c]
    s = torch.unsqueeze(s, dim=1)

    # get the effective singular values
    valid_index = (s > 0.00001).float()
    temp = torch.empty(s.shape).fill_(0.00001)
    s_effective = torch.max( s,  temp)
    sqrt_inv_s_effective = torch.sqrt(1.0 / s_effective) * valid_index

    sqrt_inv_s_effective = sqrt_inv_s_effective.cuda()
    u = u.cuda()
    v = v.cuda()

    # normalized features
    normalized_features = torch.bmm(u.permute(0, 2, 1), unbiased_features)
    normalized_features = sqrt_inv_s_effective.permute(0, 2, 1) * normalized_features
    normalized_features = torch.bmm(v, normalized_features)
    normalized_features = normalized_features.view(shape)
    
    return normalized_features

def zca_normalization(features):
    # [b, c, h, w]
    shape = features.shape

    # reshape the features to orderless feature vectors
    mean_features = torch.mean(features, dim=(2, 3), keepdims=True)
    unbiased_features = (features - mean_features).view(shape[0], shape[1], -1) # [b, c, h*w]

    # get the convariance matrix
    gram = torch.bmm(unbiased_features, unbiased_features.permute(0, 2, 1)) # [b, c, c]
    gram = gram / (shape[1] * shape[2] * shape[3])

    # converting the feature spaces
    u, s, v = torch.svd(gram.cpu(), compute_uv=True)
    # u: [b, c, c], s: [b, c], v: [b, c, c]
    s = torch.unsqueeze(s, dim=1)

    # get the effective singular values
    valid_index = (s > 0.00001).float()
    temp = torch.empty(s.shape).fill_(0.00001)
    s_effective = torch.max( s,  temp)
    sqrt_s_effective = torch.sqrt(s_effective) * valid_index
    sqrt_inv_s_effective = torch.sqrt(1.0 / s_effective) * valid_index

    sqrt_inv_s_effective = sqrt_inv_s_effective.cuda()
    sqrt_s_effective = sqrt_s_effective.cuda()
    u = u.cuda()
    v = v.cuda()

    # colorization functions
    colorization_kernel = torch.bmm((u * sqrt_s_effective), v.permute(0, 2, 1))

    # normalized features
    normalized_features = torch.bmm(u.permute(0, 2, 1), unbiased_features)
    normalized_features = sqrt_inv_s_effective.permute(0, 2, 1) * normalized_features
    normalized_features = torch.bmm(v, normalized_features)
    normalized_features = normalized_features.view(shape)
    return normalized_features, colorization_kernel, mean_features

def zca_colorization(normalized_features, colorization_kernel, mean_features):
    # broadcasting the tensors for matrix multiplication
    shape = normalized_features.shape
    normalized_features = normalized_features.view(shape[0], shape[1], -1) # [b, c, h*w]

    colorized_features = torch.bmm(normalized_features.permute(0, 2, 1), colorization_kernel).permute(0, 2, 1)
    colorized_features = colorized_features.view(shape) + mean_features

    return colorized_features

def hw_flatten(x):
    # [b, c, h, w] -> [b, c, h * w]
    return x.view(x.shape[0], x.shape[1], -1)

def batch_mean_image_subtraction(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if len(images.shape) != 4:
        raise ValueError('Input must be of size [batch, height, width, C>0')
    num_channels = images.shape[1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = list(torch.split(images, 1, dim=1))
    # print(images)
    # print(channels)
    for i in range(num_channels):
        channels[i] = channels[i] - means[i]
    return torch.cat(channels, dim=1)

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    
def project_features(features, projection_module='ZCA'):
    if projection_module == 'ZCA':
        return zca_normalization(features)
    elif projection_module == 'AdaIN':
        return adain_normalization(features)
    else:
        return features, None, None

def reconstruct_features(projected_features, feature_kernels, mean_features, reconstruction_module='ZCA'):
    if reconstruction_module == 'ZCA':
        return zca_colorization(projected_features, feature_kernels, mean_features)
    elif reconstruction_module == 'AdaIN':
        return adain_colorization(projected_features, feature_kernels, mean_features)
    else:
        return projected_features

# https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
def KMeans(x, K=10, Niter=80, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids

    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = x.unsqueeze(1)  # (Npoints, 1, D)
    for i in range(Niter):
        c_j = c.unsqueeze(0)  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(torch.float64)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    c, cl = c.sort(axis=0)
    return cl, c

if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn(1, 512, 32, 32)
    b, c, d = project_features(a, 'ZCA')
    e = reconstruct_features(b, c, d, 'ZCA')
    print(e - a)