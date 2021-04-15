import torch
import torch.nn as nn
import torchvision
import net.utils as utils
import torch.nn.functional as F
FEATURE_CHANNEL = 512

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

vgg_reverse = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
    )

# Encoder
class Encoder(nn.Module):
    def __init__(self, use_gpu=True, encoder = vgg):
        super(Encoder, self).__init__()
        encoder = encoder
        if use_gpu == True:
            encoder.cuda()
        enc_layers = list(encoder.children())
        self.layers = [nn.Sequential(*enc_layers[:4]),
                      nn.Sequential(*enc_layers[4:11]),
                      nn.Sequential(*enc_layers[11:18]),
                      nn.Sequential(*enc_layers[18:31]),
                      nn.Sequential(*enc_layers[31:44])]
        self.layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        
    def forward(self, x):
        out = {}
        for i in range(0, len(self.layers)):
            x = self.layers[i](x)
            out[self.layer_names[i]] = x
        return out
    
    def __str__(self):
        ans = ''
        for layer in self.layers:
            ans += (layer.__str__() + '\n')
        return ans

class Correlation(nn.Module):
    def __init__(self, in_channel = 512, hidden_channel = 512):
        super(Correlation, self).__init__()
        self.f = nn.Conv2d(in_channel, hidden_channel, kernel_size=1) # [b, floor(c/2), h, w]
        self.g = nn.Conv2d(in_channel, hidden_channel, kernel_size=1) # [b, floor(c/2), h, w]
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        f = utils.hw_flatten(self.f(x)).permute(0, 2, 1) # [b, nc, c']
        g = utils.hw_flatten(self.g(y)) # [b, c', ns]

        energy = torch.bmm(f, g) # [b, nc, ns]
        correlation = self.softmax(energy) # [b, nc, ns]
        attention = torch.mean(correlation.permute(0, 2, 1), 2, keepdim=True).permute(0, 2, 1)
        return correlation, attention

# Self Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_channel = 512):
        super(SelfAttention, self).__init__()
        self.corr = Correlation(in_channel = in_channel, hidden_channel = in_channel//2)
        self.h = nn.Conv2d(in_channel, in_channel, kernel_size=1)      # [b, c, h, w]

    def forward(self, x):
        x_size = x.shape
        correlation, attention = self.corr(x, x)
        h = utils.hw_flatten(self.h(x)) # [b, c, ns]

        residual = torch.bmm(h, correlation.permute(0, 2, 1)) # [b, c, nc]
        residual = residual.view(x_size)# [b, c, h, w]
        attention = attention.view(x_size[0], 1, x_size[2], x_size[3])
        return residual, attention

# SAVANet Module
class SAVANet(nn.Module):
    def __init__(self, in_channel = 512, self_attn = None, alpha=0.5, filter=False):
        super(SAVANet, self).__init__()
        self.feat_corr = Correlation(in_channel = in_channel, hidden_channel = in_channel)
        self.h = nn.Conv2d(in_channel, in_channel, kernel_size=1)      # [b, c, h, w]
        self.alpha = alpha
        self.filter = filter

        self.self_attn = self_attn
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channel, in_channel, (1, 1))

    def attention_filter(self, attention_feature_map, kernel_size=3, mean=6, stddev=5):
        attention_map = torch.abs(attention_feature_map)

        attention_mask = attention_map > 2 * torch.mean(attention_map)
        attention_mask = attention_mask.float()

        w = torch.randn(kernel_size, kernel_size)
        utils.truncated_normal_(w, mean, stddev)
        w = w / torch.sum(w)

        # [in_channels, out_channels, filter_height, filter_width]
        w = torch.unsqueeze(w, 0)
        w = w.repeat(attention_mask.shape[1], 1, 1)

        w = torch.unsqueeze(w, 0)
        w = w.repeat(attention_mask.shape[1], 1, 1, 1)

        gaussian_filter = nn.Conv2d(attention_mask.shape[1], attention_mask.shape[1], (kernel_size, kernel_size))
        gaussian_filter.weight.data = w
        gaussian_filter.weight.requires_grad = False
        pad_filter = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            gaussian_filter
        )
        pad_filter.cuda()
        attention_map = pad_filter(attention_mask)
        attention_map = attention_map - torch.min(attention_map)
        attention_map = attention_map / torch.max(attention_map)
        return attention_map

    def attn_corr(self, x, y):
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        y = torch.reshape(y, (y.shape[0], y.shape[1], -1))
        n_x = x.shape[-1]
        n_y = y.shape[-1]
        x = x.repeat(1, n_y, 1).permute(0, 2, 1)
        y = y.repeat(1, n_x, 1)
        dist = torch.abs(x - y)
        correlation = self.softmax(1.0 - dist)
        return correlation

    def attn_mask(self, x):
        x_size = x.shape
        x = x.reshape(x_size[0], x_size[1], -1)
        _, indices = x.topk(int(x.shape[2] * 0.925), dim=2, largest=False, sorted=True)
        attn = torch.ones(x.shape)
        attn[:, :, indices] = 0
        attn = attn.view(x_size).type(torch.cuda.FloatTensor)
        return attn
    
    def corr_mask(self, x, y):
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        y = torch.reshape(y, (y.shape[0], y.shape[1], -1))
        n_x = x.shape[-1]
        n_y = y.shape[-1]
        x = x.repeat(1, n_y, 1).permute(0, 2, 1)
        y = y.repeat(1, n_x, 1)
        # dist = torch.abs(x - y)
        mask = (torch.eq(x, y)).type(torch.cuda.FloatTensor)
        return mask

    def forward(self, x, y):
        x_size = x.shape
        y_size = y.shape
        x_norm_adain, _, _ = utils.project_features(x, "AdaIN")
        y_norm_adain, _, _ = utils.project_features(y, "AdaIN")
        x_norm_zca, _, _ = utils.project_features(x, "ZCA")
        y_norm_zca, _, _ = utils.project_features(y, "ZCA")

        correlation, _ = self.feat_corr(x_norm_adain, y_norm_adain)

        _, atten_x = self.self_attn(x_norm_zca)
        _, atten_y = self.self_attn(y_norm_zca)

        del x_norm_adain
        torch.cuda.empty_cache()
        del y_norm_adain
        torch.cuda.empty_cache()
        del x_norm_zca
        torch.cuda.empty_cache()
        del y_norm_zca
        torch.cuda.empty_cache()


        if self.filter:
            atten_x = self.attention_filter(atten_x)
            atten_y = self.attention_filter(atten_y)
            atten_x /= atten_x.sum()
            atten_y /= atten_y.sum()
        
        # attention_correlation = self.attn_corr(atten_x, atten_y)
        # atten_x_mask = (atten_x > 0.0001).type(torch.cuda.FloatTensor)
        # atten_y_mask = (atten_y > 0.0001).type(torch.cuda.FloatTensor)
        atten_x_mask = self.attn_mask(atten_x)
        atten_y_mask = self.attn_mask(atten_y)
        correlation_mask = self.corr_mask(atten_x_mask, atten_y_mask)
        # del atten_x
        # torch.cuda.empty_cache()
        # del atten_y
        # torch.cuda.empty_cache()

        # del atten_x_mask
        # torch.cuda.empty_cache()
        # del atten_y_mask
        # torch.cuda.empty_cache()
        # print(correlation.get_device())
        # correlation = self.alpha * feature_correlation + (1 - self.alpha) * attention_correlation
        # correlation_mask = correlation_mask.cpu()
        # correlation = correlation.cpu()
        correlation = correlation_mask * correlation
        correlation = F.normalize(correlation, p=1, dim=2)

        h = utils.hw_flatten(self.h(y)) # [b, c, ns]

        residual = torch.bmm(h, correlation.permute(0, 2, 1)) # [b, c, nc]
        residual = residual.view(x_size)# [b, c, hc, wc]
        residual = self.out_conv(residual)

        output = x + residual
        return output, atten_x, atten_y

class SANet(nn.Module): 
    def __init__(self, in_channel = 512):
        super(SANet, self).__init__()
        self.feat_corr = Correlation(in_channel = in_channel, hidden_channel = in_channel)
        self.h = nn.Conv2d(in_channel, in_channel, kernel_size=1)      # [b, c, h, w]

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channel, in_channel, (1, 1))
    def forward(self, x, y):
        x_size = x.shape
        x_norm_adain, _, _ = utils.project_features(x, "AdaIN")
        y_norm_adain, _, _ = utils.project_features(y, "AdaIN")
        # x_norm_zca, _, _ = utils.project_features(x, "ZCA")
        # y_norm_zca, _, _ = utils.project_features(y, "ZCA")

        feature_correlation, _ = self.feat_corr(x_norm_adain, y_norm_adain)

        correlation = feature_correlation

        h = utils.hw_flatten(self.h(y)) # [b, c, ns]

        residual = torch.bmm(h, correlation.permute(0, 2, 1)) # [b, c, nc]
        residual = residual.view(x_size)# [b, c, hc, wc]
        residual = self.out_conv(residual)

        output = x + residual
        return output

# Reconstruction Network with Self-attention Module
class AttentionNet(nn.Module):
    def __init__(self, attn = None, encoder = None, decoder = vgg_reverse):
        super(AttentionNet, self).__init__()
        # self.perceptual_loss_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        self.perceptual_loss_layers = ['conv1', 'conv2', 'conv3', 'conv4']

        self.recons_weight = 10
        self.perceptual_weight = 1
        self.tv_weight = 10
        self.attention_weight = 6

        self.encode = Encoder() if encoder == None else Encoder(encoder)
        self.self_attn = SelfAttention(in_channel=FEATURE_CHANNEL) if attn == None else attn
        self.decode = decoder

        self.mse_loss = nn.MSELoss()

    def get_encoder(self):
        return self.encode

    def self_attention_autoencoder(self, x, projection_method='AdaIN', mode = 'add_multiply'): # in case kernels are not seperated
        input_features = self.encode(x)
        projected_hidden_feature, colorization_kernels, mean_features = utils.project_features(input_features['conv4'], projection_method)
        # projected_hidden_feature = utils.whitening(input_features['conv4'])
        # print("haha")
        residual, attention = self.self_attn(projected_hidden_feature)

        if mode == 'add_multiply':
            hidden_feature = projected_hidden_feature * residual + projected_hidden_feature
        elif mode == 'add':
            hidden_feature = residual + projected_hidden_feature

        hidden_feature = utils.reconstruct_features(hidden_feature, colorization_kernels, mean_features, projection_method)


        output = self.decode(hidden_feature)
        return output, residual, attention

    def calc_recon_loss(self, x, target):
        recons_loss = self.mse_loss(x, target)
        return recons_loss
    
    def calc_perceptual_loss(self, x, target):
        input_feat = self.encode(x)
        output_feat = self.encode(target)
        
        perceptual_loss = 0.0

        for layer in self.perceptual_loss_layers:
            input_per_feat = input_feat[layer]
            output_per_feat = output_feat[layer]
            perceptual_loss += self.mse_loss(input_per_feat, output_per_feat)
        return perceptual_loss

    def calc_tv_loss(self, x):
        tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) 
        tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return tv_loss
    
    def calc_attention_loss(self, att_map):
        return torch.norm(att_map, p=1)

    def attention_filter(self, attention_feature_map, kernel_size=3, mean=6, stddev=5):
        attention_map = torch.abs(attention_feature_map)

        attention_mask = attention_map > 2 * torch.mean(attention_map)
        attention_mask = attention_mask.float()

        w = torch.randn(kernel_size, kernel_size)
        utils.truncated_normal_(w, mean, stddev)
        w = w / torch.sum(w)

        # [in_channels, out_channels, filter_height, filter_width]
        w = torch.unsqueeze(w, 0)
        w = w.repeat(attention_mask.shape[1], 1, 1)

        w = torch.unsqueeze(w, 0)
        w = w.repeat(attention_mask.shape[1], 1, 1, 1)

        gaussian_filter = nn.Conv2d(attention_mask.shape[1], attention_mask.shape[1], (kernel_size, kernel_size))
        gaussian_filter.weight.data = w
        gaussian_filter.weight.requires_grad = False
        pad_filter = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            gaussian_filter
        )
        pad_filter.cuda()
        attention_map = pad_filter(attention_mask)
        attention_map = attention_map - torch.min(attention_map)
        attention_map = attention_map / torch.max(attention_map)
        return attention_map

    def test(self, x, projection_method='ZCA'):
        output, attention_feature_map, attention_map = self.self_attention_autoencoder(x, projection_method = projection_method)
        output = utils.batch_mean_image_subtraction(output)

        recon_loss = self.calc_recon_loss(x, output) * (255**2 / 4)
        perceptual_loss =  self.calc_perceptual_loss(x, output) * (255**2 / 4)
        tv_loss = self.calc_tv_loss(output) * (255 / 2)
        attention_loss = self.calc_attention_loss(attention_feature_map)

        total_loss = recon_loss * self.recons_weight + perceptual_loss * self.perceptual_weight \
                    + tv_loss * self.tv_weight + attention_loss * self.attention_weight
        loss_dict = {'total': total_loss, 'construct': recon_loss, 'percept': perceptual_loss, 'tv': tv_loss, 'attn': attention_loss}

        attention_map = attention_map.repeat(1, 512, 1, 1)
        print(attention_map.shape)
        attention_map = self.attention_filter(attention_map)
        attention_feature_map = self.attention_filter(attention_feature_map)
        return loss_dict, output, attention_feature_map, attention_map

    def forward(self, x, projection_method='ZCA', mode='add_multiply'):
        # returns loss, output, residual
        # seperate must be False in this case
        output, attention_feature_map, _ = self.self_attention_autoencoder(x, projection_method = projection_method, mode = mode)
        output = utils.batch_mean_image_subtraction(output)

        recon_loss = self.calc_recon_loss(x, output) * (255**2 / 4)
        perceptual_loss =  self.calc_perceptual_loss(x, output) * (255**2 / 4)
        tv_loss = self.calc_tv_loss(output) * (255 / 2)
        attention_loss = self.calc_attention_loss(attention_feature_map)

        total_loss = recon_loss * self.recons_weight + perceptual_loss * self.perceptual_weight \
                    + tv_loss * self.tv_weight + attention_loss * self.attention_weight
        loss_dict = {'total': total_loss, 'construct': recon_loss, 'percept': perceptual_loss, 'tv': tv_loss, 'attn': attention_loss}
        return loss_dict, output, attention_feature_map


    
    

