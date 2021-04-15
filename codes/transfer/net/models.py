from net.network import SelfAttention, SAVANet, Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import net.utils as utils
FEATURE_CHANNEL = 512

class Transform(nn.Module):
    def __init__(self, in_channel = 512, self_attn = None, alpha=0.5, filter=False):
        super(Transform, self).__init__()
        self.savanet4_1 = SAVANet(in_channel=in_channel, self_attn = self_attn, alpha=alpha, filter = filter)
        self.savanet5_1 = SAVANet(in_channel=in_channel, self_attn = self_attn, alpha=alpha, filter = filter)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_channel, in_channel, (3, 3))

    def fusion(self, swapped4_1, swapped5_1):
        return self.merge_conv( self.merge_conv_pad( swapped4_1 + self.upsample5_1(swapped5_1) ) )

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        swapped4_1, content_attn4_1, style_attn4_1= self.savanet4_1(content4_1, style4_1)
        swapped5_1, content_attn5_1, style_attn5_1= self.savanet5_1(content5_1, style5_1)
        fused = self.fusion(swapped4_1, swapped5_1)
        return fused, content_attn4_1, style_attn4_1, content_attn5_1, style_attn5_1

class SAVA_test(nn.Module):
    def __init__(self, transformer = None, encoder = None, decoder = None):
        super(SAVA_test, self).__init__()
        self.transformer = transformer
        self.encode = Encoder() if encoder == None else Encoder(encoder)
        self.decode = decoder

        self.content_weight = 1.0
        self.style_weight = 3.0
        self.identity1_weight = 50
        self.identity2_weight = 1.0
        self.perceptual_loss_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

        self.mse_loss = nn.MSELoss()

    def calc_content_loss(self, x, y, norm = False):
        if norm == False:
            return self.mse_loss(x, y)
        else:
            x_norm_adain, _, _ = utils.project_features(x, "AdaIN")
            y_norm_adain, _, _ = utils.project_features(y, "AdaIN")
            return self.mse_loss(x_norm_adain, y_norm_adain)
    
    def calc_style_loss(self, x, y):
        x_mean, x_std = utils.calc_mean_std(x)
        y_mean, y_std = utils.calc_mean_std(y)

        return self.mse_loss(x_mean, y_mean) + \
               self.mse_loss(x_std, y_std)

    def calc_void_loss(self, input, output):
        pass

    def transfer(self, contents, styles):
        content_features = self.encode(contents)
        style_features = self.encode(styles)

        content_hidden_feature_4 = content_features[self.perceptual_loss_layers[-2]]
        style_hidden_feature_4 = style_features[self.perceptual_loss_layers[-2]]
        content_hidden_feature_5 = content_features[self.perceptual_loss_layers[-1]]
        style_hidden_feature_5 = style_features[self.perceptual_loss_layers[-1]]

        swapped_features, content_attn4_1, style_attn4_1, content_attn5_1, style_attn5_1 = self.transformer(content_hidden_feature_4, style_hidden_feature_4, content_hidden_feature_5, style_hidden_feature_5)

        output = self.decode(swapped_features)
        return output, swapped_features, [content_attn4_1, style_attn4_1, content_attn5_1, style_attn5_1]
    
    def forward(self, contents, styles):
        content_features = self.encode(contents)
        style_features = self.encode(styles)

        content_hidden_feature_4 = content_features[self.perceptual_loss_layers[-2]]
        style_hidden_feature_4 = style_features[self.perceptual_loss_layers[-2]]
        content_hidden_feature_5 = content_features[self.perceptual_loss_layers[-1]]
        style_hidden_feature_5 = style_features[self.perceptual_loss_layers[-1]]

        swapped_features, _, _, _, _ = self.transformer(content_hidden_feature_4, style_hidden_feature_4, content_hidden_feature_5, style_hidden_feature_5)
        I_cs = self.decode(swapped_features)
        F_cs = self.encode(I_cs)

        swapped_features, _, _, _, _ = self.transformer(content_hidden_feature_4, content_hidden_feature_4, content_hidden_feature_5, content_hidden_feature_5)
        I_cc = self.decode(swapped_features)
        F_cc = self.encode(I_cc)

        swapped_features, _, _, _, _ = self.transformer(style_hidden_feature_4, style_hidden_feature_4, style_hidden_feature_5, style_hidden_feature_5)
        I_ss = self.decode(swapped_features)
        F_ss = self.encode(I_ss)

        loss_c = self.calc_content_loss(content_hidden_feature_4, F_cs['conv4'], norm=True) + \
                 self.calc_content_loss(content_hidden_feature_5, F_cs['conv5'], norm=True)

        loss_s = 0.0
        for layer in self.perceptual_loss_layers:
            loss_s += self.calc_style_loss(style_features[layer], F_cs[layer])

        loss_identity1 = self.calc_content_loss(I_cc, contents) + self.calc_content_loss(I_ss, styles)

        loss_identity2 = 0.0
        for layer in self.perceptual_loss_layers:
            loss_identity2 += self.calc_content_loss(F_cc[layer], content_features[layer]) + \
                              self.calc_content_loss(F_ss[layer], style_features[layer])

        total_loss = self.content_weight * loss_c + self.style_weight * loss_s + \
                     self.identity1_weight * loss_identity1 + self.identity2_weight * loss_identity2
        loss_dict = {'total': total_loss, 'content': loss_c, 'style': loss_s, 'identity1': loss_identity1, 'identity2': loss_identity2}
        return loss_dict





