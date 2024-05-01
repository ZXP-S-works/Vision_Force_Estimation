from collections import OrderedDict

import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np
from model.cnn import ResUNet, ResNet, MLP, ResNetCubic, MlpResNet, TransformerResNet
from model.tf import SimpleViT, posemb_sincos_2d_v2
import torchvision.transforms.functional as TF
import polygenerator 
from PIL import Image, ImageDraw
import random, copy
import os
import matplotlib.pyplot as plt

def flatten_x(x, w):
    return x[:, 0] * w + x[:, 1]

def unflatten_x(x_flat, w):
    return torch.stack((x_flat // w, x_flat % w), dim=1)


def get_f_from_f_map(f_map, x_flat):
    x_flat = x_flat.repeat_interleave(2)
    return f_map.flatten(start_dim=2).reshape(len(x_flat), -1)[torch.arange(len(x_flat)), x_flat].reshape(-1, 2)

def generate_random_patch(N_indices_range, width_range, height_range, center):
    N_indices = random.choice(np.arange(N_indices_range[0],N_indices_range[1])) 
    polygon_width = random.uniform(width_range[0],width_range[1])
    polygon_height = random.uniform(height_range[0],height_range[1])
    # normalized_polygon = polygenerator.random_polygon(num_points = N_indices)
    normalized_polygon = polygenerator.random_convex_polygon(num_points = N_indices)
    polygon = []
    for pt in normalized_polygon:
        polygon.append((int(pt[0]*polygon_width + center[0]), int(pt[1]*polygon_height + center[1])))
    return polygon

aug_transforms = v2.Compose([
    v2.ColorJitter(brightness = 0.1, saturation = 0.1),])

class VisionForceEstimator:
    def __init__(self, args, net):
        self.network = net
        if args.network in ['tf', 'transformerresnet']:
            self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.h, self.w = args.h, args.w
        self.device = args.device
        self.l1_weight = args.l1_weight
        self.l2_weight = args.l2_weight
        self.l1_loss = args.l1_loss
        self.l2_loss = args.l2_loss
        self.resolution = args.resolution
        self.n_history = args.n_history
        self.n_servo_info = args.n_servo_info
        self.loss_on_entire_history = args.loss_on_entire_history
        self.segmented_imgs = args.segmented_imgs
        if hasattr(args, 'segmentation_aug'):
            self.segmentation_aug = args.segmentation_aug
        else:
            self.segmentation_aug = False
        self.background_img_folder = args.background_img_folder
        self.args = args
        ## position feature 
        self.position_feature_tensor = torch.zeros((2, 160, 90))
        for i in range(self.position_feature_tensor.shape[1]):
            for j in range(self.position_feature_tensor.shape[2]):
                self.position_feature_tensor[0,:,:] = i
                self.position_feature_tensor[1,:,:] = j
        self.position_feature_tensor[0,:,:] = self.position_feature_tensor[0,:,:]/self.position_feature_tensor.shape[1] - 0.5
        self.position_feature_tensor[1,:,:] = self.position_feature_tensor[1,:,:]/self.position_feature_tensor.shape[2] - 0.5
        self.position_feature_tensor = self.position_feature_tensor.to(self.device)

        ## position embedding
        self.position_emb_tensor = torch.zeros((args.position_embedding_dim, 160, 90))
        self.position_emb_tensor = posemb_sincos_2d_v2(self.position_emb_tensor, temperature=60).to(self.device)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def getModelStr(self):
        return str(self.network)

    def loadModel(self, path):
        self.network.load_state_dict(torch.load(path, map_location=torch.device(self.device)))

    def saveModel(self, path):
        torch.save(self.network.state_dict(), '{}.pt'.format(path))

    def forward(self, s):
        maps = self.network(s)
        return maps[:, :1], maps[:, 1:]

    def normalize_rgb(self, img):
        img[:,0:3,...] /= 255
        img[:,0:3,...] -= 0.5
        return img

    def normalize_f(self, f):
        f = f / 10
        return f

    def load_batch(self, batch, is_train=False, return_entire_history = False, use_position_feature = False,\
                   use_position_embedding = False):
        # ToDo: update and simplify this function, especially image, theta, force
        img, x, f, servo_info = [], [], [], []
        if np.random.rand() > 0.5:
            mask_img = self.get_aug_segmented_mask(batch[0].img)
            #augment entire batch of images together
            for item in batch:
                item_img = item.img
                if is_train and self.segmentation_aug:
                    item_img = self.aug_segmented(item_img, self.background_img_folder, patch_mask=mask_img) 
                if item_img.shape[0] != 3:
                    item_img = np.swapaxes(item_img, 0, 2)
                img.append(torch.tensor(item_img, dtype=torch.float)[:3].unsqueeze(0))
        else:
            for item in batch:
                item_img = item.img
                if item_img.shape[0] != 3:
                    item_img = np.swapaxes(item_img, 0, 2)
                img.append(torch.tensor(item_img, dtype=torch.float)[:3].unsqueeze(0))

        for item in batch:
            if item.x is not None:
                x.append(torch.tensor(item.x, dtype=torch.float).reshape(1, -1))
            else:
                x.append(torch.tensor([[0]]))
            f.append(torch.tensor(item.f, dtype=torch.float).reshape(1, -1))


        x = torch.cat(x)
        f = torch.cat(f)
        img = torch.cat(img)
        f = f[:,1:3] #raw data contains 6D F/T readings

        ## photo metric augmentation:
        if is_train:
            img = self.aug_by_batch(img/255, self.segmented_imgs)
            img *= 255
        if not return_entire_history:
            f = f[self.n_history - 1::self.n_history]  # only retrieve the last force data in the history
        img, x, f = img.to(self.device), x.to(self.device), f.to(self.device)
        if use_position_feature:
            position_feature_tensor = self.position_feature_tensor.detach().clone()
            position_feature_tensor = torch.tile(position_feature_tensor, (img.shape[0],1,1,1))
            img = torch.concatenate((img, position_feature_tensor), axis = 1)
        elif use_position_embedding:
            position_emb_tensor = self.position_emb_tensor.detach().clone()
            position_emb_tensor = torch.tile(position_emb_tensor, (img.shape[0],1,1,1))

        return img, x, f, None

    def aug_by_batch(self, img, segmented_imgs = False):
        n_batches = int(img.shape[0]/self.n_history)
        for i in range(n_batches):
            brightness_factor = np.random.uniform(1 - 0.5,1 + 0.5) #TODO 0.5 - 1.5; Hue 
            contrast_factor = np.random.uniform(1 - 0.5,1 + 0.5)
            saturation_factor = np.random.uniform(1 - 0.5,1 + 0.5)
            hue_factor = np.random.uniform(- 0.5,0.5)
            # if not segmented_imgs:
            img[i*self.n_history:(i+1)*self.n_history] = \
                TF.adjust_brightness(img[i*self.n_history:(i+1)*self.n_history], brightness_factor)
            img[i*self.n_history:(i+1)*self.n_history] = \
                TF.adjust_contrast(img[i*self.n_history:(i+1)*self.n_history], contrast_factor)
            img[i*self.n_history:(i+1)*self.n_history] = \
                TF.adjust_saturation(img[i*self.n_history:(i+1)*self.n_history], saturation_factor)
            img[i*self.n_history:(i+1)*self.n_history] = \
                TF.adjust_hue(img[i*self.n_history:(i+1)*self.n_history], hue_factor)
        return img

    def get_aug_segmented_mask(self, img):
        if img.shape[0] == 3:
            tmp_img = Image.new('L', [img.shape[1], img.shape[2]], 0)
        else:
            tmp_img = Image.new('L', [img.shape[1], img.shape[0]], 0)
        center_x = random.randint(0,20)
        center_y = random.randint(0,20)
        polygon = generate_random_patch(N_indices_range = [4,5], width_range=[80,160], \
            height_range=[10,80], center = [center_x, center_y])
        if img.shape[0] == 3:
            tmp_img = Image.new('L', [img.shape[1], img.shape[2]], 0)
        else:
            tmp_img = Image.new('L', [img.shape[1], img.shape[0]], 0)
        ImageDraw.Draw(tmp_img).polygon(polygon, outline=1, fill=1)
        ## randomly flip the mask
        if np.random.rand() > 0.5:
            tmp_img = tmp_img.transpose(Image.FLIP_LEFT_RIGHT)
        patch_mask = np.array(tmp_img)
        return patch_mask

    def aug_segmented(self, img, background_img_folder, patch_mask = None):
        import matplotlib.pyplot as plt
        if img.shape[0] == 3:
            img = np.swapaxes(img, 0, 2)

        if patch_mask is None:
            center_x = random.randint(0,20)
            center_y = random.randint(0,20)
            polygon = generate_random_patch(N_indices_range = [4,5], width_range=[140,160], \
                height_range=[20,80], center = [center_x, center_y])

            tmp_img = Image.new('L', [img.shape[1], img.shape[0]], 0)
            ImageDraw.Draw(tmp_img).polygon(polygon, outline=1, fill=1)
            patch_mask = np.array(tmp_img)
        for k in range(3):
            img_slice = img[:,:,k]
            img_slice[np.where(patch_mask>0.5)] = 0.

        if img.shape[2] == 3:
            img = np.swapaxes(img, 0, 2)
        return img
        

class VisionForceEstimatorRegression(VisionForceEstimator):
    def __init__(self, args, net):
        super().__init__(args, net)

    def train_batch(self, batch, do_step = True, num_batches = 1):
        img, x, f, servo_info = self.load_batch(batch, is_train=True, \
            return_entire_history = self.loss_on_entire_history, \
            use_position_feature=self.args.use_position_feature,\
            use_position_embedding=self.args.use_position_embedding)
        img = self.normalize_rgb(img)
        if self.n_servo_info > 0:
            f_hat, z = self.network(img, servo_info, \
                return_entire_history = self.loss_on_entire_history)
        else:
            f_hat, z = self.network(img, \
                return_entire_history = self.loss_on_entire_history)
        f_hat = f_hat.view((-1, f_hat.shape[-1]))
        if self.l2_loss == 'MSE':
            l2 = F.mse_loss(f_hat, f)
        elif self.l2_loss == 'smooth_l1':
            l2 = F.smooth_l1_loss(f_hat, f, beta=0.5)
        elif self.l2_loss == 'MSE_contrastive':
            l2 = F.mse_loss(f_hat, f)
            b, n = z.shape
            dz = z.unsqueeze(1).expand(b, b, n)
            dz = dz - dz.permute(1, 0, 2)
            dz = torch.norm(dz, dim=-1)
            # dz = dz / dz.sum()
            b, n = f.shape
            df = f.unsqueeze(1).expand(b, b, n)
            df = df - df.permute(1, 0, 2)
            df = torch.norm(df, dim=-1)
            expand = df > 1
            shrink = df < 0.5
            # df = df / df.sum()
            l2 += dz[shrink].mean() - dz[expand].mean() + torch.norm(dz, dim=-1).mean()
        else:
            raise NotImplementedError
        loss = l2 * self.l2_weight / num_batches
        loss.backward()
        if do_step:
            for param in self.network.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item(), 0, l2.item()

    def test(self, batch, plotting):
        img, _, f, servo_info  = self.load_batch(batch, \
            use_position_feature=self.args.use_position_feature,\
            use_position_embedding=self.args.use_position_embedding)
        img = self.normalize_rgb(img)
        f = f[:, :2]
        with torch.no_grad():
            self.network.eval()
            if self.n_servo_info > 0:
                f_hat, z = self.network(img, servo_info)
            else:
                f_hat, z = self.network(img)
            df = torch.linalg.norm(f_hat[:, :2] - f, dim=1)
            norm_f = torch.linalg.norm(f, dim=1)
            df_normal_f = df / norm_f
            norm_std_f = torch.linalg.norm(f.std(dim=0))
            self.network.train()
        if plotting:
            plot_quantitative_results(f, f_hat)
            print('testing finished')
            return None
        return df.mean().cpu(), norm_std_f.cpu(), df_normal_f.mean().cpu()

    def test_verbose(self, batch):
        img, _, f, servo_info  = self.load_batch(batch, use_position_feature=self.args.use_position_feature)
        img = self.normalize_rgb(img)
        f = f[:, :2]
        with torch.no_grad():
            self.network.eval()
            if self.n_servo_info > 0:
                f_hat, z = self.network(img, servo_info)
            else:
                f_hat, z = self.network(img)
            self.network.train()

        return f.cpu(), f_hat.cpu()


def create_estimator(args):
    if args.network == 'mlpresnet':
        net = MlpResNet(n_input_channel=args.n_input_channel,
                        n_output_channel=args.n_output_channel,
                        n_hidden=args.n_hidden,
                        kernel_size=args.kernel_size,
                        resolution=args.resolution,
                        dropout=args.dropout,
                        n_history=args.n_history)
        net.build()
    elif args.network == 'transformerresnet':
        net = TransformerResNet(n_input_channel=args.n_input_channel,
                                n_output_channel=args.n_output_channel,
                                n_hidden=args.n_hidden,
                                n_servo_info=args.n_servo_info,
                                kernel_size=args.kernel_size,
                                resolution=args.resolution,
                                dropout=args.dropout,
                                n_history=args.n_history,
                                num_classes=args.n_output_channel,
                                dim=128,
                                depth=4,
                                heads=8,
                                mlp_dim=256,
                                dim_head=args.n_hidden,
                                grad_on_one_frame = args.grad_on_one_frame)
    else:
        raise NotImplementedError
    net.to(args.device)
    return VisionForceEstimatorRegression(args, net)


def plot_quantitative_results(f, f_hat):
    # valid f vs f_hat
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 3), dpi=300)
    plt.plot(f[:, 0].clone().cpu(), 'r', label=r'$f_x$')
    plt.plot(f[:, 1].clone().cpu(), 'b', label=r'$f_z$')
    plt.plot(f_hat[:, 0].clone().cpu(), 'r:', label=r'$\hat{f}_x$', alpha=0.6)
    plt.plot(f_hat[:, 1].clone().cpu(), 'b:', label=r'$\hat{f}_z$', alpha=0.6)
    plt.xlabel('image frame')
    plt.ylabel('force magnitude (N)')
    plt.legend()
    plt.show()


def visualize_estimation(img, x, f, x_hat, f_hat, i):
    import numpy as np
    import matplotlib.pyplot as plt
    i = 20
    plt.figure()
    plt.imshow(img[i, 0])
    plt.title('F = ' + str(np.round(np.sqrt(f[i, 0] ** 2 + f[i, 1] ** 2).item(), 2)) + 'N')
    plt.quiver(x[i, 1], x[i, 0], f[i, 1], f[i, 0], scale=10, color='r', angles='xy')
    plt.quiver(x_hat[i, 1], x_hat[i, 0], f_hat[i, 1], f_hat[i, 0], scale=10, color='w', angles='xy')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # x map
    import numpy as np
    import matplotlib.pyplot as plt
    i = 20
    plt.figure()
    plt.imshow(torch.zeros((50, 50)))
    # plt.title('F = ' + str(np.round(np.sqrt(f[i, 0] ** 2 + f[i, 1] ** 2).item(), 2)) + 'N')
    # plt.quiver(x[i, 1], x[i, 0], f[i, 1], f[i, 0], scale=10, color='r', angles='xy')
    plt.quiver(torch.ones(500) * 25, torch.ones(500) * 25, x_hat[:, 1] - x[:, 1], x_hat[:, 0] - x[:, 0],
               scale_units='xy', scale=1, color='y')
    plt.grid(False)
    # plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # f map
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure()
    plt.quiver(torch.zeros(500), torch.zeros(500), -(f[:, 0] - f_hat[:, 0]), f[:, 1] - f_hat[:, 1], units='xy',
               angles='xy', scale_units='xy', scale=1, color='r', alpha=0.05)
    plt.grid(False)
    plt.ylim(-8, 4)
    plt.xlim(-4, 2)
    axs = plt.gca()
    axs.set_aspect('equal')
    plt.title('F - F_hat over 500 data')
    plt.xlabel('X (N)')
    plt.ylabel('Z (N)')
    plt.tight_layout()
    plt.show()

    # error max img
    import matplotlib.pyplot as plt

    maxs = (-F.mse_loss(f_hat, f, reduction='none').sum(-1)).topk(3)
    ii = maxs.indices
    print(maxs.values)
    for i in ii:
        plt.figure()
        plt.imshow(img[i].clone().detach().permute(1, 2, 0) + 0.5)
        plt.grid(False)
        plt.tight_layout()
    plt.show()

    # error-f
    import matplotlib.pyplot as plt

    error = F.mse_loss(f_hat, f, reduction='none')
    emag = error.sum(1).squeeze().pow(0.5)
    error = error.pow(0.5)
    fmag = f.pow(2).sum(1).squeeze().pow(0.5)

    emag, error_idx = torch.sort(emag)
    plt.figure()
    plt.plot(emag, label='magnitude')
    plt.plot(error[:, 0][error_idx], label='Fx')
    plt.plot(error[:, 1][error_idx], label='Fy')
    plt.plot(error[:, 2][error_idx], label='Fz')
    plt.legend()
    plt.figure()
    plt.plot(fmag[error_idx], label='magnitude')
    plt.plot(f[:, 0][error_idx], label='Fx')
    plt.plot(f[:, 1][error_idx], label='Fy')
    plt.plot(f[:, 2][error_idx], label='Fz')
    plt.legend()
    plt.show()

    # magnitude
    import matplotlib.pyplot as plt
    error = F.mse_loss(f_hat, f, reduction='none')[:, :2]
    emag = error.sum(1).squeeze().pow(0.5)
    fxy = f[:, :2]
    fmag = fxy.pow(2).sum(1).squeeze().pow(0.5)
    fmagmean = fmag.mean()
    n_emag = emag / fmagmean * 100
    # Calculate the number of bins based on the bin width
    num_bins = 50

    # Create the bar plot
    plt.hist(fmag, bins=num_bins, color='b')

    # Set labels and title
    plt.xlabel('F magnitude (N)')
    plt.ylabel('Frequency')

    # Display the plot
    plt.show()

    # angle error
    import matplotlib.pyplot as plt
    def angle_between_vectors(v1, v2):
        dot_product = np.sum(v1 * v2, axis=1)
        magnitude_product = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        cosine_similarity = dot_product / magnitude_product
        angle = np.arccos(cosine_similarity)
        return np.degrees(angle)

    angles = angle_between_vectors(f.numpy(), f_hat.numpy())

    # Calculate the number of bins based on the bin width
    num_bins = 10
    # Create the bar plot
    plt.hist(angles, bins=num_bins, color='b')
    # Set labels and title
    plt.xlabel('Angle error (degree)')
    plt.ylabel('Frequency')
    # Display the plot
    plt.show()

    # valid f vs f_hat
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 4), dpi=300)
    plt.plot(f[:, 0].clone().cpu(), label=r'f_x')
    plt.plot(f[:, 1].clone().cpu(), label=r'f_z')
    plt.plot(f_hat[:, 0].clone().cpu(), label=r'$\hat{f}_x$', alpha=0.6)
    plt.plot(f_hat[:, 1].clone().cpu(), label=r'$\hat{f}_z$', alpha=0.6)
    plt.legend()
    plt.show()

    # f_hat - f at f
    import matplotlib.pyplot as plt
    colors = df
    plt.figure()
    plt.title(r'$||\hat{F} - F||^2_2$ at $F$')
    plt.scatter(f[:, 0], f[:, 1], c=colors)
    plt.colorbar()
    plt.xlabel(r'$F_x$')
    plt.ylabel(r'$F_y$')
    plt.axis('equal')
    plt.show()

    # df at df
    import matplotlib.pyplot as plt
    colors = df
    idx = norm_f < 0.5
    plt.figure()
    plt.title(r'$||d\hat{F} - dF||^2_2$ at $dF$')
    plt.scatter(f[idx, 0], f[idx, 1], c=colors[idx])
    plt.colorbar()
    plt.xlabel(r'$dF_x$')
    plt.ylabel(r'$dF_y$')
    plt.axis('equal')
    plt.show()
