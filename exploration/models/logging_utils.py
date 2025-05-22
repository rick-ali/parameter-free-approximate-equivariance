import torch
import torchvision



def log_reconstructed_and_decodedlatent(model, x1, x2, decoded_transformed_latent, reconstructed, stage, n_display=8, data='CIFAR'):
        comparison = torch.cat([reconstructed[:n_display], x1[:n_display], decoded_transformed_latent[:n_display], x2[:n_display]])
        if data == 'CIFAR':
                grid = torchvision.utils.make_grid(comparison, nrow=n_display, normalize=True)
        else:
                grid = torchvision.utils.make_grid(comparison, nrow=n_display)
        model.logger.experiment.add_image(f"{stage}: D(E(x1)) vs x1 vs D(WE(x1)) vs x2", grid, model.global_step)