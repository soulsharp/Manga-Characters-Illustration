import torch
import config


def wgan_gp_loss(real_data, fake_data, discriminator):
    # Generate random epsilon for gradient penalty calculation
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(config.DEVICE)
    
    # Interpolates between real and fake samples
    x_hat = epsilon * real_data + (1 - epsilon) * fake_data
    x_hat.requires_grad_(True)
    
    # Calculates critic scores for interpolated samples
    critic_interpolated = discriminator(x_hat)
    
    # Compute gradient penalty
    gradients = torch.autograd.grad(outputs=critic_interpolated,
                                    inputs=x_hat,
                                    grad_outputs=torch.ones_like(critic_interpolated.size()).to(config.DEVICE),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return config.LAMBDA_GP * gradient_penalty