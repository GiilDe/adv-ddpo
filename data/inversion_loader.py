from pathlib import Path
import torch

def save_inversion_data(images, labels, latents, save_dir, batch_num):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(images, f'{save_dir}/test_images_{batch_num}.pt')
    torch.save(labels, f'{save_dir}/test_labels_{batch_num}.pt')
    torch.save(latents, f'{save_dir}/test_latents_{batch_num}.pt')


def load_inversion_data(save_dir, batch_num):
    images = torch.load(f'{save_dir}/test_images_{batch_num}.pt')
    labels = torch.load(f'{save_dir}/test_labels_{batch_num}.pt')
    latents = torch.load(f'{save_dir}/test_latents_{batch_num}.pt')
    
    return images, labels, latents