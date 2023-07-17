from diffusers import DDIMInverseScheduler, DDIMPipeline, DDIMScheduler
import torch
import numpy as np
import PIL

from models import DDIMPipelineGivenImage, MnistClassifier
import torchvision


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=1,
    shuffle=False,
)



mnist_classifier = MnistClassifier()
mnist_classifier.load_state_dict(torch.load("model.pth"))
mnist_classifier.eval()
for data, target in test_loader:
    pred = mnist_classifier(data).argmax(dim=1)
    print(target)
    print(pred)
    break

# num_steps = 1000
# model_id = "nabdan/mnist_20_epoch"
# # model_id = "google/ddpm-cifar10-32"

# pipeline = DDIMPipeline.from_pretrained(model_id)
# pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
# pipeline.scheduler.set_timesteps(num_steps)
# pipeline_after_inv = DDIMPipelineGivenImage.from_pretrained(model_id)
# pipeline_after_inv.scheduler.set_timesteps(num_steps)

# def process_image_mnist(image):
#   image = np.array(image).astype(np.float32) / 255.0
#   # image = np.expand_dims(image, axis=(0))
#   image = np.expand_dims(image, axis=(0, 1))
#   # image = image.transpose(0, 3, 1, 2)
#   image = 2.0 * image - 1.0
#   image = torch.from_numpy(image)
#   # image = image.reshape(1, 3, 32, 32)
#   return image


# scheduler_inv = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
# pipeline_inv = DDIMPipelineGivenImage.from_pretrained(model_id, scheduler=scheduler_inv)
# pipeline_inv.scheduler = scheduler_inv
# pipeline_inv.scheduler.set_timesteps(num_steps)
# image = pipeline(num_inference_steps=num_steps).images[0]

# # image = PIL.Image.open("sample_image-300x298.png")
# # image = image.resize((28, 28))
# # image = image.convert("L")

# # image = PIL.Image.open("cifar-10-img.png")
# # image = image.convert("RGB")
# # image = image.resize((32, 32))
# image.save("image.png")

# image_tensor = process_image_mnist(image)
# image_inv = pipeline_inv(image_=image_tensor, num_inference_steps=num_steps).images[0]
# image_inv.save("image_inv.png")
# image_inv_tensor = process_image_mnist(image_inv)
# image_after_inv = pipeline_after_inv(image_=image_inv_tensor, num_inference_steps=num_steps).images[0]
# image_after_inv.save("image_after_inv.png")

# image_after_inv2 = pipeline_after_inv(image_=image_inv_tensor, num_inference_steps=num_steps).images[0]
# image_after_inv.save("image_after_inv2.png")

# image_after_inv3 = pipeline_after_inv(image_=image_inv_tensor, num_inference_steps=num_steps).images[0]
# image_after_inv.save("image_after_inv3.png")
