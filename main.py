import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, datasets
from typing import Callable
from torch.utils import data
from torch import nn, optim, cuda
from tqdm import tqdm

from model import model

device = "cuda" if cuda.is_available() else "cpu"

transformers = transforms.Compose([
    # ændre størrelsen på billedet til 224x224
    transforms.Resize((512, 512)),
    # for at forbedre træningsmodellen, kan vi tilfældeligt flippe
    # billedet for at skabe større differens i billederne. der er 
    # 50% chance for at billedet flipper.
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    # ændre bit værdierne for billederne til tensor værdier.
    transforms.ToTensor()
])

training_transformers = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root="./races", transform=transformers)
test_data = datasets.ImageFolder(root="./test", transform=training_transformers)

torch.manual_seed(42)

NUM_WORKERS = os.cpu_count()

train_dataloader = data.DataLoader(dataset=train_data, batch_size=16, num_workers=0, shuffle=True)
test_dataloader = data.DataLoader(dataset=test_data, batch_size=16, num_workers=0, shuffle=False)

def train_model(model: nn.Module, dataloader: data.DataLoader) -> None:
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    epochs = []
    for epoch in tqdm(range(5)):
        result = {"train_acc": 0, "train_loss": 0, "test_acc": 0, "test_loss": 0}

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            logps = model(x)
            loss = loss_fn(logps, y)
            result["train_loss"] += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logps_class = torch.argmax(torch.softmax(logps, dim=1), dim=1)
            result["train_acc"] += (logps_class == y).sum().item()/len(logps)
        print("train step is done")
        with torch.inference_mode():
            for batch, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                
                logps = model(x)
                print(logps, "train model", type(logps))
                loss = loss_fn(logps, y)
                result["test_loss"] += loss

                logps_class = torch.argmax(logps, dim=1)
                result["test_acc"] += (logps_class == y).sum().item()/len(logps)
        print("test step is done")
        epochs.append(result)
        print(result)

    return epochs

def main():
    epochs = train_model(model, train_dataloader)
    print("done training")
    print(epochs)
    for epoch in epochs:
        print(epoch)


if __name__ == "__main__":
    main()

# fig = plt.figure()

# def visualize_transformed_data(image_path: str, transform: Callable) -> None:
#     for dirpath, dirname, filenames in os.walk(image_path):
#         for i, filename in enumerate(filenames):
#             real_path = os.path.join(image_path, filename)
#             print(real_path)
            
#             with Image.open(real_path) as image:
#                 ax = plt.subplot(4, 3, i + 1)
#                 plt.tight_layout()
#                 ax.set_title(filename)
#                 ax.axis('off')
#                 image = transform(image).permute(1, 2, 0)
#                 plt.imshow(image)

# visualize_transformed_data("races/viszla", transformers)

# plt.show()


# trainset = DogRaceDataset("dog_races.csv", "dog_races/", transform=v2.Resize(size=(256, 256), antialias=False))
# trainloader = data.DataLoader(trainset, batch_size=10, shuffle=True)

# model = nn.Sequential(nn.Linear(7680, 256),
#                       nn.ReLU(),
#                       nn.Linear(256, 256))

# criterion = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.003)

# for e in range(5):
#     running_loss = 0
#     for viszla in trainloader:
#         print(viszla["image"].shape[::-1])
#         images = viszla["image"].view(viszla["image"].shape[::-1][0], -1)
#         print(images)
    
#         # TODO: Training pass
#         optimizer.zero_grad()
#         output = model(images.to(torch.float32))
#         loss = criterion(output, viszla["race"])
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#     else:
#         print(f"Training loss: {running_loss/len(trainloader)}")

# # # Get our data
# # viszla = next(iter(trainloader))
# # # Flatten images
# # images = images.view(images.shape[0], -1)

# # # Forward pass, get our logits
# # logits = model(images)
# # # Calculate the loss with the logits and the labels
# # loss = criterion(logits, labels)

# fig = plt.figure()

# for i, viszla in enumerate(training_data):
#     print(viszla["pathname"])
    # ax = plt.subplot(4, 3, i + 1)
    # plt.tight_layout()
    # ax.set_title(viszla["pathname"])
    # ax.axis('off')

#     viszla["image"] = viszla["image"].swapaxes(0, 1)
#     viszla["image"] = viszla["image"].swapaxes(1, 2)

#     plt.imshow(viszla["image"])

# plt.show()  

