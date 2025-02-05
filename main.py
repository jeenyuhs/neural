import os
import torch

from PIL import Image
from torchvision import transforms, datasets
from typing import Any, Callable
from torch.utils import data
from torch import nn, optim, cuda
from tqdm import tqdm

from model import DogBreedClassifier

device = "cuda" if cuda.is_available() else "cpu"

def train_step(
    model: nn.Module, 
    dataloader: data.DataLoader,
    criterion: Callable, 
    optimizer: Callable
) -> dict[str, Any]:
    running_loss, running_acc = 0, 0

    for image, label in dataloader:
        image, label = image.to(device), label.to(device)

        score = model(image)

        loss = criterion(score, label)
        running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score_class = torch.argmax(torch.softmax(score, dim=1), dim=1)
        running_acc += (score_class == label).sum().item()/len(score_class)
    
    return {"train_loss": running_loss, "train_acc": running_acc}

def test_step(
    model: nn.Module, 
    dataloader: data.DataLoader,
    criterion: Callable
) -> dict[str, Any]:
    running_loss, running_acc = 0, 0

    for image, label in dataloader:
        image, label = image.to(device), label.to(device)

        score = model(image)

        loss = criterion(score, label)
        running_loss += loss

        score_class = torch.argmax(score, dim=1)
        running_acc += (score_class == label).sum().item()/len(score_class)
    
    return {"test_loss": running_loss, "test_acc": running_acc}

def train_model(
    model: nn.Module, 
    dataloader: data.DataLoader, 
    criterion: Callable, 
    optimizer: Callable, 
    epoch_iterations: int = 5
) -> list[dict[str, Any]]:
    model.train()

    epochs = []
    for epoch in tqdm(range(epoch_iterations)):
        step = train_step(model, dataloader, criterion, optimizer)
        epochs.append(step)

        if epoch == epoch_iterations - 1:
            torch.save({
                "model_state_dict": model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
            }, f"models/checkpoint_last_epoch.pt")

    return epochs

def test_model(model: nn.Module, dataloader: data.DataLoader) -> list[dict[str, Any]]:
    model.eval()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    epochs = []

    with torch.no_grad():
        step = test_step(model, dataloader, criterion)
        epochs.append(step)

    return epochs

def any_available_models() -> bool:
    return len(os.listdir("models")) > 0

def predict_random_dog(image_path: str, model: nn.Module, transform: Callable) -> str:
    model.eval()

    image = Image.open(image_path).convert('RGB')
    augmented_image = transform(image).unsqueeze(dim=0).to(device)

    with torch.no_grad():
        score = model(augmented_image)

    _, preds = torch.max(score, 1)

    print(f"{image_path} is a {"Viszla" if preds[0] == 1 else "Doberman"}")

def main():
    transformers = transforms.Compose([
        # ændre størrelsen på billedet til 224x224
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    training_transformers = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root="./races", transform=transformers)
    test_data = datasets.ImageFolder(root="./test", transform=training_transformers)

    torch.manual_seed(42)

    NUM_WORKERS = os.cpu_count()

    train_dataloader = data.DataLoader(dataset=train_data, batch_size=64, num_workers=NUM_WORKERS, shuffle=True)
    test_dataloader = data.DataLoader(dataset=test_data, batch_size=64, num_workers=NUM_WORKERS, shuffle=False)

    model = DogBreedClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    
    if not any_available_models():
        print("no saved checkpoints. running training program")
        train_model(model, train_dataloader, criterion, optimizer, epoch_iterations=200)
        test_model(model, test_dataloader)

        print("finished training. try running the program again to test model")
        raise SystemExit(1)
    
    
    checkpoint = torch.load("models/checkpoint_last_epoch.pt")
    print("found model checkpoint")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for _, _, filenames in os.walk("untrained_pictures"):
        for filename in filenames:
            real_path = os.path.join("untrained_pictures", filename)

            predict_random_dog(real_path, model, transformers)

if __name__ == "__main__":
    main()
