import torch
from torchvision import datasets, transforms
from model import CNN
import numpy as np
import matplotlib.pyplot as plt

# Device
device = torch.device("cpu")

# Transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load test dataset
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,   # faster than 1
    shuffle=False
)

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("MNIST_CNN_Model.pth", map_location=device))
model.eval()


# Collect predictions

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


# Accuracy

accuracy = (all_preds == all_labels).sum() / len(all_labels)
print(f"\nAccuracy: {accuracy:.4f}")


# Confusion Matrix (10x10)

num_classes = 10
cm = np.zeros((num_classes, num_classes), dtype=int)

for actual, pred in zip(all_labels, all_preds):
    cm[actual][pred] += 1


# Precision, Recall, F1

precision = []
recall = []
f1 = []

for i in range(num_classes):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp

    p = tp / (tp + fp) if (tp + fp) != 0 else 0
    r = tp / (tp + fn) if (tp + fn) != 0 else 0
    f = 2 * p * r / (p + r) if (p + r) != 0 else 0

    precision.append(p)
    recall.append(r)
    f1.append(f)

print("\nPer-class Precision:", np.round(precision, 4))
print("Per-class Recall   :", np.round(recall, 4))
print("Per-class F1 Score :", np.round(f1, 4))

print("\nMacro Precision:", np.mean(precision))
print("Macro Recall   :", np.mean(recall))
print("Macro F1 Score :", np.mean(f1))


# Plot Confusion Matrix

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.colorbar()

# Tick marks
plt.xticks(np.arange(num_classes))
plt.yticks(np.arange(num_classes))

# Annotate cells
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j],
                 ha='center', va='center')

plt.tight_layout()
plt.show()