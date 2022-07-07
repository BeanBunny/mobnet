# %%
# !git clone "https://github.com/BeanBunny/mobnet"

# %%
import torch
import numpy as np
import sklearn.model_selection
from tqdm import tqdm
import os
import torchvision

lr = 0.01
EPOCHS = 200
BATCH_SIZE = 20
WORKERS = 0
SPLIT = 0.8

dataset_path = os.path.join(".", "JS.zip")
paths = os.path.join(".", "dataset")
print(dataset_path, paths)

# %%
import zipfile
with zipfile.ZipFile(dataset_path, "r") as zip_ref:
    zip_ref.extractall("./dataset")

# %%
model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
class MobNet(torch.nn.Module):
    def __init__(self):
        super(MobNet, self).__init__()
        flatten = torch.nn.Flatten()
        drop = torch.nn.Dropout(0.2)
        linear = torch.nn.Linear(62720, 7840)
        linear2 = torch.nn.Linear(7840, 980)
        linear3 = torch.nn.Linear(980, 2)
        soft = torch.nn.Softmax(dim=1)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.fc = torch.nn.Sequential(*(flatten, drop, linear, linear2, linear3, soft))
    def forward(self, x):
        x1 = self.model(x)
        x2 = self.fc(x1)
        return x2
finalModel = MobNet()

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
finalModel.to(device)
device

# %%
def splitter(path):
    temp = path.split('/')[-1]
    temp = temp.split('\\')[-1]
    return temp

# %%
# try using the 1000 layer too
# cnn style final layers instead of the fully connected

# %%
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
images = torchvision.datasets.ImageFolder(paths, transform=transform)

train, test = sklearn.model_selection.train_test_split(images, train_size=SPLIT)

print('Training set has {} instances'.format(len(train)))
print('Testing set has {} instances'.format(len(test)))

tr_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
te_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

# %%
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(finalModel.parameters(), lr=lr)

# %%
def training():
    lastl = 0
    for _, data in enumerate(tr_loader):
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = finalModel(inputs)
        outputs = torch.argmax(outputs, dim=1)
        outputs = torch.reshape(outputs.type(torch.DoubleTensor), (1, BATCH_SIZE))
        outputs.requires_grad=True
        labels = torch.reshape(labels.type(torch.DoubleTensor), (1, BATCH_SIZE))
        labels.requires_grad=True
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        lastl = loss.item()
    return lastl

# %%
def run_training():
    for epoch in tqdm(range(EPOCHS)):
        print('EPOCH {}:'.format(epoch + 1))
        finalModel.train(True)
        avg_loss = training()
        finalModel.train(False)
        print('loss: {}'.format(avg_loss))
        if epoch == 0:
            best_loss = avg_loss
        if best_loss > avg_loss:
            best_loss = avg_loss
            bestweights = finalModel.state_dict()
            bestepoch = epoch + 1
    print(bestepoch)
    model_path = "model_{}.pt".format(bestepoch)
    torch.save(bestweights, model_path)
    return model_path

model_path = run_training()

# %% [markdown]
# <h1>Testing</h1>

# %%
import sklearn.metrics
finalModel.load_state_dict(torch.load(model_path))
finalModel.eval()
print(device)

# %%
def testing():
    result = []
    result2 = []
    for _, data in enumerate(te_loader):
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device, dtype=torch.float)
        outputs = finalModel(inputs)
        labels = labels.cpu().detach().numpy()
        inputs = inputs.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.argmax(axis=1)
        result.append(outputs)
        result2.append(labels)
    return np.concatenate(result), np.concatenate(result2)
outputs, grounds = testing()

# %%
print(sklearn.metrics.accuracy_score(grounds, outputs))

# %%
print(sklearn.metrics.confusion_matrix(grounds, outputs))

# %%
input = torch.randn(1,3,224,224, device="cuda")
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]
torch.onnx.export(finalModel, input, "mobilenet_v2.onnx", input_names=input_names, output_names=output_names)


