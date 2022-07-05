# %%
# !git clone "https://github.com/BeanBunny/mobnet"

# %%

import torch
import cv2
import numpy as np
import glob
import sklearn.model_selection
from tqdm import tqdm
import os

lr = 0.01
EPOCHS = 200
BATCH_SIZE = 20
WORKERS = 0
SPLIT = 0.8
INSTANCESPERBATCH = 0

dataset_path = os.path.join(".", "JS.zip")
paths = [os.path.join(".", "cat", "*.jpg"), os.path.join(".", "dog", "*.jpg")]
print(dataset_path, paths)

# %%
import zipfile
with zipfile.ZipFile(dataset_path, "r") as zip_ref:
    zip_ref.extractall("./")

# %%
model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
class MobNet(torch.nn.Module):
    def __init__(self):
        super(MobNet, self).__init__()
        flatten = torch.nn.Flatten()
        drop = torch.nn.Dropout(0.2)
        linear = torch.nn.Linear(62720, 2)
        soft = torch.nn.Softmax(dim=1)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.fc = torch.nn.Sequential(*(flatten, drop, linear, soft))
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
labels = {"cat": 0, "dog": 1}
def func(path):
    result = []
    for j in glob.glob(path):
        i = cv2.imread(j, cv2.COLOR_BGR2RGB)
        i = i/255
        i = np.transpose(i)
        i = (i, labels[splitter(os.path.split(path)[0])])
        result.append(i)
    return result

images = func(paths[0])
images2 = func(paths[1])
images3 = images + images2

# %%
train, test = sklearn.model_selection.train_test_split(images3, train_size=SPLIT)

print('Training set has {} instances'.format(len(train)))
print('Testing set has {} instances'.format(len(test)))

INSTANCESPERBATCH = len(train)/BATCH_SIZE

tr_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
te_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

# %%
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(finalModel.parameters(), lr=lr)

# %%
def training():
    runl = 0
    lastl = 0
    for i, data in enumerate(tr_loader):
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
        
        runl += loss.item()
        if i % INSTANCESPERBATCH == INSTANCESPERBATCH-1:
            lastl = runl / INSTANCESPERBATCH
            print('  batch {} loss: {}'.format(i + 1, lastl))
            runl = 0
    return lastl

# %%
def run_training():
    for epoch in tqdm(range(EPOCHS)):
        print('EPOCH {}:'.format(epoch + 1))
        finalModel.train(True)
        avg_loss = training()
        print('loss: {}'.format(avg_loss))
        finalModel.train(False)
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


