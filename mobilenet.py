# %%
import torch
import cv2
import numpy as np
import glob
import sklearn.model_selection
from tqdm import tqdm
from torchsummary import summary

lr = 0.0001
EPOCHS = 20
BATCH_SIZE = 20

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
paths = ["./JS/cat/*.jpg", "./JS/dog/*.jpg"]
labels = {"cat": 0, "dog": 1}
def func(path):
    result = []
    for j in glob.glob(path):
        i = cv2.imread(j, cv2.COLOR_BGR2RGB)
        i = i/255
        i = np.transpose(i)
        i = (i, labels[path.split('/')[2]])
        result.append(i)
    return result

images = func(paths[0])
images2 = func(paths[1])
images3 = images + images2

# %%
train, test = sklearn.model_selection.train_test_split(images3, train_size=0.8)

print('Training set has {} instances'.format(len(train)))
print('Testing set has {} instances'.format(len(test)))

tr_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
te_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

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
        if i % 100 == 99:
            lastl = runl / 100
            print('  batch {} loss: {}'.format(i + 1, lastl))
            runl = 0
    return lastl

# %%
def run_training():
    for epoch in tqdm(range(EPOCHS)):
        print('EPOCH {}:'.format(epoch + 1))
        finalModel.train(True)
        avg_loss = training()
        finalModel.train(False)
        if epoch == 0:
            best_loss = avg_loss
        if best_loss > avg_loss:
            best_loss = avg_loss
            bestweights = finalModel.state_dict()
            bestepoch = epoch + 1
    model_path = "model_{}.pt".format(bestepoch)
    torch.save(bestweights, model_path)

run_training()

# %% [markdown]
# <h1>Testing</h1>

# %%
import sklearn.metrics
finalModel.load_state_dict(torch.load("./model_20.pt"))
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
        result.append(outputs)
        result2.append(labels)
    return np.concatenate(result), np.concatenate(result2)
outputs, grounds = testing()
print(outputs, grounds)
outputs = np.argmax(outputs, axis=0)

# %%
print(len(outputs))
print(outputs[0].shape)
print(outputs)
print(grounds)

# %%
print(sklearn.metrics.accuracy_score(grounds, outputs))

# %%
print(sklearn.metrics.confusion_matrix(grounds, outputs))

# %%
input = torch.randn(1,3,224,224, device="cuda")
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]
torch.onnx.export(finalModel, input, "mobilenet_v2.onnx", input_names=input_names, output_names=output_names)

# %%



