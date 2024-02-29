import torch
from typing import Callable
from tqdm.auto import tqdm


def train_step(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                accuracy_fn: Callable):
  model.train()
  train_loss, train_acc = 0, 0
  sample = 0

  for X, target in data_loader:
    X, target = X.to(device), target.to(device).type(torch.float)
    sample += len(X)

    y_pred = model(X).squeeze()
    y_label = torch.round(torch.sigmoid(y_pred))

    loss = loss_fn(y_pred, target)
    train_loss += loss
    train_acc += accuracy_fn(y_label, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if sample % 320 == 0:
      print(f"Train || Examined {sample}/{len(data_loader.dataset)}")

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)

  return train_loss, train_acc


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device, 
              accuracy_fn: Callable):
  model.eval()
  test_loss, test_acc = 0, 0
  sample = 0

  with torch.inference_mode():
    for X, target in data_loader:
      X, target = X.to(device), target.to(device).type(torch.float)
      sample += len(X)

      y_pred = model(X).squeeze()
      y_label = torch.round(torch.sigmoid(y_pred))

      loss = loss_fn(y_pred, target)
      test_loss += loss
      test_acc += accuracy_fn(y_label, target)
    
      if sample % 64 == 0:
        print(f"Test || Examined {sample}/{len(data_loader.dataset)}")

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    return test_loss, test_acc


def train_model(model: torch.nn.Module,
                train_data_loader: torch.utils.data.DataLoader,
                test_data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                epochs: int,
                device: torch.device,
                accuracy_fn: Callable):

  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []
             }

  test_loss, test_acc = test_step(model=model,
                                    data_loader=test_data_loader,
                                    loss_fn=loss_fn,
                                    device=device,
                                    accuracy_fn=accuracy_fn)
  print(f"Epoch: n/a | Train loss: n/a | Train accuracy: n/a |Test loss: {test_loss:.5f} | Test accuracy: {test_acc*100:.2f}%")
  results["test_loss"].append(test_loss)
  results["test_acc"].append(test_acc)

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       data_loader=train_data_loader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device,
                                       accuracy_fn=accuracy_fn)
    test_loss, test_acc = test_step(model=model,
                                    data_loader=test_data_loader,
                                    loss_fn=loss_fn,
                                    device=device,
                                    accuracy_fn=accuracy_fn)

    print(f"Epoch: {epoch+1} | Train loss: {train_loss:.5f} | Train accuracy: {train_acc*100:.2f}% | Test loss: {test_loss:.5f} | Test accuracy: {test_acc*100:.2f}%")
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results