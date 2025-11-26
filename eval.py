import torch

# Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # todo...
            # Compute outputs and loss
            outputs = model(inputs)
            loss = criterion(outputs, targets) # Compute loss
            val_loss += loss.item()
            # now we have to take the predictions selecting the class with max score
            # example : outputs = we have all the scores for each class and we take the max
            _, predicted = outputs.max(1) # Get predictions because max returns (values, indices of classes)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}%')
    return val_accuracy