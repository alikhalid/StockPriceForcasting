import torch
from sklearn.metrics import mean_squared_error


def train(epoch, X, y, model, optimizer, criterion, writer):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)

    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    writer.add_scalar('train/loss', loss.item(), epoch)

    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')


def test(epoch, X, y, model, writer):
    model.eval()

    with torch.no_grad():
        test_outputs = model(X).squeeze()
        test_loss = mean_squared_error(y.numpy(), test_outputs.numpy())
        print(f'Test Loss (MSE): {test_loss:.4f}')
        writer.add_scalar('test/loss', test_loss, epoch)

        return test_outputs, y
