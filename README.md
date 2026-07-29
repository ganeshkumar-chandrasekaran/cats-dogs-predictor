# Cat or Dog? — Dual Model Predictor

Live site: https://ganeshkumar-chandrasekaran.github.io/cats-dogs-predictor/

On-device browser inference comparing:

- **Custom CNN** (`model/custom_cnn.onnx`) — MiniResNet trained from scratch
- **Transfer Learning** (`model/transfer_resnet18.onnx`) — ResNet18 + GAP head

A MobileNet gate marks non-pet images as **Neither**. Photos never leave the device.
