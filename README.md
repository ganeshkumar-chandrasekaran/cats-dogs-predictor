# Cat or Dog Predictor

Browser-based Cats vs Dogs classifier using a custom MiniResNet (ONNX Runtime Web).

- Images are processed **locally in the browser**
- Uploaded images are **cleared from memory after prediction**
- Hosted on GitHub Pages

## Local preview

```bash
cd cats-dogs-predictor
python3 -m http.server 8080
```

Open http://localhost:8080
