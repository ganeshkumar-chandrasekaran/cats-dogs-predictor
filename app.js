(() => {
  const CLASS_NAMES = ["Cat", "Dog"];
  const IMG_SIZE = 224;
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];
  // Works for both project-root Pages and /web/ local paths
  const MODEL_URL = new URL("./model/miniresnet_cats_dogs.onnx", window.location.href).href;

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("preview");
  const previewWrap = document.getElementById("previewWrap");
  const placeholder = document.getElementById("placeholder");
  const predictBtn = document.getElementById("predictBtn");
  const clearBtn = document.getElementById("clearBtn");
  const statusEl = document.getElementById("status");
  const resultEl = document.getElementById("result");
  const labelOut = document.getElementById("labelOut");
  const catBar = document.getElementById("catBar");
  const dogBar = document.getElementById("dogBar");
  const catPct = document.getElementById("catPct");
  const dogPct = document.getElementById("dogPct");
  const canvas = document.getElementById("workCanvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  let session = null;
  let objectUrl = null;
  let selectedFile = null;

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  function revokePreview() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    preview.removeAttribute("src");
    previewWrap.classList.add("hidden");
    placeholder.classList.remove("hidden");
  }

  function wipeImageData() {
    // Clear canvas pixels
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Drop file references so GC can reclaim the blob
    selectedFile = null;
    fileInput.value = "";
    revokePreview();
  }

  async function loadModel() {
    try {
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";
      setStatus("Warming up the model…");
      session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
      });
      setStatus("Ready — drop a cat or dog photo.");
    } catch (err) {
      console.error(err);
      setStatus("Could not load the model. Refresh and try again.");
    }
  }

  function showPreview(file) {
    revokePreview();
    selectedFile = file;
    objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    previewWrap.classList.remove("hidden");
    placeholder.classList.add("hidden");
    predictBtn.disabled = !session;
    clearBtn.disabled = false;
    resultEl.classList.add("hidden");
    setStatus("Image selected. Click Predict.");
  }

  function preprocessToTensor() {
    // Draw cover-cropped into 224x224
    const img = preview;
    const iw = img.naturalWidth;
    const ih = img.naturalHeight;
    const scale = Math.max(IMG_SIZE / iw, IMG_SIZE / ih);
    const sw = IMG_SIZE / scale;
    const sh = IMG_SIZE / scale;
    const sx = (iw - sw) / 2;
    const sy = (ih - sh) / 2;
    ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, IMG_SIZE, IMG_SIZE);

    const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    const float32 = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
    // NCHW + ImageNet normalize
    let i = 0;
    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        const p = (y * IMG_SIZE + x) * 4;
        const r = data[p] / 255;
        const g = data[p + 1] / 255;
        const b = data[p + 2] / 255;
        float32[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        float32[IMG_SIZE * IMG_SIZE + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        float32[2 * IMG_SIZE * IMG_SIZE + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        i++;
      }
    }
    return new ort.Tensor("float32", float32, [1, 3, IMG_SIZE, IMG_SIZE]);
  }

  async function predict() {
    if (!session || !selectedFile) return;
    predictBtn.disabled = true;
    setStatus("Running inference…");
    try {
      const input = preprocessToTensor();
      const feeds = {};
      const inputName = session.inputNames[0];
      feeds[inputName] = input;
      const out = await session.run(feeds);
      const outputName = session.outputNames[0];
      const probs = Array.from(out[outputName].data);
      const cat = probs[0];
      const dog = probs[1];
      const predIdx = cat >= dog ? 0 : 1;
      const confidence = Math.max(cat, dog);

      labelOut.textContent = `It’s a ${CLASS_NAMES[predIdx]}  ·  ${(confidence * 100).toFixed(1)}%`;
      // restart bar animation
      catBar.style.width = "0%";
      dogBar.style.width = "0%";
      requestAnimationFrame(() => {
        catBar.style.width = `${(cat * 100).toFixed(1)}%`;
        dogBar.style.width = `${(dog * 100).toFixed(1)}%`;
      });
      catPct.textContent = `${(cat * 100).toFixed(1)}%`;
      dogPct.textContent = `${(dog * 100).toFixed(1)}%`;
      resultEl.classList.remove("hidden");
      setStatus("Done — image cleared from this device.");
    } catch (err) {
      console.error(err);
      setStatus("Prediction failed. Try another image.");
    } finally {
      // Privacy: delete / discard uploaded image after prediction
      wipeImageData();
      predictBtn.disabled = true;
      clearBtn.disabled = true;
    }
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) showPreview(file);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) showPreview(file);
  });

  predictBtn.addEventListener("click", predict);
  clearBtn.addEventListener("click", () => {
    wipeImageData();
    resultEl.classList.add("hidden");
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    setStatus("Cleared. Upload another image when ready.");
  });

  loadModel().then(() => {
    if (session && selectedFile) predictBtn.disabled = false;
  });
})();
