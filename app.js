(() => {
  const IMG_SIZE = 224;
  const RESIZE_SIDE = 256;
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];
  const CUSTOM_URL = new URL("./model/custom_cnn.onnx", window.location.href).href;
  const TRANSFER_URL = new URL("./model/transfer_resnet18.onnx", window.location.href).href;

  const CAT_HINTS = [
    "cat", "kitten", "tabby", "tiger cat", "persian cat", "siamese cat",
    "egyptian cat", "lynx", "cougar", "lion", "tiger", "leopard", "cheetah", "jaguar",
  ];
  const DOG_HINTS = [
    "dog", "puppy", "retriever", "terrier", "spaniel", "hound", "poodle", "shepherd",
    "beagle", "bulldog", "chihuahua", "husky", "pug", "boxer", "collie", "dalmatian",
    "labrador", "corgi", "mastiff", "pinscher", "setter", "sheepdog", "wolfhound",
    "malamute", "samoyed", "pomeranian", "rottweiler", "doberman", "greyhound",
    "whippet", "pekinese", "papillon", "toy terrier", "affenpinscher", "bloodhound",
    "bluetick", "coonhound", "walker hound", "english foxhound", "redbone",
    "borzoi", "irish wolfhound", "italian greyhound", "ibizan hound",
    "norwegian elkhound", "otterhound", "saluki", "scottish deerhound", "weimaraner",
    "staffordshire", "cairn", "australian terrier", "dandie", "boston bull",
    "miniature schnauzer", "giant schnauzer", "standard schnauzer",
    "kelpie", "briard", "komondor", "old english sheepdog", "shetland sheepdog",
    "border collie", "bouvier", "german shepherd",
    "cardigan", "pembroke", "toy poodle", "miniature poodle", "standard poodle",
    "mexican hairless", "timber wolf", "white wolf", "red wolf", "coyote", "dingo",
    "dhole", "african hunting dog",
  ];
  const FALSE_CAT = ["caterpillar", "catamaran", "catheter", "catalina", "cation"];

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("preview");
  const previewWrap = document.getElementById("previewWrap");
  const placeholder = document.getElementById("placeholder");
  const predictBtn = document.getElementById("predictBtn");
  const clearBtn = document.getElementById("clearBtn");
  const statusEl = document.getElementById("status");
  const resultEl = document.getElementById("result");
  const canvas = document.getElementById("workCanvas");

  const panels = {
    custom: {
      labelOut: document.getElementById("customLabelOut"),
      catBar: document.getElementById("customCatBar"),
      dogBar: document.getElementById("customDogBar"),
      noneBar: document.getElementById("customNoneBar"),
      catPct: document.getElementById("customCatPct"),
      dogPct: document.getElementById("customDogPct"),
      nonePct: document.getElementById("customNonePct"),
    },
    transfer: {
      labelOut: document.getElementById("transferLabelOut"),
      catBar: document.getElementById("transferCatBar"),
      dogBar: document.getElementById("transferDogBar"),
      noneBar: document.getElementById("transferNoneBar"),
      catPct: document.getElementById("transferCatPct"),
      dogPct: document.getElementById("transferDogPct"),
      nonePct: document.getElementById("transferNonePct"),
    },
  };

  function getCtx() {
    return canvas.getContext("2d", { willReadFrequently: true, alpha: false });
  }

  let customSession = null;
  let transferSession = null;
  let gateModel = null;
  let objectUrl = null;
  let selectedFile = null;

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  function modelsReady() {
    return !!(customSession && transferSession && gateModel);
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
    getCtx().clearRect(0, 0, canvas.width || IMG_SIZE, canvas.height || IMG_SIZE);
    selectedFile = null;
    fileInput.value = "";
    revokePreview();
  }

  function labelLooksLikeCat(name) {
    const n = name.toLowerCase();
    if (FALSE_CAT.some((x) => n.includes(x))) return false;
    return CAT_HINTS.some((h) => n.includes(h));
  }

  function labelLooksLikeDog(name) {
    const n = name.toLowerCase();
    return DOG_HINTS.some((h) => n.includes(h));
  }

  function getExifOrientation(file) {
    return new Promise((resolve) => {
      const type = (file && file.type) || "";
      if (type && !/jpe?g/i.test(type)) {
        resolve(1);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const view = new DataView(reader.result);
          if (view.byteLength < 2 || view.getUint16(0, false) !== 0xffd8) {
            resolve(1);
            return;
          }
          let offset = 2;
          while (offset + 4 < view.byteLength) {
            const marker = view.getUint16(offset, false);
            offset += 2;
            if (marker === 0xffe1) {
              if (
                view.getUint32(offset + 2, false) !== 0x45786966 ||
                view.getUint16(offset + 6, false) !== 0x0000
              ) {
                offset += view.getUint16(offset, false);
                continue;
              }
              const tiff = offset + 8;
              const little = view.getUint16(tiff, false) === 0x4949;
              const ifd0 = tiff + view.getUint32(tiff + 4, little);
              const entries = view.getUint16(ifd0, little);
              for (let i = 0; i < entries; i++) {
                const entry = ifd0 + 2 + i * 12;
                if (view.getUint16(entry, little) === 0x0112) {
                  resolve(view.getUint16(entry + 8, little) || 1);
                  return;
                }
              }
              resolve(1);
              return;
            }
            if ((marker & 0xff00) !== 0xff00 || marker === 0xffda) break;
            offset += view.getUint16(offset, false);
          }
          resolve(1);
        } catch (_) {
          resolve(1);
        }
      };
      reader.onerror = () => resolve(1);
      reader.readAsArrayBuffer(file.slice(0, 128 * 1024));
    });
  }

  async function waitForPreview() {
    if (!preview.src) throw new Error("No preview image");
    if (preview.decode) {
      try {
        await preview.decode();
      } catch (_) {
        /* ignore */
      }
    }
    if (!preview.complete || !preview.naturalWidth) {
      await new Promise((resolve, reject) => {
        preview.onload = resolve;
        preview.onerror = reject;
      });
    }
  }

  /**
   * Match notebook test pipeline: Resize(256) -> CenterCrop(224) -> Normalize
   */
  async function prepareSquareCanvasFromFile(file) {
    await waitForPreview();
    const orientation = await getExifOrientation(file);

    let source = null;
    let shouldClose = false;

    if (orientation !== 1 && typeof createImageBitmap === "function") {
      try {
        source = await createImageBitmap(file, { imageOrientation: "from-image" });
        shouldClose = true;
      } catch (_) {
        source = null;
      }
    }

    if (!source && typeof createImageBitmap === "function") {
      try {
        source = await createImageBitmap(preview);
        shouldClose = true;
      } catch (_) {
        source = null;
      }
    }

    if (!source) source = preview;

    const sw = source.width || source.naturalWidth || source.videoWidth;
    const sh = source.height || source.naturalHeight || source.videoHeight;
    const scale = RESIZE_SIDE / Math.min(sw, sh);
    const rw = Math.round(sw * scale);
    const rh = Math.round(sh * scale);
    const sx = Math.max(0, Math.floor((rw - IMG_SIZE) / 2));
    const sy = Math.max(0, Math.floor((rh - IMG_SIZE) / 2));

    const tmp = document.createElement("canvas");
    tmp.width = rw;
    tmp.height = rh;
    const tctx = tmp.getContext("2d", { alpha: false });
    tctx.drawImage(source, 0, 0, rw, rh);

    canvas.width = IMG_SIZE;
    canvas.height = IMG_SIZE;
    const ctx = getCtx();
    ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
    ctx.drawImage(tmp, sx, sy, IMG_SIZE, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE);

    if (shouldClose && typeof source.close === "function") source.close();
    return canvas;
  }

  async function detectPetSignal(canvasEl) {
    const preds = await gateModel.classify(canvasEl, 10);
    let catScore = 0;
    let dogScore = 0;
    let catMass = 0;
    let dogMass = 0;
    for (const p of preds) {
      if (labelLooksLikeCat(p.className)) {
        catScore = Math.max(catScore, p.probability);
        catMass += p.probability;
      }
      if (labelLooksLikeDog(p.className)) {
        dogScore = Math.max(dogScore, p.probability);
        dogMass += p.probability;
      }
    }
    const petScore = Math.max(catScore, dogScore);
    const petMass = catMass + dogMass;
    const isPet =
      petScore >= 0.12 ||
      petMass >= 0.15 ||
      (catScore >= 0.05 && dogScore >= 0.05);
    return { isPet, petScore, petMass, catScore, dogScore, catMass, dogMass, preds };
  }

  function canvasToTensor() {
    const ctx = getCtx();
    const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    const float32 = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
    let i = 0;
    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        const p = (y * IMG_SIZE + x) * 4;
        float32[i] = (data[p] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        float32[IMG_SIZE * IMG_SIZE + i] =
          (data[p + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        float32[2 * IMG_SIZE * IMG_SIZE + i] =
          (data[p + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        i++;
      }
    }
    return new ort.Tensor("float32", float32, [1, 3, IMG_SIZE, IMG_SIZE]);
  }

  async function runOnnx(session, input) {
    const out = await session.run({ [session.inputNames[0]]: input });
    return Array.from(out[session.outputNames[0]].data);
  }

  async function loadModels() {
    try {
      setStatus("Loading Custom CNN + Transfer Learning models…");
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";
      if (window.tf) await tf.ready();
      const [custom, transfer, gate] = await Promise.all([
        ort.InferenceSession.create(CUSTOM_URL, { executionProviders: ["wasm"] }),
        ort.InferenceSession.create(TRANSFER_URL, { executionProviders: ["wasm"] }),
        mobilenet.load({ version: 2, alpha: 1.0 }),
      ]);
      customSession = custom;
      transferSession = transfer;
      gateModel = gate;
      setStatus("Ready — drop a photo (cat, dog, or anything else).");
      if (selectedFile) predictBtn.disabled = false;
    } catch (err) {
      console.error(err);
      setStatus("Could not load models. Refresh and try again.");
    }
  }

  function showPreview(file) {
    revokePreview();
    selectedFile = file;
    objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    previewWrap.classList.remove("hidden");
    placeholder.classList.add("hidden");
    predictBtn.disabled = !modelsReady();
    clearBtn.disabled = false;
    resultEl.classList.add("hidden");
    setStatus("Image selected. Click Predict.");
  }

  function normalizeScores(cat, dog, neither) {
    const sum = cat + dog + neither;
    if (sum <= 0) return { cat: 0, dog: 0, neither: 1 };
    return { cat: cat / sum, dog: dog / sum, neither: neither / sum };
  }

  function renderPanel(panel, cat, dog, neither, labelOverride) {
    const s = normalizeScores(cat, dog, neither);
    const ranked = [
      { name: "Cat", v: s.cat },
      { name: "Dog", v: s.dog },
      { name: "Neither", v: s.neither },
    ].sort((a, b) => b.v - a.v);
    const top = ranked[0];
    const pct = (top.v * 100).toFixed(1);
    const label =
      labelOverride ||
      (top.name === "Neither"
        ? `Neither  ·  ${pct}%`
        : `It’s a ${top.name}  ·  ${pct}%`);

    panel.labelOut.textContent = label;
    panel.labelOut.classList.toggle("is-neither", top.name === "Neither");
    panel.catBar.style.width = "0%";
    panel.dogBar.style.width = "0%";
    panel.noneBar.style.width = "0%";
    requestAnimationFrame(() => {
      panel.catBar.style.width = `${(s.cat * 100).toFixed(1)}%`;
      panel.dogBar.style.width = `${(s.dog * 100).toFixed(1)}%`;
      panel.noneBar.style.width = `${(s.neither * 100).toFixed(1)}%`;
    });
    panel.catPct.textContent = `${(s.cat * 100).toFixed(1)}%`;
    panel.dogPct.textContent = `${(s.dog * 100).toFixed(1)}%`;
    panel.nonePct.textContent = `${(s.neither * 100).toFixed(1)}%`;
  }

  async function predict() {
    if (!modelsReady() || !selectedFile) return;
    predictBtn.disabled = true;
    setStatus("Preparing image…");
    try {
      const square = await prepareSquareCanvasFromFile(selectedFile);

      setStatus("Checking if this looks like a cat or dog…");
      const gate = await detectPetSignal(square);

      if (!gate.isPet) {
        const neither = Math.min(0.96, Math.max(0.82, 1 - gate.petScore));
        const residual = 1 - neither;
        const neitherLabel = `Neither — not a cat or dog  ·  ${(neither * 100).toFixed(1)}%`;
        renderPanel(panels.custom, residual / 2, residual / 2, neither, neitherLabel);
        renderPanel(panels.transfer, residual / 2, residual / 2, neither, neitherLabel);
        resultEl.classList.remove("hidden");
        setStatus("Done. Upload another photo or press Clear to remove this image.");
        return;
      }

      setStatus("Pet detected — running Custom CNN + Transfer Learning…");
      const input = canvasToTensor();
      const [customProbs, transferProbs] = await Promise.all([
        runOnnx(customSession, input),
        runOnnx(transferSession, input),
      ]);

      renderPanel(panels.custom, customProbs[0], customProbs[1], 0, null);
      renderPanel(panels.transfer, transferProbs[0], transferProbs[1], 0, null);
      resultEl.classList.remove("hidden");
      setStatus("Done. Upload another photo or press Clear to remove this image.");
    } catch (err) {
      console.error(err);
      setStatus("Prediction failed. Try another image.");
    } finally {
      predictBtn.disabled = !selectedFile;
      clearBtn.disabled = !selectedFile;
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

  loadModels();
})();
