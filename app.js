(() => {
  const CLASS_NAMES = ["Cat", "Dog", "Neither"];
  const IMG_SIZE = 224;
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];
  const MIN_PET_CONFIDENCE = 0.55; // kept for reference; pet override uses 0.58 below
  const MODEL_URL = new URL("./model/miniresnet_cats_dogs.onnx", window.location.href).href;

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
  const labelOut = document.getElementById("labelOut");
  const catBar = document.getElementById("catBar");
  const dogBar = document.getElementById("dogBar");
  const noneBar = document.getElementById("noneBar");
  const catPct = document.getElementById("catPct");
  const dogPct = document.getElementById("dogPct");
  const nonePct = document.getElementById("nonePct");
  const canvas = document.getElementById("workCanvas");

  function getCtx() {
    return canvas.getContext("2d", { willReadFrequently: true, alpha: false });
  }

  let session = null;
  let gateModel = null;
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
   * Match training: transforms.Resize((224, 224)) stretches to exact size (no center-crop).
   * Orientation: use preview pixels when EXIF is normal; otherwise decode with from-image.
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
        // Displayed <img> is usually already upright on desktop & modern mobile
        source = await createImageBitmap(preview);
        shouldClose = true;
      } catch (_) {
        source = null;
      }
    }

    if (!source) source = preview;

    canvas.width = IMG_SIZE;
    canvas.height = IMG_SIZE;
    const ctx = getCtx();
    ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
    // Stretch to 224×224 — same as training Resize((224, 224))
    ctx.drawImage(source, 0, 0, IMG_SIZE, IMG_SIZE);

    if (shouldClose && typeof source.close === "function") source.close();
    return canvas;
  }

  async function detectPetSignal(canvasEl) {
    // top-10 helps when both cat and dog appear (labels can be split)
    const preds = await gateModel.classify(canvasEl, 10);
    let catScore = 0;
    let dogScore = 0;
    let animalHint = 0;
    for (const p of preds) {
      const name = p.className.toLowerCase();
      if (labelLooksLikeCat(p.className)) catScore = Math.max(catScore, p.probability);
      if (labelLooksLikeDog(p.className)) dogScore = Math.max(dogScore, p.probability);
      // Broad animal fallback (fox, wolf already in dog hints; add generic)
      if (
        /(animal|pet|kitten|puppy|feline|canine|mammal)/.test(name) ||
        labelLooksLikeCat(p.className) ||
        labelLooksLikeDog(p.className)
      ) {
        animalHint = Math.max(animalHint, p.probability);
      }
    }
    // Sum cat+dog mass across top preds — better for cat+dog photos
    let catMass = 0;
    let dogMass = 0;
    for (const p of preds) {
      if (labelLooksLikeCat(p.className)) catMass += p.probability;
      if (labelLooksLikeDog(p.className)) dogMass += p.probability;
    }
    const petScore = Math.max(catScore, dogScore, catMass + dogMass, animalHint);
    const isPet = petScore >= 0.06 || catMass + dogMass >= 0.08;
    return { isPet, petScore, catScore, dogScore, catMass, dogMass, preds };
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

  async function loadModels() {
    try {
      setStatus("Loading models…");
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";
      if (window.tf) await tf.ready();
      const [sess, gate] = await Promise.all([
        ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] }),
        mobilenet.load({ version: 2, alpha: 1.0 }),
      ]);
      session = sess;
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
    predictBtn.disabled = !(session && gateModel);
    clearBtn.disabled = false;
    resultEl.classList.add("hidden");
    setStatus("Image selected. Click Predict.");
  }

  function normalizeScores(cat, dog, neither) {
    const sum = cat + dog + neither;
    if (sum <= 0) return { cat: 0, dog: 0, neither: 1 };
    return { cat: cat / sum, dog: dog / sum, neither: neither / sum };
  }

  function renderScores(cat, dog, neither, labelOverride) {
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

    labelOut.textContent = label;
    labelOut.classList.toggle("is-neither", top.name === "Neither");
    catBar.style.width = "0%";
    dogBar.style.width = "0%";
    noneBar.style.width = "0%";
    requestAnimationFrame(() => {
      catBar.style.width = `${(s.cat * 100).toFixed(1)}%`;
      dogBar.style.width = `${(s.dog * 100).toFixed(1)}%`;
      noneBar.style.width = `${(s.neither * 100).toFixed(1)}%`;
    });
    catPct.textContent = `${(s.cat * 100).toFixed(1)}%`;
    dogPct.textContent = `${(s.dog * 100).toFixed(1)}%`;
    nonePct.textContent = `${(s.neither * 100).toFixed(1)}%`;
    resultEl.classList.remove("hidden");
  }

  async function predict() {
    if (!session || !gateModel || !selectedFile) return;
    predictBtn.disabled = true;
    setStatus("Preparing image…");
    try {
      const square = await prepareSquareCanvasFromFile(selectedFile);

      setStatus("Checking if this looks like a cat or dog…");
      const gate = await detectPetSignal(square);

      // Always run Cat/Dog model — needed for cat+dog photos and gate misses
      setStatus("Classifying…");
      const input = canvasToTensor();
      const out = await session.run({ [session.inputNames[0]]: input });
      const probs = Array.from(out[session.outputNames[0]].data);
      const cat = probs[0];
      const dog = probs[1];
      const confidence = Math.max(cat, dog);

      // Neither only for non-pets. If MiniResNet is fairly sure, trust it even
      // when MobileNet gate is weak (common with cat+dog in one frame).
      const treatAsPet =
        gate.isPet ||
        confidence >= 0.58 ||
        gate.petScore >= 0.04 ||
        (gate.catMass || 0) + (gate.dogMass || 0) >= 0.05;

      if (!treatAsPet) {
        const neither = Math.min(0.96, Math.max(0.80, 1 - Math.max(gate.petScore, confidence * 0.5)));
        const residual = 1 - neither;
        renderScores(
          residual / 2,
          residual / 2,
          neither,
          `Neither — not a cat or dog  ·  ${(neither * 100).toFixed(1)}%`
        );
      } else {
        // Always pick Cat or Dog for pet-like images (no Neither takeover)
        renderScores(cat, dog, 0, null);
      }
      setStatus("Done — image cleared from this device.");
    } catch (err) {
      console.error(err);
      setStatus("Prediction failed. Try another image.");
    } finally {
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

  loadModels();
})();
