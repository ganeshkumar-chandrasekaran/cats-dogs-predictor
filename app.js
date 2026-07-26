(() => {
  const CLASS_NAMES = ["Cat", "Dog", "Neither"];
  const IMG_SIZE = 224;
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];
  const MIN_PET_CONFIDENCE = 0.60;
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
  // Re-get context after resizes; keep options when recreating
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
    const ctx = getCtx();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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

  /** Read JPEG EXIF orientation (1–8). PNG/WebP → 1. */
  function getExifOrientation(file) {
    return new Promise((resolve) => {
      if (!file || !file.type || !file.type.includes("jpeg") && file.type !== "image/jpg") {
        // Still try for .jpg with empty type
        if (file && file.type && !/jpe?g/i.test(file.type) && file.type !== "") {
          resolve(1);
          return;
        }
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
              // APP1
              const appLength = view.getUint16(offset, false);
              if (offset + appLength > view.byteLength) break;
              if (
                view.getUint32(offset + 2, false) !== 0x45786966 ||
                view.getUint16(offset + 6, false) !== 0x0000
              ) {
                offset += appLength;
                continue;
              }
              const tiffOffset = offset + 8;
              const little = view.getUint16(tiffOffset, false) === 0x4949;
              if (view.getUint16(tiffOffset + 2, little) !== 0x002a) {
                resolve(1);
                return;
              }
              const ifd0 = tiffOffset + view.getUint32(tiffOffset + 4, little);
              const entries = view.getUint16(ifd0, little);
              for (let i = 0; i < entries; i++) {
                const entry = ifd0 + 2 + i * 12;
                if (entry + 12 > view.byteLength) break;
                const tag = view.getUint16(entry, little);
                if (tag === 0x0112) {
                  resolve(view.getUint16(entry + 8, little));
                  return;
                }
              }
              resolve(1);
              return;
            }
            if ((marker & 0xff00) !== 0xff00) break;
            if (marker === 0xffda) break; // SOS
            const len = view.getUint16(offset, false);
            offset += len;
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

  /** Draw source onto a new canvas with EXIF orientation applied once. */
  function bakeOrientation(source, orientation) {
    const w = source.width;
    const h = source.height;
    const out = document.createElement("canvas");
    const octx = out.getContext("2d");
    const swap = orientation >= 5 && orientation <= 8;
    out.width = swap ? h : w;
    out.height = swap ? w : h;

    switch (orientation) {
      case 2: octx.transform(-1, 0, 0, 1, w, 0); break;
      case 3: octx.transform(-1, 0, 0, -1, w, h); break;
      case 4: octx.transform(1, 0, 0, -1, 0, h); break;
      case 5: octx.transform(0, 1, 1, 0, 0, 0); break;
      case 6: octx.transform(0, 1, -1, 0, h, 0); break; // 90° CW
      case 7: octx.transform(0, -1, -1, 0, h, w); break;
      case 8: octx.transform(0, -1, 1, 0, 0, w); break; // 90° CCW
      default: break;
    }
    octx.drawImage(source, 0, 0);
    return out;
  }

  async function decodeRawBitmap(file) {
    // Prefer raw pixels (no auto-orient) so we apply EXIF exactly once.
    if (typeof createImageBitmap === "function") {
      try {
        const bitmap = await createImageBitmap(file, { imageOrientation: "none" });
        return { bitmap, appliesExifOurselves: true };
      } catch (_) {
        try {
          // Browser may have already oriented — do not apply EXIF again
          const bitmap = await createImageBitmap(file);
          return { bitmap, appliesExifOurselves: false };
        } catch (_) {
          /* fall through */
        }
      }
    }
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.decoding = "async";
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = url;
      });
      if (img.decode) {
        try { await img.decode(); } catch (_) { /* ignore */ }
      }
      // <img> decode is typically already upright on modern browsers
      return { bitmap: img, appliesExifOurselves: false };
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function coverCropToWorkCanvas(source) {
    const iw = source.width;
    const ih = source.height;
    if (!iw || !ih) throw new Error("Invalid image dimensions");

    canvas.width = IMG_SIZE;
    canvas.height = IMG_SIZE;
    const ctx = getCtx();
    ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);

    const scale = Math.max(IMG_SIZE / iw, IMG_SIZE / ih);
    const sw = IMG_SIZE / scale;
    const sh = IMG_SIZE / scale;
    const sx = (iw - sw) / 2;
    const sy = (ih - sh) / 2;
    ctx.drawImage(source, sx, sy, sw, sh, 0, 0, IMG_SIZE, IMG_SIZE);
  }

  async function prepareSquareCanvasFromFile(file) {
    const { bitmap, appliesExifOurselves } = await decodeRawBitmap(file);
    let upright = bitmap;

    if (appliesExifOurselves) {
      const orientation = await getExifOrientation(file);
      if (orientation && orientation !== 1) {
        upright = bakeOrientation(bitmap, orientation);
        if (typeof bitmap.close === "function") bitmap.close();
      }
    }

    coverCropToWorkCanvas(upright);
    if (typeof upright.close === "function") upright.close();
    return canvas;
  }

  async function detectPetSignal(canvasEl) {
    const preds = await gateModel.classify(canvasEl, 5);
    let catScore = 0;
    let dogScore = 0;
    for (const p of preds) {
      if (labelLooksLikeCat(p.className)) catScore = Math.max(catScore, p.probability);
      if (labelLooksLikeDog(p.className)) dogScore = Math.max(dogScore, p.probability);
    }
    const petScore = Math.max(catScore, dogScore);
    const isPet = petScore >= 0.12;
    return { isPet, petScore, catScore, dogScore, preds };
  }

  function canvasToTensor() {
    const ctx = getCtx();
    const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    const float32 = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
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

      if (!gate.isPet) {
        const neither = Math.min(0.96, Math.max(0.80, 1 - gate.petScore));
        const residual = 1 - neither;
        renderScores(
          residual / 2,
          residual / 2,
          neither,
          `Neither — not a cat or dog  ·  ${(neither * 100).toFixed(1)}%`
        );
        setStatus("Done — image cleared from this device.");
        return;
      }

      setStatus("Pet detected — classifying Cat vs Dog…");
      const input = canvasToTensor();
      const feeds = { [session.inputNames[0]]: input };
      const out = await session.run(feeds);
      const probs = Array.from(out[session.outputNames[0]].data);
      const cat = probs[0];
      const dog = probs[1];
      const confidence = Math.max(cat, dog);

      if (confidence < MIN_PET_CONFIDENCE) {
        renderScores(cat, dog, 1 - confidence, null);
      } else {
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
