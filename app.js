(() => {
  const CLASS_NAMES = ["Cat", "Dog", "Neither"];
  const IMG_SIZE = 224;
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];
  // MiniResNet must be fairly sure after the pet gate passes
  const MIN_PET_CONFIDENCE = 0.60;
  const MODEL_URL = new URL("./model/miniresnet_cats_dogs.onnx", window.location.href).href;

  // ImageNet / MobileNet labels that indicate a domestic cat or dog
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
    "borzoi", "irish wolfhound", "italian greyhound", "whippet", "ibizan hound",
    "norwegian elkhound", "otterhound", "saluki", "scottish deerhound", "weimaraner",
    "staffordshire", "cairn", "australian terrier", "dandie", "boston bull",
    "miniature schnauzer", "giant schnauzer", "standard schnauzer",
    "kelpie", "briard", "komondor", "old english sheepdog", "shetland sheepdog",
    "collie", "border collie", "bouvier", "rottweiler", "german shepherd",
    "cardigan", "pembroke", "toy poodle", "miniature poodle", "standard poodle",
    "mexican hairless", "timber wolf", "white wolf", "red wolf", "coyote", "dingo",
    "dhole", "african hunting dog",
  ];
  // Avoid accidental substring matches
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
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

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

  async function detectPetSignal(imgEl) {
    // MobileNet gate: reject humans / objects that are not cat/dog-like
    const preds = await gateModel.classify(imgEl, 5);
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

  async function loadModels() {
    try {
      setStatus("Loading models…");
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";
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

  function preprocessToTensor() {
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

  function renderScores(cat, dog, neither, label) {
    labelOut.textContent = label;
    labelOut.classList.toggle("is-neither", neither >= Math.max(cat, dog));
    catBar.style.width = "0%";
    dogBar.style.width = "0%";
    noneBar.style.width = "0%";
    requestAnimationFrame(() => {
      catBar.style.width = `${(cat * 100).toFixed(1)}%`;
      dogBar.style.width = `${(dog * 100).toFixed(1)}%`;
      noneBar.style.width = `${(neither * 100).toFixed(1)}%`;
    });
    catPct.textContent = `${(cat * 100).toFixed(1)}%`;
    dogPct.textContent = `${(dog * 100).toFixed(1)}%`;
    nonePct.textContent = `${(neither * 100).toFixed(1)}%`;
    resultEl.classList.remove("hidden");
  }

  async function predict() {
    if (!session || !gateModel || !selectedFile) return;
    predictBtn.disabled = true;
    setStatus("Checking if this looks like a cat or dog…");
    try {
      // Wait for image decode
      if (!preview.complete) {
        await new Promise((resolve, reject) => {
          preview.onload = resolve;
          preview.onerror = reject;
        });
      }

      const gate = await detectPetSignal(preview);

      if (!gate.isPet) {
        // Not a pet → Neither
        const neither = Math.max(0.75, 1 - gate.petScore);
        renderScores(0.05, 0.05, neither, "Neither — not a cat or dog");
        setStatus("Done — image cleared from this device.");
        return;
      }

      setStatus("Pet detected — classifying Cat vs Dog…");
      const input = preprocessToTensor();
      const feeds = { [session.inputNames[0]]: input };
      const out = await session.run(feeds);
      const probs = Array.from(out[session.outputNames[0]].data);
      let cat = probs[0];
      let dog = probs[1];
      const confidence = Math.max(cat, dog);

      if (confidence < MIN_PET_CONFIDENCE) {
        const neither = 1 - confidence;
        renderScores(cat * 0.5, dog * 0.5, neither, "Neither — unsure if cat or dog");
      } else {
        const predIdx = cat >= dog ? 0 : 1;
        // Keep a small Neither slice so the third bar is meaningful
        const neither = Math.max(0.02, 1 - confidence);
        const scale = (1 - neither) / (cat + dog);
        cat *= scale;
        dog *= scale;
        renderScores(
          cat,
          dog,
          neither,
          `It’s a ${CLASS_NAMES[predIdx]}  ·  ${(confidence * 100).toFixed(1)}%`
        );
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
