const $ = (selector) => document.querySelector(selector);
const form = $('#form'), input = $('#file'), dropzone = $('#dropzone');
const button = $('#process'), status = $('#status'), results = $('#results');
let selected = null, preview = null, busy = false;

function message(text, type = '') {
  status.textContent = text;
  status.className = `status ${type}`;
}

function clearResult() {
  results.hidden = true;
  $('#empty').hidden = false;
  $('#result-tag').textContent = 'Ready when you are';
  $('#download').removeAttribute('href');
}

function chooseFile(file) {
  if (busy) return;
  clearResult();
  selected = null;
  button.disabled = true;
  $('#selection').hidden = true;
  if (preview) { URL.revokeObjectURL(preview); preview = null; }
  if (!file) { message('Choose an image to get started.'); return; }
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    message('Please choose a JPG, PNG or WebP image.', 'error'); return;
  }
  if (!file.size || file.size > 20 * 1024 * 1024) {
    message('Choose a non-empty image smaller than 20 MB.', 'error'); return;
  }
  selected = file;
  preview = URL.createObjectURL(file);
  $('#thumbnail').src = preview;
  $('#filename').textContent = file.name;
  $('#filesize').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  $('#selection').hidden = false;
  button.disabled = false;
  message('Image ready. Select Analyze image to continue.');
}

input.addEventListener('change', () => chooseFile(input.files[0]));
$('#remove').addEventListener('click', () => { input.value = ''; chooseFile(null); });
['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => {
  event.preventDefault(); if (!busy) dropzone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => {
  event.preventDefault(); dropzone.classList.remove('dragging');
}));
dropzone.addEventListener('drop', event => {
  if (busy) return;
  input.value = '';
  chooseFile(event.dataTransfer.files[0]);
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!selected || busy) return;
  busy = true;
  button.disabled = input.disabled = $('#remove').disabled = true;
  $('#button-label').textContent = 'Analyzing…';
  $('#analysis').setAttribute('aria-busy', 'true');
  clearResult();
  $('#result-tag').textContent = 'Analyzing image';
  message('Finding and counting detections…', 'busy');
  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: selected
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Image analysis failed. Please try again.');
    $('#original').src = preview;
    const image = `data:image/jpeg;base64,${data.image}`;
    $('#output').src = image;
    $('#download').href = image;
    $('#total').textContent = data.count;
    const counts = $('#class-counts');
    counts.replaceChildren();
    for (const entry of data.classes) {
      const badge = document.createElement('div'); badge.className = 'class-count';
      const swatch = document.createElement('span'); swatch.className = 'swatch';
      swatch.style.backgroundColor = entry.color;
      const name = document.createElement('span'); name.className = 'class-name'; name.textContent = entry.name;
      const count = document.createElement('strong'); count.textContent = entry.count;
      badge.append(swatch, name, count); counts.append(badge);
    }
    $('#result-note').textContent = data.count === 0
      ? 'No detections found in this image. You can still download the result.'
      : '';
    $('#result-note').hidden = data.count !== 0;
    $('#empty').hidden = true;
    results.hidden = false;
    $('#result-tag').textContent = 'Analysis complete';
    message('All done. Your result is ready to download.');
  } catch (error) {
    $('#result-tag').textContent = 'Try again';
    message(error.message || 'Could not connect. Check that the server is running.', 'error');
  } finally {
    busy = false;
    button.disabled = input.disabled = $('#remove').disabled = false;
    $('#button-label').textContent = 'Analyze image';
    $('#analysis').setAttribute('aria-busy', 'false');
  }
});
