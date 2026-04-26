const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("previewSection");
const img = document.getElementById("previewImage");

uploadBox.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", function () {
  const file = this.files[0];

  if (file) {
    img.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
    uploadBox.style.display = "none";

    // Send to backend
    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
      method: "POST",
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      showResult(data);
    });
  }
});

function showResult(data) {
  const resultHTML = `
    <div class="result-card">
      <h2>${data.class.toUpperCase()}</h2>
      <p>Confidence: ${data.confidence}%</p>
    </div>
  `;

  preview.insertAdjacentHTML("beforeend", resultHTML);
}

function reset() {
  location.reload();
}