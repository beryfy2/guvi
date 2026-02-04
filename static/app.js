const modelSelect = document.getElementById("modelSelect");
const modelName = document.getElementById("modelName");
const modelDesc = document.getElementById("modelDesc");
const modelInput = document.getElementById("modelInput");
const textField = document.getElementById("textField");
const fileField = document.getElementById("fileField");
const fileInput = document.getElementById("fileInput");
const textInput = document.getElementById("textInput");

const byId = new Map(MODELS.map((model) => [model.id, model]));

function updateCard(modelId) {
  const model = byId.get(modelId) || MODELS[0];
  modelName.textContent = model.name;
  modelDesc.textContent = model.description;
  modelInput.textContent = `Input: ${model.input}`;

  const isText = model.input === "text";
  if (textField) {
    textField.style.display = isText ? "flex" : "none";
  }
  if (fileField) {
    fileField.style.display = isText ? "none" : "flex";
  }

  if (fileInput) {
    let accept = "";
    if (model.input === "image") {
      accept = "image/*";
    } else if (model.input === "audio") {
      accept = "audio/*";
    } else if (model.input === "tabular") {
      accept = ".csv";
    }
    fileInput.accept = accept;
  }

  if (textInput) {
    if (model.input === "text") {
      textInput.placeholder = "Paste URL, email content, or transaction features...";
    } else {
      textInput.placeholder = "Text input disabled for this model.";
    }
  }
}

updateCard(currentId);

modelSelect.addEventListener("change", (event) => {
  updateCard(event.target.value);
});
