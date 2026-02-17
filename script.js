let model;
let uploadedImage;

// Charger le modèle ESRGAN
async function loadModel() {
    model = await tf.loadGraphModel(
        "https://tfhub.dev/captain-pool/esrgan-tf2/1",
        { fromTFHub: true }
    );
    console.log("Modèle chargé !");
}

loadModel();

// Gestion upload image
document.getElementById("upload").addEventListener("change", function(event) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById("preview");
        img.src = e.target.result;
        uploadedImage = img;
    };
    reader.readAsDataURL(event.target.files[0]);
});

// Amélioration IA
async function enhanceImage() {
    if (!model || !uploadedImage) {
        alert("Modèle ou image non chargés !");
        return;
    }

    const tensor = tf.browser.fromPixels(uploadedImage)
        .toFloat()
        .div(255.0)
        .expandDims();

    const output = await model.predict(tensor);
    const enhanced = output.squeeze();

    const canvas = document.getElementById("outputCanvas");
    await tf.browser.toPixels(enhanced, canvas);

    alert("Image améliorée !");
}
