let model;
// Función asincrónica para cargar el modelo desde un archivo JSON
async function loadModel() {
     // Utiliza TensorFlow.js para cargar el modelo desde el archivo 'model.json'
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/SaulDael/ClasificadorDeRopa/main/model.json');
    console.log('Modelo Cargado.....');
}

// Función para preprocesar la imagen antes de pasarla al modelo
function preprocessImage(image) {
    return tf.tidy(() => {

        const tensor = tf.browser.fromPixels(image);

        const grayscale = tensor.mean(2);

        const expanded = grayscale.expandDims(2);

        const resized = tf.image.resizeBilinear(expanded, [28, 28]);

        const normalized = resized.div(255.0);

        const squeezed = normalized.squeeze();

        const reshaped = squeezed.expandDims(0);

        return reshaped;
    });
}

// Nombres de las clases para las predicciones del modelo
const classNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"];

// Función principal para ejecutar el modelo cuando se selecciona una imagen
function runModel() {
    const imageInput = document.getElementById('imageInput');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const classificationParagraph = document.getElementById('classification');

    const file = imageInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        const image = new Image();
        image.onload = function () {
           
            context.drawImage(image, 0, 0, canvas.width, canvas.height);

            const preprocessedImage = preprocessImage(image);

            const output = model.predict(preprocessedImage);

            const predictions = output.arraySync()[0];
            const maxPredictionIndex = predictions.indexOf(Math.max(...predictions));

            const classificationResult = "Clasificación: " + classNames[maxPredictionIndex];

            classificationParagraph.textContent = classificationResult;

            output.print();

            preprocessedImage.dispose();
        };
        image.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Carga el modelo cuando se carga la página
loadModel();
