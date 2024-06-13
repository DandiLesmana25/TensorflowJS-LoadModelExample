/*
inference: merujuk pada proses menggunakan(load) model yang telah dilatih untuk membuat prediksi.
dan menghasilkan output dari data yang belum pernah dilihat dalam proses training atau pelatihan model.
*/ 
const tfjs = require('@tensorflow/tfjs-node');

const loadModel = () => {
    const modelURL = "file://models/model.json" //path model

    // Load a model composed of Layer objects, including its topology and optionally weights. 
    return tfjs.loadLayersModel(modelURL);

}

//  decoding and preprocessing data
function predict(model, imageBuffer) {
    const tensor = tfjs.node.decodeJpeg(imageBuffer).resizeNearestNeighbor([150, 150]).expandDims()
    .toFloat();

    return model.predict(tensor).data();
}

module.exports = {loadModel, predict};