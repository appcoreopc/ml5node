
import * as tf from '@tensorflow/tfjs';
import * as data from './data';
import * as loader from "./loader";

let model;

async function trainModel(xTrain : tf.Tensor<tf.Rank>, yTrain: tf.Tensor<tf.Rank>, xTest:tf.Tensor<tf.Rank>, yTest: tf.Tensor<tf.Rank>) {
  
  console.log('Training model... Please wait.');

  const params = { learningRate: 0.555, epochs: 200 };

  xTrain.shape[1];

  // Define the topology of the model: two dense layers.
  const model = tf.sequential();
  
  // model.add(tf.layers.dense(
  //   { units: 10, activation: 'sigmoid', inputShape: xTrain.shape }));
  // model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  
  model.add(tf.layers.dense(
    {units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]!]}));
  
    model.add(tf.layers.dense({units: 3, activation: 'softmax'}));

  model.summary();

  const optimizer = tf.train.adam(params.learningRate);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const trainLogs = [];
  //const lossContainer = document.getElementById('lossCanvas');
  //const accContainer = document.getElementById('accuracyCanvas');
  //const beginMs = performance.now();
  // Call `model.fit` to train the model.
  const history = await model.fit(xTrain, yTrain, {
    epochs: params.epochs,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        //const secPerEpoch =
        //  (performance.now() - beginMs) / (1000 * (epoch + 1));
        console.log(`Training model... Approximating time per epoch`)
        trainLogs.push(logs);
        //tfvis.show.history(lossContainer, trainLogs, ['loss', 'val_loss'])
        //tfvis.show.history(accContainer, trainLogs, ['acc', 'val_acc'])
        calculateAndDrawConfusionMatrix(model, xTest, yTest);
      },
    }
  });
  
  //const secPerEpoch = (performance.now() - beginMs) / (1000 * params.epochs);
  console.log(
    `Model training complete`);
  return model;
}

async function calculateAndDrawConfusionMatrix(model:tf.Sequential, xTest:tf.Tensor<tf.Rank>, yTest:tf.Tensor<tf.Rank>) {
  const [preds, labels] = tf.tidy(() => {
    const preds = model.predict(xTest); //argMax(-1);
    const labels = yTest.argMax(-1);
    return [preds, labels];
  });

  //const confMatrixData = await tfvis.metrics.confusionMatrix(labels, preds);
  //const container = document.getElementById('confusion-matrix');
  //tfvis.render.confusionMatrix(
  //    container,
  //    {values: confMatrixData, labels: data.IRIS_CLASSES},
  //    {shadeDiagonal: true},
  //);

  tf.dispose([preds, labels]);
}

async function predictOnManualInput(model: tf.LayersModel | undefined) {
  if (model == null) {
    console.log('ERROR: Please load or train model first.');
    return;
  }

  // Use a `tf.tidy` scope to make sure that WebGL memory allocated for the
  // `predict` call is released at the end.
  tf.tidy(() => {
    // Prepare input data as a 2D `tf.Tensor`.
    const inputData = [2.0, 3.0, 4.0, 2.1]
    const input = tf.tensor2d([inputData], [1, 4]);

    // Call `model.predict` to get the prediction output as probabilities for
    // the Iris flower categories.

    const predictOut = model.predict(input) as tf.Tensor<tf.Rank>;
    const logits = Array.from(predictOut.dataSync());
    const winner = data.IRIS_CLASSES[predictOut.argMax(-1).dataSync()[0]];
    //ui.setManualInputWinnerMessage(winner);
    //ui.renderLogitsForManualInput(logits);
  });
}

async function evaluateModelOnTestData(model:tf.Sequential, xTest:tf.Tensor, yTest:tf.Tensor) {
  //  ui.clearEvaluateTable();

  tf.tidy(() => {
    const xData = xTest.dataSync();
    const yTrue = yTest.argMax(-1).dataSync();
    const predictOut = model.predict(xTest) as tf.Tensor<tf.Rank>;
    const yPred = predictOut.argMax(-1);
    //ui.renderEvaluateTable(
    //    xData, yTrue, yPred.dataSync(), predictOut.dataSync());
    calculateAndDrawConfusionMatrix(model, xTest, yTest);
  });

   predictOnManualInput(model);
}

const HOSTED_MODEL_JSON_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json';

async function iris() {
   
  const [xTrain, yTrain, xTest, yTest] = data.getIrisData(0.15);

  //const localLoadButton = document.getElementById('load-local');
  //const localSaveButton = document.getElementById('save-local');
  //const localRemoveButton = document.getElementById('remove-local');

  model = await trainModel(xTrain, yTrain, xTest, yTest);
  await evaluateModelOnTestData(model, xTest, yTest);
  //localSaveButton.disabled = false;

  if (await loader.urlExists(HOSTED_MODEL_JSON_URL)) {
    console.log('Model available: ' + HOSTED_MODEL_JSON_URL);
    //const button = document.getElementById('load-pretrained-remote');
    //button.addEventListener('click', async () => {
    //  ui.clearEvaluateTable();
      model = await loader.loadHostedPretrainedModel(HOSTED_MODEL_JSON_URL);
      await predictOnManualInput(model);
      //localSaveButton.disabled = false;
    }
}

iris();
