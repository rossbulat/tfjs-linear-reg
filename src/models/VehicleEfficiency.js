import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

export class VehicleEfficiency {

  tfvis = true;
  dataUrl = 'https://storage.googleapis.com/tfjs-tutorials/carsData.json';
  rawData;
  model = '';
  tensorData = {
    inputs: [],
    labels: []
  };
  batchSize;
  epochs;

  constructor (config) {
    if (config === undefined)
      return;

    const { batchSize, epochs, tfvis } = config;

    // model config
    batchSize !== undefined ? this.batchSize = batchSize : this.batchSize = 32;
    epochs !== undefined ? this.epochs = epochs : this.epochs = 50;

    // show tfvis plots
    tfvis !== undefined ? this.tfvis = true : this.tfvis = false;
  }

  async getData () {

    const carsDataReq = await fetch(this.dataUrl);
    const carsData = await carsDataReq.json();

    // return only the values we need for the data: mpg and horsepower
    // filter to only include items where mpg and horsepower exist
    const cleaned = carsData.map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
      .filter(car => (car.mpg != null && car.horsepower != null));

    this.rawData = cleaned;

    // display tfvis graph if turned on
    if (this.tfvis) {

      const values = this.rawData.map(d => ({
        x: d.horsepower,
        y: d.mpg,
      }));

      tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values },
        {
          xLabel: 'Horsepower',
          yLabel: 'MPG',
          height: 300
        }
      );
    }

    return true;
  }

  // Create a sequential model
  createModel () {

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    model.add(tf.layers.dense({ units: 1, useBias: true }));
    this.model = model;

    //display model summary if tfvis is on
    if (this.tfvis) {
      tfvis.show.modelSummary({ name: 'Model Summary' }, this.model);
    }
  }


  dataToTensors () {

    const tensorData = tf.tidy(() => {

      // shuffle raw data
      tf.util.shuffle(this.rawData);

      // convert data to Tensors
      const inputs = this.rawData.map(d => d.horsepower)
      const labels = this.rawData.map(d => d.mpg);

      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      // normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      };
    });

    this.tensorData = tensorData;
  }


  async train () {

    const { inputs, labels } = this.tensorData;

    // prepare the model for training.  
    this.model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });

    // include tfvis callback if turned on
    let callbacks = null;
    if (this.tfvis) {
      callbacks = tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    }

    // train model
    return await this.model.fit(inputs, labels, {
      batchSize: this.batchSize,
      epochs: this.epochs,
      shuffle: true,
      callbacks: callbacks
    });
  }


  predict (inputData) {

    const {
      inputMax,
      inputMin,
      labelMin,
      labelMax } = this.tensorData;

    const [xs, preds] = tf.tidy(() => {

      // feed inputData inputs to make the predictions
      const preds = this.model.predict(inputData.reshape([100, 1]));

      // un-normalize the data by doing the inverse of the min-max scaling 
      const unNormXs = inputData
        .mul(inputMax.sub(inputMin))
        .add(inputMin);

      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);

      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    if (this.tfvis) {

      const predictedPoints =
        Array
          .from(xs)
          .map((val, i) => {
            return { x: val, y: preds[i] }
          });

      const originalPoints =
        this.rawData
          .map(d => ({
            x: d.horsepower, y: d.mpg,
          }));

      tfvis.render.scatterplot(
        { name: 'Model Predictions vs Original Data' },
        { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
        {
          xLabel: 'Horsepower',
          yLabel: 'MPG',
          height: 300
        }
      );
    }

    return {
      inputs: xs,
      preds: preds
    }
  }

  async init () {
    await this.getData(); // get and clean data
    this.createModel(); // Create the model
    this.dataToTensors(); // format tensors
    await this.train(); // Train the model
  }
}

export default VehicleEfficiency;
