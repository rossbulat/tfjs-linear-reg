import * as tf from '@tensorflow/tfjs';
import { VehicleEfficiency } from './models/VehicleEfficiency';

async function run () {

  // Initialise the model class
  const MyModel = new VehicleEfficiency();

  // init loads and cleans data
  // normalises and shuffles data
  // trains model ready for predictionsx
  await MyModel.init();

  // Generate predictions for a uniform range of numbers between 0 and 1;
  const testInputs = tf.linspace(0, 1, 100);

  // Make some predictions using the model and compare them to the original data
  MyModel.predict(testInputs);
}

document.addEventListener('DOMContentLoaded', run);
