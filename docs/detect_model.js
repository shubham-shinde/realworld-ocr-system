const detection_model_shape = [32, 128]
const detection_model_text = '. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
class DetectionModel {
	constructor() {
		// Define the Detection model architecture
		this.model = this.createModel();
	}

	createModel() {
		// Define the Detection model architecture using TensorFlow.js layers
		const model = tf.sequential();

		// Example architecture:
		// Add your own layers and configuration based on your Detection implementation
		// model.add(tf.layers.someLayer());

		return model;
	}

	async loadModel() {
		// Load the Detection model weights from a file or URL
		const modelPath = 'web_model/model.json';
		this.model = await tf.loadGraphModel(modelPath);
	}

	async detectText(images) {
		// Preprocess the image data for Detection model input
		let modelInput = await Promise.all(images.map(this.preprocessing))
		let modelOutput = await Promise.all(modelInput.map(e => this.model.executeAsync(e)))
		modelOutput = await tf.stack(modelOutput).squeeze(2).argMax(-1).array()
		let texts = []
		for (let i of modelOutput) {
			let full_text = []
			let text = []
			for (let j_i in i) {
				j_i = Number(j_i)
				let j = detection_model_text[i[j_i]]
				full_text.push(j)
				if (j_i > 0 && detection_model_text[i[j_i - 1]] === j) continue;
				else if (j == '.') continue;
				else text.push(j)
			}
			text = text.join('')
			texts.push(text)
			console.log(full_text.join(''))
		}
		return texts
	}

	async preprocessing(tensorData) {
		let img_shape = tensorData.shape
		let resize = Math.min(detection_model_shape[0] / img_shape[0], detection_model_shape[1] / img_shape[1])
		let resized_shape = [Math.ceil(img_shape[0] * resize), Math.ceil(img_shape[1] * resize)]
		const resizedData = tf.image.resizeBilinear(
			tensorData,
			resized_shape
		);
		let pad = [
			[Math.round(((detection_model_shape[0] - resized_shape[0]) / 2) + 0.1), Math.round(((detection_model_shape[0] - resized_shape[0]) / 2) - 0.1)],
			[Math.round(((detection_model_shape[1] - resized_shape[1]) / 2) + 0.1), Math.round(((detection_model_shape[1] - resized_shape[1]) / 2) - 0.1)],
		]

		// Pad the image to make it 640x640 with a value of 114
		const paddedData = tf.pad(resizedData, [...pad, [0, 0]], 114)
		const grayScale = paddedData.mean(-1).expandDims(-1)

		// Normalize pixel values to range from 0 to 1
		const normalizedData = grayScale.div(255.0);

		// Add a batch dimension to the tensor
		const batchedData = normalizedData.expandDims(0)
		return batchedData

	}

}


const detectionModel = new DetectionModel();
// detectionModel.loadModel()
