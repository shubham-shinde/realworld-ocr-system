// Define the YOLO model class
const model_shape = [640, 640]
const conf = 0.1
class YOLOModel {
	constructor() {
		// Define the YOLO model architecture
		this.model = this.createModel();
	}

	createModel() {
		// Define the YOLO model architecture using TensorFlow.js layers
		const model = tf.sequential();

		// Example architecture:
		// Add your own layers and configuration based on your YOLO implementation
		// model.add(tf.layers.someLayer());

		return model;
	}

	async loadModel() {
		// Load the YOLO model weights from a file or URL
		const modelPath = 'last_web_model/model.json';
		this.model = await tf.loadGraphModel(modelPath);
	}

	async detectObjects(imageData) {
		// Preprocess the image data for YOLO model input
		const [inputTensor, pad, resize] = preprocessImageData(imageData);
		// Run the input tensor through the YOLO model
		console.log(inputTensor.shape, imageData.shape)
		const outputTensor = await this.model.predict(inputTensor);

		// Process the YOLO model output to obtain bounding box predictions
		const predictions = await postprocessOutputTensor(outputTensor, pad, resize);

		return predictions;
	}
}

function preprocessImageData(tensorData) {
	// Convert image data to tensor format
	// const tensorData = tf.browser.fromPixels(imageData);

	// Resize the image to 640x640
	let img_shape = tensorData.shape
	let resize = Math.min(model_shape[0] / img_shape[0], model_shape[1] / img_shape[1])
	let resized_shape = [Math.ceil(img_shape[0] * resize), Math.ceil(img_shape[1] * resize)]
	const resizedData = tf.image.resizeBilinear(
		tensorData,
		resized_shape
	);
	let pad = [
		[Math.round(((model_shape[0] - resized_shape[0]) / 2) + 0.1), Math.round(((model_shape[0] - resized_shape[0]) / 2) + 0.1)],
		[Math.round(((model_shape[1] - resized_shape[1]) / 2) + 0.1), Math.round(((model_shape[1] - resized_shape[1]) / 2) + 0.1)],
	]

	// Pad the image to make it 640x640 with a value of 114
	const paddedData = tf.pad(resizedData, [...pad, [0, 0]], 114)

	// Normalize pixel values to range from 0 to 1
	const normalizedData = paddedData.div(255.0);

	// Add a batch dimension to the tensor
	const batchedData = normalizedData.expandDims(0);

	return [batchedData, pad, resize];
}

// Function to process YOLO model output to obtain bounding box predictions
async function postprocessOutputTensor(outputTensor, pad, resize) {
	let model_conf = outputTensor.slice([0, 4], [1, 1])
	let model_boxes = outputTensor.slice([0, 0], [1, 4])
		.sub(tf.tensor([[[pad[1][0]], [pad[0][0]], [0], [0]]]))
		.div(tf.tensor([[[resize], [resize], [resize], [resize]]]))
	let sub = tf.stack([
		model_boxes.slice([0, 2], [1, 1]).squeeze().div(2),
		model_boxes.slice([0, 3], [1, 1]).squeeze().div(2),
		tf.zeros([model_boxes.shape[2]]),
		tf.zeros([model_boxes.shape[2]])
	]).expandDims(0)
	let div = tf.tensor([[[1], [1], [2], [2]]])
	let add = tf.stack([
		tf.zeros([model_boxes.shape[2]]),
		tf.zeros([model_boxes.shape[2]]),
		model_boxes.slice([0, 0], [1, 1]).squeeze(),
		model_boxes.slice([0, 1], [1, 1]).squeeze()
	]).expandDims(0)
	let model_boxes_xyxy = model_boxes.sub(sub).div(div).add(add)

	let all_boxes = model_boxes_xyxy.squeeze().transpose()
	let box_indexes = await tf.image.nonMaxSuppressionAsync(
		all_boxes,
		model_conf.squeeze(),
		40, // no of boxes out
		0.1, // iou threshold
		conf //score threshold
	)
	let boxes = all_boxes.gather(box_indexes)
	let predictions = await boxes.array()
	return predictions
}

const yoloModel = new YOLOModel();
yoloModel.loadModel()

