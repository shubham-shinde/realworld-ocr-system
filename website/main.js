// import {YOLOModel} from './yolo_model.js';

// Global variables
let videoStream;
let canvas;
let context;
// Create an instance of the YOLO model
// const yoloModel = new YOLOModel();
// yoloModel.loadModel()

// Function to start the camera and capture an image
async function captureImage() {
	try {
		const video = document.createElement('video');
		const constraints = {video: {facingMode: 'environment'}};

		// Prompt the user for camera access
		videoStream = await navigator.mediaDevices.getUserMedia(constraints);
		video.srcObject = videoStream;
		// await video.play();

		// Create a canvas element for preview
		canvas = document.getElementById('preview');
		context = canvas.getContext('2d');
		canvas.width = video.videoWidth;
		canvas.height = video.videoHeight;

		// Draw the video frame on the canvas
		context.drawImage(video, 0, 0, canvas.width, canvas.height);

		// Capture the image data from the canvas
		// const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

		// Perform object detection using YOLO model
		// const predictions = await performObjectDetection(imageData);

		// Display the bounding boxes
		// displayBoundingBoxes(predictions);

		// Clean up resources
		videoStream.getTracks().forEach(track => track.stop());
		video.srcObject = null;
	} catch (error) {
		console.error('Error capturing image:', error);
	}
}

// Function to perform object detection using the YOLO model
async function performObjectDetection(imageData) {
	// Convert image data to tensor format or perform any necessary preprocessing

	// Perform object detection using YOLO model (implement this according to your YOLO implementation)
	const predictions = yoloModel.detectObjects(imageData);

	// Perform any postprocessing if needed

	return predictions;
}

// Function to display the bounding boxes on the canvas
function displayBoundingBoxes(predictions) {
	// Clear the canvas
	context.clearRect(0, 0, canvas.width, canvas.height);

	// Draw bounding boxes
	for (const prediction of predictions) {
		const [x, y, width, height] = prediction.bbox;
		const label = `${prediction.class}: ${prediction.score.toFixed(2)}`;

		// Draw rectangle
		context.strokeStyle = 'red';
		context.lineWidth = 2;
		context.beginPath();
		context.rect(x, y, width, height);
		context.stroke();

		// Draw label
		context.fillStyle = 'red';
		context.fillText(label, x, y > 10 ? y - 5 : 10);
	}
}

window.captureImage = captureImage
