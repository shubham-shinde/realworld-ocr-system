// Function to start video capturing and display it on canvas
function startVideoCapture() {
	const videoContainer = document.getElementById('videoContainer');
	const canvasContainer = document.getElementById('canvasContainer');
	const captureBtn = document.getElementById('capture-btn')
	const resetBtn = document.getElementById('reset-btn')
	captureBtn.style.display = 'block'
	videoContainer.style.display = 'flex'
	resetBtn.style.display = 'None'
	canvasContainer.style.display = 'None'

	navigator.mediaDevices.getUserMedia({video: {facingMode: 'environment'}, audio: false})
		.then((stream) => {
			const videoElement = document.getElementById('video');
			videoElement.srcObject = stream;
		})
		.catch((error) => {
			console.error('Error accessing camera:', error);
		});
	const videoElement = document.getElementById('video');
	const canvasElement = document.getElementById('canvas');
	const context = canvasElement.getContext('2d');
	// function drawImage(video) {
	// 	context.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
	// }
	// canvasInterval = window.setInterval(() => {
	// 	drawImage(videoElement);
	// }, 1000 / 1000);
}

// Function to stop video capturing and display captured image on canvas
async function stopVideoCapture() {
	const videoElement = document.getElementById('video');
	const canvasElement = document.getElementById('canvas');
	const context = canvasElement.getContext('2d');
	const videoContainer = document.getElementById('videoContainer');
	const canvasContainer = document.getElementById('canvasContainer');
	const captureBtn = document.getElementById('capture-btn')
	const resetBtn = document.getElementById('reset-btn')
	let height = videoElement.offsetHeight
	let width = videoElement.offsetWidth
	// Pause video playback
	videoElement.pause();

	// Draw current video frame on the canvas
	canvasElement.height = height
	canvasElement.width = width

	canvasElement.width = videoElement.videoWidth;
	canvasElement.height = videoElement.videoHeight;
	context.drawImage(videoElement, 0, 0);

	captureBtn.style.display = 'None'
	videoContainer.style.display = 'None'
	resetBtn.style.display = 'block'
	canvasContainer.style.display = 'flex'

	// Convert the captured image to TensorFlow.js tensor
	const imageTensor = tf.browser.fromPixels(canvasElement);
	const predictions = await yoloModel.detectObjects(imageTensor)
	displayBoundingBoxes(predictions, context);

	// Perform further processing with the imageTensor using your TensorFlow.js model
	// ...
	// Call your TensorFlow.js model functions or perform predictions here

	// Cleanup: Stop video stream and remove video element srcObject
	videoElement.srcObject.getTracks()[0].stop();
	videoElement.srcObject = null;
}

// Function to display the bounding boxes on the canvas
function displayBoundingBoxes(predictions, context) {
	// Draw bounding boxes
	for (const prediction of predictions) {
		const [x, y, width, height] = prediction;
		console.log(x, y, width, height)

		// Draw rectangle
		context.strokeStyle = 'red';
		context.lineWidth = 2;
		context.beginPath();
		context.rect(x, y, width - x, height - y);
		context.stroke();

		// Draw label
		context.fillStyle = 'red';
	}
}

// Add event listener to the capture button
const captureBtn = document.getElementById('capture-btn');
captureBtn.addEventListener('click', stopVideoCapture);
const resetBtn = document.getElementById('reset-btn');
resetBtn.addEventListener('click', startVideoCapture);


// Start video capturing on page load
window.addEventListener('load', startVideoCapture);
