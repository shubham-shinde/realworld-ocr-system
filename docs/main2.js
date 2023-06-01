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
	// function drawImage(video) {
	// 	context.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
	// }
	// canvasInterval = window.setInterval(() => {
	// 	drawImage(videoElement);
	// }, 1000 / 1000);
}

// Function to stop video capturing and display captured image on canvas
function stopVideoCapture(drawImage) {
	const videoElement = document.getElementById('video');
	const videoContainer = document.getElementById('videoContainer');
	const canvasContainer = document.getElementById('canvasContainer');
	const captureBtn = document.getElementById('capture-btn')
	const resetBtn = document.getElementById('reset-btn')
	// Pause video playback
	videoElement.pause();

	drawImage(videoElement);

	captureBtn.style.display = 'None'
	videoContainer.style.display = 'None'
	resetBtn.style.display = 'block'
	canvasContainer.style.display = 'flex'

	// Perform further processing with the imageTensor using your TensorFlow.js model
	// ...
	// Call your TensorFlow.js model functions or perform predictions here

	// Cleanup: Stop video stream and remove video element srcObject
	if (videoElement.srcObject) {
		videoElement.srcObject.getTracks()[0].stop();
		videoElement.srcObject = null;
	}
}

async function textDetection() {
	const canvasElement = document.getElementById('canvas');
	let imageTensor = tf.browser.fromPixels(canvasElement);

	// Convert the captured image to TensorFlow.js tensor
	const predictions = await yoloModel.detectObjects(imageTensor)
	let textImages = []



	for (let box of predictions) {
		let [x1, y1, x2, y2] = box;
		x1 = Math.floor(x1)
		x2 = Math.ceil(x2)
		y1 = Math.floor(y1)
		y2 = Math.ceil(y2)
		let [ih, iw, _] = imageTensor.shape
		let [x, y] = [Math.max(0, x1), Math.max(0, y1)]
		let [w, h] = [Math.min(x2 - x, iw - x), Math.min(y2 - y, ih - y)]
		const img = imageTensor.slice([y, x], [h, w])
		textImages.push(img)
	}
	let texts = await detectionModel.detectText(textImages)
	displayBoundingBoxes(predictions, texts);
}

async function loadCaptureImage() {
	const canvasElement = document.getElementById('canvas');
	const context = canvasElement.getContext('2d');

	stopVideoCapture((videoElement) => {
		canvasElement.width = videoElement.videoWidth;
		canvasElement.height = videoElement.videoHeight;
		context.drawImage(videoElement, 0, 0);
		return canvasElement
	})

	await textDetection()

	// Convert the captured image to TensorFlow.js tensor
}

async function loadExampleImage(btn_id) {
	const canvasElement = document.getElementById('canvas');
	const context = canvasElement.getContext('2d');

	stopVideoCapture((_videoElement) => {
		const image = document.getElementById("img" + btn_id)
		canvasElement.width = image.width;
		canvasElement.height = image.height;
		context.drawImage(image, 0, 0);
	})

	await textDetection()
}

// Function to display the bounding boxes on the canvas
function displayBoundingBoxes(predictions, texts) {
	const canvasElement = document.getElementById('canvas');
	const context = canvasElement.getContext('2d');
	context.font = "8px Arial";
	context.fillStyle = 'green';
	context.strokeStyle = 'red';
	// Draw rectangle
	// Draw bounding boxes
	for (let i in predictions) {
		i = Number(i)
		prediction = predictions[i]
		let text = texts[i]
		const [x1, y1, x2, y2] = prediction;
		console.log(x1, y1, x2, y2)

		// Draw rectangle
		context.lineWidth = 2;
		context.beginPath();
		context.rect(x1, y1, x2 - x1, y2 - y1);
		// context.fillText(text, x1 - 2, y1 - 2);
		drawTextBG(context, text, '8px Arial', x1, y1)
		context.stroke();

		// Draw label
	}
}

function drawTextBG(ctx, txt, font, x, y) {
	ctx.font = font;
	ctx.textBaseline = 'bottom';
	ctx.fillStyle = 'red';
	var width = ctx.measureText(txt).width;
	ctx.fillRect(x, y - parseInt(font, 10), width, parseInt(font, 10));
	ctx.fillStyle = '#000';
	ctx.fillText(txt, x, y);
}

// Add event listener to the capture button
const captureBtn = document.getElementById('capture-btn');
captureBtn.addEventListener('click', loadCaptureImage);
const resetBtn = document.getElementById('reset-btn');
resetBtn.addEventListener('click', startVideoCapture);
const imgBtns = document.getElementsByClassName('thumbnail')
for (let imgBtn of imgBtns) imgBtn.addEventListener('click', (e) => loadExampleImage(e.target.id))


// Start video capturing on page load
window.addEventListener('load', startVideoCapture);
yoloModel.loadModel()
detectionModel.loadModel()
// testing
// window.addEventListener('load', async () => {
// 	await yoloModel.loadModel()
// 	await detectionModel.loadModel()
// 	await loadExampleImage(2)
// });
