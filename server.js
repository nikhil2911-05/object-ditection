import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { pipeline, env, RawImage } from '@xenova/transformers';

// Disable local models to fetch from HuggingFace
env.allowLocalModels = false;

const app = express();
const port = 8000;

app.use(cors());
app.use(express.json());

// Set up Multer for handling file uploads in memory
const upload = multer({ storage: multer.memoryStorage() });

// In-memory store for history
const detectionHistory = [];
let historyIdCounter = 1;

let detector = null;

// Initialize the model
async function initModel() {
    try {
        console.log('Loading high-accuracy Object Detection model (DETR-ResNet-50)...');
        console.log('This may take a moment the first time as it downloads the model.');
        
        // We use Xenova/detr-resnet-50 for high accuracy object detection
        detector = await pipeline('object-detection', 'Xenova/detr-resnet-50');
        console.log('Model loaded successfully!');
    } catch (err) {
        console.error('Error loading model:', err);
    }
}

initModel();

app.post('/detect', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image provided' });
    }

    if (!detector) {
        return res.status(503).json({ error: 'Model is still loading, please try again in a few seconds.' });
    }

    try {
        // Convert image buffer to a Blob and read it directly to avoid Node.js fetch data URI issues
        const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
        const image = await RawImage.fromBlob(blob);

        // Run object detection (threshold to ensure high accuracy)
        const output = await detector(image, { threshold: 0.5 });
        
        /* 
         * output format from transformers.js:
         * [{ score: 0.99, label: 'person', box: { xmin, ymin, xmax, ymax } }, ...]
         */
        
        const detected_objects = output.map(item => {
            const width = item.box.xmax - item.box.xmin;
            const height = item.box.ymax - item.box.ymin;
            return {
                label: item.label,
                confidence: item.score,
                box: [item.box.xmin, item.box.ymin, width, height]
            };
        });

        // Record history
        detectionHistory.unshift({
            id: historyIdCounter++,
            date: new Date().toISOString(),
            objectCount: detected_objects.length,
            imageUrl: null // In a production app, save to S3/disk and return URL
        });

        // Keep last 20
        if (detectionHistory.length > 20) {
            detectionHistory.pop();
        }

        res.json({ objects: detected_objects });
    } catch (error) {
        console.error('Error processing image:', error);
        res.status(500).json({ error: 'Failed to process image' });
    }
});

app.get('/detections', (req, res) => {
    res.json({ history: detectionHistory });
});

app.listen(port, () => {
    console.log(`Node.js Backend server listening at http://localhost:${port}`);
});
