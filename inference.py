"""Run `python inference.py`, then visit http://127.0.0.1:8000."""

import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
import yaml
from ultralytics.utils.nms import non_max_suppression


class Yolov9Inference:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        # This project uses a trusted full-model checkpoint. Unpickling a
        # checkpoint from an untrusted source can execute arbitrary code.
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = checkpoint['model'].to(device).float().eval()
        with Path(__file__).with_name('data.yaml').open(encoding='utf-8') as stream:
            names = yaml.safe_load(stream)['names']
        self.class_names = (dict(enumerate(names)) if isinstance(names, list)
                            else {int(key): value for key, value in names.items()})
        palette = ['#e53935', '#1e88e5', '#8e24aa', '#00a67d', '#f58b00', '#d63384']
        self.class_colors = {key: palette[index % len(palette)]
                             for index, key in enumerate(self.class_names)}

    def infer(self, image_path, img_size=640):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError('Could not read the uploaded image.')
        self.original_shape = image.shape[:2]
        self.img_size = img_size
        image = cv2.resize(image, (img_size, img_size))
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(tensor)

    def postprocess(self, predictions, conf_thres=0.25, iou_thres=0.45):
        detections = non_max_suppression(predictions, conf_thres, iou_thres)[0]
        detections = detections.cpu().numpy()
        height, width = self.original_shape
        detections[:, [0, 2]] *= width / self.img_size
        detections[:, [1, 3]] *= height / self.img_size
        return detections

    def draw_boxes(self, image_path, predictions, output_path):
        image = cv2.imread(image_path)
        for x1, y1, x2, y2, confidence, class_id in predictions:
            class_id = int(class_id)
            color_hex = self.class_colors.get(class_id, '#00ff00').lstrip('#')
            color = tuple(int(color_hex[index:index + 2], 16) for index in (4, 2, 0))
            label = f'{self.class_names.get(class_id, str(class_id))} {confidence:.2f}'
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, label, (max(0, int(x1)), max(18, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if not cv2.imwrite(output_path, image):
            raise OSError('Could not save the annotated image.')


MODEL_PATH = Path(__file__).resolve().parent / 'best.pt'
MAX_UPLOAD = 20 * 1024 * 1024

FRONTEND_DIR = Path(__file__).resolve().parent


def process_image(image_bytes, detector):
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not read this image. Please upload a JPG, PNG or WebP image.')
    with TemporaryDirectory(prefix='acne_inference_') as directory:
        source = Path(directory) / 'input.png'
        output = Path(directory) / 'output.jpg'
        if not cv2.imwrite(str(source), image):
            raise OSError('Could not prepare the uploaded image.')
        predictions = detector.postprocess(detector.infer(str(source)))
        detector.draw_boxes(str(source), predictions, str(output))
        return {'image': base64.b64encode(output.read_bytes()).decode('ascii'),
                'count': len(predictions),
                'classes': [{'name': name, 'color': detector.class_colors[class_id],
                             'count': int(np.count_nonzero(predictions[:, 5] == class_id))}
                            for class_id, name in detector.class_names.items()]}


class InferenceHandler(BaseHTTPRequestHandler):
    detector = None

    def respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, status, data):
        self.respond(status, json.dumps(data).encode('utf-8'), 'application/json')

    def do_GET(self):
        routes = {
            '/': ('index.html', 'text/html; charset=utf-8'),
            '/styles.css': ('styles.css', 'text/css; charset=utf-8'),
            '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
        }
        route = routes.get(self.path.split('?', 1)[0])
        if route is None:
            self.send_error(404)
            return
        filename, content_type = route
        try:
            body = (FRONTEND_DIR / filename).read_bytes()
        except FileNotFoundError:
            self.send_error(404, 'Frontend file not found')
            return
        self.respond(200, body, content_type)

    def do_POST(self):
        if self.path != '/predict':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.json_response(400, {'error': 'Invalid upload size.'})
            return
        if not 0 < length <= MAX_UPLOAD:
            self.json_response(413, {'error': 'Upload an image between 1 byte and 20 MB.'})
            return
        try:
            result = process_image(self.rfile.read(length), self.detector)
        except ValueError as error:
            self.json_response(400, {'error': str(error)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.json_response(500, {'error': 'Image processing failed. See the terminal for details.'})
        else:
            self.json_response(200, result)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {MODEL_PATH}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading model on {device}…')
    InferenceHandler.detector = Yolov9Inference(str(MODEL_PATH), device=device)
    with HTTPServer(('127.0.0.1', 8000), InferenceHandler) as server:
        print('Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nServer stopped.')


if __name__ == '__main__':
    main()
