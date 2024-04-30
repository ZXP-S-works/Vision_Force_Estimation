from segment_anything import sam_model_registry, SamPredictor

#ONNX stuff
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic



checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)


image = cv2.imread('/vast/palmer/home.grace/yz2379/Data/0811_raw_data/val/rgb_0.png')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("dogs_embedding.npy", image_embedding)