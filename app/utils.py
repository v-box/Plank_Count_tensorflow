import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils


def load_image_into_numpy_array(path):
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def load_model():
    category_index = {
    1: {'id': 1, 'name': 'plank'},
    }
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load('saved_model')
    return detect_fn,category_index

def infer_image(image,detect_fn,category_index):
    
    #for imls in range(0,len(image_list)):
      #image_path = image_list[imls]
      #fil = image_path.split('.')[0].split('/')[-1]
      
    image_np = load_image_into_numpy_array(image)
    input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    scores=detections['detection_scores'][0].numpy()
    num_detections = int(detections['num_detections'][0].numpy())
    count = 0
    for i in range(num_detections):
      if scores[i]>=.1:
        count = count + 1
    print(count)
  
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          detections['detection_classes'][0].numpy().astype(np.int32),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=2000,
          min_score_thresh=.1,
          agnostic_mode=False)
    dat1 = Image.fromarray(image_np_with_detections)
    ImageDraw.Draw(dat1).text((100, 10),f' Total {str(count)} Planks ',(255,255,255))
  
    #dat2 = dat1.save(f'boxed_{fil}.png')
    
    return count,dat1
