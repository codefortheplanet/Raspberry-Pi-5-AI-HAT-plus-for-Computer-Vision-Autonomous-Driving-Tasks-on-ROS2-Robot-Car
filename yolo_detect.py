from math import frexp
from traceback import print_tb
from torch import imag
#from yolov5 import YOLOv5
from ultralytics import YOLO
import rclpy
import yolov5_ros2.fps as fps
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np
#from sdk import common

from yolov5_ros2.cv_tool import px2xy
import os
from interfaces.msg import ObjectInfo, ObjectsInfo

from std_srvs.srv import Trigger

# HAILO IMPORTS - Added for Hailo support
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                                ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Hailo platform not available, using CPU/CUDA inference")

# Get the ROS distribution version and set the shared directory for YoloV5 configuration files.
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')

# Create a ROS 2 Node class YoloV5Ros2.
class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")
        self.fps = fps.FPS()

        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu, options: cuda:0, hailo"))

        self.declare_parameter("model", "yolov5s", ParameterDescriptor(
            name="model", description="Default model selection: yolov5s"))

        self.declare_parameter("image_topic", "/ascamera/camera_publisher/rgb0/image", ParameterDescriptor(
            name="image_topic", description="Image topic, default: /ascamera/camera_publisher/rgb0/image"))

        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Whether to display detection results, default: False"))

        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Whether to publish detection result images, default: False"))

        self.create_service(Trigger, '/yolov5/start', self.start_srv_callback)
        self.create_service(Trigger, '/yolov5/stop', self.stop_srv_callback) 
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        # Load the model.
        device = self.get_parameter('device').value
        
        # HAILO SUPPORT - Check if Hailo device is requested
        self.use_hailo = (device == "hailo" and HAILO_AVAILABLE)
        
        if self.use_hailo:
            # Load Hailo model (.hef file)
            model_path = package_share_directory + "/config/" + self.get_parameter('model').value + ".hef"
            self.get_logger().info(f"Loading Hailo model: {model_path}")
            self.init_hailo(model_path)
        else:
            # Load PyTorch model (.pt file)
            model_path = package_share_directory + "/config/" + self.get_parameter('model').value + ".pt"
            self.get_logger().info(f"Loading PyTorch model: {model_path}")
            self.yolov5 = YOLO(model_path)
            self.yolov5.to(device)
        
        # Create publishers.
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()
        self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        # Create an image subscriber with the updated topic.
        image_topic = self.get_parameter('image_topic').value
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)

        # Image format conversion (using cv_bridge).
        self.bridge = CvBridge()

        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

    # HAILO INITIALIZATION - New method for Hailo setup
    def init_hailo(self, hef_path):
        """Initialize Hailo device and load model"""
        try:
            # Create VDevice
            params = VDevice.create_params()
            self.hailo_device = VDevice(params)
            
            # Load HEF
            self.hef = HEF(hef_path)
            
            # Configure network
            self.network_group = self.hailo_device.configure(self.hef)[0]
            self.network_group_params = self.network_group.create_params()
            
            # Get input/output info
            self.input_vstreams_params = InputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.UINT8)
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            
        # ADD THIS LINE - Store input name
        # Get input info from HEF (more reliable)
            input_vstream_infos = self.hef.get_input_vstream_infos()
        
        # Store input name from vstream info
            self.input_name = input_vstream_infos[0].name
            
            # Get input shape
            #input_info = self.hef.get_input_vstream_infos()[0]
            self.input_shape = input_vstream_infos[0].shape
            self.input_height = self.input_shape[0]
            self.input_width = self.input_shape[1]
            
            # COCO class names (same as original YOLOv5)
            self.class_names = [
                'addedLane', 'curveLeft', 'curveRight', 'dip', 'doNotEnter', 'doNotPass', 'intersection', 'keepRight', 'laneEnds', 'merge', 'noLeftTurn', 'noRightTurn', 'pedestrianCrossing', 'rampSpeedAdvisory20', 'rampSpeedAdvisory35', 'rampSpeedAdvisory40', 'rampSpeedAdvisory45', 'rampSpeedAdvisory50', 'rampSpeedAdvisoryUrdbl', 'rightLaneMustTurn', 'roundabout', 'school', 'schoolSpeedLimit25', 'signalAhead', 'slow', 'speedLimit15', 'speedLimit25', 'speedLimit30', 'speedLimit35', 'speedLimit40', 'speedLimit45', 'speedLimit50', 'speedLimit55', 'speedLimit65', 'speedLimitUrdbl', 'stop', 'stopAhead', 'thruMergeLeft', 'thruMergeRight', 'thruTrafficMergeLeft', 'truckSpeedLimit55', 'turnLeft', 'turnRight', 'yield', 'yieldAhead', 'zoneAhead25', 'zoneAhead45'
            ]
            
            self.get_logger().info(f'Hailo model loaded: input shape {self.input_shape}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Hailo: {e}')
            raise

    # HAILO INFERENCE - New method for Hailo inference
    def hailo_inference(self, image):
        
        """Run inference using Hailo accelerator"""
        import time
        
        self.get_logger().info('=== Starting Hailo inference ===')
        start_time = time.time()
        
        # Preprocess image
        self.get_logger().info('Preprocessing...')
        """Run inference using Hailo accelerator"""
        # Preprocess image
        resized = cv2.resize(image, (self.input_width, self.input_height))
        #rgb = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)  # Convert back to BGR for model
        rgb = resized.astype(np.uint8)
        #normalized = rgb.astype(np.float32) / 255.0
        #normalized = rgb.astype(np.float32)
        input_data = np.expand_dims(rgb, axis=0)
 
        self.get_logger().info(f'Input data shape: {input_data.shape}')
        
        # Run inference
        self.get_logger().info('Running inference on Hailo...')
        try:
            # Run inference
            with InferVStreams(self.network_group, self.input_vstreams_params, 
                              self.output_vstreams_params) as infer_pipeline:
                # Use stored input name
                input_dict = {self.input_name: input_data}
                
                with self.network_group.activate(self.network_group_params):
                    output = infer_pipeline.infer(input_dict)
                
                outputs = list(output.values())
        except Exception as e:
            self.get_logger().error(f'Hailo inference failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return np.array([]), np.array([]), np.array([])
    
        elapsed = time.time() - start_time
        self.get_logger().info(f'Hailo inference took {elapsed:.3f}s')
        
        # Post-process outputs to match YOLOv5 format
        return self.parse_hailo_output(outputs, image.shape[:2])

    # HAILO POST-PROCESSING - New method to parse Hailo outputs
    def parse_hailo_output_old(self, outputs, orig_shape):
        """Parse Hailo output to match YOLOv5 format"""
        orig_h, orig_w = orig_shape
        boxes_list = []
        scores_list = []
        categories_list = []
        
        self.get_logger().info(f'Parsing {len(outputs)} output layers')
        
        # Process each output layer
        for idx, output in enumerate(outputs):
            self.get_logger().info(f'Output {idx} type: {type(output)}, shape: {np.shape(output) if hasattr(output, "shape") else "N/A"}')
            
            # Handle different output formats
            if isinstance(output, (list, tuple)):
                # Output is a list/tuple, process each element
                for item in output:
                    if isinstance(item, np.ndarray):
                        self.get_logger().info(f'  Item shape: {item.shape}')
                        # Process this detection
                        if len(item.shape) >= 1 and item.shape[-1] >= 6:
                            # Format: [x1, y1, x2, y2, confidence, class_id] or similar
                            for detection in item:
                                if len(detection) >= 6:
                                    x1, y1, x2, y2, conf, cls = detection[:6]
                                    
                                    if conf > 0.25:  # Confidence threshold
                                        boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
                                        scores_list.append(float(conf))
                                        categories_list.append(int(cls))
            
            elif isinstance(output, np.ndarray):
                self.get_logger().info(f'Output {idx} is ndarray with shape: {output.shape}')
                
                # Remove batch dimension if present
                while len(output.shape) > 2 and output.shape[0] == 1:
                    output = output[0]
                
                if len(output.shape) == 2:
                    # Each row is a detection
                    for detection in output:
                        if len(detection) >= 6:
                            x1, y1, x2, y2, conf, cls = detection[:6]
                            
                            if conf > 0.25:
                                boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
                                scores_list.append(float(conf))
                                categories_list.append(int(cls))
                
                elif len(output.shape) == 1 and output.shape[0] >= 85:
                    # Standard YOLOv5 output format
                    confidences = output[4]
                    class_scores = output[5:]
                    class_id = np.argmax(class_scores)
                    final_conf = confidences * class_scores[class_id]
                    
                    if final_conf > 0.25:
                        x_center, y_center, w, h = output[:4]
                        x1 = (x_center - w / 2) * orig_w
                        y1 = (y_center - h / 2) * orig_h
                        x2 = (x_center + w / 2) * orig_w
                        y2 = (y_center + h / 2) * orig_h
                        
                        boxes_list.append([x1, y1, x2, y2])
                        scores_list.append(float(final_conf))
                        categories_list.append(int(class_id))
        
        self.get_logger().info(f'Parsed {len(boxes_list)} detections')
        
        # Return as numpy arrays (handle empty case)
        if len(boxes_list) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(boxes_list), np.array(scores_list), np.array(categories_list)

    def parse_hailo_output(self, outputs, orig_shape):
        """Parse Hailo NMS output to match YOLOv5 format"""
        orig_h, orig_w = orig_shape
        boxes_list = []
        scores_list = []
        categories_list = []
        
        for output in outputs:
            # Remove batch dimension
            if isinstance(output, (list, tuple)) and len(output) == 1:
                output = output[0]
            
            # Iterate through each class
            if isinstance(output, (list, tuple)):
                for class_id, class_detections in enumerate(output):
                    if isinstance(class_detections, np.ndarray) and class_detections.size > 0:
                        # Handle multiple detections per class
                        if len(class_detections.shape) == 2:
                            for detection in class_detections:
                                y_min, x_min, y_max, x_max, conf = detection
                                
                                if conf < 0.25:
                                    continue
                                
                                x1 = float(x_min) * orig_w
                                y1 = float(y_min) * orig_h
                                x2 = float(x_max) * orig_w
                                y2 = float(y_max) * orig_h
                                
                                boxes_list.append([x1, y1, x2, y2])
                                scores_list.append(float(conf))
                                categories_list.append(class_id)
                        
                        elif len(class_detections.shape) == 1 and len(class_detections) == 5:
                            y_min, x_min, y_max, x_max, conf = class_detections
                            
                            if conf >= 0.25:
                                x1 = float(x_min) * orig_w
                                y1 = float(y_min) * orig_h
                                x2 = float(x_max) * orig_w
                                y2 = float(y_max) * orig_h
                                
                                boxes_list.append([x1, y1, x2, y2])
                                scores_list.append(float(conf))
                                categories_list.append(class_id)
        
        if len(boxes_list) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Apply OpenCV's NMS
        boxes_array = np.array(boxes_list)
        scores_array = np.array(scores_list)
        
        # Convert [x1, y1, x2, y2] to [x, y, w, h] for cv2.dnn.NMSBoxes
        boxes_xywh = []
        for box in boxes_array:
            x1, y1, x2, y2 = box
            boxes_xywh.append([x1, y1, x2-x1, y2-y1])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, 
            scores_array.tolist(), 
            score_threshold=0.25,
            nms_threshold=0.45
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            boxes_array = boxes_array[indices]
            scores_array = scores_array[indices]
            categories_array = np.array(categories_list)[indices]
            
            return boxes_array, scores_array, categories_array
        
        return np.array([]), np.array([]), np.array([])
 
    def get_node_state(self, request, response):
        response.success = True
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start yolov5 detect")
        self.start = True
        response.success = True
        response.message = "start"
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop yolov5 detect")
        self.start = False
        response.success = True
        response.message = "stop"
        return response

    def image_callback(self, msg: Image):
        self.get_logger().info('========== IMAGE CALLBACK STARTED ==========')
        try:
        # 5. Detect and publish results.
            image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            
            # HAILO OR PYTORCH INFERENCE - Choose based on device
            if self.use_hailo:
                self.get_logger().info('Running Hailo inference...')
                # Use Hailo inference
                boxes, scores, categories = self.hailo_inference(image)
                self.get_logger().info(f'Hailo inference complete: {len(boxes)} detections')
                names_dict = {i: name for i, name in enumerate(self.class_names)}
            else:
                # Use PyTorch inference (original code)
                detect_result = self.yolov5.predict(image)
                predictions = detect_result[0]
                boxes = predictions.boxes.xyxy.cpu().numpy()
                scores = predictions.boxes.conf.cpu().numpy()
                categories = predictions.boxes.cls.cpu().numpy()
                names_dict = predictions.names

            self.result_msg.detections.clear()
            self.result_msg.header.frame_id = "camera"
            self.result_msg.header.stamp = self.get_clock().now().to_msg()

            # Parse the results (same for both Hailo and PyTorch).
            for index in range(len(categories)):
                name = names_dict[int(categories[index])]
                detection2d = Detection2D()
                detection2d.id = name
                x1, y1, x2, y2 = boxes[index]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                if ros_distribution == 'galactic':
                    detection2d.bbox.center.x = center_x
                    detection2d.bbox.center.y = center_y
                else:
                    detection2d.bbox.center.position.x = center_x
                    detection2d.bbox.center.position.y = center_y

                detection2d.bbox.size_x = float(x2 - x1)
                detection2d.bbox.size_y = float(y2 - y1)

                obj_pose = ObjectHypothesisWithPose()
                obj_pose.hypothesis.class_id = name
                obj_pose.hypothesis.score = float(scores[index])

                detection2d.results.append(obj_pose)
                self.result_msg.detections.append(detection2d)

                # Draw results.
                if self.show_result or self.pub_result_img:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{name}:{obj_pose.hypothesis.score:.2f}", (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.waitKey(1)

                objects_info = []
                h, w = image.shape[:2]

                object_info = ObjectInfo()
                object_info.class_name = name
                object_info.box = [int(coord) for coord in [x1, y1, x2, y2]]
                object_info.score = round(float(scores[index]), 2)
                object_info.width = w
                object_info.height = h
                objects_info.append(object_info)

                object_msg = ObjectsInfo()
                object_msg.objects = objects_info
                self.object_pub.publish(object_msg)

            # Display results if needed.
            if self.show_result:
                self.fps.update()
                image = self.fps.show_fps(image)
                cv2.imshow('result', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if self.pub_result_img:
                result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
                result_img_msg.header = msg.header
                self.result_img_pub.publish(result_img_msg)
            if len(categories) > 0:
                self.yolo_result_pub.publish(self.result_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc()) 
        
    def __del__(self):
        """Cleanup resources"""
        if self.use_hailo and hasattr(self, 'hailo_device'):
            self.hailo_device.release()

def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
