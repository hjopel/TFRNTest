import { Camera, CameraType } from "expo-camera";
import { useState, useEffect, useRef } from "react";
import {
  Button,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Dimensions,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import * as posedetection from "@tensorflow-models/pose-detection";
import Svg, { Circle } from "react-native-svg";

import { cameraWithTensors } from "@tensorflow/tfjs-react-native";

const IS_ANDROID = Platform.OS === "android";
const IS_IOS = Platform.OS === "ios";

// Camera preview size.
//
// From experiments, to render camera feed without distortion, 16:9 ratio
// should be used fo iOS devices and 4:3 ratio should be used for android
// devices.
//
// This might not cover all cases.
const CAM_PREVIEW_WIDTH = Dimensions.get("window").width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// The score threshold for pose detection results.
const MIN_KEYPOINT_SCORE = 0.5;

// The size of the resized output from TensorCamera.
//
// For movenet, the size here doesn't matter too much because the model will
// preprocess the input (crop, resize, etc). For best result, use the size that
// doesn't distort the image.
const OUTPUT_TENSOR_WIDTH = 180;
const OUTPUT_TENSOR_HEIGHT = OUTPUT_TENSOR_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

const TensorCamera = cameraWithTensors(Camera);
export default function App() {
  const [type, setType] = useState(CameraType.back);
  const [permission, requestPermission] = Camera.useCameraPermissions();
  const [isReady, setIsReady] = useState(false);
  const [model, setModel] = useState(null);
  const [poses, setPoses] = useState([]);
  const rafId = useRef(null);

  useEffect(() => {
    async function prepare() {
      rafId.current = null;

      // // Set initial orientation.
      // const curOrientation = await ScreenOrientation.getOrientationAsync();
      // setOrientation(curOrientation);

      // // Listens to orientation change.
      // ScreenOrientation.addOrientationChangeListener((event) => {
      //   setOrientation(event.orientationInfo.orientation);
      // });

      // Camera permission.
      await Camera.requestCameraPermissionsAsync();

      // Wait for tfjs to initialize the backend.
      await tf.ready();

      // Load movenet model.
      // https://github.com/tensorflow/tfjs-models/tree/master/pose-detection
      const movenetModelConfig = {
        modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: true,
      };
      // if (LOAD_MODEL_FROM_BUNDLE) {
      //   const modelJson = require('./offline_model/model.json');
      //   const modelWeights1 = require('./offline_model/group1-shard1of2.bin');
      //   const modelWeights2 = require('./offline_model/group1-shard2of2.bin');
      //   movenetModelConfig.modelUrl = bundleResourceIO(modelJson, [
      //     modelWeights1,
      //     modelWeights2,
      //   ]);
      // }
      const model = await posedetection.createDetector(
        posedetection.SupportedModels.MoveNet,
        movenetModelConfig
      );
      setModel(model);

      // Ready!
      setIsReady(true);
    }
    prepare();
  }, []);

  useEffect(() => {
    // Called when the app is unmounted.
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);

  if (!permission || !isReady) {
    // Camera permissions are still loading
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: "center" }}>
          We need your permission to show the camera
        </Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  function toggleCameraType() {
    setType((current) =>
      current === CameraType.back ? CameraType.front : CameraType.back
    );
  }

  const handleCameraStream = async (images, updatePreview, gl) => {
    const loop = async () => {
      // Get the tensor and run pose detection.
      const imageTensor = images.next().value;

      const startTs = Date.now();
      const poses = await model.estimatePoses(
        imageTensor,
        undefined,
        Date.now()
      );
      const latency = Date.now() - startTs;
      // setFps(Math.floor(1000 / latency));
      setPoses(poses);
      tf.dispose([imageTensor]);

      if (rafId.current === 0) {
        return;
      }

      // Render camera preview manually when autorender=false.
      updatePreview();
      gl.endFrameEXP();

      rafId.current = requestAnimationFrame(loop);
    };

    loop();
  };

  const isPortrait = () => true;

  const getOutputTensorWidth = () => {
    // On iOS landscape mode, switch width and height of the output tensor to
    // get better result. Without this, the image stored in the output tensor
    // would be stretched too much.
    //
    // Same for getOutputTensorHeight below.
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_WIDTH
      : OUTPUT_TENSOR_HEIGHT;
  };

  const getOutputTensorHeight = () => {
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_HEIGHT
      : OUTPUT_TENSOR_WIDTH;
  };
  const renderPose = () => {
    if (poses != null && poses.length > 0) {
      const keypoints = poses[0].keypoints
        .filter((k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE)
        .map((k) => {
          // Flip horizontally on android or when using back camera on iOS.
          // const flipX = IS_ANDROID || cameraType === Camera.Constants.Type.back;
          const flipX = true; // code above doesn't work correctly, if front it has to be flipped (on ios)
          const x = flipX ? getOutputTensorWidth() - k.x : k.x;
          const y = k.y;
          const cx =
            (x / getOutputTensorWidth()) *
            (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
          const cy =
            (y / getOutputTensorHeight()) *
            (isPortrait() ? CAM_PREVIEW_HEIGHT : CAM_PREVIEW_WIDTH);
          return (
            <Circle
              key={`skeletonkp_${k.name}`}
              cx={cx}
              cy={cy}
              r="4"
              strokeWidth="2"
              fill="#00AA00"
              stroke="white"
            />
          );
        });

      return <Svg style={styles.svg}>{keypoints}</Svg>;
    } else {
      return <View></View>;
    }
  };

  const getTextureRotationAngleInDegrees = () => {
    // On Android, the camera texture will rotate behind the scene as the phone
    // changes orientation, so we don't need to rotate it in TensorCamera.
    if (IS_ANDROID) {
      return 0;
    }

    // For iOS, the camera texture won't rotate automatically. Calculate the
    // rotation angles here which will be passed to TensorCamera to rotate it
    // internally.
    // switch (orientation) {
    //   // Not supported on iOS as of 11/2021, but add it here just in case.
    //   case ScreenOrientation.Orientation.PORTRAIT_DOWN:
    //     return 180;
    //   case ScreenOrientation.Orientation.LANDSCAPE_LEFT:
    //     return cameraType === Camera.Constants.Type.front ? 270 : 90;
    //   case ScreenOrientation.Orientation.LANDSCAPE_RIGHT:
    //     return cameraType === Camera.Constants.Type.front ? 90 : 270;
    //   default:
    //     return 0;
    // }
    return 0;
  };
  return (
    <View style={styles.container}>
      <TensorCamera
        style={styles.camera}
        type={CameraType.front}
        resizeWidth={getOutputTensorWidth()}
        resizeHeight={getOutputTensorHeight()}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={false}
        rotation={getTextureRotationAngleInDegrees()}
      />
      {renderPose()}
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={toggleCameraType}>
          <Text style={styles.text}>Flip Camera</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
  },
  camera: {
    width: "100%",
    height: "100%",
    zIndex: 1,
  },
  svg: {
    width: "100%",
    height: "100%",
    position: "absolute",
    zIndex: 30,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: "transparent",
    margin: 64,
  },
  button: {
    flex: 1,
    alignSelf: "flex-end",
    alignItems: "center",
  },
  text: {
    fontSize: 24,
    fontWeight: "bold",
    color: "white",
  },
});
