(function () {
  var ns = $.namespace('pskl.service');

  ns.FlowNetService = function () {
    this.model = null;
    this.isModelLoaded = false;
    this.modelPath = './models/flownet/model.json';
    // Should add fallback URLs:
    this.fallbackModelPaths = [
      './models/flownet/model.json'
      // Remove the non-working URLs for now
    ];
    // Standard size for model input
    this.targetSize = {
      width: 256,
      height: 256
    };
  };

  ns.FlowNetService.prototype.init = async function (progressCallback) {
    try {
      // Enhanced WebGL setup for TensorFlow.js
      await tf.setBackend('webgl');
      
      // Optimize WebGL for TensorFlow.js
      tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', -1);
      tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
      tf.env().set('WEBGL_PACK', true);
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
      tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);  // Aggressive cleanup
      
      // Enable float texture support
      const gl = document.createElement('canvas').getContext('webgl2');
      if (gl) {
        tf.env().set('WEBGL_VERSION', 2);
        tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);
      }

      // Add debug logging
      console.log('Attempting to load model from paths:', this.fallbackModelPaths);
      
      let loadError;
      for (const modelPath of this.fallbackModelPaths) {
        try {
          console.log('Trying to load model from:', modelPath);
          // Add fetch test
          try {
            const response = await fetch(modelPath);
            const modelJson = await response.json();
            console.log('Model JSON loaded:', modelJson);
          } catch (fetchError) {
            console.error('Failed to fetch model JSON:', fetchError);
          }

          this.model = await tf.loadGraphModel(modelPath, {
            onProgress: (fraction) => {
              const progress = Math.round(fraction * 100);
              console.log('Loading progress:', progress + '%');
              if (progressCallback) {
                progressCallback(progress);
              }
            }
          });
          console.log('Successfully loaded model from:', modelPath);
          break;
        } catch (error) {
          console.warn(`Failed to load model from ${modelPath}:`, error);
          loadError = error;
        }
      }

      if (!this.model) {
        throw loadError || new Error('Failed to load model from all paths');
      }

      console.log('FlowNet model loaded successfully');
      console.log('Model inputs:', this.model.inputs);

      // Warmup with correct input shape and count
      const dummyTensor1 = tf.zeros([1, 256, 256, 3]);
      const dummyTensor2 = tf.zeros([1, 256, 256, 3]);
      
      try {
        // Use array instead of named dict for inputs
        const warmupResult = await this.model.predict([dummyTensor1, dummyTensor2]);
        warmupResult.dispose();
      } finally {
        dummyTensor1.dispose();
        dummyTensor2.dispose();
      }
      
      this.isModelLoaded = true;
      return true;
    } catch (error) {
      console.error('Failed to initialize FlowNet:', error);
      throw error;
    }
  };

  ns.FlowNetService.prototype.preprocessFrame = function(frame) {
    // Store original dimensions for later use
    this.originalWidth = frame.getWidth();
    this.originalHeight = frame.getHeight();
    
    return tf.tidy(() => {
      const width = frame.getWidth();
      const height = frame.getHeight();
      const pixels = frame.getPixels();
      
      // Create RGB data array (3 channels)
      const data = new Float32Array(width * height * 3);
      for (let i = 0; i < pixels.length; i++) {
        const color = pixels[i];
        // Extract ABGR components (Piskel's format)
        const a = (color >>> 24) & 0xFF;
        const b = (color >>> 16) & 0xFF;
        const g = (color >>> 8) & 0xFF;
        const r = color & 0xFF;

        // Store alpha for later use
        this.alphaData = this.alphaData || new Uint8Array(pixels.length);
        this.alphaData[i] = a;

        // Convert to RGB and normalize to [-1, 1] range for better flow computation
        data[i * 3] = (r / 127.5) - 1;     // R
        data[i * 3 + 1] = (g / 127.5) - 1; // G
        data[i * 3 + 2] = (b / 127.5) - 1; // B
      }
      
      // Create tensor and reshape
      let tensor = tf.tensor3d(data, [height, width, 3]);

      // Resize if necessary
      if (width !== this.targetSize.width || height !== this.targetSize.height) {
        tensor = tf.image.resizeBilinear(tensor, [
          this.targetSize.height,
          this.targetSize.width
        ]);
      }
      
      return tensor;
    });
  };

  ns.FlowNetService.prototype.prepareFramePair = function(frame1, frame2) {
    return tf.tidy(() => {
      // Convert frames to tensors
      const tensor1 = this.preprocessFrame(frame1);
      const tensor2 = this.preprocessFrame(frame2);
      
      // Create batch dimension and stack frames
      const batched1 = tensor1.expandDims(0);
      const batched2 = tensor2.expandDims(0);
      
      // Return array of tensors as expected by model
      return [batched1, batched2];
    });
  };

  // Update compute flow to use preprocessing
  ns.FlowNetService.prototype.computeFlow = function(frame1, frame2) {
    try {
      return tf.tidy(() => {
        // 1. Prepare input frames
        const input = this.prepareFramePair(frame1, frame2);
        
        // 2. Run model inference to get flow field
        const flowField = this.model.predict(input);
        
        // Get flow dimensions and ensure correct shape
        // Model outputs (batch, height, width, 2) tensor
        const flow = flowField.squeeze(); // Remove batch dim, now (height, width, 2)
        console.log('Flow shape after squeeze:', flow.shape);
        
        // No need to reshape since model outputs correct shape [height, width, 2]
        return this.postprocessFlow(flow, frame1.getWidth(), frame1.getHeight());
      });
    } catch (error) {
      console.error('Flow computation failed:', error);
      return this.computeSimpleFlow(frame1, frame2);
    }
  };

  // Simple fallback flow computation
  ns.FlowNetService.prototype.computeSimpleFlow = function(frame1, frame2) {
    return tf.tidy(() => {
      // Simple difference-based flow as fallback
      const tensor1 = this.preprocessFrame(frame1);
      const tensor2 = this.preprocessFrame(frame2);
      return tensor2.sub(tensor1);
    });
  };

  ns.FlowNetService.prototype.postprocessFlow = function(flow, width, height) {
    return tf.tidy(() => {
      // Ensure flow is rank 3 [height, width, 2]
      let processedFlow = flow;
      if (flow.rank > 3) {
        processedFlow = flow.squeeze(); // Remove extra dimensions
      }
      
      // Denormalize flow from tanh range [-1, 1] to pixel displacements
      const denormFlow = processedFlow.mul(tf.scalar(Math.max(width, height) / 2));
      
      // Resize flow field to match frame dimensions if needed
      if (width !== this.targetSize.width || height !== this.targetSize.height) {
        // Scale flow values according to resize ratio
        const scaleX = width / this.targetSize.width;
        const scaleY = height / this.targetSize.height;
        
        // Create scale factors as a 3D tensor directly
        const scaleFactors = tf.fill(
          [denormFlow.shape[0], denormFlow.shape[1], 2],
          [scaleY, scaleX]
        );
        
        const scaledFlow = denormFlow.mul(scaleFactors);
        
        return tf.image.resizeBilinear(
          scaledFlow.expandDims(0),
          [height, width]
        ).squeeze(0);
      }
      return denormFlow;
    });
  };

  // Add method to visualize flow for debugging
  ns.FlowNetService.prototype.visualizeFlow = function(flow) {
    return tf.tidy(() => {
      // Convert flow vectors to HSV color space
      // Hue represents direction, Saturation represents magnitude
      const [flowY, flowX] = tf.split(flow, 2, -1);
      
      // Calculate magnitude and angle
      const magnitude = tf.sqrt(tf.square(flowX).add(tf.square(flowY)));
      const angle = tf.atan2(flowY, flowX);
      
      // Normalize magnitude for visualization
      const normalizedMagnitude = tf.clipByValue(
        magnitude.div(tf.maximum(magnitude.max(), 1e-6)).mul(255),
        0, 255
      );
      
      // Convert angle to hue (0-179 for OpenCV compatibility)
      const hue = angle.add(Math.PI).mul(179).div(2 * Math.PI);
      
      // Create HSV image
      const hsv = tf.stack([
        hue.squeeze(),
        tf.onesLike(hue.squeeze()).mul(255),
        normalizedMagnitude.squeeze()
      ], -1);
      
      return hsv;
    });
  };

  // Update the warpFrame method to fix transform matrix creation
  ns.FlowNetService.prototype.warpFrame = function(frame, flow, t) {
    return tf.tidy(() => {
      const tensor = this.preprocessFrame(frame).expandDims(0);
      const [batchSize, h, w, channels] = tensor.shape;
      
      // Ensure flow has correct shape [height, width, 2]
      let processedFlow = flow;
      if (flow.rank !== 3) {
        processedFlow = flow.reshape([flow.shape[0], flow.shape[1], 2]);
      }
      
      // Resize flow to match frame dimensions if they don't match
      if (processedFlow.shape[0] !== h || processedFlow.shape[1] !== w) {
        processedFlow = tf.image.resizeBilinear(
          processedFlow.expandDims(0),
          [h, w]
        ).squeeze(0);
      }
      
      // Get flow components
      const flowY = processedFlow.slice([0, 0, 0], [-1, -1, 1]).squeeze(-1);
      const flowX = processedFlow.slice([0, 0, 1], [-1, -1, 1]).squeeze(-1);
      
      // Scale flow by time factor
      const scaledFlowX = flowX.mul(t);
      const scaledFlowY = flowY.mul(t);

      // Create sampling grid
      const gridY = tf.range(0, h).reshape([h, 1]).tile([1, w]);
      const gridX = tf.range(0, w).reshape([1, w]).tile([h, 1]);
      
      // Apply flow to grid
      const sampledY = gridY.add(scaledFlowY).clipByValue(0, h - 1);
      const sampledX = gridX.add(scaledFlowX).clipByValue(0, w - 1);
      
      // Create transform matrix for affine transformation
      const transformArray = new Float32Array([
        1, 0, 0,  // First row: x scaling, x shearing, x translation
        0, 1, 0,  // Second row: y shearing, y scaling, y translation
        0, 0      // Perspective terms (excluding the implicit 1)
      ]);
      
      // Apply flow displacements
      transformArray[2] = sampledX.mean().dataSync()[0]; // x translation
      transformArray[5] = sampledY.mean().dataSync()[0]; // y translation
      
      // Create transform matrix tensor
      const transformMatrix = tf.tensor1d(transformArray).reshape([1, 8]);

      // Apply transform
      const warped = tf.image.transform(
        tensor,
        transformMatrix,
        'bilinear'
      );

      // Create and apply edge mask
      const mask = tf.buffer([1, h, w, channels]);
      const padding = 2;
      for (let i = padding; i < h - padding; i++) {
        for (let j = padding; j < w - padding; j++) {
          for (let c = 0; c < channels; c++) {
            mask.set(1, 0, i, j, c);
          }
        }
      }
      
      return this.postprocessFrame(
        warped.mul(tf.tensor(mask.values, mask.shape)).squeeze()
      );
    });
  };

  // Update smoothFlow to ensure consistent tensor shapes
  ns.FlowNetService.prototype.smoothFlow = function(flow) {
    return tf.tidy(() => {
      // Apply Gaussian blur to flow field
      const kernel = tf.tensor2d([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
      ]).div(16);

      // Get flow dimensions
      const [h, w, c] = flow.shape;
      console.log('Flow shape in smoothFlow:', [h, w, c]);

      // Split flow into Y and X components
      const flowChannels = tf.split(flow, 2, -1);
      const flowY = flowChannels[0];
      const flowX = flowChannels[1];

      // Expand kernel for 2D convolution
      const kernelExpanded = kernel.expandDims(-1).expandDims(-1);

      // Apply smoothing to each component
      const smoothedX = tf.conv2d(
        flowX.reshape([1, h, w, 1]),
        kernelExpanded,
        1,
        'same'
      ).squeeze([0]); // Remove batch dimension

      const smoothedY = tf.conv2d(
        flowY.reshape([1, h, w, 1]),
        kernelExpanded,
        1,
        'same'
      ).squeeze([0]);

      // Stack the smoothed components back together
      return tf.stack([smoothedY, smoothedX], -1);
    });
  };

  ns.FlowNetService.prototype.postprocessFrame = function(tensor) {
    return tf.tidy(() => {
      // Convert back to pixel values from [-1,1] range
      const pixels = tensor
        .add(1).mul(127.5)  // [-1,1] -> [0,255]
        .clipByValue(0, 255)
        .cast('int32');
      
      // Get original frame dimensions
      const originalWidth = this.originalWidth || this.targetSize.width;
      const originalHeight = this.originalHeight || this.targetSize.height;
      
      // Resize tensor to original dimensions if needed
      let resizedPixels = pixels;
      if (pixels.shape[0] !== originalHeight || pixels.shape[1] !== originalWidth) {
        resizedPixels = tf.image.resizeBilinear(
          pixels,
          [originalHeight, originalWidth]
        );
      }
      
      // Create frame with original dimensions
      const frame = new pskl.model.Frame(originalWidth, originalHeight);
      
      // Get pixel data
      const pixelData = resizedPixels.dataSync();
      const rgbaPixels = new Uint32Array(originalWidth * originalHeight);
      
      // Convert RGB to ABGR (Piskel's format)
      for (let i = 0; i < originalWidth * originalHeight; i++) {
        const r = Math.round(pixelData[i * 3]);
        const g = Math.round(pixelData[i * 3 + 1]);
        const b = Math.round(pixelData[i * 3 + 2]);
        // Use stored alpha or default to fully opaque
        const a = this.alphaData ? this.alphaData[i] : 255;
        
        // Pack as ABGR (Piskel's format)
        // Only set pixel if it has some opacity
        if (a > 0) {
          rgbaPixels[i] = (a << 24) | (b << 16) | (g << 8) | r;
        } else {
          rgbaPixels[i] = 0; // Fully transparent
        }
      }
      
      frame.setPixels(rgbaPixels);
      return frame;
    });
  };

  ns.FlowNetService.prototype.verifyModel = async function() {
    const testFrame1 = new pskl.model.Frame(64, 64);
    const testFrame2 = new pskl.model.Frame(64, 64);
    
    // Add test patterns
    testFrame1.setPixel(32, 32, 0xFF000000);
    testFrame2.setPixel(34, 34, 0xFF000000);
    
    try {
      const flow = await this.computeFlow(testFrame1, testFrame2);
      const magnitude = tf.sqrt(
        tf.square(flow.slice([0, 0, 0], [-1, -1, 1]))
          .add(tf.square(flow.slice([0, 0, 1], [-1, -1, 1])))
      );
      
      const maxFlow = magnitude.max().dataSync()[0];
      return maxFlow > 0 && maxFlow < 10; // Reasonable flow range
    } catch (error) {
      console.error('Model verification failed:', error);
      return false;
    }
  };

  ns.FlowNetService.prototype.testPreprocessing = function() {
    // Create test frames
    const testFrame1 = new pskl.model.Frame(64, 64);
    const testFrame2 = new pskl.model.Frame(64, 64);
    
    // Draw a simple shape in frame 1 (black square)
    for (let y = 20; y < 30; y++) {
      for (let x = 20; x < 30; x++) {
        testFrame1.setPixel(x, y, 0xFF000000);
      }
    }
    
    // Draw the same shape in frame 2, moved diagonally
    for (let y = 25; y < 35; y++) {
      for (let x = 25; x < 35; x++) {
        testFrame2.setPixel(x, y, 0xFF000000);
      }
    }

    try {
      // Test preprocessing
      const tensor1 = this.preprocessFrame(testFrame1);
      const tensor2 = this.preprocessFrame(testFrame2);
      
      // Test frame pair preparation
      const combinedTensor = this.prepareFramePair(testFrame1, testFrame2);
      
      // Log tensor information
      console.log('Preprocessing test results:', {
        tensor1Shape: tensor1.shape,
        tensor2Shape: tensor2.shape,
        combinedShape: combinedTensor.shape,
        tensor1Stats: {
          min: tensor1.min().dataSync()[0],
          max: tensor1.max().dataSync()[0]
        }
      });

      // Clean up
      tensor1.dispose();
      tensor2.dispose();
      combinedTensor.dispose();
      
      return true;
    } catch (error) {
      console.error('Preprocessing test failed:', error);
      return false;
    }
  };

  ns.FlowNetService.prototype.testPreprocessingDetailed = function() {
    const testFrame1 = new pskl.model.Frame(64, 64);
    
    // Draw different colored pixels in ABGR format
    testFrame1.setPixel(20, 20, 0xFF000000);  // Black (A=FF, B=00, G=00, R=00)
    testFrame1.setPixel(21, 20, 0xFF0000FF);  // Red (A=FF, B=00, G=00, R=FF)
    testFrame1.setPixel(22, 20, 0xFF00FF00);  // Green (A=FF, B=00, G=FF, R=00)
    testFrame1.setPixel(23, 20, 0xFFFF0000);  // Blue (A=FF, B=FF, G=00, R=00)
    
    try {
      const tensor1 = this.preprocessFrame(testFrame1);
      const data = tensor1.dataSync();
      
      // Log values for each test pixel
      console.log('Pixel Values:', {
        black: {
          r: data[20 * 64 * 3 + 20 * 3],
          g: data[20 * 64 * 3 + 20 * 3 + 1],
          b: data[20 * 64 * 3 + 20 * 3 + 2]
        },
        red: {
          r: data[20 * 64 * 3 + 21 * 3],
          g: data[20 * 64 * 3 + 21 * 3 + 1],
          b: data[20 * 64 * 3 + 21 * 3 + 2]
        },
        green: {
          r: data[20 * 64 * 3 + 22 * 3],
          g: data[20 * 64 * 3 + 22 * 3 + 1],
          b: data[20 * 64 * 3 + 22 * 3 + 2]
        },
        blue: {
          r: data[20 * 64 * 3 + 23 * 3],
          g: data[20 * 64 * 3 + 23 * 3 + 1],
          b: data[20 * 64 * 3 + 23 * 3 + 2]
        }
      });

      tensor1.dispose();
      return true;
    } catch (error) {
      console.error('Detailed preprocessing test failed:', error);
      return false;
    }
  };
})(); 