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
    this.regions = new Map();
    this.tinyRegions = [];
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
        const interpolationService = new pskl.service.InterpolationService();
        
        // Convert frames to RGB tensors
        const tensor1 = interpolationService.frameToTensor(frame1);
        const tensor2 = interpolationService.frameToTensor(frame2);
        
        // Extract RGB channels and ensure 3 channels
        const rgb1 = tensor1.slice([0, 0, 0], [-1, -1, 3]);
        const rgb2 = tensor2.slice([0, 0, 0], [-1, -1, 3]);

        // Get original dimensions
        const [originalH, originalW] = rgb1.shape;

        // Resize to 256x256 for model input (FlowNet requirement)
        const resized1 = tf.image.resizeBilinear(rgb1, [256, 256]);
        const resized2 = tf.image.resizeBilinear(rgb2, [256, 256]);
        
        // Add batch dimension
        const batched1 = resized1.expandDims(0);
        const batched2 = resized2.expandDims(0);
        
        // Run model inference
        const flowField = this.model.predict([batched1, batched2]);
        const flow = flowField.squeeze();
        
        // Resize flow back to original dimensions
        const resizedFlow = tf.image.resizeBilinear(
          flow.expandDims(0),
          [originalH, originalW]
        ).squeeze(0);

        // Scale the flow values to account for the resize
        const scaleY = originalH / 256;
        const scaleX = originalW / 256;
        
        // Apply scaling to maintain proper motion magnitude
        const scaledFlow = tf.mul(
          resizedFlow,
          tf.tensor([scaleY, scaleX]).reshape([1, 1, 2])
        ).mul(0.5);

        // Apply motion vector quantization with original frame dimensions
        const quantizedFlow = this.quantizeFlow(scaledFlow, frame1);
        
        return quantizedFlow;
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
      const interpolationService = new pskl.service.InterpolationService();
      const tensor = interpolationService.frameToTensor(frame);
      const [h, w, channels] = tensor.shape;
      
      // Detect edges
      const edgeInfo = this.detectEdges(tensor);
      
      // Ensure flow has correct shape
      let processedFlow = flow;
      if (flow.rank !== 3) {
        processedFlow = flow.reshape([h, w, 2]);
      }
      
      // Scale flow by time factor
      const scaledFlow = processedFlow.mul(t);
      
      // Get flow components
      const flowY = scaledFlow.slice([0, 0, 0], [-1, -1, 1]).squeeze(-1);
      const flowX = scaledFlow.slice([0, 0, 1], [-1, -1, 1]).squeeze(-1);

      // Create sampling grid coordinates
      const ys = tf.range(0, h);
      const xs = tf.range(0, w);
      
      // Create meshgrid
      const [gridX, gridY] = tf.meshgrid(xs, ys);
      
      // Add flow to grid coordinates
      const sampleX = gridX.add(flowX);
      const sampleY = gridY.add(flowY);

      // Create output tensor
      const output = tf.buffer([h, w, channels]);
      
      // Get data for processing
      const tensorData = tensor.arraySync();
      const sampleXData = sampleX.arraySync();
      const sampleYData = sampleY.arraySync();
      const edgeData = edgeInfo.edges.arraySync();
      
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          // Get sample coordinates
          let sx = sampleXData[y][x];
          let sy = sampleYData[y][x];
          
          // Clamp coordinates
          sx = Math.max(0, Math.min(w - 1, sx));
          sy = Math.max(0, Math.min(h - 1, sy));
          
          // Use nearest neighbor sampling for edge pixels
          if (edgeData[y][x] > 0.5) {
            const x0 = Math.round(sx);
            const y0 = Math.round(sy);
            
            for (let c = 0; c < channels; c++) {
              output.set(tensorData[y0][x0][c], y, x, c);
            }
          } else {
            // Use bilinear sampling for non-edge pixels
            const x0 = Math.floor(sx);
            const y0 = Math.floor(sy);
            const x1 = Math.min(x0 + 1, w - 1);
            const y1 = Math.min(y0 + 1, h - 1);
            
            const wx = sx - x0;
            const wy = sy - y0;
            
            for (let c = 0; c < channels; c++) {
              const v00 = tensorData[y0][x0][c];
              const v01 = tensorData[y0][x1][c];
              const v10 = tensorData[y1][x0][c];
              const v11 = tensorData[y1][x1][c];
              
              const value = (1 - wy) * ((1 - wx) * v00 + wx * v01) +
                           wy * ((1 - wx) * v10 + wx * v11);
              
              output.set(value, y, x, c);
            }
          }
        }
      }
      
      // Create edge mask with padding
      const mask = tf.buffer([h, w, channels]);
      const padding = 2;
      for (let i = padding; i < h - padding; i++) {
        for (let j = padding; j < w - padding; j++) {
          for (let c = 0; c < channels; c++) {
            mask.set(1, i, j, c);
          }
        }
      }
      
      // Apply mask and convert back to frame
      const warpedTensor = output.toTensor().mul(mask.toTensor());
      return interpolationService.tensorToFrame(warpedTensor, w, h);
    });
  };

  // Update smoothFlow to handle single-channel input
  ns.FlowNetService.prototype.smoothFlow = function(flow) {
    return tf.tidy(() => {
      const kernel = tf.tensor2d([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
      ]).div(16);

      // Ensure flow is 3D for conv2d
      const flow3D = flow.expandDims(0).expandDims(-1);
      const kernelExpanded = kernel.expandDims(-1).expandDims(-1);

      // Apply smoothing
      return tf.conv2d(
        flow3D,
        kernelExpanded,
        1,
        'same'
      ).squeeze([0, -1]); // Remove batch and channel dimensions
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

  // Add edge detection methods
  ns.FlowNetService.prototype.detectEdges = function(tensor, threshold = 0.1) {
    return tf.tidy(() => {
      // Sobel kernels for edge detection
      const sobelX = tf.tensor2d([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
      ]).expandDims(-1).expandDims(-1);

      const sobelY = tf.tensor2d([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
      ]).expandDims(-1).expandDims(-1);

      // Convert to grayscale with perceptual weights
      let grayscale;
      if (tensor.shape[2] === 4) {
        const rgb = tensor.slice([0, 0, 0], [-1, -1, 3]);
        // Use perceptual weights for better edge detection
        grayscale = rgb.mul(tf.tensor3d([0.299, 0.587, 0.114], [1, 1, 3])).sum(-1);
      } else {
        grayscale = tensor.mean(-1);
      }

      // Ensure proper shape for convolution
      grayscale = grayscale.expandDims(-1);

      // Apply Sobel with stronger edge detection
      const gx = tf.conv2d(grayscale.expandDims(0), sobelX, 1, 'same').squeeze(0);
      const gy = tf.conv2d(grayscale.expandDims(0), sobelY, 1, 'same').squeeze(0);

      // Compute edge magnitude with non-linear enhancement
      const magnitude = tf.sqrt(tf.square(gx).add(tf.square(gy))).squeeze(-1);
      const enhanced = tf.pow(magnitude, tf.scalar(1.5)); // Non-linear enhancement
      
      // Normalize and apply adaptive threshold
      const normalizedEdges = enhanced.div(enhanced.max());
      const edges = normalizedEdges.greater(threshold);

      return {
        edges: edges,
        magnitude: normalizedEdges
      };
    });
  };

  // Modify quantizeFlow to use edge information
  ns.FlowNetService.prototype.quantizeFlow = function(flow, frame) {
    return tf.tidy(() => {
      const interpolationService = new pskl.service.InterpolationService();
      const tensor = interpolationService.frameToTensor(frame);
      
      // Get dimensions from the flow tensor
      const [h, w] = flow.shape.slice(0, 2);
      const resizedTensor = tf.image.resizeBilinear(tensor, [h, w]);
      
      // More granular edge detection with three levels
      const edgeInfo1 = this.detectEdges(resizedTensor, 0.015); // Very fine details
      const edgeInfo2 = this.detectEdges(resizedTensor, 0.03);  // Medium details
      const edgeInfo3 = this.detectEdges(resizedTensor, 0.06);  // Strong edges
      
      // Split flow into components
      const [flowY, flowX] = tf.split(flow, 2, -1);
      
      // Compute flow statistics
      const flowMagnitude = tf.sqrt(tf.square(flowX).add(tf.square(flowY)));
      const maxFlow = flowMagnitude.max();
      
      // More granular quantization levels for smoother transitions
      const quantLevels = [0.05, 0.1, 0.2, 0.35, 0.5];
      const quantizedFlows = quantLevels.map(level => {
        const threshold = maxFlow.mul(level);
        // Add small random offset to break up banding
        const noiseScale = threshold.mul(0.1);
        const noise = tf.randomUniform(flowX.shape, -1, 1).mul(noiseScale);
        
        return {
          x: tf.round(flowX.add(noise).div(threshold)).mul(threshold),
          y: tf.round(flowY.add(noise).div(threshold)).mul(threshold)
        };
      });
      
      // Initialize with original flow
      let refinedX = flowX;
      let refinedY = flowY;
      
      // Progressive refinement with smoother transitions
      quantLevels.forEach((level, i) => {
        const threshold = maxFlow.mul(level);
        const magnitudeMask = flowMagnitude.greater(threshold);
        
        // Smoother transition curve using sigmoid-like function
        const transitionWidth = threshold.mul(0.3); // Wider transition region
        const transitionWeight = flowMagnitude.sub(threshold)
          .div(transitionWidth)
          .tanh()
          .add(1)
          .div(2)
          .clipByValue(0, 1);
        
        // Adaptive blending based on quantization level
        const levelWeight = Math.pow(0.8, i); // Exponential decrease in quantization influence
        
        const blendedX = quantizedFlows[i].x.mul(transitionWeight.mul(levelWeight))
          .add(refinedX.mul(tf.sub(1, transitionWeight.mul(levelWeight))));
        const blendedY = quantizedFlows[i].y.mul(transitionWeight.mul(levelWeight))
          .add(refinedY.mul(tf.sub(1, transitionWeight.mul(levelWeight))));
        
        refinedX = tf.where(magnitudeMask, blendedX, refinedX);
        refinedY = tf.where(magnitudeMask, blendedY, refinedY);
      });
      
      // Multi-level edge preservation
      const edgeMask1 = edgeInfo1.edges.expandDims(-1);
      const edgeMask2 = edgeInfo2.edges.expandDims(-1);
      const edgeMask3 = edgeInfo3.edges.expandDims(-1);
      
      // Weighted edge preservation
      const detailWeight = edgeMask1.mul(0.5)
        .add(edgeMask2.mul(0.3))
        .add(edgeMask3.mul(0.2));
      
      // Reshape tensors
      const refinedXReshaped = refinedX.reshape([h, w, 1]);
      const refinedYReshaped = refinedY.reshape([h, w, 1]);
      const flowXReshaped = flowX.reshape([h, w, 1]);
      const flowYReshaped = flowY.reshape([h, w, 1]);
      
      // Adaptive edge-aware blending
      const edgeBlendX = flowXReshaped.mul(detailWeight)
        .add(refinedXReshaped.mul(tf.sub(1, detailWeight)));
      const edgeBlendY = flowYReshaped.mul(detailWeight)
        .add(refinedYReshaped.mul(tf.sub(1, detailWeight)));
      
      // Enhanced smoothing with edge preservation
      const smoothedX = this.enhancedSmoothing(
        edgeBlendX.squeeze(-1),
        edgeMask3.squeeze(-1)
      );
      const smoothedY = this.enhancedSmoothing(
        edgeBlendY.squeeze(-1),
        edgeMask3.squeeze(-1)
      );
      
      return tf.stack([smoothedY, smoothedX], -1);
    });
  };

  // Enhanced smoothing function
  ns.FlowNetService.prototype.enhancedSmoothing = function(flow, edgeMask) {
    return tf.tidy(() => {
      // Gaussian-like kernel for smoother results
      const kernel = tf.tensor2d([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
      ]).div(256);
      
      const flow3D = flow.expandDims(0).expandDims(-1);
      const kernelExpanded = kernel.expandDims(-1).expandDims(-1);
      
      // Apply two-pass smoothing for better results
      const firstPass = tf.conv2d(
        flow3D,
        kernelExpanded,
        1,
        'same'
      ).squeeze([0, -1]);
      
      const secondPass = tf.conv2d(
        firstPass.expandDims(0).expandDims(-1),
        kernelExpanded,
        1,
        'same'
      ).squeeze([0, -1]);
      
      // Adaptive blending between original and smoothed
      const blendFactor = tf.sub(1, edgeMask).pow(tf.scalar(2));
      return flow.mul(edgeMask).add(secondPass.mul(blendFactor));
    });
  };

  ns.FlowNetService.prototype.visualizeQuantizedFlow = function(flow) {
    return tf.tidy(() => {
      // Split flow into components
      const [flowY, flowX] = tf.split(flow, 2, -1);
      
      // Get unique flow values
      const uniqueX = Array.from(new Set(flowX.dataSync()));
      const uniqueY = Array.from(new Set(flowY.dataSync()));
      
      console.log('Unique flow values:', {
        x: uniqueX.sort((a, b) => a - b),
        y: uniqueY.sort((a, b) => a - b)
      });
      
      // Create visualization
      const magnitude = tf.sqrt(tf.square(flowX).add(tf.square(flowY)));
      const direction = tf.atan2(flowY, flowX);
      
      return {
        magnitude: magnitude.arraySync(),
        direction: direction.arraySync(),
        quantizedValues: {
          x: uniqueX,
          y: uniqueY
        }
      };
    });
  };

  // Main color analysis method
  ns.FlowNetService.prototype.analyzeColorPalette = function(frame1, frame2) {
    return tf.tidy(() => {
      // Extract unique colors from both frames
      const colors1 = new Set();
      const colors2 = new Set();
      
      // Get pixel data
      const pixels1 = frame1.getPixels();
      const pixels2 = frame2.getPixels();
      
      // Collect unique non-transparent colors
      for (let i = 0; i < pixels1.length; i++) {
        if (pixels1[i] !== 0) { // Skip transparent pixels
          colors1.add(pixels1[i]);
        }
        if (pixels2[i] !== 0) {
          colors2.add(pixels2[i]);
        }
      }
      
      // Convert to arrays and sort by frequency
      const colorMap1 = this.getColorFrequencyMap(pixels1, colors1);
      const colorMap2 = this.getColorFrequencyMap(pixels2, colors2);
      
      return {
        frame1Colors: Array.from(colors1),
        frame2Colors: Array.from(colors2),
        colorMaps: {
          frame1: colorMap1,
          frame2: colorMap2
        },
        colorMatches: this.matchColors(colorMap1, colorMap2)
      };
    });
  };

  // Helper method to get color frequency map
  ns.FlowNetService.prototype.getColorFrequencyMap = function(pixels, uniqueColors) {
    const frequencyMap = new Map();
    
    // Initialize frequency map
    uniqueColors.forEach(color => {
      frequencyMap.set(color, {
        count: 0,
        color: color,
        r: color & 0xFF,
        g: (color >> 8) & 0xFF,
        b: (color >> 16) & 0xFF,
        a: (color >> 24) & 0xFF
      });
    });
    
    // Count color frequencies
    for (let i = 0; i < pixels.length; i++) {
      const color = pixels[i];
      if (color !== 0) {
        const info = frequencyMap.get(color);
        info.count++;
      }
    }
    
    return frequencyMap;
  };

  // Helper method to match colors between frames
  ns.FlowNetService.prototype.matchColors = function(colorMap1, colorMap2) {
    const matches = new Map();
    
    colorMap1.forEach((info1, color1) => {
      let bestMatch = null;
      let minDistance = Infinity;
      
      colorMap2.forEach((info2, color2) => {
        const distance = Math.sqrt(
          Math.pow(info1.r - info2.r, 2) +
          Math.pow(info1.g - info2.g, 2) +
          Math.pow(info1.b - info2.b, 2)
        );
        
        if (distance < minDistance) {
          minDistance = distance;
          bestMatch = color2;
        }
      });
      
      if (bestMatch !== null) {
        matches.set(color1, bestMatch);
      }
    });
    
    return matches;
  };

  // Add test method for color palette analysis
  ns.FlowNetService.prototype.testColorPaletteAnalysis = function(frame1, frame2) {
    try {
      const paletteInfo = this.analyzeColorPalette(frame1, frame2);
      
      console.log('Color Palette Analysis Results:', {
        'Frame 1 Colors': paletteInfo.frame1Colors.map(color => ({
          hex: '#' + color.toString(16).padStart(8, '0'),
          info: paletteInfo.colorMaps.frame1.get(color)
        })),
        'Frame 2 Colors': paletteInfo.frame2Colors.map(color => ({
          hex: '#' + color.toString(16).padStart(8, '0'),
          info: paletteInfo.colorMaps.frame2.get(color)
        })),
        'Color Matches': Array.from(paletteInfo.colorMatches).map(([color1, color2]) => ({
          from: '#' + color1.toString(16).padStart(8, '0'),
          to: '#' + color2.toString(16).padStart(8, '0'),
          fromInfo: paletteInfo.colorMaps.frame1.get(color1),
          toInfo: paletteInfo.colorMaps.frame2.get(color2)
        }))
      });

      return true;
    } catch (error) {
      console.error('Color palette analysis test failed:', error);
      return false;
    }
  };

  // Update detectConnectedRegions to handle tiny regions better
  ns.FlowNetService.prototype.detectConnectedRegions = function(frame) {
    const pixels = frame.getPixels();
    const width = frame.getWidth();
    const height = frame.getHeight();
    const regions = new Map();
    const visited = new Set();

    // Simplified color similarity check focused on pixel art
    const areSimilarColors = (color1, color2) => {
      if (color1 === color2) return true; // Exact match
      if (color1 === 0 || color2 === 0) return false; // Skip transparent

      const r1 = color1 & 0xFF;
      const g1 = (color1 >> 8) & 0xFF;
      const b1 = (color1 >> 16) & 0xFF;
      
      const r2 = color2 & 0xFF;
      const g2 = (color2 >> 8) & 0xFF;
      const b2 = (color2 >> 16) & 0xFF;

      // Check if colors are in same family
      const getColorFamily = (r, g, b) => {
        if (Math.abs(r - g) < 30 && Math.abs(g - b) < 30) return 'gray';
        const max = Math.max(r, g, b);
        if (max === r) return 'red';
        if (max === g) return 'green';
        return 'blue';
      };

      // More permissive matching
      const colorFamily1 = getColorFamily(r1, g1, b1);
      const colorFamily2 = getColorFamily(r2, g2, b2);
      
      if (colorFamily1 === colorFamily2) return true;

      // Check luminance for shading
      const getLuminance = (r, g, b) => (0.299 * r + 0.587 * g + 0.114 * b);
      const lum1 = getLuminance(r1, g1, b1);
      const lum2 = getLuminance(r2, g2, b2);
      
      return Math.abs(lum1 - lum2) < 60; // More permissive luminance threshold
    };

    // Simplified flood fill
    const floodFill = (startX, startY, baseColor) => {
      const region = [];
      const stack = [{x: startX, y: startY}];
      
      while (stack.length > 0) {
        const {x, y} = stack.pop();
        const pos = y * width + x;
        
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        if (visited.has(pos)) continue;
        
        const currentColor = pixels[pos];
        if (!areSimilarColors(currentColor, baseColor)) continue;
        
        visited.add(pos);
        region.push({x, y});
        
        // Check neighbors (including diagonals)
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            stack.push({x: x + dx, y: y + dy});
          }
        }
      }
      
      return {
        pixels: region,
        size: region.length,
        color: baseColor
      };
    };

    // First pass: collect all regions
    const allRegions = [];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pos = y * width + x;
        const color = pixels[pos];
        
        if (color === 0 || visited.has(pos)) continue;
        
        const region = floodFill(x, y, color);
        allRegions.push(region);
      }
    }

    // Sort regions by size (largest first)
    allRegions.sort((a, b) => b.size - a.size);

    // Separate tiny and normal regions
    const tinyRegions = allRegions.filter(r => r.size < 10);
    const normalRegions = allRegions.filter(r => r.size >= 10);

    // First, create groups from normal regions
    const mergedRegions = new Map();
    for (const region of normalRegions) {
      let merged = false;
      
      // Try to merge with existing groups
      for (const [existingColor, existingRegions] of mergedRegions) {
        if (this.areSimilarColors(region.color, existingColor)) {
          existingRegions.push(region);
          merged = true;
          break;
        }
      }
      
      // If no merge, create new group
      if (!merged) {
        mergedRegions.set(region.color, [region]);
      }
    }

    // Then, merge tiny regions into the closest normal region
    for (const tinyRegion of tinyRegions) {
      let bestMatch = null;
      let minDistance = Infinity;
      let bestColor = null;

      // Find closest normal region
      for (const [color, regions] of mergedRegions) {
        for (const region of regions) {
          // Calculate distance between region centers
          const tinyCenter = {
            x: tinyRegion.pixels.reduce((sum, p) => sum + p.x, 0) / tinyRegion.pixels.length,
            y: tinyRegion.pixels.reduce((sum, p) => sum + p.y, 0) / tinyRegion.pixels.length
          };
          
          const regionCenter = {
            x: region.pixels.reduce((sum, p) => sum + p.x, 0) / region.pixels.length,
            y: region.pixels.reduce((sum, p) => sum + p.y, 0) / region.pixels.length
          };

          const dist = Math.abs(tinyCenter.x - regionCenter.x) + 
                      Math.abs(tinyCenter.y - regionCenter.y);

          if (dist < minDistance) {
            minDistance = dist;
            bestMatch = region;
            bestColor = color;
          }
        }
      }

      // Always merge tiny region with its closest neighbor
      if (bestMatch) {
        mergedRegions.get(bestColor).push(tinyRegion);
      } else {
        // If no normal regions exist, create a new group
        mergedRegions.set(tinyRegion.color, [tinyRegion]);
      }
    }

    return mergedRegions;
  };

  // Update test method for connected region detection
  ns.FlowNetService.prototype.testConnectedRegions = function(frame) {
    try {
      const regions = this.detectConnectedRegions(frame);
      
      // Detailed analysis
      const analysis = {
        'Total Region Groups': regions.size,
        'Detailed Regions': Array.from(regions.entries()).map(([color, regionList]) => {
          const totalPixels = regionList.reduce((sum, region) => sum + region.pixels.length, 0);
          return {
            color: '#' + color.toString(16).padStart(8, '0'),
            numberOfRegions: regionList.length,
            totalPixels: totalPixels,
            largestRegion: regionList.reduce((largest, region) => 
              region.pixels.length > largest.pixels.length ? region : largest, 
              regionList[0]
            ),
            regions: regionList.map(region => ({
              size: region.pixels.length,
              uniqueColors: region.colors.length,
              bounds: {
                x: Math.min(...region.pixels.map(p => p.x)),
                y: Math.min(...region.pixels.map(p => p.y)),
                width: Math.max(...region.pixels.map(p => p.x)) - Math.min(...region.pixels.map(p => p.x)) + 1,
                height: Math.max(...region.pixels.map(p => p.y)) - Math.min(...region.pixels.map(p => p.y)) + 1
              }
            }))
          };
        }).sort((a, b) => b.totalPixels - a.totalPixels)
      };

      // Additional statistics
      const statistics = {
        'Average Region Size': Array.from(regions.values())
          .flat()
          .reduce((sum, region) => sum + region.pixels.length, 0) / 
          Array.from(regions.values()).flat().length,
        'Total Regions': Array.from(regions.values()).flat().length,
        'Color Groups': regions.size
      };

      console.log('Connected Regions Analysis:', analysis);
      console.log('Region Statistics:', statistics);
      
      return true;
    } catch (error) {
      console.error('Connected region detection failed:', error);
      return false;
    }
  };

  // Update areSimilarColors to be more balanced
  ns.FlowNetService.prototype.areSimilarColors = function(color1, color2) {
    const r1 = color1 & 0xFF;
    const g1 = (color1 >> 8) & 0xFF;
    const b1 = (color1 >> 16) & 0xFF;
    const a1 = (color1 >> 24) & 0xFF;
    
    const r2 = color2 & 0xFF;
    const g2 = (color2 >> 8) & 0xFF;
    const b2 = (color2 >> 16) & 0xFF;
    const a2 = (color2 >> 24) & 0xFF;
    
    // Skip transparent pixels
    if (a1 === 0 || a2 === 0) return false;
    
    // Exact match
    if (color1 === color2) return true;
    
    // Get luminance
    const getLuminance = (r, g, b) => (0.299 * r + 0.587 * g + 0.114 * b);
    const lum1 = getLuminance(r1, g1, b1);
    const lum2 = getLuminance(r2, g2, b2);
    
    // More balanced color matching
    const colorDiff = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);
    const lumDiff = Math.abs(lum1 - lum2) / 255;
    
    // Balanced shading check
    const isShading = (
      colorDiff < 150 && // More balanced threshold
      lumDiff < 0.4 &&   // More balanced luminance steps
      Math.max(
        Math.abs(r1 - r2),
        Math.abs(g1 - g2),
        Math.abs(b1 - b2)
      ) < 80 // More balanced channel differences
    );
    
    // Get color family
    const getColorFamily = (r, g, b) => {
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      
      // Balanced gray detection
      if (max - min < 40) return 'gray';
      
      // Get dominant channel
      if (r > g + 30 && r > b + 30) return 'red';
      if (g > r + 30 && g > b + 30) return 'green';
      if (b > r + 30 && b > g + 30) return 'blue';
      return 'mixed';
    };
    
    const family1 = getColorFamily(r1, g1, b1);
    const family2 = getColorFamily(r2, g2, b2);
    
    return isShading || family1 === family2;
  };

  // Update mergeAdjacentRegions to be more balanced
  ns.FlowNetService.prototype.mergeAdjacentRegions = function(regions) {
    const merged = new Map();
    const processed = new Set();
    
    // Helper to check if regions are adjacent or close
    const areClose = (region1, region2) => {
      // For small regions, use a moderate proximity threshold
      if (region1.pixels.length < 10 || region2.pixels.length < 10) {
        const center1 = {
          x: region1.pixels.reduce((sum, p) => sum + p.x, 0) / region1.pixels.length,
          y: region1.pixels.reduce((sum, p) => sum + p.y, 0) / region1.pixels.length
        };
        
        const center2 = {
          x: region2.pixels.reduce((sum, p) => sum + p.x, 0) / region2.pixels.length,
          y: region2.pixels.reduce((sum, p) => sum + p.y, 0) / region2.pixels.length
        };
        
        const dist = Math.abs(center1.x - center2.x) + Math.abs(center1.y - center2.y);
        return dist < 8; // More balanced threshold
      }
      
      // For larger regions, check pixel proximity
      for (const p1 of region1.pixels) {
        for (const p2 of region2.pixels) {
          const dx = Math.abs(p1.x - p2.x);
          const dy = Math.abs(p1.y - p2.y);
          if (dx <= 2 && dy <= 2) return true;
        }
      }
      return false;
    };
    
    // Convert to array and sort by size
    const regionsList = Array.from(regions.entries())
      .sort((a, b) => b[1].reduce((sum, r) => sum + r.pixels.length, 0) - 
                      a[1].reduce((sum, r) => sum + r.pixels.length, 0));
    
    // Process regions, starting with largest
    for (const [color1, regions1] of regionsList) {
      if (processed.has(color1)) continue;
      
      let mergedGroup = [...regions1];
      processed.add(color1);
      
      // Keep merging until no more matches found
      let changed = true;
      while (changed) {
        changed = false;
        
        for (const [color2, regions2] of regionsList) {
          if (processed.has(color2)) continue;
          
          // Check proximity and color similarity
          const shouldMerge = regions1.some(r1 => 
            regions2.some(r2 => 
              areClose(r1, r2) && this.areSimilarColors(color1, color2)
            )
          );
          
          if (shouldMerge) {
            mergedGroup = mergedGroup.concat(regions2);
            processed.add(color2);
            changed = true;
          }
        }
      }
      
      merged.set(color1, mergedGroup);
    }
    
    return merged;
  };
})(); 