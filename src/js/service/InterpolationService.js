(function () {
  var ns = $.namespace('pskl.service');

  ns.InterpolationService = function () {
    this.isModelLoaded = false;
  };

  ns.InterpolationService.prototype.init = async function () {
    try {
      // Check for WebGL support
      if (tf.getBackend() !== 'webgl') {
        try {
          // Configure WebGL before setting it as backend
          tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', -1); // Prevent duplicate kernel registration
          await tf.setBackend('webgl');
          console.log('Successfully enabled WebGL backend');
        } catch (e) {
          console.warn('WebGL not available, falling back to CPU:', e);
          await tf.setBackend('cpu');
        }
      }

      // Set WebGL flags for better performance
      if (tf.getBackend() === 'webgl') {
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        tf.env().set('WEBGL_PACK', true);
      }

      console.log('Active backend:', tf.getBackend());
      this.isModelLoaded = true;
    } catch (error) {
      console.error('Failed to initialize interpolation service:', error);
      throw error;
    }
  };

  ns.InterpolationService.prototype.frameToTensor = function (frame) {
    return tf.tidy(() => {
      const width = frame.getWidth();
      const height = frame.getHeight();
      const pixels = frame.getPixels();
      
      // Create RGBA data array
      const data = new Float32Array(width * height * 4);
      for (let i = 0; i < pixels.length; i++) {
        const color = pixels[i];
        // Extract ABGR components (Piskel's format)
        const a = (color >>> 24) & 0xFF;
        const b = (color >>> 16) & 0xFF;
        const g = (color >>> 8) & 0xFF;
        const r = color & 0xFF;

        // Convert to normalized values
        data[i * 4] = r / 255;     // R
        data[i * 4 + 1] = g / 255; // G
        data[i * 4 + 2] = b / 255; // B
        data[i * 4 + 3] = a / 255; // A
      }
      
      return tf.tensor3d(data, [height, width, 4]);
    });
  };

  ns.InterpolationService.prototype.tensorToFrame = function (tensor, width, height) {
    // Convert tensor back to Piskel frame
    const data = tensor.dataSync();
    const pixels = new Uint32Array(width * height);
    
    // Create canvas with willReadFrequently flag
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    for (let i = 0; i < pixels.length; i++) {
      // Get RGBA values and convert back to 0-255 range
      const r = Math.round(data[i * 4] * 255);
      const g = Math.round(data[i * 4 + 1] * 255);
      const b = Math.round(data[i * 4 + 2] * 255);
      const a = Math.round(data[i * 4 + 3] * 255);

      // Only set pixel if it's not fully transparent
      if (a > 0) {
        // Combine into ABGR format (Piskel's format)
        pixels[i] = 
          (a << 24) |  // Alpha in highest byte
          (b << 16) |  // Blue
          (g << 8) |   // Green
          r;           // Red in lowest byte
      } else {
        pixels[i] = 0; // Fully transparent pixel
      }
    }
    
    const frame = new pskl.model.Frame(width, height);
    frame.setPixels(pixels);
    return frame;
  };

  ns.InterpolationService.prototype.interpolateFrames = async function (frame1, frame2, numFrames) {
    if (!this.isModelLoaded) {
      throw new Error('Model not loaded');
    }

    const frames = [];
    const width = frame1.getWidth();
    const height = frame1.getHeight();

    // Convert both frames to tensors
    const tensor1 = this.frameToTensor(frame1);
    const tensor2 = this.frameToTensor(frame2);

    try {
      // Generate intermediate frames
      for (let i = 1; i <= numFrames; i++) {
        const t = i / (numFrames + 1);
        
        // Use tf.tidy for automatic memory management
        const newTensor = tf.tidy(() => {
          // Split alpha and RGB channels
          const [rgb1, a1] = tf.split(tensor1, [3, 1], -1);
          const [rgb2, a2] = tf.split(tensor2, [3, 1], -1);

          // Create binary masks for non-zero alpha
          const mask1 = tf.greater(a1, 0);
          const mask2 = tf.greater(a2, 0);

          // Interpolate RGB where both pixels are visible
          const bothVisible = tf.logicalAnd(mask1, mask2);
          const rgbInterp = tf.where(
            bothVisible,
            rgb1.mul(1 - t).add(rgb2.mul(t)),
            tf.where(mask1, rgb1, rgb2)
          );

          // Handle alpha with sharp transition at t=0.5
          const alpha = tf.where(
            bothVisible,
            tf.maximum(a1, a2),
            tf.where(
              tf.less(t, 0.5),
              tf.where(mask1, a1, tf.zeros(a1.shape)),
              tf.where(mask2, a2, tf.zeros(a2.shape))
            )
          );

          // Combine channels
          return tf.concat([rgbInterp, alpha], -1);
        });

        // Convert interpolated tensor back to frame
        const newFrame = this.tensorToFrame(newTensor, width, height);
        frames.push(newFrame);

        // Clean up intermediate tensor
        newTensor.dispose();
      }
    } finally {
      // Clean up input tensors
      tensor1.dispose();
      tensor2.dispose();
    }

    return frames;
  };

  // Add test method to verify frame-tensor conversions
  ns.InterpolationService.prototype.testFrameConversion = function (frame) {
    console.log('Testing frame-tensor conversion...');
    
    // Get original pixels for comparison
    const originalPixels = frame.getPixels();
    let firstNonZeroPixel = null;
    let pixelIndex = -1;

    // Find first non-zero pixel for detailed comparison
    for (let i = 0; i < originalPixels.length; i++) {
      if (originalPixels[i] !== 0) {
        firstNonZeroPixel = originalPixels[i];
        pixelIndex = i;
        break;
      }
    }

    if (firstNonZeroPixel === null) {
      console.log('No non-zero pixels found in frame');
      return;
    }

    // Log original pixel values
    console.log('Original pixel:', {
      index: pixelIndex,
      x: pixelIndex % frame.getWidth(),
      y: Math.floor(pixelIndex / frame.getWidth()),
      hex: firstNonZeroPixel.toString(16),
      components: {
        a: (firstNonZeroPixel >>> 24) & 0xFF,
        b: (firstNonZeroPixel >>> 16) & 0xFF,
        g: (firstNonZeroPixel >>> 8) & 0xFF,
        r: firstNonZeroPixel & 0xFF
      }
    });

    // Convert to tensor and back
    const tensor = this.frameToTensor(frame);
    const convertedFrame = this.tensorToFrame(tensor, frame.getWidth(), frame.getHeight());
    const convertedPixels = convertedFrame.getPixels();

    // Log converted pixel values
    console.log('Converted pixel:', {
      hex: convertedPixels[pixelIndex].toString(16),
      components: {
        a: (convertedPixels[pixelIndex] >>> 24) & 0xFF,
        b: (convertedPixels[pixelIndex] >>> 16) & 0xFF,
        g: (convertedPixels[pixelIndex] >>> 8) & 0xFF,
        r: convertedPixels[pixelIndex] & 0xFF
      }
    });

    // Compare all pixels
    let mismatchCount = 0;
    for (let i = 0; i < originalPixels.length; i++) {
      if (originalPixels[i] !== convertedPixels[i]) {
        mismatchCount++;
      }
    }

    console.log('Conversion test results:', {
      totalPixels: originalPixels.length,
      mismatchCount,
      tensorShape: tensor.shape,
      success: mismatchCount === 0
    });

    // Clean up
    tensor.dispose();
  };
})(); 