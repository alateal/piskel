(function () {
  var ns = $.namespace('pskl.service');

  ns.InterpolationService = function () {
    this.flowNetService = new pskl.service.FlowNetService();
    this.isModelLoaded = false;
  };

  ns.InterpolationService.prototype.init = async function () {
    try {
      await this.flowNetService.init();
      this.isModelLoaded = true;
    } catch (error) {
      console.error('Failed to initialize interpolation:', error);
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
      throw new Error('Service not initialized');
    }

    console.log('Starting frame interpolation:', {
      frame1: frame1,
      frame2: frame2,
      numFrames: numFrames
    });

    const frames = [];
    let flow = null;
    try {
      // Compute optical flow
      flow = await this.flowNetService.computeFlow(frame1, frame2);
      console.log('Flow computed:', flow);

      // Generate intermediate frames
      for (let i = 1; i <= numFrames; i++) {
        const t = i / (numFrames + 1);
        console.log('Generating frame', i, 'at t =', t);
        
        // Warp frames using flow
        const warped1 = await this.flowNetService.warpFrame(frame1, flow, t);
        const warped2 = await this.flowNetService.warpFrame(frame2, flow, 1 - t);
        
        // Blend warped frames
        const blendedFrame = this.blendFrames(warped1, warped2, t);
        frames.push(blendedFrame);
      }
    } catch (error) {
      console.error('Error during interpolation:', error);
      throw error;
    } finally {
      // Cleanup
      if (flow) flow.dispose();
    }

    console.log('Interpolation complete, generated frames:', frames);
    return frames;
  };

  ns.InterpolationService.prototype.blendFrames = function(frame1, frame2, t) {
    return tf.tidy(() => {
      // Convert frames to tensors
      const tensor1 = this.flowNetService.preprocessFrame(frame1);
      const tensor2 = this.flowNetService.preprocessFrame(frame2);
      
      // Linear interpolation
      const blended = tensor1.mul(1 - t).add(tensor2.mul(t));
      
      // Convert back to frame
      return this.flowNetService.postprocessFrame(blended);
    });
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