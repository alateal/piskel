(function () {
  var ns = $.namespace('pskl.service');

  ns.InterpolationService = function () {
    console.log('InterpolationService constructor called');
    this.flowNetService = new pskl.service.FlowNetService();
    this.isModelLoaded = false;
  };

  ns.InterpolationService.prototype.init = async function () {
    console.log('InterpolationService init started');
    try {
      if (!this.flowNetService) {
        this.flowNetService = new pskl.service.FlowNetService();
      }
      await this.flowNetService.init();
      this.isModelLoaded = true;
      console.log('InterpolationService initialized successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize interpolation:', error);
      this.isModelLoaded = false;
      // Don't throw, just log the error to prevent app initialization failure
      return false;
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
        
        // Extract ABGR components and normalize to 0-1 range
        const a = ((color >>> 24) & 0xFF) / 255;
        const b = ((color >>> 16) & 0xFF) / 255;
        const g = ((color >>> 8) & 0xFF) / 255;
        const r = (color & 0xFF) / 255;

        // Store as RGBA
        data[i * 4] = r;     // R
        data[i * 4 + 1] = g; // G
        data[i * 4 + 2] = b; // B
        data[i * 4 + 3] = a; // A
      }
      
      // Create tensor with proper shape
      return tf.tensor3d(data, [height, width, 4]);
    });
  };

  ns.InterpolationService.prototype.tensorToFrame = function (tensor, width, height) {
    // Convert tensor back to Piskel frame
    const data = tensor.dataSync();
    const pixels = new Uint32Array(width * height);
    
    for (let i = 0; i < pixels.length; i++) {
      // Get RGBA values and convert back to 0-255 range
      const r = Math.round(Math.max(0, Math.min(255, data[i * 4] * 255)));
      const g = Math.round(Math.max(0, Math.min(255, data[i * 4 + 1] * 255)));
      const b = Math.round(Math.max(0, Math.min(255, data[i * 4 + 2] * 255)));
      const a = Math.round(Math.max(0, Math.min(255, data[i * 4 + 3] * 255)));

      // Only set pixel if it has some opacity
      if (a > 0) {
        // Combine into ABGR format (Piskel's format)
        pixels[i] = 
          ((a & 0xFF) << 24) |  // Alpha
          ((b & 0xFF) << 16) |  // Blue
          ((g & 0xFF) << 8)  |  // Green
          (r & 0xFF);           // Red
      } else {
        // Fully transparent pixels
        pixels[i] = 0;
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

    // Add detailed debug logging
    const debugFrame = (frame, label) => {
      const pixels = frame.getPixels();
      const nonZeroPixel = pixels.find(p => p !== 0) || 0;
      console.log(`${label} color sample:`, this.debugPixelColor(nonZeroPixel));
    };

    debugFrame(frame1, 'Frame 1');
    debugFrame(frame2, 'Frame 2');

    console.log('Starting frame interpolation:', {
      frame1Width: frame1.getWidth(),
      frame1Height: frame1.getHeight(),
      frame2Width: frame2.getWidth(),
      frame2Height: frame2.getHeight(),
      numFrames: numFrames
    });

    const frames = [];
    let flow = null;
    try {
      // Compute optical flow
      flow = await this.flowNetService.computeFlow(frame1, frame2);
      console.log('Flow computed:', {
        shape: flow.shape,
        dtype: flow.dtype
      });

      // Generate intermediate frames
      for (let i = 1; i <= numFrames; i++) {
        const t = i / (numFrames + 1);
        console.log('Generating frame', i, 'at t =', t);
        
        // Warp frames using flow
        const warped1 = await this.flowNetService.warpFrame(frame1, flow, t);
        debugFrame(warped1, `Warped frame1 at t=${t}`);
        
        const warped2 = await this.flowNetService.warpFrame(frame2, flow, 1 - t);
        debugFrame(warped2, `Warped frame2 at t=${1-t}`);
        
        // Blend warped frames
        const blendedFrame = this.blendFrames(warped1, warped2, t);
        debugFrame(blendedFrame, `Blended frame at t=${t}`);
        
        frames.push(blendedFrame);
      }
    } catch (error) {
      console.error('Error during interpolation:', error);
      throw error;
    } finally {
      if (flow) flow.dispose();
    }

    return frames;
  };

  ns.InterpolationService.prototype.blendFrames = function(frame1, frame2, t) {
    return tf.tidy(() => {
      // Convert frames to tensors using our color-aware method
      const tensor1 = this.frameToTensor(frame1);
      const tensor2 = this.frameToTensor(frame2);
      
      // Linear interpolation
      const blended = tensor1.mul(1 - t).add(tensor2.mul(t));
      
      // Convert back using our color-aware method
      return this.tensorToFrame(blended, frame1.getWidth(), frame1.getHeight());
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

  // Add this helper method to debug color values
  ns.InterpolationService.prototype.debugPixelColor = function(pixel) {
    return {
      r: pixel & 0xFF,
      g: (pixel >> 8) & 0xFF,
      b: (pixel >> 16) & 0xFF,
      a: (pixel >> 24) & 0xFF,
      hex: '#' + pixel.toString(16).padStart(8, '0')
    };
  };
})(); 