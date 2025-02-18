(function () {
  var ns = $.namespace('pskl.service');

  ns.InterpolationService = function () {
    this.model = null;
    this.isModelLoaded = false;
  };

  ns.InterpolationService.prototype.init = async function () {
    try {
      await tf.setBackend('webgl');
      await this.loadFlowNetModel();
    } catch (error) {
      console.error('Failed to initialize interpolation service:', error);
    }
  };

  ns.InterpolationService.prototype.loadFlowNetModel = async function () {
    try {
      // Create a simple test model for development
      const model = tf.sequential();
      
      // Input shape: [height, width, channels * 2] (two frames concatenated)
      model.add(tf.layers.conv2d({
        inputShape: [null, null, 8],  // RGBA for two frames
        filters: 16,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
      }));
      
      model.add(tf.layers.conv2d({
        filters: 4,  // Output RGBA
        kernelSize: 3,
        padding: 'same',
        activation: 'sigmoid'
      }));

      this.model = model;
      this.isModelLoaded = true;
      console.log('Test model created successfully');
    } catch (error) {
      console.error('Failed to create test model:', error);
      this.isModelLoaded = false;
    }
  };

  ns.InterpolationService.prototype.frameToTensor = function (frame) {
    // Convert Piskel frame to tensor
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
  };

  ns.InterpolationService.prototype.tensorToFrame = function (tensor, width, height) {
    // Convert tensor back to Piskel frame
    const data = tensor.dataSync();
    const pixels = new Uint32Array(width * height);
    
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
    const pixels1 = frame1.getPixels();
    const pixels2 = frame2.getPixels();

    // Generate intermediate frames
    for (let i = 1; i <= numFrames; i++) {
      const t = i / (numFrames + 1);
      const newFrame = new pskl.model.Frame(width, height);
      const newPixels = new Uint32Array(width * height);

      // Clear all pixels first
      newPixels.fill(0);

      for (let p = 0; p < pixels1.length; p++) {
        const color1 = pixels1[p];
        const color2 = pixels2[p];

        // Skip fully transparent pixels
        if (color1 === 0 && color2 === 0) {
          continue;
        }

        // Extract components
        const a1 = (color1 >>> 24) & 0xFF;
        const b1 = (color1 >>> 16) & 0xFF;
        const g1 = (color1 >>> 8) & 0xFF;
        const r1 = color1 & 0xFF;

        const a2 = (color2 >>> 24) & 0xFF;
        const b2 = (color2 >>> 16) & 0xFF;
        const g2 = (color2 >>> 8) & 0xFF;
        const r2 = color2 & 0xFF;

        // Handle different cases
        if (a1 > 0 && a2 > 0) {
          // Both pixels are visible - do normal interpolation
          const r = Math.round(r1 + (r2 - r1) * t);
          const g = Math.round(g1 + (g2 - g1) * t);
          const b = Math.round(b1 + (b2 - b1) * t);
          const a = Math.max(a1, a2); // Keep full opacity

          newPixels[p] = (a << 24) | (b << 16) | (g << 8) | r;
        } else if (a1 > 0) {
          // Pixel only in first frame - sharp transition
          if (t < 0.5) {
            newPixels[p] = color1; // Keep original color and alpha
          }
        } else if (a2 > 0) {
          // Pixel only in second frame - sharp transition
          if (t >= 0.5) {
            newPixels[p] = color2; // Keep original color and alpha
          }
        }
      }

      newFrame.setPixels(newPixels);
      frames.push(newFrame);
    }

    return frames;
  };

  ns.InterpolationService.prototype.preprocessFrames = function (tensor1, tensor2) {
    // FlowNet preprocessing:
    // 1. Normalize to [-1, 1]
    // 2. Concatenate frames
    const normalized1 = tensor1.sub(0.5).mul(2);
    const normalized2 = tensor2.sub(0.5).mul(2);
    return tf.concat([normalized1, normalized2], 2).expandDims(0);
  };

  ns.InterpolationService.prototype.warpFrame = function (frame1, frame2, flow, t) {
    // Implement frame warping using the optical flow
    // This is where the actual frame interpolation happens
    return tf.tidy(() => {
      // Create sampling grid
      const [height, width] = frame1.shape;
      const [gridX, gridY] = tf.meshgrid(
        tf.linspace(0, width - 1, width),
        tf.linspace(0, height - 1, height)
      );

      // Compute intermediate positions using flow
      const pos_x = gridX.sub(flow.slice([0,0,0], [-1,-1,1]).mul(t));
      const pos_y = gridY.sub(flow.slice([0,0,1], [-1,-1,1]).mul(t));

      // Sample from both frames and blend
      const warped1 = this.bilinearSample(frame1, pos_x, pos_y);
      const warped2 = this.bilinearSample(frame2, pos_x.add(flow.slice([0,0,0], [-1,-1,1])), 
                                                  pos_y.add(flow.slice([0,0,1], [-1,-1,1])));

      return warped1.mul(1-t).add(warped2.mul(t));
    });
  };

  ns.InterpolationService.prototype.bilinearSample = function (frame, pos_x, pos_y) {
    // Implement bilinear sampling
    return tf.tidy(() => {
      const [height, width] = frame.shape;
      const x0 = tf.floor(pos_x);
      const y0 = tf.floor(pos_y);
      const x1 = x0.add(1);
      const y1 = y0.add(1);

      const q00 = frame.gather([y0, x0]);
      const q01 = frame.gather([y0, x1]);
      const q10 = frame.gather([y1, x0]);
      const q11 = frame.gather([y1, x1]);

      const wx0 = pos_x.sub(x0).mul(q01.sub(q00)).add(q00);
      const wx1 = pos_x.sub(x0).mul(q11.sub(q10)).add(q10);
      const wy0 = pos_y.sub(y0).mul(wx1.sub(wx0)).add(wx0);

      return wy0;
    });
  };
})(); 