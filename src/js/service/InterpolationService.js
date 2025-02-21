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
    
    // Create a new frame and properly initialize it
    const frame = new pskl.model.Frame(width, height);
    frame.setPixels(pixels);
    // Ensure frame has a valid hash
    frame.version = 1;
    frame.hashValue = frame.getHash();
    return frame;
  };

  ns.InterpolationService.prototype.interpolateFrames = async function (frame1, frame2, numFrames) {
    const frames = [];
    try {
      // Test RIFE server connection with proper error handling
      try {
        const response = await fetch('http://localhost:8000/health');
        const data = await response.json();
        
        if (response.ok && data.status === 'ok' && data.model_loaded) {
          console.log('RIFE server available, using RIFE for interpolation');
          return await this.interpolateWithRIFE(frame1, frame2);
        } else {
          console.log('RIFE server available but model not loaded:', data);
          throw new Error('RIFE model not loaded');
        }
      } catch (error) {
        console.log('RIFE server not available:', error);
        return await this.interpolateWithTensorFlow(frame1, frame2, numFrames);
      }
    } catch (error) {
      console.error('Error during interpolation:', error);
      throw error;
    }
  };

  // Update the interpolateWithRIFE method with better error handling
  ns.InterpolationService.prototype.interpolateWithRIFE = async function (frame1, frame2) {
    try {
      // Extract color palette from source frames
      const frame1Palette = this.extractColorPalette(frame1);
      const frame2Palette = this.extractColorPalette(frame2);
      // Combine palettes
      const combinedPalette = [...new Set([...frame1Palette, ...frame2Palette])];

      // Convert frames to blobs
      const blob1 = await this.frameToBlob(frame1);
      const blob2 = await this.frameToBlob(frame2);

      // Create form data
      const formData = new FormData();
      formData.append('frame1', blob1);
      formData.append('frame2', blob2);
      formData.append('time_step', '0.5');

      console.log('Sending request to RIFE server...');
      
      // Send request to RIFE server (updated port to 8000)
      const response = await fetch('http://localhost:8000/interpolate', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('RIFE server response:', {
          status: response.status,
          statusText: response.statusText,
          headers: Object.fromEntries(response.headers),
          error: errorText
        });
        throw new Error(`RIFE server error: ${response.statusText} (${errorText})`);
      }

      const blob = await response.blob();
      
      // Convert response blob back to frame with original size and palette
      return await this.blobToFrame(blob, {
        width: frame1.getWidth(),
        height: frame1.getHeight()
      }, combinedPalette);

    } catch (error) {
      console.error('Error in RIFE interpolation:', error);
      throw error;
    }
  };

  // Add new method to apply movement interpolation
  ns.InterpolationService.prototype.applyMovementInterpolation = async function(frame, movement, t, originalSize) {
    // Create a new frame for the result
    const result = new pskl.model.Frame(originalSize.width, originalSize.height);
    const pixels = new Uint32Array(originalSize.width * originalSize.height);
    
    // Calculate interpolated position
    const ease = this.easeInOutQuad(t);
    const currentOffset = {
        x: Math.round(movement.dx * ease),
        y: Math.round(movement.dy * ease)
    };
    
    // Get frame pixels
    const sourcePixels = frame.getPixels();
    
    // Apply movement to each pixel
    for (let y = 0; y < originalSize.height; y++) {
        for (let x = 0; x < originalSize.width; x++) {
            const destPos = y * originalSize.width + x;
            
            // Calculate source position with offset
            const srcX = x - currentOffset.x;
            const srcY = y - currentOffset.y;
            
            // Check if source position is within bounds
            if (srcX >= 0 && srcX < originalSize.width && 
                srcY >= 0 && srcY < originalSize.height) {
                const srcPos = srcY * originalSize.width + srcX;
                pixels[destPos] = sourcePixels[srcPos];
            } else {
                pixels[destPos] = 0; // Transparent if out of bounds
            }
        }
    }
    
    result.setPixels(pixels);
    return result;
  };

  // Update analyzeSpriteDifference to be more accurate
  ns.InterpolationService.prototype.analyzeSpriteDifference = function(frame1, frame2) {
    const bounds1 = this.getSpriteBounds(frame1);
    const bounds2 = this.getSpriteBounds(frame2);
    
    // Calculate centers
    const center1 = {
        x: (bounds1.minX + bounds1.maxX) / 2,
        y: (bounds1.minY + bounds1.maxY) / 2
    };
    
    const center2 = {
        x: (bounds2.minX + bounds2.maxX) / 2,
        y: (bounds2.minY + bounds2.maxY) / 2
    };
    
    // Calculate movement vector
    const dx = Math.round(center2.x - center1.x);
    const dy = Math.round(center2.y - center1.y);
    
    console.log('Movement analysis:', {
        dx, dy,
        bounds1,
        bounds2,
        center1,
        center2
    });
    
    return {
        dx,
        dy,
        bounds1,
        bounds2,
        center1,
        center2
    };
  };

  // Move existing TensorFlow implementation to new method
  ns.InterpolationService.prototype.interpolateWithTensorFlow = async function (frame1, frame2, numFrames) {
    const frames = [];
    
    // Analyze sprite movement and transformation
    const movement = this.analyzeSpriteDifference(frame1, frame2);
    
    // Generate intermediate frames
    for (let i = 1; i <= numFrames; i++) {
      const t = i / (numFrames + 1);
      console.log('Generating frame with TensorFlow', i, 'at t =', t);
      
      // Create new frame
      const width = frame1.getWidth();
      const height = frame1.getHeight();
      const result = new pskl.model.Frame(width, height);
      const pixels = new Uint32Array(width * height);
      
      // Apply easing to make movement more natural
      const ease = this.easeInOutQuad(t);
      
      // Calculate current frame position and scale
      const currentOffset = {
        x: Math.round(movement.dx * ease),
        y: Math.round(movement.dy * ease)
      };
      
      // For each pixel in the output frame
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const pos = y * width + x;
          
          // Calculate source positions with movement
          const x1 = x - currentOffset.x;
          const y1 = y - currentOffset.y;
          
          // Get colors from both frames (with bounds checking)
          const color1 = (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) 
            ? frame1.getPixel(x1, y1) 
            : 0;
            
          const x2 = x - (movement.dx - currentOffset.x);
          const y2 = y - (movement.dy - currentOffset.y);
          
          const color2 = (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height)
            ? frame2.getPixel(x2, y2)
            : 0;
          
          // Use existing pixel color determination logic
          pixels[pos] = this.determinePixelColor(color1, color2, ease);
        }
      }
      
      result.setPixels(pixels);
      frames.push(result);
    }
    
    return frames;
  };

  // Add helper method for pixel color determination
  ns.InterpolationService.prototype.determinePixelColor = function(color1, color2, ease) {
    // Both transparent
    if (color1 === 0 && color2 === 0) {
      return 0;
    }
    
    // Handle transitioning pixels
    if (color1 === 0) {
      // Fade in color2
      const alpha = ((color2 >> 24) & 0xFF) * ease;
      return (Math.round(alpha) << 24) | (color2 & 0x00FFFFFF);
    }
    
    if (color2 === 0) {
      // Fade out color1
      const alpha = ((color1 >> 24) & 0xFF) * (1 - ease);
      return (Math.round(alpha) << 24) | (color1 & 0x00FFFFFF);
    }
    
    // For non-transparent pixels, use the color based on timing
    return ease > 0.5 ? color2 : color1;
  };

  ns.InterpolationService.prototype.blendFrames = function(frame1, frame2, t) {
    return tf.tidy(() => {
      const width = frame1.getWidth();
      const height = frame1.getHeight();
      
      // Get flow field from FlowNetService
      const flowNetService = pskl.app.flowNetService;
      const flow = flowNetService.computeFlow(frame1, frame2);
      
      // Create output frame
      const result = new pskl.model.Frame(width, height);
      const pixels = new Uint32Array(width * height);
      const confidence = new Float32Array(width * height);
      
      // First pass: Initial color selection with confidence map
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const pos = y * width + x;
          
          // Get flow vector at this position
          const flowVec = {
            x: flow.gather([y, x, 1]).dataSync()[0],
            y: flow.gather([y, x, 0]).dataSync()[0]
          };
          
          // Calculate sample positions with flow
          const sampleX = Math.round(x + flowVec.x * t);
          const sampleY = Math.round(y + flowVec.y * t);
          
          // Get colors from both frames
          const color1 = frame1.getPixel(x, y);
          const color2 = frame2.getPixel(x, y);
          
          // Get sample colors (with bounds checking)
          const sampleColor1 = (sampleX >= 0 && sampleX < width && sampleY >= 0 && sampleY < height) 
            ? frame1.getPixel(sampleX, sampleY) 
            : color1;
            
          const sampleColor2 = (sampleX >= 0 && sampleX < width && sampleY >= 0 && sampleY < height)
            ? frame2.getPixel(sampleX, sampleY)
            : color2;
          
          // Handle transparency
          if (sampleColor1 === 0 && sampleColor2 === 0) {
            pixels[pos] = 0;
            confidence[pos] = 1;
            continue;
          }
          
          // Calculate color confidence
          const colorConfidence = this.calculateColorConfidence(
            sampleColor1, 
            sampleColor2,
            x, y,
            frame1, frame2
          );
          
          confidence[pos] = colorConfidence;
          
          // More conservative threshold for keeping original colors
          if (colorConfidence > 0.3) { // Lowered from 0.7
            pixels[pos] = t > 0.5 ? sampleColor2 : sampleColor1;
            continue;
          }
          
          // For very low confidence pixels, check if they're part of the main sprite
          const isPartOfSprite = this.isPartOfMainSprite(x, y, frame1, frame2);
          if (isPartOfSprite) {
            pixels[pos] = t > 0.5 ? sampleColor2 : sampleColor1;
            confidence[pos] = 0.8; // Boost confidence for sprite parts
            continue;
          }
          
          // Only make potential artifacts transparent
          if (colorConfidence < 0.2 && this.isBandingArtifact(x, y, pixels, confidence, width, height)) {
            pixels[pos] = 0;
          } else {
            pixels[pos] = t > 0.5 ? sampleColor2 : sampleColor1;
          }
        }
      }
      
      // Second pass: Only clean up definite artifacts
      const cleanedPixels = new Uint32Array(pixels);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const pos = y * width + x;
          
          if (pixels[pos] === 0) continue;
          
          // Only remove completely isolated pixels
          if (this.isCompletelyIsolated(x, y, pixels, width, height)) {
            cleanedPixels[pos] = 0;
            continue;
          }
        }
      }
      
      result.setPixels(cleanedPixels);
      return result;
    });
  };

  // Add helper method to calculate color confidence
  ns.InterpolationService.prototype.calculateColorConfidence = function(color1, color2, x, y, frame1, frame2) {
    // If colors are very similar, high confidence
    if (this.areColorsSimilar(color1, color2)) {
      return 1.0;
    }
    
    // Check if color exists in original frames near this position
    const radius = 2;
    let matchCount = 0;
    let totalChecks = 0;
    
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        
        if (nx >= 0 && nx < frame1.getWidth() && ny >= 0 && ny < frame1.getHeight()) {
          totalChecks += 2;
          if (this.areColorsSimilar(color1, frame1.getPixel(nx, ny))) matchCount++;
          if (this.areColorsSimilar(color2, frame2.getPixel(nx, ny))) matchCount++;
        }
      }
    }
    
    return matchCount / totalChecks;
  };

  // Add helper method to detect banding artifacts
  ns.InterpolationService.prototype.isBandingArtifact = function(x, y, pixels, confidence, width, height) {
    // Check for characteristic banding pattern
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    let bandingScore = 0;
    
    for (const [dx, dy] of directions) {
      const nx = x + dx;
      const ny = y + dy;
      
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const neighborPos = ny * width + nx;
        
        // Increase score if neighbor is also low confidence
        if (confidence[neighborPos] < 0.5) {
          bandingScore++;
        }
        
        // Check for alternating colors (characteristic of banding)
        if (pixels[neighborPos] !== 0 && 
            !this.areColorsSimilar(pixels[y * width + x], pixels[neighborPos])) {
          bandingScore++;
        }
      }
    }
    
    return bandingScore >= 3; // Threshold for banding detection
  };

  // Add helper method to detect isolated pixels
  ns.InterpolationService.prototype.isIsolatedPixel = function(x, y, pixels, width, height) {
    // Count non-transparent neighbors
    let neighbors = 0;
    
    // Check 8 surrounding pixels
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        
        const nx = x + dx;
        const ny = y + dy;
        
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          if (pixels[ny * width + nx] !== 0) {
            neighbors++;
          }
        }
      }
    }
    
    // Consider pixel isolated if it has fewer than 2 neighbors
    return neighbors < 2;
  };

  // Add helper method to detect edge pixels
  ns.InterpolationService.prototype.isEdgePixel = function(data, x, y, width, height) {
    const idx = (y * width + x) * 4;
    
    // Check immediate neighbors (4-connected)
    const neighbors = [
        [0, -1], // top
        [-1, 0], // left
        [1, 0],  // right
        [0, 1]   // bottom
    ];
    
    // Get center pixel values
    const centerR = data[idx];
    const centerG = data[idx + 1];
    const centerB = data[idx + 2];
    const centerA = data[idx + 3];
    
    if (centerA < 128) return false;
    
    for (const [dx, dy] of neighbors) {
      const nx = x + dx;
      const ny = y + dy;
      
        // Check bounds
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            return true; // Consider pixels at image boundaries as edges
        }
        
        const nidx = (ny * width + nx) * 4;
        const neighborA = data[nidx + 3];
        
        // If neighbor is transparent, this is an edge
        if (neighborA < 128) {
          return true;
        }
        
        // Check for significant color difference
        const dr = Math.abs(centerR - data[nidx]);
        const dg = Math.abs(centerG - data[nidx + 1]);
        const db = Math.abs(centerB - data[nidx + 2]);
        
        if (dr > 30 || dg > 30 || db > 30) {
            return true;
      }
    }
    
    return false;
  };

  // Add helper method to check if colors are similar
  ns.InterpolationService.prototype.areColorsSimilar = function(color1, color2) {
    const r1 = color1 & 0xFF;
    const g1 = (color1 >> 8) & 0xFF;
    const b1 = (color1 >> 16) & 0xFF;
    
    const r2 = color2 & 0xFF;
    const g2 = (color2 >> 8) & 0xFF;
    const b2 = (color2 >> 16) & 0xFF;
    
    // Calculate color difference using weighted components
    const rDiff = Math.abs(r1 - r2);
    const gDiff = Math.abs(g1 - g2);
    const bDiff = Math.abs(b1 - b2);
    
    // Use a stricter threshold for pixel art
    return (rDiff + gDiff + bDiff) < 30;
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

  // Add helper method to check if a pixel is part of the main sprite
  ns.InterpolationService.prototype.isPartOfMainSprite = function(x, y, frame1, frame2) {
    const radius = 2;
    let solidNeighbors = 0;
    
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        
        if (nx >= 0 && nx < frame1.getWidth() && ny >= 0 && ny < frame1.getHeight()) {
          if (frame1.getPixel(nx, ny) !== 0 || frame2.getPixel(nx, ny) !== 0) {
            solidNeighbors++;
          }
        }
      }
    }
    
    // Consider it part of the sprite if it has enough solid neighbors
    return solidNeighbors >= 4;
  };

  // Update to be more strict about what's considered isolated
  ns.InterpolationService.prototype.isCompletelyIsolated = function(x, y, pixels, width, height) {
    let neighbors = 0;
    
    // Check immediate neighbors only
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    
    for (const [dx, dy] of directions) {
      const nx = x + dx;
      const ny = y + dy;
      
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        if (pixels[ny * width + nx] !== 0) {
          neighbors++;
        }
      }
    }
    
    // Only consider completely isolated pixels (no immediate neighbors)
    return neighbors === 0;
  };

  // Add helper method to scale flow field
  ns.InterpolationService.prototype.scaleFlow = function(flow, scale) {
    return tf.tidy(() => {
      return flow.mul(tf.scalar(scale));
    });
  };

  // Add motion refinement method
  ns.InterpolationService.prototype.refineMotion = function(frame, frame1, frame2, t) {
    const width = frame.getWidth();
    const height = frame.getHeight();
    const result = new pskl.model.Frame(width, height);
    const pixels = new Uint32Array(width * height);
    
    // Create motion map for smoother transitions
    const motionMap = this.createMotionMap(frame1, frame2);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pos = y * width + x;
        const currentPixel = frame.getPixel(x, y);
        
        if (currentPixel === 0) {
          pixels[pos] = 0;
          continue;
        }
        
        // Get motion strength at this position
        const motionStrength = motionMap[pos];
        
        if (motionStrength > 0.2) { // Pixel is part of moving area
          // Calculate motion-adjusted position
          const progress = this.smoothstep(t); // Apply easing function
          const dx = Math.round(motionStrength * (x - width/2) * (progress - 0.5) * 2);
          const dy = Math.round(motionStrength * (y - height/2) * (progress - 0.5) * 2);
          
          // Sample from source or destination based on position
          const sourceX = x - dx;
          const sourceY = y - dy;
          
          if (sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height) {
            const sourcePixel = t < 0.5 ? frame1.getPixel(sourceX, sourceY) : frame2.getPixel(sourceX, sourceY);
            pixels[pos] = sourcePixel !== 0 ? sourcePixel : currentPixel;
          } else {
            pixels[pos] = currentPixel;
          }
        } else {
          pixels[pos] = currentPixel;
        }
      }
    }
    
    result.setPixels(pixels);
    return result;
  };

  // Add helper method to create motion map
  ns.InterpolationService.prototype.createMotionMap = function(frame1, frame2) {
    const width = frame1.getWidth();
    const height = frame1.getHeight();
    const motionMap = new Float32Array(width * height);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pos = y * width + x;
        const color1 = frame1.getPixel(x, y);
        const color2 = frame2.getPixel(x, y);
        
        if (color1 === 0 && color2 === 0) {
          motionMap[pos] = 0;
          continue;
        }
        
        // Calculate local motion strength
        let motionStrength = 0;
        const radius = 2;
        
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const neighborPos = ny * width + nx;
              const neighborColor1 = frame1.getPixel(nx, ny);
              const neighborColor2 = frame2.getPixel(nx, ny);
              
              if (neighborColor1 !== neighborColor2) {
                motionStrength += 1;
              }
            }
          }
        }
        
        motionMap[pos] = motionStrength / ((2 * radius + 1) * (2 * radius + 1));
      }
    }
    
    return motionMap;
  };

  // Add smoothstep easing function
  ns.InterpolationService.prototype.smoothstep = function(t) {
    // Smooth interpolation curve
    return t * t * (3 - 2 * t);
  };

  // Add motion-based color selection
  ns.InterpolationService.prototype.shouldUseColor2 = function(x, y, movement, t) {
    // Calculate which direction the sprite is moving
    const movingRight = movement.dx > 0;
    const movingDown = movement.dy > 0;
    
    // For horizontal movement
    if (Math.abs(movement.dx) > Math.abs(movement.dy)) {
      return movingRight ? (x >= movement.bounds1.maxX * t) : (x <= movement.bounds2.maxX * (1 - t));
    }
    
    // For vertical movement
    return movingDown ? (y >= movement.bounds1.maxY * t) : (y <= movement.bounds2.maxY * (1 - t));
  };

  // Add easing function for smoother transitions
  ns.InterpolationService.prototype.easeInOutQuad = function(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  };

  // Add helper method to get sprite bounds
  ns.InterpolationService.prototype.getSpriteBounds = function(frame) {
    const width = frame.getWidth();
    const height = frame.getHeight();
    
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (frame.getPixel(x, y) !== 0) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }
    
    return { minX, minY, maxX, maxY };
  };

  // Update frameToBlob to maintain higher resolution
  ns.InterpolationService.prototype.frameToBlob = async function (frame) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Use original dimensions directly
        const width = frame.getWidth();
        const height = frame.getHeight();
        
        // Set canvas to power-of-2 size for RIFE
        const targetSize = 256; // RIFE's expected size
        canvas.width = targetSize;
        canvas.height = targetSize;
        
        // Disable smoothing
        ctx.imageSmoothingEnabled = false;
        
        // Draw frame directly at original size first
        const imageData = ctx.createImageData(width, height);
        const pixels = frame.getPixels();
        
        // Direct pixel transfer without intermediate scaling
        for (let i = 0; i < pixels.length; i++) {
            const color = pixels[i];
            const offset = i * 4;
            
            if (color) {
                imageData.data[offset] = color & 0xFF;         // R
                imageData.data[offset + 1] = (color >> 8) & 0xFF;  // G
                imageData.data[offset + 2] = (color >> 16) & 0xFF; // B
                imageData.data[offset + 3] = (color >> 24) & 0xFF; // A
            }
        }
        
        // Create temporary canvas at original size
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = width;
        tempCanvas.height = height;
        tempCtx.imageSmoothingEnabled = false;
        
        // Put pixels at original size
        tempCtx.putImageData(imageData, 0, 0);
        
        // Center the sprite in the target canvas
        const scale = Math.min(
            targetSize / width,
            targetSize / height
        );
        
        const scaledWidth = Math.round(width * scale);
        const scaledHeight = Math.round(height * scale);
        const offsetX = Math.floor((targetSize - scaledWidth) / 2);
        const offsetY = Math.floor((targetSize - scaledHeight) / 2);
        
        // Clear canvas
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.fillRect(0, 0, targetSize, targetSize);
        
        // Single scaling operation
        ctx.drawImage(tempCanvas, 
            0, 0, width, height,
            offsetX, offsetY, scaledWidth, scaledHeight
        );
        
        canvas.toBlob(resolve, 'image/png', 1.0);
    });
  };

  // Add color palette management for pixel art
  ns.InterpolationService.prototype.extractColorPalette = function(frame) {
    const pixels = frame.getPixels();
    const palette = new Set();
    
    for (let i = 0; i < pixels.length; i++) {
        const color = pixels[i];
        if (color !== 0) { // Skip transparent pixels
            palette.add(color);
        }
    }
    
    return Array.from(palette);
  };

  // Find closest color in palette
  ns.InterpolationService.prototype.findClosestColor = function(r, g, b, palette) {
    if (!Array.isArray(palette) || palette.length === 0) {
      // Return black if no palette is provided
      return 0xFF000000;
    }

    let minDistance = Infinity;
    let closestColor = palette[0];
    
    for (const color of palette) {
      const pr = color & 0xFF;
      const pg = (color >> 8) & 0xFF;
      const pb = (color >> 16) & 0xFF;
      
      // Calculate color distance (using simple RGB distance)
      const distance = Math.pow(r - pr, 2) + Math.pow(g - pg, 2) + Math.pow(b - pb, 2);
      
      if (distance < minDistance) {
        minDistance = distance;
        closestColor = color;
      }
    }
    
    return closestColor;
  };

  // Update dithering to use color palette
  ns.InterpolationService.prototype.applyFloydSteinbergDithering = function(imageData, palette) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    
    // Create buffer for error diffusion
    const buffer = new Float32Array(width * height * 3);
    
    // Copy image data to buffer
    for (let i = 0; i < width * height; i++) {
        const offset = i * 4;
        const bufferOffset = i * 3;
        buffer[bufferOffset] = data[offset];
        buffer[bufferOffset + 1] = data[offset + 1];
        buffer[bufferOffset + 2] = data[offset + 2];
    }
    
    // Apply dithering with palette
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            const offset = i * 4;
            const bufferOffset = i * 3;
            
            // Skip transparent pixels
            if (data[offset + 3] < 128) continue;
            
            // Get current color
            const r = Math.max(0, Math.min(255, buffer[bufferOffset]));
            const g = Math.max(0, Math.min(255, buffer[bufferOffset + 1]));
            const b = Math.max(0, Math.min(255, buffer[bufferOffset + 2]));
            
            // Find closest palette color
            const newColor = this.findClosestColor(r, g, b, palette);
            const nr = newColor & 0xFF;
            const ng = (newColor >> 8) & 0xFF;
            const nb = (newColor >> 16) & 0xFF;
            
            // Set pixel to palette color
            data[offset] = nr;
            data[offset + 1] = ng;
            data[offset + 2] = nb;
            data[offset + 3] = 255;
            
            // Calculate error
            const errorR = r - nr;
            const errorG = g - ng;
            const errorB = b - nb;
            
            // Distribute error with reduced coefficients
            const distribution = [
                [x + 1, y, 5/16],
                [x - 1, y + 1, 3/16],
                [x, y + 1, 5/16],
                [x + 1, y + 1, 3/16]
            ];
            
            for (const [nx, ny, factor] of distribution) {
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    const ni = ny * width + nx;
                    const nbOffset = ni * 3;
                    buffer[nbOffset] += errorR * factor;
                    buffer[nbOffset + 1] += errorG * factor;
                    buffer[nbOffset + 2] += errorB * factor;
                }
            }
        }
    }
  };

  // Update processFramesForRIFE to handle RGB and alpha separately
  ns.InterpolationService.prototype.processFramesForRIFE = async function(frame1, frame2) {
    // Verify frames have same dimensions
    if (frame1.getWidth() !== frame2.getWidth() || frame1.getHeight() !== frame2.getHeight()) {
        throw new Error('Frames must have the same dimensions');
    }

    // Extract alpha masks first
    const alphaMask1 = this.extractAlphaMask(frame1);
    const alphaMask2 = this.extractAlphaMask(frame2);

    // Create RGB-only versions (set alpha to 255 for non-transparent pixels)
    const rgbBlob1 = await this.frameToRGBOnlyBlob(frame1);
    const rgbBlob2 = await this.frameToRGBOnlyBlob(frame2);

    return {
        blob1: rgbBlob1,
        blob2: rgbBlob2,
        alphaMask1,
        alphaMask2,
        frame1,
        frame2,
        originalSize: {
            width: frame1.getWidth(),
            height: frame1.getHeight()
        }
    };
};

  // Add method to convert frame to RGB-only blob
  ns.InterpolationService.prototype.frameToRGBOnlyBlob = async function(frame) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    canvas.width = frame.getWidth();
    canvas.height = frame.getHeight();
    
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const pixels = frame.getPixels();
    
    // Copy pixels, setting alpha to 255 for non-transparent pixels
    for (let i = 0; i < pixels.length; i++) {
        const pixel = pixels[i];
        const offset = i * 4;
        const alpha = (pixel >>> 24) & 0xFF;
        
        if (alpha > 128) {
            // Copy RGB values
            imageData.data[offset] = pixel & 0xFF;         // R
            imageData.data[offset + 1] = (pixel >> 8) & 0xFF;  // G
            imageData.data[offset + 2] = (pixel >> 16) & 0xFF; // B
            imageData.data[offset + 3] = 255;  // Set alpha to fully opaque
        } else {
            // For transparent pixels, set to black with full alpha
            imageData.data[offset] = 0;
            imageData.data[offset + 1] = 0;
            imageData.data[offset + 2] = 0;
            imageData.data[offset + 3] = 255;
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Create upscaled canvas with nearest-neighbor
    const upscaledCanvas = document.createElement('canvas');
    const upCtx = upscaledCanvas.getContext('2d', { 
        willReadFrequently: true,
        imageSmoothingEnabled: false
    });
    
    const scale = Math.max(256 / canvas.width, 256 / canvas.height);
    upscaledCanvas.width = Math.round(canvas.width * scale);
    upscaledCanvas.height = Math.round(canvas.height * scale);
    
    // Force nearest-neighbor scaling
    upCtx.imageSmoothingEnabled = false;
    upCtx.webkitImageSmoothingEnabled = false;
    upCtx.mozImageSmoothingEnabled = false;
    upCtx.msImageSmoothingEnabled = false;
    
    upCtx.drawImage(canvas, 0, 0, upscaledCanvas.width, upscaledCanvas.height);
    
    return new Promise(resolve => {
        upscaledCanvas.toBlob(resolve, 'image/png', 1.0);
    });
};

  // Update blobToFrame to handle transparency better
  ns.InterpolationService.prototype.blobToFrame = async function(blob, originalSize, sourcePalette) {
    try {
      const img = await createImageBitmap(blob);
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = originalSize.width;
      canvas.height = originalSize.height;
      
      // Disable smoothing for pixel art
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      const outputPixels = new Uint32Array(canvas.width * canvas.height);

      // Process each pixel
      for (let i = 0; i < outputPixels.length; i++) {
        const offset = i * 4;
        
        // Get RGBA values
        const r = data[offset];
        const g = data[offset + 1];
        const b = data[offset + 2];
        const a = data[offset + 3];

        // Skip fully transparent pixels
        if (a < 128) {
          outputPixels[i] = 0;
          continue;
        }

        // Find closest palette color if palette is provided
        let finalColor;
        if (sourcePalette && sourcePalette.length > 0) {
          finalColor = this.findClosestColor(r, g, b, sourcePalette);
        } else {
          finalColor = (b << 16) | (g << 8) | r;
        }

        // Combine with alpha
        outputPixels[i] = (a << 24) | finalColor;
      }

      // Create new frame with processed pixels
      const newFrame = new pskl.model.Frame(originalSize.width, originalSize.height);
      newFrame.setPixels(outputPixels);
      return newFrame;

    } catch (error) {
      console.error('Error converting blob to frame:', error);
      throw error;
    }
  };

  // Add method to get influences from surrounding tiles
  ns.InterpolationService.prototype.getTileInfluences = function(x, y, tileMotions, tileSize) {
    const influences = [];
    const radius = 2; // Consider 2 tiles in each direction

    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            const tileX = Math.floor((x + dx * tileSize) / tileSize);
            const tileY = Math.floor((y + dy * tileSize) / tileSize);
            const tileId = `${tileX}_${tileY}`;
            
            const motion = tileMotions[tileId];
            if (motion) {
                // Calculate distance-based weight
                const centerX = motion.x + tileSize / 2;
                const centerY = motion.y + tileSize / 2;
                const distance = Math.sqrt(
                    Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2)
                );
                const weight = Math.max(0, 1 - distance / (tileSize * 2));

                influences.push({
                    motion,
                    weight: weight * motion.confidence
                });
            }
        }
    }

    return influences;
  };

  // Add method to calculate weighted offset
  ns.InterpolationService.prototype.calculateWeightedOffset = function(influences, timeStep) {
    if (influences.length === 0) {
        return { x: 0, y: 0 };
    }

    let totalWeight = 0;
    let weightedX = 0;
    let weightedY = 0;

    influences.forEach(({ motion, weight }) => {
        weightedX += motion.dx * weight;
        weightedY += motion.dy * weight;
        totalWeight += weight;
    });

    return {
        x: Math.round((weightedX / totalWeight) * timeStep),
        y: Math.round((weightedY / totalWeight) * timeStep)
    };
  };

  // Add anti-banding color sampling
  ns.InterpolationService.prototype.getSourceColorsWithAntiband = function(x, y, offset, pixels1, pixels2, width, height) {
    const samples = [];
    const sampleOffsets = [
        [0, 0], [0.25, 0.25], [-0.25, 0.25],
        [0.25, -0.25], [-0.25, -0.25]
    ];

    for (const [dx, dy] of sampleOffsets) {
        const sx = x - offset.x + dx;
        const sy = y - offset.y + dy;
        
        if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
            const i = Math.floor(sy) * width + Math.floor(sx);
            samples.push(pixels1[i]);
        }
    }

    // Get average color from samples
    const color1 = this.averageColors(samples);
    const color2 = pixels2[y * width + x];

    return {
        color1,
        color2,
        alpha1: (color1 >>> 24) & 0xFF,
        alpha2: (color2 >>> 24) & 0xFF,
        x: x,  // Add coordinates
        y: y,
        dx: offset.x,  // Add offset information
        dy: offset.y
    };
  };

  // Add color averaging helper
  ns.InterpolationService.prototype.averageColors = function(colors) {
    if (colors.length === 0) return 0;

    let r = 0, g = 0, b = 0, a = 0;
    colors.forEach(color => {
        r += color & 0xFF;
        g += (color >> 8) & 0xFF;
        b += (color >> 16) & 0xFF;
        a += (color >>> 24) & 0xFF;
    });

    return (Math.round(a / colors.length) << 24) |
           (Math.round(b / colors.length) << 16) |
           (Math.round(g / colors.length) << 8) |
           Math.round(r / colors.length);
  };

  // Update sprite movement calculation with segment handling
  ns.InterpolationService.prototype.calculateSpriteMovement = function(frame1, frame2) {
    const width = frame1.getWidth();
    const height = frame1.getHeight();
    const pixels1 = frame1.getPixels();
    const pixels2 = frame2.getPixels();

    // Segment the sprite into regions (head, body, limbs)
    const segments1 = this.segmentSprite(pixels1, width, height);
    const segments2 = this.segmentSprite(pixels2, width, height);

    // Calculate movement for each segment
    const movements = {};
    for (const segmentName in segments1) {
        const segment1 = segments1[segmentName];
        const segment2 = segments2[segmentName];

        if (segment1 && segment2) {
            movements[segmentName] = {
                dx: segment2.center.x - segment1.center.x,
                dy: segment2.center.y - segment1.center.y,
                bounds1: segment1.bounds,
                bounds2: segment2.bounds
            };
        }
    }

    return movements;
  };

  // Add sprite segmentation
  ns.InterpolationService.prototype.segmentSprite = function(pixels, width, height) {
    const segmentMap = new Uint8Array(width * height);
    const visited = new Set();

    // Adjusted color ranges for better segment detection
    const segments = {
        head: {
            colors: [[200, 150, 100], [255, 220, 180]], // Broader skin tone range
            bounds: { left: width, right: 0, top: height, bottom: 0 },
            center: { x: 0, y: 0 },
            pixels: new Set()
        },
        dress: {
            colors: [[200, 50, 100], [255, 180, 220]], // Broader pink range
            bounds: { left: width, right: 0, top: height, bottom: 0 },
            center: { x: 0, y: 0 },
            pixels: new Set()
        },
        hair: {
            colors: [[200, 150, 0], [255, 255, 150]], // Broader blonde range
            bounds: { left: width, right: 0, top: height, bottom: 0 },
            center: { x: 0, y: 0 },
            pixels: new Set()
        }
    };

    // Flood fill to find connected components
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            if (visited.has(i)) continue;

            const pixel = pixels[i];
            const alpha = (pixel >>> 24) & 0xFF;
            if (alpha < 128) continue;

            const r = pixel & 0xFF;
            const g = (pixel >> 8) & 0xFF;
            const b = (pixel >> 16) & 0xFF;

            // Find matching segment
            for (const [segmentName, segment] of Object.entries(segments)) {
                if (this.isColorInRange(r, g, b, segment.colors[0], segment.colors[1])) {
                    this.floodFillSegment(x, y, pixels, width, height, segment, visited);
                    break;
                }
            }
        }
    }

    // Calculate center points for each segment
    for (const segment of Object.values(segments)) {
        if (segment.pixels.size > 0) {
            segment.center = {
                x: (segment.bounds.left + segment.bounds.right) / 2,
                y: (segment.bounds.top + segment.bounds.bottom) / 2
            };
        }
    }

    return segments;
  };

  // Add color range check helper
  ns.InterpolationService.prototype.isColorInRange = function(r, g, b, min, max) {
    return r >= min[0] && r <= max[0] &&
           g >= min[1] && g <= max[1] &&
           b >= min[2] && b <= max[2];
  };

  // Add flood fill for segments
  ns.InterpolationService.prototype.floodFillSegment = function(startX, startY, pixels, width, height, segment, visited) {
    const stack = [[startX, startY]];
    const startPixel = pixels[startY * width + startX];
    const startR = startPixel & 0xFF;
    const startG = (startPixel >> 8) & 0xFF;
    const startB = (startPixel >> 16) & 0xFF;
    const colorThreshold = 45; // Increased threshold for better segment connection

    while (stack.length > 0) {
        const [x, y] = stack.pop();
        const i = y * width + x;

        if (visited.has(i)) continue;
        visited.add(i);

        const pixel = pixels[i];
        const r = pixel & 0xFF;
        const g = (pixel >> 8) & 0xFF;
        const b = (pixel >> 16) & 0xFF;
        const alpha = (pixel >>> 24) & 0xFF;

        // Check if pixel is similar to start pixel
        if (alpha < 128 || 
            Math.abs(r - startR) > colorThreshold ||
            Math.abs(g - startG) > colorThreshold ||
            Math.abs(b - startB) > colorThreshold) {
            continue;
        }

        // Update segment bounds
        segment.bounds.left = Math.min(segment.bounds.left, x);
        segment.bounds.right = Math.max(segment.bounds.right, x);
        segment.bounds.top = Math.min(segment.bounds.top, y);
        segment.bounds.bottom = Math.max(segment.bounds.bottom, y);
        segment.pixels.add(i);

        // Add neighbors to stack
        for (const [dx, dy] of [[-1,0], [1,0], [0,-1], [0,1]]) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                stack.push([nx, ny]);
            }
        }
    }
  };

  // Add improved dithering pattern
  ns.InterpolationService.prototype.getBayerMatrix = function() {
    return [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ].map(row => row.map(x => (x / 16) - 0.5)); // Normalize to [-0.5, 0.5] range
  };

  // Add improved palette color matching with dithering
  ns.InterpolationService.prototype.findClosestColorWithDither = function(r, g, b, palette, x, y) {
    let closestColor = palette[0];
    let minDistance = Number.MAX_VALUE;
    
    // Add slight spatial variation to color matching
    const variation = ((x + y) % 2) * 2 - 1; // Alternating +1/-1 pattern
    
    for (const color of palette) {
        const pr = color & 0xFF;
        const pg = (color >> 8) & 0xFF;
        const pb = (color >> 16) & 0xFF;
        
        // Calculate weighted color distance with spatial variation
        const dr = (r - pr + variation) * 0.299; // Weight red less
        const dg = (g - pg + variation) * 0.587; // Weight green more
        const db = (b - pb + variation) * 0.114; // Weight blue less
        
        const distance = dr * dr + dg * dg + db * db;
        
        if (distance < minDistance) {
            minDistance = distance;
            closestColor = color;
        }
    }
    
    return closestColor;
  };

  // Add tile-based motion matching
  ns.InterpolationService.prototype.calculateTileMotion = function(pixels1, pixels2, width, height, tileSize = 8) {
    const tiles = {};
    const numTilesX = Math.ceil(width / tileSize);
    const numTilesY = Math.ceil(height / tileSize);

    // For each tile
    for (let ty = 0; ty < numTilesY; ty++) {
        for (let tx = 0; tx < numTilesX; tx++) {
            const tileId = `${tx}_${ty}`;
            const tileX = tx * tileSize;
            const tileY = ty * tileSize;

            // Skip empty tiles
            if (!this.isTileVisible(pixels1, tileX, tileY, tileSize, width, height)) {
                continue;
            }

            // Find best matching position in frame 2 with stricter matching
            const motion = this.findBestTileMatch(
                pixels1, pixels2, 
                tileX, tileY, 
                tileSize, width, height,
                8  // Reduced search radius for more stable movement
            );

            if (motion) {
                tiles[tileId] = {
                    x: tileX,
                    y: tileY,
                    dx: motion.dx,
                    dy: motion.dy,
                    confidence: motion.confidence
                };
            }
        }
    }

    // Smooth out tile motions to prevent breaking
    this.smoothTileMotions(tiles, numTilesX, numTilesY);
    return tiles;
  };

  // Add motion smoothing to prevent sprite breaking
  ns.InterpolationService.prototype.smoothTileMotions = function(tiles, numTilesX, numTilesY) {
    const smoothed = {};
    
    for (const [tileId, tile] of Object.entries(tiles)) {
        const [tx, ty] = tileId.split('_').map(Number);
        let avgDx = tile.dx;
        let avgDy = tile.dy;
        let count = 1;
        
        // Average with neighboring tiles
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                
                const neighborId = `${tx + dx}_${ty + dy}`;
                const neighbor = tiles[neighborId];
                
                if (neighbor) {
                    avgDx += neighbor.dx;
                    avgDy += neighbor.dy;
                    count++;
                }
            }
        }
        
        // Update motion with smoothed values
        smoothed[tileId] = {
            ...tile,
            dx: Math.round(avgDx / count),
            dy: Math.round(avgDy / count)
        };
    }
    
    // Apply smoothed motions back to tiles
    Object.assign(tiles, smoothed);
  };

  // Update findBestTileMatch with better confidence calculation
  ns.InterpolationService.prototype.findBestTileMatch = function(pixels1, pixels2, tileX, tileY, tileSize, width, height, searchRadius = 8) {
    let bestMatch = { dx: 0, dy: 0, confidence: 0 };
    let lowestDiff = Infinity;

    // Search nearby positions
    for (let dy = -searchRadius; dy <= searchRadius; dy++) {
        for (let dx = -searchRadius; dx <= searchRadius; dx++) {
            const diff = this.compareTiles(
                pixels1, pixels2,
                tileX, tileY,
                tileX + dx, tileY + dy,
                tileSize, width, height
            );

            // Add distance penalty to prefer smaller movements
            const distancePenalty = (dx * dx + dy * dy) / (searchRadius * searchRadius);
            const adjustedDiff = diff * (1 + distancePenalty * 0.5);

            if (adjustedDiff < lowestDiff) {
                lowestDiff = adjustedDiff;
                bestMatch = {
                    dx: dx,
                    dy: dy,
                    confidence: 1 - (diff / (tileSize * tileSize * 255))
                };
            }
        }
    }

    // Increased confidence threshold
    return bestMatch.confidence > 0.7 ? bestMatch : null;
  };

  // Add tile comparison method
  ns.InterpolationService.prototype.compareTiles = function(pixels1, pixels2, x1, y1, x2, y2, tileSize, width, height) {
    let totalDiff = 0;
    let validPixels = 0;

    // Compare corresponding pixels in both tiles
    for (let dy = 0; dy < tileSize; dy++) {
        for (let dx = 0; dx < tileSize; dx++) {
            const px1 = x1 + dx;
            const py1 = y1 + dy;
            const px2 = x2 + dx;
            const py2 = y2 + dy;

            // Check bounds
            if (px1 >= 0 && px1 < width && py1 >= 0 && py1 < height &&
                px2 >= 0 && px2 < width && py2 >= 0 && py2 < height) {
                
                const i1 = py1 * width + px1;
                const i2 = py2 * width + px2;

                const pixel1 = pixels1[i1];
                const pixel2 = pixels2[i2];

                const alpha1 = (pixel1 >>> 24) & 0xFF;
                const alpha2 = (pixel2 >>> 24) & 0xFF;

                // Only compare visible pixels
                if (alpha1 > 128 && alpha2 > 128) {
                    // Compare RGB values
                    const r1 = pixel1 & 0xFF;
                    const g1 = (pixel1 >> 8) & 0xFF;
                    const b1 = (pixel1 >> 16) & 0xFF;

                    const r2 = pixel2 & 0xFF;
                    const g2 = (pixel2 >> 8) & 0xFF;
                    const b2 = (pixel2 >> 16) & 0xFF;

                    // Calculate weighted color difference
                    const dr = Math.abs(r1 - r2) * 0.299;
                    const dg = Math.abs(g1 - g2) * 0.587;
                    const db = Math.abs(b1 - b2) * 0.114;

                    totalDiff += dr + dg + db;
                    validPixels++;
                }
            }
        }
    }

    // Return average difference per valid pixel, or Infinity if no valid pixels
    return validPixels > 0 ? totalDiff / validPixels : Infinity;
  };

  // Add helper methods for tile processing
  ns.InterpolationService.prototype.isTileVisible = function(pixels, tileX, tileY, tileSize, width, height) {
    for (let y = tileY; y < Math.min(tileY + tileSize, height); y++) {
        for (let x = tileX; x < Math.min(tileX + tileSize, width); x++) {
            const alpha = (pixels[y * width + x] >>> 24) & 0xFF;
            if (alpha > 128) return true;
        }
    }
    return false;
  };

  // Update postProcessFrame to include pixel count based artifact removal
  ns.InterpolationService.prototype.postProcessFrame = function(interpolatedPixels, frame1, frame2, timeStep, width, height) {
    // Store frame references for getRegionColors
    this.frame1 = frame1;
    this.frame2 = frame2;
    
    const pixels1 = frame1.getPixels();
    const pixels2 = frame2.getPixels();
    const processedPixels = new Uint32Array(width * height);
    const colorMap = new Map();

    // First pass: Process colors and transparency
    for (let i = 0; i < interpolatedPixels.length; i++) {
        processedPixels[i] = interpolatedPixels[i];
    }

    // Apply transparency mask
    this.applyTransparencyMask(processedPixels, pixels1, pixels2, timeStep, width, height);
    
    // Remove artifacts based on pixel count
    this.removeArtifactsByPixelCount(processedPixels, frame1, frame2, width, height);
    
    return processedPixels;
};

  // Add color region identification
  ns.InterpolationService.prototype.identifyColorRegions = function(pixels1, pixels2, width, height) {
    const regions = new Uint32Array(width * height);
    const visited = new Set();
    let regionId = 1;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            if (visited.has(i)) continue;

            const pixel1 = pixels1[i];
            const pixel2 = pixels2[i];
            const alpha1 = (pixel1 >>> 24) & 0xFF;
            const alpha2 = (pixel2 >>> 24) & 0xFF;

            if (alpha1 > 128 || alpha2 > 128) {
                // Flood fill to find connected region
                this.floodFillRegion(
                    x, y, pixels1, pixels2,
                    regions, visited, regionId,
                    width, height
                );
                regionId++;
            }
        }
    }

    return regions;
};

  // Add color bleeding detection
  ns.InterpolationService.prototype.isColorBleeding = function(pixel, region) {
    if (!region) return false;

    const r = pixel & 0xFF;
    const g = (pixel >> 8) & 0xFF;
    const b = (pixel >> 16) & 0xFF;

    // Check if color is significantly different from region colors
    const regionColors = this.getRegionColors(region);
    return !regionColors.some(color => {
        const dr = Math.abs(r - (color & 0xFF));
        const dg = Math.abs(g - ((color >> 8) & 0xFF));
        const db = Math.abs(b - ((color >> 16) & 0xFF));
        return (dr + dg + db) < 30; // Adjust threshold as needed
    });
};

  // Add color correction
  ns.InterpolationService.prototype.correctPixelColor = function(x, y, pixel, pixels1, pixels2, region, timeStep, width, height, colorMap) {
    const key = `${x},${y},${region}`;
    if (colorMap.has(key)) {
        return colorMap.get(key);
    }

    // Get dominant colors from the region
    const regionColors = this.getRegionColors(region);
    
    // Find closest valid color
    const r = pixel & 0xFF;
    const g = (pixel >> 8) & 0xFF;
    const b = (pixel >> 16) & 0xFF;
    const a = (pixel >>> 24) & 0xFF;

    let bestColor = pixel;
    let minDiff = Infinity;

    for (const color of regionColors) {
        const dr = r - (color & 0xFF);
        const dg = g - ((color >> 8) & 0xFF);
        const db = b - ((color >> 16) & 0xFF);
        
        const diff = (dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114);
        
        if (diff < minDiff) {
            minDiff = diff;
            bestColor = color;
        }
    }

    // Preserve alpha
    const correctedColor = (a << 24) | (bestColor & 0x00FFFFFF);
    colorMap.set(key, correctedColor);
    return correctedColor;
};

  // Add edge preservation
  ns.InterpolationService.prototype.preserveEdges = function(pixels, regions, width, height) {
    const edgePixels = new Set();

    // Identify edge pixels
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const i = y * width + x;
            if (this.isEdgePixel(pixels, regions, x, y, width)) {
                edgePixels.add(i);
            }
        }
    }

    // Enhance edge pixels
    for (const i of edgePixels) {
        const pixel = pixels[i];
        const alpha = (pixel >>> 24) & 0xFF;
        if (alpha > 128) {
            // Sharpen edges by increasing contrast
            const r = Math.min(255, Math.max(0, ((pixel & 0xFF) * 1.2)));
            const g = Math.min(255, Math.max(0, (((pixel >> 8) & 0xFF) * 1.2)));
            const b = Math.min(255, Math.max(0, (((pixel >> 16) & 0xFF) * 1.2)));
            pixels[i] = (alpha << 24) | (b << 16) | (g << 8) | r;
        }
    }
};

  // Add flood fill region method
  ns.InterpolationService.prototype.floodFillRegion = function(startX, startY, pixels1, pixels2, regions, visited, regionId, width, height) {
    const stack = [[startX, startY]];
    const startPixel1 = pixels1[startY * width + startX];
    const startPixel2 = pixels2[startY * width + startX];
    const colorThreshold = 30;

    while (stack.length > 0) {
        const [x, y] = stack.pop();
        const i = y * width + x;

        if (visited.has(i)) continue;
        visited.add(i);

        // Get colors from both frames
        const pixel1 = pixels1[i];
        const pixel2 = pixels2[i];
        const alpha1 = (pixel1 >>> 24) & 0xFF;
        const alpha2 = (pixel2 >>> 24) & 0xFF;

        // Skip transparent pixels
        if (alpha1 < 128 && alpha2 < 128) continue;

        // Check color similarity with start pixel
        if (this.isColorSimilar(pixel1, startPixel1, colorThreshold) ||
            this.isColorSimilar(pixel2, startPixel2, colorThreshold)) {
            
            // Mark pixel as part of region
            regions[i] = regionId;

            // Add neighbors to stack
            for (const [dx, dy] of [[-1,0], [1,0], [0,-1], [0,1]]) {
                const nx = x + dx;
                const ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    stack.push([nx, ny]);
                }
            }
        }
    }
};

  // Add color similarity check helper
  ns.InterpolationService.prototype.isColorSimilar = function(color1, color2, threshold) {
    const r1 = color1 & 0xFF;
    const g1 = (color1 >> 8) & 0xFF;
    const b1 = (color1 >> 16) & 0xFF;
    
    const r2 = color2 & 0xFF;
    const g2 = (color2 >> 8) & 0xFF;
    const b2 = (color2 >> 16) & 0xFF;

    const dr = Math.abs(r1 - r2);
    const dg = Math.abs(g1 - g2);
    const db = Math.abs(b1 - b2);

    return (dr + dg + db) < threshold;
};

  // Add method to get region colors
  ns.InterpolationService.prototype.getRegionColors = function(regionId) {
    if (!regionId) return [];

    const colors = new Set();
    const pixels1 = this.frame1.getPixels();
    const pixels2 = this.frame2.getPixels();
    const width = this.frame1.getWidth();
    const height = this.frame1.getHeight();

    // Sample colors from both frames
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            if (this.regions[i] === regionId) {
                const pixel1 = pixels1[i];
                const pixel2 = pixels2[i];
                const alpha1 = (pixel1 >>> 24) & 0xFF;
                const alpha2 = (pixel2 >>> 24) & 0xFF;

                if (alpha1 > 128) colors.add(pixel1 & 0x00FFFFFF);
                if (alpha2 > 128) colors.add(pixel2 & 0x00FFFFFF);
            }
        }
    }

    return Array.from(colors);
};

  // Add helper to apply transparency mask
  ns.InterpolationService.prototype.applyTransparencyMask = function(pixels, pixels1, pixels2, timeStep, width, height) {
    for (let i = 0; i < pixels.length; i++) {
        const alpha1 = (pixels1[i] >>> 24) & 0xFF;
        const alpha2 = (pixels2[i] >>> 24) & 0xFF;

        // If pixel should be transparent in both frames, make it transparent
        if (alpha1 < 128 && alpha2 < 128) {
            pixels[i] = 0;
            continue;
        }

        // If pixel is transitioning between transparent and opaque
        if ((alpha1 < 128) !== (alpha2 < 128)) {
            const currentAlpha = (pixels[i] >>> 24) & 0xFF;
            const targetAlpha = Math.round(
                (alpha1 < 128 ? 0 : alpha1) * (1 - timeStep) +
                (alpha2 < 128 ? 0 : alpha2) * timeStep
            );
            
            // Update alpha while preserving RGB
            pixels[i] = (targetAlpha << 24) | (pixels[i] & 0x00FFFFFF);
        }
    }
};

  // Update removeArtifactsByPixelCount to be more aggressive with low intensity artifacts
  ns.InterpolationService.prototype.removeArtifactsByPixelCount = function(interpolatedPixels, frame1, frame2, width, height) {
    const pixels1 = frame1.getPixels();
    const pixels2 = frame2.getPixels();
    
    // Get color intensity ranges from original frames
    const intensityRanges = this.getColorIntensityRanges(pixels1, pixels2);
    const { minIntensity, maxIntensity, avgIntensity } = intensityRanges;
    
    // Count non-transparent pixels in original frames
    const count1 = this.countNonTransparentPixels(pixels1);
    const count2 = this.countNonTransparentPixels(pixels2);
    const targetCount = Math.round((count1 + count2) / 2);
    
    // Process pixels
    const pixelsToRemove = [];
    let currentCount = 0;
    
    // First pass: Mark low intensity pixels and calculate local intensity
    const localIntensities = new Float32Array(width * height);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            const pixel = interpolatedPixels[i];
            const alpha = (pixel >>> 24) & 0xFF;
            
            if (alpha > 128) {
                currentCount++;
                
                // Calculate pixel intensity
                const r = pixel & 0xFF;
                const g = (pixel >> 8) & 0xFF;
                const b = (pixel >> 16) & 0xFF;
                const intensity = (r * 0.299 + g * 0.587 + b * 0.114);
                
                // Calculate local average intensity
                const localIntensity = this.calculateLocalIntensity(x, y, interpolatedPixels, width, height);
                localIntensities[i] = localIntensity;
                
                // Check if pixel exists in original frames
                const existsInOriginal = (pixels1[i] >>> 24) > 128 || (pixels2[i] >>> 24) > 128;
                
                // Calculate intensity deviation from both global and local averages
                const globalDeviation = Math.abs(intensity - avgIntensity) / (maxIntensity - minIntensity);
                const localDeviation = Math.abs(intensity - localIntensity) / localIntensity;
                
                // More aggressive criteria for removal:
                // 1. Not in original frames AND
                // 2. Either significantly deviates from global intensity OR
                // 3. Significantly deviates from local intensity OR
                // 4. Is an isolated pixel with low intensity
                if (!existsInOriginal && (
                    globalDeviation > 0.2 || // Reduced threshold (was 0.3)
                    localDeviation > 0.25 ||
                    (this.isIsolatedPixel(x, y, interpolatedPixels, width, height) && intensity < avgIntensity) ||
                    intensity < minIntensity * 1.2 // Remove very low intensity pixels
                )) {
                    pixelsToRemove.push({
                        index: i,
                        deviation: Math.max(globalDeviation, localDeviation),
                        intensity: intensity
                    });
                }
            }
        }
    }
    
    // Sort pixels by deviation AND intensity (prioritize removing low intensity pixels)
    pixelsToRemove.sort((a, b) => {
        // Prioritize low intensity pixels more heavily
        const intensityWeight = 0.7;
        const deviationWeight = 0.3;
        
        const scoreA = (intensityWeight * (1 - a.intensity/255)) + (deviationWeight * a.deviation);
        const scoreB = (intensityWeight * (1 - b.intensity/255)) + (deviationWeight * b.deviation);
        return scoreB - scoreA;
    });
    
    // Remove pixels more aggressively
    const excessCount = Math.max(
        currentCount - targetCount,
        Math.floor(pixelsToRemove.length * 0.8) // Remove at least 80% of suspicious pixels
    );
    
    for (let i = 0; i < excessCount && i < pixelsToRemove.length; i++) {
        interpolatedPixels[pixelsToRemove[i].index] = 0;
    }
    
    return interpolatedPixels;
};

  // Add helper to calculate local intensity
  ns.InterpolationService.prototype.calculateLocalIntensity = function(x, y, pixels, width, height) {
    let totalIntensity = 0;
    let count = 0;
    const radius = 2; // Check 5x5 neighborhood
    
    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const pixel = pixels[ny * width + nx];
                const alpha = (pixel >>> 24) & 0xFF;
                
                if (alpha > 128) {
                    const r = pixel & 0xFF;
                    const g = (pixel >> 8) & 0xFF;
                    const b = (pixel >> 16) & 0xFF;
                    totalIntensity += (r * 0.299 + g * 0.587 + b * 0.114);
                    count++;
                }
            }
        }
    }
    
    return count > 0 ? totalIntensity / count : 0;
};

  // Add helper to get color intensity ranges from original frames
  ns.InterpolationService.prototype.getColorIntensityRanges = function(pixels1, pixels2) {
    let minIntensity = Infinity;
    let maxIntensity = -Infinity;
    let totalIntensity = 0;
    let validPixels = 0;
    
    // Process both frames
    for (const pixels of [pixels1, pixels2]) {
        for (let i = 0; i < pixels.length; i++) {
            const alpha = (pixels[i] >>> 24) & 0xFF;
            if (alpha > 128) {
                const r = pixels[i] & 0xFF;
                const g = (pixels[i] >> 8) & 0xFF;
                const b = (pixels[i] >> 16) & 0xFF;
                const intensity = (r * 0.299 + g * 0.587 + b * 0.114);
                
                minIntensity = Math.min(minIntensity, intensity);
                maxIntensity = Math.max(maxIntensity, intensity);
                totalIntensity += intensity;
                validPixels++;
            }
        }
    }
    
    return {
        minIntensity,
        maxIntensity,
        avgIntensity: totalIntensity / validPixels
    };
};

  // Add helper to count non-transparent pixels
  ns.InterpolationService.prototype.countNonTransparentPixels = function(pixels) {
    let count = 0;
    for (let i = 0; i < pixels.length; i++) {
        if ((pixels[i] >>> 24) & 0xFF > 128) {
            count++;
        }
    }
    return count;
};

  // Add mask-based region tracking
  ns.InterpolationService.prototype.createRegionMasks = function(frame1, frame2) {
    const width = frame1.getWidth();
    const height = frame1.getHeight();
    const pixels1 = frame1.getPixels();
    const pixels2 = frame2.getPixels();
    
    // Create binary masks for both frames
    const mask1 = new Uint8Array(width * height);
    const mask2 = new Uint8Array(width * height);
    
    // Create region labels
    const regions1 = new Uint32Array(width * height);
    const regions2 = new Uint32Array(width * height);
    
    // Track unique regions and their properties
    const regionProps = new Map();
    let nextRegionId = 1;

    // First pass: Create binary masks and initial regions
    for (let i = 0; i < pixels1.length; i++) {
        // Create binary masks (1 for non-transparent pixels)
        mask1[i] = (pixels1[i] >>> 24) > 128 ? 1 : 0;
        mask2[i] = (pixels2[i] >>> 24) > 128 ? 1 : 0;
        
        if (mask1[i]) {
            const x = i % width;
            const y = Math.floor(i / width);
            const regionId = this.floodFillRegion(x, y, pixels1, width, height, regions1, nextRegionId);
            
            if (regionId === nextRegionId) {
                // New region found
                regionProps.set(regionId, {
                    color: pixels1[i] & 0x00FFFFFF,
                    bounds: { minX: x, maxX: x, minY: y, maxY: y },
                    pixels: new Set([i])
                });
                nextRegionId++;
            } else {
                // Add to existing region
                const props = regionProps.get(regionId);
                props.bounds.minX = Math.min(props.bounds.minX, x);
                props.bounds.maxX = Math.max(props.bounds.maxX, x);
                props.bounds.minY = Math.min(props.bounds.minY, y);
                props.bounds.maxY = Math.max(props.bounds.maxY, y);
                props.pixels.add(i);
            }
        }
    }

    return {
        masks: { frame1: mask1, frame2: mask2 },
        regions: { frame1: regions1, frame2: regions2 },
        props: regionProps
    };
};

  // Update blobToFrame to use mask-based interpolation
  ns.InterpolationService.prototype.blobToFrame = async function(blob, originalSize, sourcePalette, alphaMask1, alphaMask2, timeStep, frame1, frame2) {
    try {
        const img = await createImageBitmap(blob, { resizeQuality: 'pixelated' });
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = originalSize.width;
        canvas.height = originalSize.height;
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        const pixels1 = frame1.getPixels();
        const pixels2 = frame2.getPixels();
        const outputPixels = new Uint32Array(canvas.width * canvas.height);

        // Calculate tile motion
        const tileMotions = this.calculateTileMotion(
            pixels1, pixels2,
            canvas.width, canvas.height,
            8  // Increased from 4 to 8
        );

        // Process pixels
        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const i = y * canvas.width + x;
                
                // Find which tile this pixel belongs to
                const tileX = Math.floor(x / 8);  // Match new tile size
                const tileY = Math.floor(y / 8);
                const tileId = `${tileX}_${tileY}`;
                const tileMotion = tileMotions[tileId];

                // Calculate offset based on tile motion with stability check
                const offset = tileMotion ? {
                    x: Math.round(tileMotion.dx * timeStep * tileMotion.confidence),
                    y: Math.round(tileMotion.dy * timeStep * tileMotion.confidence)
                } : { x: 0, y: 0 };

                // Get source position with motion compensation
                const sx = x - offset.x;
                const sy = y - offset.y;

                // Get colors with bounds checking
                const color1 = (sx >= 0 && sx < canvas.width && sy >= 0 && sy < canvas.height) 
                    ? pixels1[sy * canvas.width + sx] 
                    : pixels1[i];  // Fall back to original position if out of bounds
                const color2 = pixels2[i];

                const alpha1 = (color1 >>> 24) & 0xFF;
                const alpha2 = (color2 >>> 24) & 0xFF;

                if (alpha1 < 128 && alpha2 < 128) {
                    outputPixels[i] = 0;
                    continue;
                }

                // Blend colors
                const r1 = color1 & 0xFF;
                const g1 = (color1 >> 8) & 0xFF;
                const b1 = (color1 >> 16) & 0xFF;

                const r2 = color2 & 0xFF;
                const g2 = (color2 >> 8) & 0xFF;
                const b2 = (color2 >> 16) & 0xFF;

                const r = Math.round(r1 * (1 - timeStep) + r2 * timeStep);
                const g = Math.round(g1 * (1 - timeStep) + g2 * timeStep);
                const b = Math.round(b1 * (1 - timeStep) + b2 * timeStep);
                const a = Math.round(alpha1 * (1 - timeStep) + alpha2 * timeStep);

                if (sourcePalette && sourcePalette.length > 0) {
                    const finalColor = this.findClosestColorWithDither(r, g, b, sourcePalette, x, y);
                    outputPixels[i] = (a << 24) | (finalColor & 0x00FFFFFF);
                } else {
                    outputPixels[i] = (a << 24) | (b << 16) | (g << 8) | r;
                }
            }
        }

        // Apply post-processing
        const processedPixels = this.postProcessFrame(
            outputPixels,
            frame1,
            frame2,
            timeStep,
            canvas.width,
            canvas.height
        );

        const newFrame = new pskl.model.Frame(originalSize.width, originalSize.height);
        newFrame.setPixels(processedPixels);
        return newFrame;

    } catch (error) {
        console.error('Error converting blob to frame:', error);
        throw error;
    }
  };

  // Add helper to calculate region motions using tiles
  ns.InterpolationService.prototype.calculateRegionMotions = function(regionData, frame1, frame2, width, height) {
    const motions = new Map();
    const tileSize = 8; // Use 8x8 tiles for motion detection

    for (const [regionId, props] of regionData.props.entries()) {
        // Calculate tile-based motion for this region
        const regionTiles = this.getTilesForRegion(props, tileSize, width, height);
        const tileMotions = this.calculateTileMotion(
            frame1.getPixels(), frame2.getPixels(),
            width, height, tileSize,
            regionTiles // Pass region tiles to limit search area
        );

        // Aggregate tile motions for this region
        const avgMotion = this.aggregateRegionMotion(tileMotions, props);
        motions.set(regionId, avgMotion);
    }

    return motions;
};

  // Add helper to fill gaps between regions
  ns.InterpolationService.prototype.fillRegionGaps = function(pixels, frame1, frame2, timeStep, width, height) {
    for (let i = 0; i < pixels.length; i++) {
        if ((pixels[i] >>> 24) < 128) {
            // Find nearest non-transparent pixel from both frames
            const x = i % width;
            const y = Math.floor(i / width);
            const nearest1 = this.findNearestPixel(x, y, frame1, width, height);
            const nearest2 = this.findNearestPixel(x, y, frame2, width, height);

            if (nearest1 && nearest2) {
                // Use nearest pixel from appropriate frame based on timeStep
                pixels[i] = timeStep < 0.5 ? nearest1 : nearest2;
            }
        }
    }
  };

  // Add helper method to extract alpha mask
  ns.InterpolationService.prototype.extractAlphaMask = function(frame) {
    const width = frame.getWidth();
    const height = frame.getHeight();
    const pixels = frame.getPixels();
    const mask = new Uint8Array(width * height);
    
    for (let i = 0; i < pixels.length; i++) {
        const alpha = (pixels[i] >>> 24) & 0xFF;
        // Create binary mask (255 for non-transparent, 0 for transparent)
        mask[i] = alpha > 128 ? 255 : 0;
    }

    return mask;
};

  // Add back generateTimeSteps method
  ns.InterpolationService.prototype.generateTimeSteps = function(numFrames) {
    const timeSteps = [];
    // Generate evenly spaced time steps between 0 and 1
    for (let i = 1; i <= numFrames; i++) {
        const t = i / (numFrames + 1);
        timeSteps.push(t);
    }
    console.log('Generated time steps:', timeSteps);
    return timeSteps;
};

  // Add back sendRIFERequest method
  ns.InterpolationService.prototype.sendRIFERequest = async function(blob1, blob2, timeStep) {
    const formData = new FormData();
    formData.append('frame1', blob1, 'frame1.png');
    formData.append('frame2', blob2, 'frame2.png');
    formData.append('time_step', timeStep.toString());

    const response = await fetch('http://localhost:8000/interpolate', {
        method: 'POST',
        body: formData,
        mode: 'cors',
        headers: {
            'Accept': 'image/png'
        }
    });

    console.log('RIFE server response:', {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers)
    });

    return response;
  };

  // Add new method to extract and validate color palette
  ns.InterpolationService.prototype.extractAndValidatePalette = function(frame1, frame2) {
    // ... existing code ...

    // Add improved palette extraction
    const palette = new Set();
    const pixels1 = frame1.getPixels();
    const pixels2 = frame2.getPixels();

    // Extract unique colors from both frames
    for (const pixels of [pixels1, pixels2]) {
      for (const pixel of pixels) {
        if ((pixel >>> 24) & 0xFF > 128) { // Only include non-transparent colors
          palette.add(pixel & 0x00FFFFFF); // Store RGB only
        }
      }
    }

    return Array.from(palette);
  };

  // Update findClosestColor to be more accurate for pixel art
  ns.InterpolationService.prototype.findClosestColor = function(r, g, b, palette) {
    let bestMatch = palette[0];
    let minDistance = Infinity;

    // Weight factors for RGB components (human perception)
    const rWeight = 0.299;
    const gWeight = 0.587; 
    const bWeight = 0.114;

    for (const color of palette) {
      const pr = color & 0xFF;
      const pg = (color >> 8) & 0xFF;
      const pb = (color >> 16) & 0xFF;

      // Calculate weighted color distance
      const distance = 
        rWeight * Math.pow(r - pr, 2) +
        gWeight * Math.pow(g - pg, 2) +
        bWeight * Math.pow(b - pb, 2);

      if (distance < minDistance) {
        minDistance = distance;
        bestMatch = color;
      }
    }

    return bestMatch;
  };

  // Add method to handle color transitions
  ns.InterpolationService.prototype.interpolateColors = function(color1, color2, t, palette) {
    // Extract components
    const r1 = color1 & 0xFF;
    const g1 = (color1 >> 8) & 0xFF;
    const b1 = (color1 >> 16) & 0xFF;
    const a1 = (color1 >>> 24) & 0xFF;

    const r2 = color2 & 0xFF;
    const g2 = (color2 >> 8) & 0xFF;
    const b2 = (color2 >> 16) & 0xFF;
    const a2 = (color2 >>> 24) & 0xFF;

    // If either color is transparent, handle specially
    if (a1 < 128 && a2 < 128) {
      return 0; // Both transparent
    }
    
    if (a1 < 128 || a2 < 128) {
      // One color is transparent - use hard transition at t=0.5
      return t < 0.5 ? color1 : color2;
    }

    // For palette colors, find closest match
    if (palette && palette.length > 0) {
      // Use hard transition at t=0.5 to avoid color blending
      return t < 0.5 ? color1 : color2;
    }

    // For non-palette colors, interpolate smoothly
    const r = Math.round(r1 * (1 - t) + r2 * t);
    const g = Math.round(g1 * (1 - t) + g2 * t);
    const b = Math.round(b1 * (1 - t) + b2 * t);
    const a = Math.round(a1 * (1 - t) + a2 * t);

    return (a << 24) | (b << 16) | (g << 8) | r;
  };
})(); 