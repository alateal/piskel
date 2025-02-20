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
    if (!this.isModelLoaded) {
      throw new Error('Service not initialized');
    }

    const frames = [];
    try {
      // Analyze sprite movement and transformation
      const movement = this.analyzeSpriteDifference(frame1, frame2);
      
      // Generate intermediate frames
      for (let i = 1; i <= numFrames; i++) {
        const t = i / (numFrames + 1);
        console.log('Generating frame', i, 'at t =', t);
        
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
            
            // Determine pixel color based on movement and timing
            if (color1 === 0 && color2 === 0) {
              pixels[pos] = 0;
              continue;
            }
            
            // Handle transitioning pixels
            if (color1 === 0) {
              // Fade in color2
              const alpha = ((color2 >> 24) & 0xFF) * ease;
              pixels[pos] = (Math.round(alpha) << 24) | (color2 & 0x00FFFFFF);
              continue;
            }
            
            if (color2 === 0) {
              // Fade out color1
              const alpha = ((color1 >> 24) & 0xFF) * (1 - ease);
              pixels[pos] = (Math.round(alpha) << 24) | (color1 & 0x00FFFFFF);
              continue;
            }
            
            // For overlapping areas, use motion-based blending
            const useColor2 = this.shouldUseColor2(x, y, movement, ease);
            pixels[pos] = useColor2 ? color2 : color1;
          }
        }
        
        result.setPixels(pixels);
        frames.push(result);
      }
    } catch (error) {
      console.error('Error during interpolation:', error);
      throw error;
    }

    return frames;
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
  ns.InterpolationService.prototype.isEdgePixel = function(x, y, pixels, width, height) {
    const centerColor = pixels[y * width + x];
    
    // Check 4 adjacent pixels
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    
    for (const [dx, dy] of directions) {
      const nx = x + dx;
      const ny = y + dy;
      
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const neighborColor = pixels[ny * width + nx];
        
        // If neighbor is transparent or very different color, this is an edge
        if (neighborColor === 0 || !this.areColorsSimilar(centerColor, neighborColor)) {
          return true;
        }
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

  // Add helper method to analyze sprite movement and transformation
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
    
    return {
      dx: Math.round(center2.x - center1.x),
      dy: Math.round(center2.y - center1.y),
      bounds1,
      bounds2
    };
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
})(); 