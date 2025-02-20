(function () {
  var ns = $.namespace('pskl.controller.dialogs');

  ns.GenerateFramesController = function (piskelController) {
    this.piskelController = piskelController;
    this.startFrame = null;
    this.endFrame = null;
    this.interpolationService = new pskl.service.InterpolationService();
  };

  pskl.utils.inherit(ns.GenerateFramesController, ns.AbstractDialogController);

  ns.GenerateFramesController.prototype.init = function () {
    this.superclass.init.call(this);

    // Add RIFE button
    this.generateRifeButton = document.querySelector('.generate-rife-button');
    if (this.generateRifeButton) {
      this.generateRifeButton.addEventListener('click', this.onRifeButtonClick.bind(this));
    }
  };

  /**
   * @override
   */
  ns.GenerateFramesController.prototype.onShow = async function () {
    console.log('GenerateFramesController: Dialog show triggered');
    
    // Get DOM elements first
    this.frameSelector = document.querySelector('.generate-frames .frame-selector');
    this.frameCountSlider = document.querySelector('.generate-frames .frame-count-slider');
    this.sliderValue = document.querySelector('.generate-frames .slider-value');
    this.generateButton = document.querySelector('.generate-frames .generate-button');
    this.cancelButton = document.querySelector('.generate-frames .cancel-button');
    
    // Add event listeners
    this.addEventListener(this.generateButton, 'click', this.onGenerateClick_);
    this.addEventListener(this.cancelButton, 'click', this.closeDialog);
    this.addEventListener(this.frameSelector, 'click', this.onFrameClick_);
    this.addEventListener(this.frameCountSlider, 'input', this.onSliderChange_);

    // Initialize frame selector
    this.populateFrameSelector_();
    
    // Initialize interpolation service if not already initialized
    if (!this.interpolationService.isModelLoaded) {
      try {
        await this.interpolationService.init();
        this.generateButton.disabled = false;
      } catch (error) {
        console.error('Failed to initialize frame interpolation:', error);
        this.showError_('Failed to initialize frame interpolation. Please try again.');
        this.generateButton.disabled = true;
      }
    }
  };

  ns.GenerateFramesController.prototype.populateFrameSelector_ = function () {
    // Get frames from the current layer
    var layer = this.piskelController.getCurrentLayer();
    var frames = layer.getFrames();
    var frameCount = frames.length;
    
    // Clear existing frames
    this.frameSelector.innerHTML = '';
    
    // Create preview for each frame
    for (var i = 0; i < frameCount; i++) {
      var frame = frames[i];
      var preview = document.createElement('div');
      preview.className = 'frame-preview';
      preview.setAttribute('data-frame-index', i);
      
      // Create canvas and render frame
      var canvas = pskl.utils.FrameUtils.toImage(frame);
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      preview.appendChild(canvas);
      
      this.frameSelector.appendChild(preview);
    }
  };

  ns.GenerateFramesController.prototype.onFrameClick_ = function (event) {
    var target = event.target.closest('.frame-preview');
    if (!target) return;

    var frameIndex = parseInt(target.getAttribute('data-frame-index'), 10);
    
    // Handle frame selection logic
    if (this.startFrame === null) {
      this.startFrame = frameIndex;
      target.classList.add('selected');
    } else if (this.endFrame === null && frameIndex !== this.startFrame) {
      // Ensure end frame comes after start frame
      if (frameIndex < this.startFrame) {
        var temp = frameIndex;
        frameIndex = this.startFrame;
        this.startFrame = temp;
        // Update UI to reflect the swap
        this.frameSelector.querySelectorAll('.frame-preview').forEach(preview => {
          preview.classList.remove('selected');
        });
        this.frameSelector.querySelector(`[data-frame-index="${this.startFrame}"]`).classList.add('selected');
      }
      this.endFrame = frameIndex;
      target.classList.add('selected');
      this.generateButton.disabled = false;
    } else {
      // Reset selection
      this.startFrame = frameIndex;
      this.endFrame = null;
      this.frameSelector.querySelectorAll('.frame-preview').forEach(preview => {
        preview.classList.remove('selected');
      });
      target.classList.add('selected');
      this.generateButton.disabled = true;
    }
  };

  ns.GenerateFramesController.prototype.clearFrameSelection_ = function () {
    this.startFrame = null;
    this.endFrame = null;
    this.generateButton.disabled = true;
    var selected = this.frameSelector.querySelectorAll('.selected');
    selected.forEach(function(el) {
      el.classList.remove('selected');
    });
  };

  ns.GenerateFramesController.prototype.onSliderChange_ = function (event) {
    this.sliderValue.textContent = event.target.value;
  };

  ns.GenerateFramesController.prototype.onFrameSelected = function (frame) {
    // Store selected frame
    this.selectedFrame = frame;
    
    // Get next frame in animation
    const frameIndex = this.piskelController.getCurrentLayer().getFrameIndex(frame);
    this.nextFrame = this.piskelController.getCurrentLayer().getFrameAt(frameIndex + 1);
    
    // Enable generate button only if we have both frames
    if (this.selectedFrame && this.nextFrame) {
      this.generateButton.removeAttribute('disabled');
    }
  };

  ns.GenerateFramesController.prototype.onGenerateClick_ = function () {
    const numFrames = parseInt(this.frameCountSlider.value, 10);
    console.log('Generate clicked, frames to generate:', numFrames);
    
    // Get the selected frames
    var layer = this.piskelController.getCurrentLayer();
    var frames = layer.getFrames();
    var frame1 = frames[this.startFrame];
    var frame2 = frames[this.endFrame];
    
    console.log('Selected frames:', {
      startFrame: this.startFrame,
      endFrame: this.endFrame,
      frame1: frame1,
      frame2: frame2
    });

    if (frame1 && frame2) {
      // Show loading state
      this.generateButton.disabled = true;
      this.generateButton.textContent = 'Generating...';

      this.interpolationService.interpolateFrames(frame1, frame2, numFrames)
        .then(frames => {
          console.log('Frames generated:', frames);
          // Insert the generated frames after the start frame
          frames.forEach((frame, i) => {
            layer.addFrameAt(frame, this.startFrame + 1 + i);
          });
          
          // Update UI without clearing selection
          $.publish(Events.PISKEL_RESET);
        })
        .catch(error => {
          console.error('Frame generation failed:', error);
          this.showError_('Failed to generate frames: ' + error.message);
        })
        .finally(() => {
          // Reset button state without clearing selection
          this.generateButton.disabled = false;
          this.generateButton.textContent = 'Generate';
        });
    } else {
      this.showError_('Please select two consecutive frames');
    }
  };

  ns.GenerateFramesController.prototype.onRifeButtonClick = async function () {
    // Get the selected frames
    var layer = this.piskelController.getCurrentLayer();
    var frames = layer.getFrames();
    var frame1 = frames[this.startFrame];
    var frame2 = frames[this.endFrame];
    
    if (!frame1 || !frame2) {
        this.showError_('Please select two frames first');
        return;
    }

    try {
        // Show loading state
        this.generateButton.disabled = true;
        this.generateRifeButton.disabled = true;
        this.generateRifeButton.textContent = 'Generating...';

        // Get number of frames from slider
        const numFrames = parseInt(this.frameCountSlider.value, 10);
        
        // Use interpolation service to generate frames
        const generatedFrames = await this.interpolationService.interpolateFrames(frame1, frame2, numFrames);
        
        if (!generatedFrames || generatedFrames.length === 0) {
            throw new Error('No frames were generated');
        }
        
        // Insert the generated frames after the start frame
        generatedFrames.forEach((frame, i) => {
            layer.addFrameAt(frame, this.startFrame + 1 + i);
        });
        
        // Update UI
        $.publish(Events.PISKEL_RESET);
    } catch (error) {
        console.error('RIFE frame generation failed:', error);
        this.showError_('Failed to generate frames: ' + error.message);
    } finally {
        // Reset button states
        this.generateButton.disabled = false;
        this.generateRifeButton.disabled = false;
        this.generateRifeButton.textContent = 'Generate with RIFE';
    }
  };

  ns.GenerateFramesController.prototype.destroy = function () {
    this.superclass.destroy.call(this);
  };

  // Helper method to show errors in the dialog
  ns.GenerateFramesController.prototype.showError_ = function (message) {
    var dialogContent = document.querySelector('.generate-frames .dialog-content');
    
    // Create error message element if it doesn't exist
    if (!this.errorMessageEl) {
      this.errorMessageEl = document.createElement('div');
      this.errorMessageEl.className = 'generate-frames-error';
      dialogContent.insertBefore(this.errorMessageEl, this.frameSelector);
    }
    
    this.errorMessageEl.textContent = message;
    this.errorMessageEl.style.display = 'block';
    
    // Hide error after 3 seconds
    setTimeout(() => {
      this.errorMessageEl.style.display = 'none';
    }, 3000);
  };
})(); 