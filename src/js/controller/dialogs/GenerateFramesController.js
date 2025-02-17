(function () {
  var ns = $.namespace('pskl.controller.dialogs');

  ns.GenerateFramesController = function (piskelController) {
    this.piskelController = piskelController;
    this.startFrame = null;
    this.endFrame = null;
  };

  pskl.utils.inherit(ns.GenerateFramesController, ns.AbstractDialogController);

  ns.GenerateFramesController.prototype.init = function () {
    this.superclass.init.call(this);
  };

  /**
   * @override
   */
  ns.GenerateFramesController.prototype.onShow = function () {
    // Get DOM elements after dialog is shown
    this.frameSelector = document.querySelector('.frame-selector');
    this.frameCountSlider = document.querySelector('.frame-count-slider');
    this.sliderValue = document.querySelector('.slider-value');
    this.generateButton = document.querySelector('.generate-button');
    this.cancelButton = document.querySelector('.cancel-button');

    // Add event listeners
    this.addEventListener(this.generateButton, 'click', this.onGenerateClick_);
    this.addEventListener(this.cancelButton, 'click', this.closeDialog);
    this.addEventListener(this.frameSelector, 'click', this.onFrameClick_);
    this.addEventListener(this.frameCountSlider, 'input', this.onSliderChange_);

    // Initialize frame selector
    this.populateFrameSelector_();
  };

  ns.GenerateFramesController.prototype.populateFrameSelector_ = function () {
    var frameCount = this.piskelController.getFrameCount();
    var layer = this.piskelController.getCurrentLayer();
    
    for (var i = 0; i < frameCount; i++) {
      var frame = layer.getFrameAt(i);
      var preview = this.createFramePreview_(frame, i);
      this.frameSelector.appendChild(preview);
    }
  };

  ns.GenerateFramesController.prototype.createFramePreview_ = function (frame, index) {
    var preview = document.createElement('div');
    preview.className = 'frame-preview';
    preview.setAttribute('data-frame-index', index);
    
    // Create a scaled-down version of the frame using CanvasRenderer
    var size = 80;
    var zoom = size / frame.getWidth();
    
    var renderer = new pskl.rendering.CanvasRenderer(frame, zoom);
    var canvas = renderer.render();
    canvas.classList.add('frame-preview-canvas');
    
    preview.appendChild(canvas);
    return preview;
  };

  ns.GenerateFramesController.prototype.onFrameClick_ = function (evt) {
    var target = evt.target.closest('.frame-preview');
    if (!target) return;

    var index = parseInt(target.getAttribute('data-frame-index'), 10);
    
    if (this.startFrame === null) {
      this.startFrame = index;
      target.classList.add('selected');
    } else if (this.endFrame === null) {
      this.endFrame = index;
      target.classList.add('selected');
      this.generateButton.disabled = false;
    } else {
      // Reset selection
      this.clearFrameSelection_();
      this.startFrame = index;
      target.classList.add('selected');
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

  ns.GenerateFramesController.prototype.onSliderChange_ = function (evt) {
    this.sliderValue.textContent = evt.target.value;
  };

  ns.GenerateFramesController.prototype.onGenerateClick_ = function () {
    if (this.startFrame === null || this.endFrame === null) {
      return;
    }

    var frameCount = parseInt(this.frameCountSlider.value, 10);
    
    // TODO: Add frame generation logic here
    
    this.closeDialog();
  };

  ns.GenerateFramesController.prototype.destroy = function () {
    this.superclass.destroy.call(this);
  };
})(); 