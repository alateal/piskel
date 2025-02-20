(function () {
  var ns = $.namespace('pskl.service.pensize');

  var MIN_PENSIZE = 1;
  var MAX_PENSIZE = 32;

  /**
   * Service to retrieve and modify the current pen size.
   */
  ns.PenSizeService = function () {
    this.size = 1;
    this.displaySize = 1; // Add display size for UI
  };

  ns.PenSizeService.prototype.init = function () {
    this.size = pskl.UserSettings.get(pskl.UserSettings.PEN_SIZE) || 1;
    this.updateDisplaySize();
    $.publish(Events.PEN_SIZE_CHANGED);

    var shortcuts = pskl.service.keyboard.Shortcuts;
    pskl.app.shortcutService.registerShortcut(shortcuts.MISC.INCREASE_PENSIZE, this.increasePenSize_.bind(this));
    pskl.app.shortcutService.registerShortcut(shortcuts.MISC.DECREASE_PENSIZE, this.decreasePenSize_.bind(this));
  };

  ns.PenSizeService.prototype.increasePenSize_ = function () {
    this.setPenSize(this.size + 1);
  };

  ns.PenSizeService.prototype.decreasePenSize_ = function () {
    this.setPenSize(this.size - 1);
  };

  ns.PenSizeService.prototype.getPenSize = function () {
    return this.size;
  };

  ns.PenSizeService.prototype.getDisplaySize = function () {
    return this.displaySize;
  };

  ns.PenSizeService.prototype.setPenSize = function (size) {
    if (this.isPenSizeValid_(size) && size != this.size) {
      this.size = size;
      this.updateDisplaySize();
      pskl.UserSettings.set(pskl.UserSettings.PEN_SIZE, size);
      $.publish(Events.PEN_SIZE_CHANGED);
    }
  };

  ns.PenSizeService.prototype.updateDisplaySize = function () {
    var zoom = pskl.app.drawingController ? pskl.app.drawingController.getZoom() : 1;
    
    // Always show exact size 1
    if (this.size <= 1) {
      this.displaySize = 1;
      return;
    }
    
    // For other sizes, scale with zoom
    this.displaySize = Math.max(1, Math.floor(this.size));
    
    console.log('Pen size updated:', {
      actualSize: this.size,
      displaySize: this.displaySize,
      zoom: zoom
    });
  };

  ns.PenSizeService.prototype.isPenSizeValid_ = function (size) {
    if (isNaN(size)) {
      return false;
    }

    return size >= MIN_PENSIZE && size <= MAX_PENSIZE;
  };

  /**
   * Get the actual pixel size for drawing
   * This accounts for the current zoom level and scale
   */
  ns.PenSizeService.prototype.getActualPenSize = function () {
    // Ensure size 1 is always exactly 1
    if (this.size <= 1) {
      return 1;
    }
    
    // For other sizes, return the exact size
    return Math.max(1, Math.floor(this.size));
  };

})();
