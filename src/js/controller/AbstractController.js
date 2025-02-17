(function () {
  var ns = $.namespace('pskl.controller');

  ns.AbstractController = function () {};

  ns.AbstractController.prototype.init = function () {};

  ns.AbstractController.prototype.destroy = function () {
    this.removeAllEventListeners();
  };

  /**
   * Add an event listener for the provided event type and target.
   */
  ns.AbstractController.prototype.addEventListener = function (target, type, callback) {
    pskl.utils.Event.addEventListener(target, type, callback, this);
  };

  /**
   * Remove all event listeners registered using addEventListener
   */
  ns.AbstractController.prototype.removeAllEventListeners = function () {
    pskl.utils.Event.removeAllEventListeners(this);
  };
})(); 