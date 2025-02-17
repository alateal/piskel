(function () {
  var ns = $.namespace('pskl.utils');

  var listeners = {};
  var listenerCount = 0;

  ns.Event = {
    addEventListener : function (target, type, callback, scope) {
      if (typeof target === 'string') {
        // Support for using selectors as target
        var elements = document.querySelectorAll(target);
        for (var i = 0 ; i < elements.length ; i++) {
          this.addEventListener(elements[i], type, callback, scope);
        }
        return;
      }

      var listener = {
        type: type,
        callback: callback,
        scope: scope,
        target: target
      };

      var listenerId = listenerCount++;
      listeners[listenerId] = listener;

      var boundCallback = callback.bind(scope);
      listener.boundCallback = boundCallback;
      target.addEventListener(type, boundCallback);
    },

    removeAllEventListeners : function (scope) {
      for (var listenerId in listeners) {
        if (listeners[listenerId].scope === scope) {
          var listener = listeners[listenerId];
          listener.target.removeEventListener(
            listener.type, 
            listener.boundCallback
          );
          delete listeners[listenerId];
        }
      }
    }
  };
})();
