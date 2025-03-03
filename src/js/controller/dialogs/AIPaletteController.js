(function () {
  var ns = $.namespace('pskl.controller.dialogs');

  ns.AIPaletteController = function () {
    this.paletteService = pskl.app.paletteService;
  };

  pskl.utils.inherit(ns.AIPaletteController, ns.AbstractDialogController);

  ns.AIPaletteController.prototype.init = function () {
    // Call superclass init first to set up the close button handler
    this.superclass.init.call(this);

    // Wait for next tick to ensure dialog is rendered
    setTimeout(() => {
      this.apiKeyInput = document.querySelector('.ai-palette-api-key-input');
      this.themeInput = document.querySelector('.ai-palette-theme-input');
      this.sizeInput = document.querySelector('.ai-palette-size-input');
      this.colorsContainer = document.querySelector('.colors-list');
      
      var buttonsContainer = document.querySelector('.ai-palette-actions');
      this.generateButton = document.querySelector('button[data-action="generate"]');
      this.saveButton = document.querySelector('button[data-action="save"]');
      
      // Add event listeners to prevent clipboard events from propagating
      if (this.apiKeyInput) {
        this.addEventListener(this.apiKeyInput, 'paste', this.onApiKeyPaste_);
        this.addEventListener(this.apiKeyInput, 'cut', this.onApiKeyClipboard_);
        this.addEventListener(this.apiKeyInput, 'copy', this.onApiKeyClipboard_);
      }
      
      if (buttonsContainer) {
        this.addEventListener(buttonsContainer, 'click', this.onButtonClick_);
      }
      
      this.colors = [];
    }, 0);
  };

  ns.AIPaletteController.prototype.onApiKeyPaste_ = function(evt) {
    evt.stopPropagation();
    evt.preventDefault();
    
    // Get the clipboard data and set it directly
    var clipboardData = evt.clipboardData || window.clipboardData;
    var pastedData = clipboardData.getData('text');
    this.apiKeyInput.value = pastedData;
  };

  ns.AIPaletteController.prototype.onApiKeyClipboard_ = function(evt) {
    evt.stopPropagation();
    evt.preventDefault();
  };

  ns.AIPaletteController.prototype.destroy = function () {
    this.superclass.destroy.call(this);
  };

  ns.AIPaletteController.prototype.onButtonClick_ = function (evt) {
    var target = evt.target;
    var action = target.getAttribute('data-action');
    if (action === 'generate') {
      this.generatePalette_();
    } else if (action === 'save') {
      this.savePalette_();
    } else if (action === 'cancel') {
      this.closeDialog();
    }
  };

  ns.AIPaletteController.prototype.generatePalette_ = function () {
    var apiKey = this.apiKeyInput.value.trim();
    var theme = this.themeInput.value;
    var size = parseInt(this.sizeInput.value);

    if (!apiKey) {
      this.showError_('Please enter your OpenAI API key');
      return;
    }

    if (!theme) {
      this.showError_('Please enter a theme or description');
      return;
    }

    if (isNaN(size) || size < 2 || size > 32) {
      this.showError_('Please enter a valid number of colors (2-32)');
      return;
    }

    this.generateButton.disabled = true;
    this.generateButton.textContent = 'Generating...';

    // Call OpenAI API to generate colors
    fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey
      },
      body: JSON.stringify({
        model: "gpt-4",
        messages: [{
          role: "system",
          content: "You are a color palette generator. Respond only with a JSON array of exactly " + size + " hexadecimal color codes that match the theme. Format: [\"#RRGGBB\", ...]"
        }, {
          role: "user",
          content: "Generate a color palette for theme: " + theme
        }]
      })
    })
    .then(response => {
      if (!response.ok) {
        return response.json().then(err => {
          throw new Error(err.error?.message || 'API request failed');
        });
      }
      return response.json();
    })
    .then(data => {
      if (!data.choices || !data.choices[0] || !data.choices[0].message || !data.choices[0].message.content) {
        throw new Error('Invalid API response format');
      }
      
      try {
        var content = data.choices[0].message.content;
        // Remove any extra whitespace or text around the JSON array
        content = content.trim().replace(/^[^[]*/, '').replace(/[^\]]*$/, '');
        this.colors = JSON.parse(content);
        
        if (!Array.isArray(this.colors) || this.colors.length !== size) {
          throw new Error('Invalid color array format');
        }

        // Validate each color is a proper hex code
        if (!this.colors.every(color => /^#[0-9A-Fa-f]{6}$/.test(color))) {
          throw new Error('Invalid color format');
        }

        this.displayColors_();
        this.generateButton.style.display = 'none';
        this.saveButton.style.display = 'inline-block';
      } catch (e) {
        throw new Error('Failed to parse color data: ' + e.message);
      }
    })
    .catch(error => {
      this.showError_('Failed to generate palette: ' + error.message);
      console.error('AI Palette Error:', error);
    })
    .finally(() => {
      this.generateButton.disabled = false;
      this.generateButton.textContent = 'Generate';
    });
  };

  ns.AIPaletteController.prototype.displayColors_ = function () {
    var html = this.colors.map(function (color) {
      return '<div class="preview-color" style="background-color:' + color + ';"></div>';
    }).join('');
    this.colorsContainer.innerHTML = html;
  };

  ns.AIPaletteController.prototype.savePalette_ = function () {
    var name = this.themeInput.value + ' AI Palette';
    var uuid = pskl.utils.Uuid.generate();
    var palette = new pskl.model.Palette(uuid, name, this.colors);
    
    this.paletteService.savePalette(palette);
    pskl.UserSettings.set(pskl.UserSettings.SELECTED_PALETTE, uuid);
    
    this.closeDialog();
  };

  ns.AIPaletteController.prototype.showError_ = function (message) {
    $.publish(Events.SHOW_NOTIFICATION, [{
      'content': message
    }]);
    window.setTimeout($.publish.bind($, Events.HIDE_NOTIFICATION), 2000);
  };
})(); 