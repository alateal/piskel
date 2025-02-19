import tensorflow as tf
import tensorflow_hub as hub
import tensorflowjs as tfjs
import os
import sys

def ensure_directories():
    """Create necessary directories for model storage"""
    dirs = [
        "dest/prod/models/flownet"
    ]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

def create_model():
    """Create a Keras model for optical flow"""
    try:
        print("Creating model...")
        
        # Define input shapes for two consecutive frames
        input_shape = (256, 256, 3)  # Height, width, channels
        
        # Create inputs for two frames
        frame1_input = tf.keras.layers.Input(shape=input_shape, name='frame1')
        frame2_input = tf.keras.layers.Input(shape=input_shape, name='frame2')
        
        # Stack frames along channel dimension
        stacked_frames = tf.keras.layers.Concatenate(axis=-1)([frame1_input, frame2_input])
        
        # Create convolutional layers for flow estimation
        x = tf.keras.layers.Conv2D(64, 7, 2, padding='same', activation='relu')(stacked_frames)
        x = tf.keras.layers.Conv2D(128, 5, 2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation='relu')(x)
        
        # Upsampling layers
        x = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(64, 5, 2, padding='same', activation='relu')(x)
        
        # Final flow prediction (2 channels for x and y flow)
        flow = tf.keras.layers.Conv2DTranspose(2, 7, 2, padding='same', activation='tanh')(x)
        
        # Create model
        model = tf.keras.Model(inputs=[frame1_input, frame2_input], outputs=flow, name='flownet')
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        return model
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        sys.exit(1)

def convert_model():
    """Convert model to TensorFlow.js format"""
    try:
        ensure_directories()
        
        print("Creating model...")
        model = create_model()
        
        print("Converting model to TensorFlow.js format...")
        output_dir = "dest/prod/models/flownet"
        
        # First save as SavedModel format
        tf.saved_model.save(model, output_dir + "_tmp")
        
        # Then convert to TensorFlow.js format
        tfjs.converters.convert_tf_saved_model(
            output_dir + "_tmp",
            output_dir
        )
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(output_dir + "_tmp")
        
        print("Model conversion complete!")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    convert_model() 