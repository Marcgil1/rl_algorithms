def setup_tf():
    """
    CUDA complains if I don't explicitly `allow_growth`. Probably not necessary
    in other computers.
    """
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

def setup_keras():
    """
    Layers must deal with `float64` entries.
    """
    import tensorflow as tf

    tf.keras.backend.set_floatx('float64')
