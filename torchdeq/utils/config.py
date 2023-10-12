import argparse
import warnings


class DEQConfig:
    """
    A configuration class that can accept various types of input config.

    Args:
        config (Union[argparse.Namespace, dict, DEQConfig, Any]): The configuration values.
    """
    def __init__(self, config):
        if config is None:
            config = dict()

        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, argparse.Namespace):
            self.config = vars(config)
        elif isinstance(config, DEQConfig):
            self.config = config.config
        else:
            warnings.warn(f"Unrecognized config type: {type(config)}. Processed using get_attr.")
            self.config = vars(config)

    def get(self, key, default=None):
        """
        Retrieves a configuration value.

        Args:
            key (str): The configuration key.
            default (optional): The default value to return if the key is not found.

        Returns:
            The configuration value, or the default value if the key is not found.
        """
        return self.config.get(key, default)
    
    def update(self, **kwargs):
        """
        Updates the configuration with new values.

        Args:
            **kwargs: The new configuration values as keyword arguments.
        """
        self.config.update(kwargs)