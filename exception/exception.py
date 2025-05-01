class EasyLaborPredictionException(Exception):
    """Custom Exception Class for the project."""
    def __init__(self, message, error=None):
        super().__init__(message)
        self.error = error
