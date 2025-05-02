class BaseStep:
    def __init__(self, name, parameters=None, diagnostics=False):
        self.name = name
        self.parameters = parameters or {}
        self.diagnostics = diagnostics
        self.context = {}

    def run(self, context=None):
        """To be implemented by subclasses"""
        if context:
            self.context = context
        raise NotImplementedError(f"Step '{self.name}' must implement a run() method.")
        return self.context

    def generate_diagnostics(self):
        """Hook for diagnostics (optional)"""
        pass
