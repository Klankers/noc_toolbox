class BaseStep:
    def __init__(self, name, parameters=None, diagnostics=False, context=None):
        self.name = name
        self.parameters = parameters or {}
        self.diagnostics = diagnostics
        self.context = context or {}

        # add attrs from parameters to self
        for key, value in self.parameters.items():
            setattr(self, key, value)

    def run(self):
        """To be implemented by subclasses"""
        raise NotImplementedError(f"Step '{self.name}' must implement a run() method.")
        return self.context

    def generate_diagnostics(self):
        """Hook for diagnostics (optional)"""
        pass
