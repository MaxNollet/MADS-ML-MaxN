class Calculator:
    def add(self, x: float, y: float) -> float:
        """Add two numbers together."""
        return x + y

    def divide(self, x: float, y: float) -> float:
        """Divide x by y."""
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
