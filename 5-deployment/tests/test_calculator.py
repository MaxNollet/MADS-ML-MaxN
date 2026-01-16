import pytest
from calculator import Calculator


@pytest.mark.unit
class TestCalculator:
    @pytest.fixture
    def calculator(self):
        """Fixture to create a Calculator instance for each test."""
        return Calculator()

    def test_add_positive_numbers(self, calculator):
        """Test adding two positive numbers."""
        result = calculator.add(2, 3)
        assert result == 5

    def test_add_negative_numbers(self, calculator):
        """Test adding negative numbers."""
        result = calculator.add(-1, -1)
        assert result == -2

    def test_add_zero(self, calculator):
        """Test adding zero."""
        result = calculator.add(5, 0)
        assert result == 5

    def test_divide_positive_numbers(self, calculator):
        """Test division with positive numbers."""
        result = calculator.divide(6, 2)
        assert result == 3

    def test_divide_by_zero(self, calculator):
        """Test division by zero raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            calculator.divide(5, 0)
        assert str(exc_info.value) == "Cannot divide by zero"
