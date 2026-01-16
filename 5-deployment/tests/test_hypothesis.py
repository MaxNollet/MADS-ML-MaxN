from math import isinf, isnan

import pytest
from calculator import Calculator
from hypothesis import assume, given
from hypothesis import strategies as st


@pytest.mark.hypothesis
class TestCalculator:
    def setup_class(self):
        self.calculator = Calculator()

    # Property-based tests using Hypothesis
    @given(
        x=st.floats(min_value=-1e6, max_value=1e6),
        y=st.floats(min_value=-1e6, max_value=1e6),
    )
    def test_add_properties(self, x, y):
        """
        Property-based test for addition with random floats.
        Tests properties like commutativity and identity.
        """
        # Filter out NaN and infinity
        assume(
            not any(
                map(lambda n: isinstance(n, float) and (isnan(n) or isinf(n)), [x, y])
            )
        )

        # Test commutativity: a + b = b + a
        assert self.calculator.add(x, y) == self.calculator.add(y, x)

        # Test identity: a + 0 = a
        assert self.calculator.add(x, 0) == x

        # Test associativity: (a + b) + c = a + (b + c)
        c = 42  # Fixed value for testing associativity
        epsilon = 1e-12
        # epsilon = 0
        left = self.calculator.add(self.calculator.add(x, y), c)
        right = self.calculator.add(x, self.calculator.add(y, c))
        assert abs(left - right) < epsilon

    @given(
        x=st.floats(min_value=-1e6, max_value=1e6),
        y=st.floats(min_value=-1e6, max_value=1e6),
    )
    def test_divide_properties(self, x, y):
        """
        Property-based test for division with random floats.
        Tests properties and edge cases of division.
        """
        # Filter out NaN, infinity, and zero for denominator
        assume(
            not any(
                map(lambda n: isinstance(n, float) and (isnan(n) or isinf(n)), [x, y])
            )
        )
        assume(y != 0)
        assume(abs(y) > 1e-300)

        result = self.calculator.divide(x, y)

        # Test division property: (a/b) * b ≈ a (within floating-point precision)
        epsilon = max(1e-10 * abs(x), 1e-10)  # Scale epsilon with input magnitude
        assert abs(result * y - x) < epsilon

        # Test division by 1 identity
        assert self.calculator.divide(x, 1) == x

    # Traditional unit tests for specific edge cases
    def test_add_zero(self):
        """Test adding zero."""
        result = self.calculator.add(5, 0)
        assert result == 5

    def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self.calculator.divide(5, 0)
        assert str(exc_info.value) == "Cannot divide by zero"

    @given(x=st.floats(min_value=-1e6, max_value=1e6))
    def test_divide_by_one(self, x):
        """Property-based test for division by one."""
        assume(not (isinstance(x, float) and (isnan(x) or isinf(x))))
        assert self.calculator.divide(x, 1) == x
