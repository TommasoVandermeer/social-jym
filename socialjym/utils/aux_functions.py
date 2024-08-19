def is_multiple(number, dividend, tolerance=1e-7):
    """
    Checks if a number (also a float) is a multiple of another number within a given tolerance error.
    """
    mod = number % dividend
    return (abs(mod) <= tolerance) or (abs(dividend - mod) <= tolerance)