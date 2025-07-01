"""Demo-Modul für Pynguin – generiert automatisch Tests."""

def add(x: int, y: int):
    """
    >>> add(2, 20)
    22
    
    >>> add(0, 5)
    5


    Preconditions
    x > 1
    y > 10
    """
    if x <= 0 or y <= 0:
        raise ValueError("x und y müssen > 0 sein")
    return x + y

