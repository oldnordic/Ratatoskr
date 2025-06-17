# This file is for testing the AI Coder Assistant's scan functionality.

def calculate_area(length, width):
    """
    Calculates the area of a rectangle.
    """
    area = length * width
    return area

def main_program():
    side1 = 10
    side2 = 5

    # This line has an intentional error: 'lenght' is misspelled
    result = calculate_area(lenght, side2)

    print(f"The area is: {result}")

if __name__ == "__main__":
    main_program()
