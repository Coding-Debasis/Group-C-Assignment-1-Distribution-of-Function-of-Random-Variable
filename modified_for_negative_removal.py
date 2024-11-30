
import numpy as np
import matplotlib.pyplot as plt
import warnings

# ------------------------------- PART 1: Distribution Functions -------------------------------

def geometric(p, vals):
    """Geometric Distribution: Probability mass function."""
    return [(1 - p)**(x - 1) * p if x > 0 else 0 for x in vals]

def binomial(n, p, vals):
    """Binomial Distribution: Probability mass function."""
    from math import comb
    return [comb(n, x) * (p**x) * ((1 - p)**(n - x)) if 0 <= x <= n else 0 for x in vals]

def poisson_dist(mu, vals):
    """Poisson Distribution: Probability mass function."""
    from math import exp, factorial
    return [(mu**x * exp(-mu)) / factorial(x) if x >= 0 else 0 for x in vals]

def uniform_dist(low, high, vals):
    """Uniform Distribution: Probability density function."""
    return [1 / (high - low) if low <= x <= high else 0 for x in vals]

# ------------------------------- PART 2: Generalized Function -------------------------------

def filter_values(vals):
    """
    Filters out negative values and displays what is removed and what is kept.
    :param vals: List of input values.
    :return: List of filtered (positive) values.
    """
    removed_values = [v for v in vals if v < 0]  # Collect negative values
    remaining_values = [v for v in vals if v >= 0]  # Collect positive values

    # Display the filtering process
    if removed_values:
        print(f"Removed negative values: {removed_values}")
    print(f"Positive values after filtering : {remaining_values}")

    if not remaining_values:
        raise ValueError("All input values are negative. Cannot compute probabilities.")

    return remaining_values


def get_probability_distribution(name, parameters, vals):
    """
    Generalized function to fetch probability values for a given distribution.
    :param name: Name of the distribution ('geometric', 'binomial', 'poisson', 'uniform').
    :param parameters: Parameters specific to the chosen distribution (dict).
    :param vals: List or array of values (x).
    :return: Filtered positive values and their probabilities.
    """
    # Filter and remove negative values
    filtered_vals = filter_values(vals)

    # Calculate probabilities based on distribution
    if name == 'geometric':
        probabilities = geometric(parameters['p'], filtered_vals)
    elif name == 'binomial':
        probabilities = binomial(parameters['n'], parameters['p'], filtered_vals)
    elif name == 'poisson':
        probabilities = poisson_dist(parameters['mu'], filtered_vals)
    elif name == 'uniform':
        probabilities = uniform_dist(parameters['low'], parameters['high'], filtered_vals)
    else:
        raise ValueError("Unsupported distribution name.")

    return filtered_vals, probabilities

# ------------------------------- PART 3: Plotting Function -------------------------------

def plot_distribution(name, parameters, vals):
    """
    Plot the distribution using matplotlib.
    :param name: Name of the distribution ('geometric', 'binomial', 'poisson', 'uniform').
    :param parameters: Parameters specific to the chosen distribution (dict).
    :param vals: List or array of values (x).
    """
    try:
        # Get filtered values and probabilities
        filtered_vals, probabilities = get_probability_distribution(name, parameters, vals)

        # Display calculated probabilities
        print("\nCalculated Probabilities:")
        for v, prob in zip(filtered_vals, probabilities):
            print(f"Value: {v}, Probability: {prob:.4f}")

        # Plot the filtered values and probabilities
        plt.style.use('ggplot')
        plt.bar(filtered_vals, probabilities, alpha=0.7, label=name.capitalize())
        plt.xlabel("Values")
        plt.ylabel("Probability")
        plt.title(f"{name.capitalize()} Distribution")
        plt.legend()
        plt.show()
    except ValueError as e:
        print(f"Error: {e}")

# ------------------------------- PART 4: Input from User -------------------------------

def get_user_input():
    """
    Get distribution name, parameters, and range of values from the user.
    """
    name = input("Enter the name of the distribution (geometric, binomial, poisson, uniform): ").strip().lower()

    # Get the range of x values
    start = int(input("Enter the start of the range of x values (integer): "))
    end = int(input("Enter the end of the range of x values (integer): "))
    vals = np.arange(start, end + 1)

    # Get distribution-specific parameters
    parameters = {}
    if name == 'geometric':
        parameters['p'] = float(input("Enter the probability of success (p): "))
    elif name == 'binomial':
        parameters['n'] = int(input("Enter the number of trials (n): "))
        parameters['p'] = float(input("Enter the probability of success (p): "))
    elif name == 'poisson':
        parameters['mu'] = float(input("Enter the mean (mu): "))
    elif name == 'uniform':
        parameters['low'] = float(input("Enter the lower bound (low): "))
        parameters['high'] = float(input("Enter the upper bound (high): "))
    else:
        raise ValueError("Invalid distribution name.")

    return name, parameters, vals

# ------------------------------- PART 5: Main Execution -------------------------------

if __name__ == "__main__":
    try:
        # Get user input
        name, parameters, vals = get_user_input()

        # Plot the distribution
        plot_distribution(name, parameters, vals)
    except ValueError as e:
        print(f"Error: {e}")
