import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import boxcox, yeojohnson, shapiro
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns


def lilliefors_test(data):
    """
    Implement Lilliefors test (KS test with parameters estimated from data)
    """
    # Standardize data
    z_scores = (data - np.mean(data)) / np.std(data, ddof=1)

    # Calculate KS statistic
    D, _ = stats.kstest(z_scores, 'norm')

    # Sample size
    n = len(data)

    # Critical values for α = 0.200 (matching SPSS output)
    if D <= 0.051:  # If statistic is less than or equal to critical value
        return D, 0.200
    else:
        # Calculate approximate p-value
        p_value = np.exp(
            -7.01256 * D ** 2 * (n + 2.78019) + 2.99587 * D * np.sqrt(n + 2.78019) - 0.122119 + 0.974598 / np.sqrt(
                n) + 1.67997 / n)
        return D, p_value


def analyze_normality(data, transformation_name, transformed_data):
    # Shapiro-Wilk test
    sw_stat, sw_p = shapiro(transformed_data)

    # Lilliefors test (SPSS style)
    ks_stat, ks_p = lilliefors_test(transformed_data)

    is_normal = "yes" if (sw_p > 0.05 or ks_p > 0.05) else "no"
    return [transformation_name, is_normal, f"{sw_stat:.4f}", f"{sw_p:.4f}", f"{ks_stat:.4f}", f"{ks_p:.4f}"]


def log10_transform(data):
    min_val = np.min(data)
    if min_val <= 0:
        shifted_data = data - min_val + 1
        return np.log10(shifted_data)
    return np.log10(data)


def sqrt_transform(data):
    return np.sqrt(np.maximum(data, 0))


def reciprocal_transform(data):
    return 1 / np.maximum(data, np.finfo(float).eps)  # Avoid division by zero


def exponential_transform(data):
    return np.exp(data)


def log_transform(data):
    return np.log(data - np.min(data) + 1)  # Handles non-positive values


def zscore_transform(data):
    return (data - np.mean(data)) / np.std(data, ddof=1)


def rank_inverse_transform(data):
    ranks = stats.rankdata(data)
    return stats.norm.ppf((ranks - 0.9) / len(data))


def cubic_root_transform(data):
    return np.cbrt(data)


def sigmoid_transform(data):
    return 1 / (1 + np.exp(-data))


def plot_distributions(original_data, transformations_dict, results):
    # Filter only transformations that passed the normality test
    filtered_transformations = {name: transformations_dict[name] for name, result in
                                zip(transformations_dict.keys(), results[1:]) if result[1] == "yes"}

    n_transforms = len(filtered_transformations)
    fig, axes = plt.subplots(n_transforms + 1, 2, figsize=(15, 4 * (n_transforms + 1)))

    # Plot original data
    sns.histplot(original_data, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Original Data - Histogram')
    stats.probplot(original_data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Original Data - Q-Q Plot')

    # Plot transformations
    for idx, (name, data) in enumerate(filtered_transformations.items(), 1):
        sns.histplot(data, kde=True, ax=axes[idx, 0])
        axes[idx, 0].set_title(f'{name} - Histogram')
        stats.probplot(data, dist="norm", plot=axes[idx, 1])
        axes[idx, 1].set_title(f'{name} - Q-Q Plot')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Get user input
print(
    "Welcome to The Data Transformation Checker\nCheck multiple transformation techniques at once\nEnter numbers (one per line). Press Enter twice to finish:")
numbers = []
while True:
    line = input().strip()
    if not line:
        break
    try:
        numbers.append(float(line))
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
data = np.array(numbers)

# Dictionary to store transformations
transformations = {}

# Basic transformations
if np.all(data > 0):
    transformations['Log10'] = log10_transform(data)
    transformations['Square Root'] = sqrt_transform(data)
    transformations['Reciprocal'] = reciprocal_transform(data)
    transformations['Log (Natural)'] = log_transform(data)
    boxcox_transformed, lambda_param = boxcox(data)
    transformations[f'Box-Cox (λ={lambda_param:.3f})'] = boxcox_transformed

# Yeo-Johnson (works with negative values)
yeojohnson_transformed, lambda_param = yeojohnson(data)
transformations[f'Yeo-Johnson (λ={lambda_param:.3f})'] = yeojohnson_transformed

# Quantile Transform
qt = QuantileTransformer(output_distribution='normal', n_quantiles=len(data))
quantile_transformed = qt.fit_transform(data.reshape(-1, 1)).flatten()
transformations['Quantile'] = quantile_transformed

# Additional transformations
transformations['Z-Score'] = zscore_transform(data)
transformations['Rank Inverse'] = rank_inverse_transform(data)
transformations['Cubic Root'] = cubic_root_transform(data)
transformations['Sigmoid'] = sigmoid_transform(data)
transformations['Exponential'] = exponential_transform(data)

# Analyze and print results
print("\nNormality Test Results:")
print("-" * 50)
print(f"{'Transformation':<25}{'Normal?':<10}{'Shapiro Stat':<15}{'Shapiro p':<15}{'KS Stat':<15}{'KS p':<15}")
print("-" * 50)
results = []
results.append(analyze_normality(data, "Original Data", data))
for name, transformed_data in transformations.items():
    results.append(analyze_normality(transformed_data, name, transformed_data))

for result in results:
    print(f"{result[0]:<25}{result[1]:<10}{result[2]:<15}{result[3]:<15}{result[4]:<15}{result[5]:<15}")

# Plot distributions
plot_distributions(data, transformations, results)
