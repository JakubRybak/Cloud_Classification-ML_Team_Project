# evaluation_utils.py
"""Functions for model evaluation and results visualization."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf # Needed for model.evaluate() potentially using TF operations

def plot_training_history(history, metrics_to_plot=['accuracy', 'loss', 'precision', 'recall', 'weighted_weather_penalty', 'auc']):
    """
    Plots training and validation metrics from a Keras History object
    in a 2x3 grid layout. Sets the Y-axis limit to [0, 1] for specified plots.

    Args:
        history (keras.callbacks.History): History object returned by model.fit().
        metrics_to_plot (list): List of metric names (strings) to plot.
                                Should ideally contain 6 or fewer metrics.
                                Assumes validation metrics are prefixed with 'val_'.
    """
    num_metrics = len(metrics_to_plot)
    if num_metrics == 0:
        print("No metrics specified for plotting.")
        return

    plt.figure(figsize=(18, 10)) # Figure size for 2x3 grid

    # Define indices (0-based) of plots to have fixed Y-axis [0, 1]
    # Corresponds to 1st, 3rd, 4th, 5th, 6th plot in the 2x3 grid
    fixed_ylim_indices = {0, 2, 3, 4, 5}

    # Limit plotting to max 6 metrics for the 2x3 grid
    if num_metrics > 6:
        print("Warning: More than 6 metrics specified. Only the first 6 will be plotted.")
        num_metrics_to_plot = 6
    else:
        num_metrics_to_plot = num_metrics

    for i in range(num_metrics_to_plot):
        metric = metrics_to_plot[i]
        plt.subplot(2, 3, i + 1) # Create subplot in 2x3 grid

        plot_metric = False
        # Check and plot training metric
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
            plot_metric = True

        # Check and plot validation metric
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Validation {metric.capitalize()}')
            plot_metric = True

        if not plot_metric:
            print(f"Warning: Metric '{metric}' or '{val_metric}' not found in history for subplot {i+1}.")
            plt.title(f'{metric.capitalize()} (No Data)')
            continue # Skip rest of the loop for this subplot

        plt.title(f'{metric.capitalize()} History')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

        # --- ZMIANA: Ustawienie limitu osi Y ---
        # Check if the current plot index `i` is in the set of indices requiring fixed Y-lim
        if i in fixed_ylim_indices:
            plt.ylim(0, 1) # Set Y-axis limits from 0 to 1
        plt.xlim(left=0)

    plt.tight_layout(pad=3.0) # Adjust layout
    plt.show()

def evaluate_model(model, validation_generator, custom_metric_name='weighted_weather_penalty'):
    """
    Performs comprehensive model evaluation using model.evaluate() and sklearn metrics.

    Displays standard metrics, confusion matrix, and classification report.

    Args:
        model (keras.Model): The trained Keras model to evaluate.
        validation_generator (keras.preprocessing.image.DirectoryIterator):
            Generator for the validation data. Should have shuffle=False.
        custom_metric_name (str, optional): The name of the custom metric (like WeightedWeatherPenalty)
                                             to specifically look for and report from evaluate results.
                                             Defaults to 'weighted_weather_penalty'.

    Returns:
        dict or None: Results from model.evaluate() as a dictionary, or None if evaluation failed.
    """

    print("\n" + "="*20 + " Starting Model Evaluation " + "="*20)
    evaluation_results = None # Initialize results

    # --- 1. Use model.evaluate() - Generally the most accurate method ---
    print("\nRunning model.evaluate()...")
    try:
        # Ensure the generator is at the beginning before evaluation
        validation_generator.reset()
        evaluation_results = model.evaluate(
            validation_generator,
            verbose=1,
            return_dict=True # Return results as a dictionary {metric_name: value}
        )
        print("model.evaluate() finished.")
        print("\n--- Results from model.evaluate() ---")
        if evaluation_results:
            for name, value in evaluation_results.items():
                print(f"{name}: {value:.4f}")
        else:
            print("model.evaluate() did not return results.")
        print("-------------------------------------")

        # Display the specific custom metric result if found
        if evaluation_results and custom_metric_name in evaluation_results:
             penalty_score = evaluation_results[custom_metric_name]
             print(f"\nResult for '{custom_metric_name}': {penalty_score:.4f}")
             print("(Remember: Lower values are typically better for penalty metrics!)")
        elif evaluation_results:
             print(f"\nWarning: Custom metric '{custom_metric_name}' not found in evaluate() results.")

    except Exception as e:
        print(f"\nError during model.evaluate(): {e}")
        print("Could not calculate metrics using evaluate(). Check generator and model compatibility.")
        evaluation_results = None # Ensure results are None after error

    # --- 2. Calculate Confusion Matrix and Classification Report using predictions ---
    # Note: This relies on generator batch order matching labels if shuffle=False was set correctly.
    print("\nGenerating predictions for Confusion Matrix and Classification Report...")
    try:
        validation_generator.reset() # Reset generator again just in case
        y_pred_probs = model.predict(validation_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1) # Get predicted class indices
        y_true = validation_generator.classes      # Get true class indices
        class_names = list(validation_generator.class_indices.keys()) # Get class names

        # Display warning if shuffle is enabled (shouldn't be for validation)
        if validation_generator.shuffle:
            print("\n*** WARNING ***")
            print("Validation generator has shuffle=True. Confusion Matrix and Classification Report might be inaccurate.")

        # Calculate and plot Confusion Matrix
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        # Calculate and print Classification Report
        print("\n--- Classification Report ---")
        # Set zero_division=0 to avoid warnings for classes with no predicted samples
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False, zero_division=0)
        print(report)
        print("---------------------------")

    except Exception as e:
        print(f"\nError during Confusion Matrix or Classification Report generation: {e}")

    print("="*25 + " Evaluation Finished " + "="*25 + "\n")

    # Return results from evaluate() for potential further use
    return evaluation_results