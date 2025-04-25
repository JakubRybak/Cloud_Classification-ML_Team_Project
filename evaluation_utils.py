# evaluation_utils.py
"""Functions for model evaluation and results visualization."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf # Needed for model.evaluate() potentially using TF operations

def plot_training_history(history, metrics_to_plot=['accuracy', 'loss', 'precision', 'recall', 'weighted_weather_penalty']):
    """
    Plots training and validation metrics from a Keras History object.

    Args:
        history (keras.callbacks.History): History object returned by model.fit().
        metrics_to_plot (list): List of metric names (strings) to plot.
                                Assumes validation metrics are prefixed with 'val_'.
    """
    num_metrics = len(metrics_to_plot)
    if num_metrics == 0:
        print("No metrics specified for plotting.")
        return

    plt.figure(figsize=(5 * num_metrics, 5))

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, num_metrics, i + 1)
        plot_metric = False
        # Check if the training metric exists in the history
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
            plot_metric = True

        # Check if the corresponding validation metric exists
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Validation {metric.capitalize()}')
            plot_metric = True

        if not plot_metric:
            # Neither train nor validation metric was found
             print(f"Warning: Metric '{metric}' or '{val_metric}' not found in history.")
             plt.title(f'{metric.capitalize()} (No Data)')
             continue # Continue to the next metric

        plt.title(f'{metric.capitalize()} History')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()

    plt.tight_layout() # Adjust subplots to prevent overlap
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