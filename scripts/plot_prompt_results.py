from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    project_path = Path(__file__).resolve().parents[1]

    output_folder = project_path / 'outputs'

    results_path = output_folder / 'prompt_attack_results.csv'

    results = pd.read_csv(results_path)

    total_examples = len(results)

    neutral_correct = (
        results['neutral_answer'] == results['real_class']
    ).sum()

    manipulative_correct = (
        results['manipulative_answer'] == results['real_class']
    ).sum()

    classifier_correct = (
        results['classifier_class'] == results['real_class']
    ).sum()

    final_correct = (
        results['final_answer'] == results['real_class']
    ).sum()

    neutral_accuracy = neutral_correct / total_examples
    manipulative_accuracy = manipulative_correct / total_examples
    classifier_accuracy = classifier_correct / total_examples
    final_accuracy = final_correct / total_examples

    successful_attacks = results['attack_success'].sum()
    recovered_examples = results['recovery_success'].sum()

    if neutral_correct > 0:
        attack_success_rate = successful_attacks / neutral_correct
    else:
        attack_success_rate = 0

    if successful_attacks > 0:
        recovery_rate = recovered_examples / successful_attacks
    else:
        recovery_rate = 0

    classifier_agreement = (
        results['final_answer'] == results['classifier_class']
    ).sum()

    classifier_agreement_rate = (
        classifier_agreement / total_examples
    )

    accuracy_names = [
        'Neutralen\nSmolVLM2',
        'Po\nnapadot',
        'ViT',
        'Konecen\nodgovor'
    ]

    accuracy_values = [
        neutral_accuracy * 100,
        manipulative_accuracy * 100,
        classifier_accuracy * 100,
        final_accuracy * 100
    ]

    colors = [
        'cornflowerblue',
        'tomato',
        'orange',
        'mediumseagreen'
    ]

    figure, axes = plt.subplots(figsize=(9, 6))

    bars = axes.bar(
        accuracy_names,
        accuracy_values,
        color=colors
    )

    axes.set_title('Sporedba na tochnosta')
    axes.set_ylabel('Accuracy (%)')
    axes.set_ylim(0, 110)

    for bar, value in zip(bars, accuracy_values):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f'{value:.1f}%',
            ha='center'
        )

    figure.tight_layout()

    accuracy_graph_path = output_folder / 'accuracy_comparison.png'

    figure.savefig(
        accuracy_graph_path,
        dpi=300
    )

    plt.close(figure)

    metric_names = [
        'Uspeshen\nnapad',
        'Oporavuvanje',
        'Soglasnost\nso ViT'
    ]

    metric_values = [
        attack_success_rate * 100,
        recovery_rate * 100,
        classifier_agreement_rate * 100
    ]

    figure, axes = plt.subplots(figsize=(8, 6))

    bars = axes.bar(
        metric_names,
        metric_values,
        color=['tomato', 'mediumseagreen', 'cornflowerblue']
    )

    axes.set_title('Vlijanie na promptot i ViT')
    axes.set_ylabel('Procent (%)')
    axes.set_ylim(0, 110)

    for bar, value in zip(bars, metric_values):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f'{value:.1f}%',
            ha='center'
        )

    figure.tight_layout()

    metrics_graph_path = output_folder / 'influence_metrics.png'

    figure.savefig(
        metrics_graph_path,
        dpi=300
    )

    plt.close(figure)

    print('Vkupno primeri:', total_examples)
    print('Neutralna accuracy:', neutral_accuracy)
    print('Accuracy po napadot:', manipulative_accuracy)
    print('ViT accuracy:', classifier_accuracy)
    print('Finalna accuracy:', final_accuracy)
    print('Attack success rate:', attack_success_rate)
    print('Recovery rate:', recovery_rate)
    print('Soglasnost so ViT:', classifier_agreement_rate)
    print()
    print('Grafikonite se zacuvani vo:')
    print(accuracy_graph_path)
    print(metrics_graph_path)
