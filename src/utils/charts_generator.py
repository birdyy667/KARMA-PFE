"""
Data visualization module for negotiation experiments.
Generates performance, time bounds, financial, and profile analysis charts.
"""

import logging
from pathlib import Path
from typing import List, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure module-level logger
logger = logging.getLogger(__name__)

# Fallback output directory based on current architecture (app/static/images)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "static" / "images"


def _ensure_directory_exists(file_path: Path) -> None:
    """Helper to ensure the target directory exists before saving a plot."""
    file_path.parent.mkdir(parents=True, exist_ok=True)


# --- 1. GENERAL PERFORMANCE GRAPH (Vertical Layout + English) ---
def generate_negotiation_metrics(data_source: List[List[Any]],
                                 output_filename: Union[str, Path] = "static/images/rapport_negociation.png") -> None:
    """
    Generates a vertical bar chart comparing success rates and times between Manual and AI modes.

    Args:
        data_source (List[List[Any]]): The raw simulation data.
        output_filename (Union[str, Path]): Target file path for the generated PNG.
    """
    if not data_source:
        logger.warning("No data provided to generate_negotiation_metrics. Skipping plot.")
        return

    columns = ['Mode', 'Success', 'Time', 'Slot', 'Profile', 'Savings']
    df = pd.DataFrame(data_source, columns=columns)


    #df['Mode'] = df['Mode'].map({'Manuel': 'Manual', 'Agent IA': 'AI Agent'})

    df_success = df.groupby('Mode')['Success'].mean().reset_index()
    df_success['Value'] = df_success['Success'] * 100

    df_time = df[df['Success'] == 1].groupby('Mode')['Time'].mean().reset_index()
    df_time['Value'] = df_time['Time']

    sns.set_theme(style="whitegrid")

    # Stacking vertically (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    fig.suptitle("Negotiation Performance: Manual vs AI Agent", fontsize=14, weight='bold', y=0.98)

    # Top Chart: Success Rate
    barplot1 = sns.barplot(x='Mode', y='Value', data=df_success, ax=axes[0], palette=['#10b981', '#b3cde3'])
    axes[0].set_title('Successful Reservations Rate', fontsize=11, weight='600')
    axes[0].set_ylabel('Percentage (%)', fontsize=9)
    axes[0].set_xlabel('Negotiation Mode', fontsize=9)
    axes[0].set_ylim(0, 110)
    for p in barplot1.patches:
        axes[0].annotate(f"{int(p.get_height())}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                         ha='center', va='center', color='black', weight='bold', fontsize=10)

    # Bottom Chart: Average Time
    barplot2 = sns.barplot(x='Mode', y='Value', data=df_time, ax=axes[1], palette=['#10b981', '#b3cde3'])
    axes[1].set_title('Average Time to Reach an Agreement', fontsize=11, weight='600')
    axes[1].set_ylabel('Time (seconds)', fontsize=9)
    axes[1].set_xlabel('Negotiation Mode', fontsize=9)
    axes[1].set_ylim(0, 70)
    for p in barplot2.patches:
        axes[1].annotate(f"{int(p.get_height())}s", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                         ha='center', va='center', color='black', weight='bold', fontsize=10)

    plt.tight_layout()

    target_path = Path(output_filename) if Path(output_filename).is_absolute() else DEFAULT_OUTPUT_DIR / Path(
        output_filename).name
    _ensure_directory_exists(target_path)

    plt.savefig(target_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- 2. BOUNDED TIME SIMULATION (English + Small Fonts) ---
def generate_bounded_time_stats(data_source: List[List[Any]],
                                output_filename: Union[str, Path] = "static/images/stats_temps_borne.png") -> None:
    """
    Generates a line chart evaluating AI vs Manual success rates under strict time limits.

    Args:
        data_source (List[List[Any]]): The raw simulation data.
        output_filename (Union[str, Path]): Target file path for the generated PNG.
    """
    if not data_source:
        logger.warning("No data provided to generate_bounded_time_stats. Skipping plot.")
        return

    columns = ['Mode', 'Success', 'Time', 'Slot', 'Profile', 'Savings']
    df = pd.DataFrame(data_source, columns=columns)

    time_limits = [1, 5, 10, 30, 60]
    simulation_results = []

    for limit in time_limits:
        for mode in ['Manual', 'AI Agent']:
            sub_df = df[df['Mode'] == mode]
            fixed_success = sub_df.apply(lambda r: 1 if (r['Success'] == 1 and r['Time'] <= limit) else 0, axis=1)
            success_rate = fixed_success.mean() * 100 if len(fixed_success) > 0 else 0.0

            mode_en = mode
            simulation_results.append({
                'Time Limit': f"{limit}s",
                'Mode': mode_en,
                'Success Rate (%)': success_rate
            })

    df_sim = pd.DataFrame(simulation_results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))

    sns.lineplot(x='Time Limit', y='Success Rate (%)', hue='Mode', data=df_sim, marker="o", linewidth=2.5,
                 palette={'Manual': '#b3cde3', 'AI Agent': '#10b981'})

    plt.title("Simulation: Success Rate under Strict Time Constraints", fontsize=10, weight='bold', pad=20)
    plt.ylim(-5, 105)
    plt.ylabel("Success Rate (%)", fontsize=9)
    plt.xlabel("Maximum allowed time to conclude negotiation", fontsize=9)

    target_path = Path(output_filename) if Path(output_filename).is_absolute() else DEFAULT_OUTPUT_DIR / Path(
        output_filename).name
    _ensure_directory_exists(target_path)

    plt.savefig(target_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- 3. FINANCIAL ANALYSIS GRAPH (Vertical Layout + English) ---
def generate_cost_comparison_stats(data_source: List[List[Any]],
                                   output_filename: Union[str, Path] = "static/images/stats_couts.png") -> None:
    """
    Generates bar charts comparing the financial impact for both the EV User and the Station.

    Args:
        data_source (List[List[Any]]): The raw simulation data.
        output_filename (Union[str, Path]): Target file path for the generated PNG.
    """
    if not data_source:
        logger.warning("No data provided to generate_cost_comparison_stats. Skipping plot.")
        return

    columns = ['Mode', 'Success', 'Time', 'Slot', 'Profile', 'Savings']
    df = pd.DataFrame(data_source, columns=columns)

    # Baseline hypotheses
    SPOT_PRICE = 0.55
    OCCUPANCY_RATE_NO_NEGO = 0.30
    OCCUPANCY_RATE_WITH_NEGO = 0.90

    # 1. Calculation for the EV User
    df['Paid_Price'] = df.apply(
        lambda r: SPOT_PRICE * (1 - (r['Savings'] / 100.0)) if r['Success'] == 1 else SPOT_PRICE, axis=1)
    avg_ev_cost = df['Paid_Price'].mean()

    # 2. Calculation for the Station
    avg_revenue_no_nego = SPOT_PRICE * OCCUPANCY_RATE_NO_NEGO
    avg_revenue_with_nego = avg_ev_cost * OCCUPANCY_RATE_WITH_NEGO

    labels = ['Without Negotiation\n(Fixed Spot Prices)', 'With AI Agent\n(Dynamic Pricing)']

    sns.set_theme(style="whitegrid")

    # Vertical format: 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    fig.suptitle("Financial Analysis", fontsize=14, weight='bold', y=0.98)

    # Top Graph: EV User perspective
    barplot1 = sns.barplot(x=labels, y=[SPOT_PRICE, avg_ev_cost], ax=axes[0], palette=['#ef4444', '#10b981'])
    axes[0].set_title('Average kWh Cost for the EV User', fontsize=11, weight='600')
    axes[0].set_ylabel('Price paid (€ / kWh)', fontsize=9)
    axes[0].set_ylim(0, 0.7)
    for p in barplot1.patches:
        axes[0].annotate(f"€{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                         ha='center', color='black', weight='bold', fontsize=10)

    # Bottom Graph: Station perspective
    barplot2 = sns.barplot(x=labels, y=[avg_revenue_no_nego, avg_revenue_with_nego], ax=axes[1],
                           palette=['#cbd5e1', '#3b82f6'])
    axes[1].set_title('Average Charging Station Revenue per Slot', fontsize=11, weight='600')
    axes[1].set_ylabel('Smoothed Revenue (€)', fontsize=9)
    axes[1].set_ylim(0, 0.7)
    for p in barplot2.patches:
        axes[1].annotate(f"€{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                         ha='center', color='black', weight='bold', fontsize=10)

    plt.tight_layout()

    target_path = Path(output_filename) if Path(output_filename).is_absolute() else DEFAULT_OUTPUT_DIR / Path(
        output_filename).name
    _ensure_directory_exists(target_path)

    plt.savefig(target_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- 4. PROFILE BENEFITS GRAPH (English) ---
def generate_profile_stats(data_source: List[List[Any]],
                           output_filename: Union[str, Path] = "static/images/stats_profils.png") -> None:
    """
    Generates a bar chart comparing the economic savings based on the user's loyalty profile.

    Args:
        data_source (List[List[Any]]): The raw simulation data.
        output_filename (Union[str, Path]): Target file path for the generated PNG.
    """
    if not data_source:
        logger.warning("No data provided to generate_profile_stats. Skipping plot.")
        return

    columns = ['Mode', 'Success', 'Time', 'Slot', 'Profile', 'Savings']
    df = pd.DataFrame(data_source, columns=columns)

    df['Profile'] = df['Profile'].map({'Standard': 'Standard', 'Regular': 'Regular'})
    df_profile = df.groupby('Profile')['Savings'].mean().reset_index()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))

    barplot = sns.barplot(x='Profile', y='Savings', data=df_profile, palette=['#b3cde3', '#10b981'])

    plt.title("Average Financial Advantage by User Profile", fontsize=11, weight='bold', pad=15)
    plt.ylabel("Saved Energy Cost (%)", fontsize=9)
    plt.xlabel("Subscription Type", fontsize=9)

    plt.ylim(0, max(df_profile['Savings'].max() + 15, 50.0))

    for p in barplot.patches:
        plt.annotate(f"{int(p.get_height())}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                     ha='center', va='center', color='black', weight='bold', fontsize=11)

    plt.tight_layout()

    target_path = Path(output_filename) if Path(output_filename).is_absolute() else DEFAULT_OUTPUT_DIR / Path(
        output_filename).name
    _ensure_directory_exists(target_path)

    plt.savefig(target_path, dpi=300, bbox_inches='tight')
    plt.close()