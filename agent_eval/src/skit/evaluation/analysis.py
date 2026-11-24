"""
Analysis and visualization functions for agent evaluation.

Provides comprehensive analysis including condition rating accuracy,
answer correctness evaluation, and image relevance metrics with
plotly visualizations.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .metrics import calculate_jaccard_similarity


def get_display_name(agent_name: str, agent_name_mapping: Dict[str, str] | None = None) -> str:
    """
    Get display name for an agent from mapping, with fallback to original name.

    Args:
        agent_name: Original agent configuration name (e.g., "embodied_sg_oracle_both")
        agent_name_mapping: Optional mapping of agent names to display names

    Returns:
        Human-readable display name if mapping exists, otherwise original name

    Example:
        >>> mapping = {"embodied_sg_oracle_both": "Embodied Agent w/ SG + Oracle"}
        >>> get_display_name("embodied_sg_oracle_both", mapping)
        'Embodied Agent w/ SG + Oracle'
        >>> get_display_name("unknown_agent", mapping)
        'unknown_agent'
    """
    if agent_name_mapping is None:
        return agent_name
    return agent_name_mapping.get(agent_name, agent_name)


def analyze_agent_responses(
    df: pd.DataFrame,
    output_prefix: str = "",
    agent_name_mapping: Dict[str, str] | None = None
):
    """
    Comprehensive analysis of agent responses with visualizations.

    Generates:
    - Histogram of condition rating distributions
    - Error distribution plots
    - Performance metrics (MAE, RMSE)
    - Accuracy metrics (exact match, within ±1)

    Args:
        df: DataFrame with agent results
        output_prefix: Prefix for output file names (e.g., "grok-4-fast_")
        agent_name_mapping: Optional mapping of agent names to display names
    """
    # Get agent rating columns
    agent_rating_columns = [
        col for col in df.columns
        if col.endswith('_condition_rating') and col != 'reference_condition_rating'
    ]

    if not agent_rating_columns:
        print("No agent rating columns found. Skipping condition rating analysis.")
        return

    # 1. Histogram: Reference vs Agent Ratings
    fig = go.Figure()

    # Add reference condition rating histogram
    fig.add_trace(go.Histogram(
        x=df['reference_condition_rating'],
        name='Reference',
        opacity=0.9,
        nbinsx=10,
        histnorm='probability'
    ))

    # Add agent condition rating histograms
    for col in agent_rating_columns:
        agent_name = col.replace('_condition_rating', '')
        display_name = get_display_name(agent_name, agent_name_mapping)
        fig.add_trace(go.Histogram(
            x=df[col],
            name=display_name,
            opacity=0.9,
            nbinsx=10,
            histnorm='probability'
        ))

    fig.update_layout(
        title='Condition Rating Distribution: Reference vs Agents',
        xaxis_title='Condition Rating',
        yaxis_title='Probability',
        barmode='group',
        bargap=0.1,
        bargroupgap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )

    filename = f"{output_prefix}condition_rating_distribution.png"
    fig.write_image(filename, width=1200, height=600)
    print(f"\nSaved histogram to {filename}")

    # 2. Error Distribution Histogram
    fig_error = go.Figure()

    for col in agent_rating_columns:
        agent_name = col.replace('_condition_rating', '')
        display_name = get_display_name(agent_name, agent_name_mapping)

        # Calculate errors (Agent - Reference)
        mask = df[col].notna() & df['reference_condition_rating'].notna()
        errors = df.loc[mask, col] - df.loc[mask, 'reference_condition_rating']

        fig_error.add_trace(go.Histogram(
            x=errors,
            name=display_name,
            opacity=0.7,
            nbinsx=15
        ))

    fig_error.update_layout(
        title='Error Distribution: Agent - Reference Rating',
        xaxis_title='Error (Agent - Reference)',
        yaxis_title='Count',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )

    # Add vertical line at zero error
    fig_error.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Perfect")

    filename = f"{output_prefix}error_distribution.png"
    fig_error.write_image(filename, width=1200, height=500)
    print(f"Saved error distribution to {filename}")

    # 3. Metrics Summary Bar Chart
    metrics_data = {'Agent': [], 'MAE': [], 'RMSE': []}

    for col in agent_rating_columns:
        agent_name = col.replace('_condition_rating', '')
        display_name = get_display_name(agent_name, agent_name_mapping)

        # Filter out NaN values
        mask = df[col].notna() & df['reference_condition_rating'].notna()
        ref_values = df.loc[mask, 'reference_condition_rating']
        agent_values = df.loc[mask, col]

        if len(ref_values) > 0:
            mae = mean_absolute_error(ref_values, agent_values)
            rmse = np.sqrt(mean_squared_error(ref_values, agent_values))

            metrics_data['Agent'].append(display_name)
            metrics_data['MAE'].append(mae)
            metrics_data['RMSE'].append(rmse)

    metrics_df = pd.DataFrame(metrics_data)

    # Create subplots for metrics
    fig_metrics = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Mean Absolute Error (Lower is Better)',
                       'Root Mean Squared Error (Lower is Better)'),
        horizontal_spacing=0.15
    )

    # MAE
    fig_metrics.add_trace(
        go.Bar(x=metrics_df['Agent'], y=metrics_df['MAE'], name='MAE', marker_color='indianred'),
        row=1, col=1
    )

    # RMSE
    fig_metrics.add_trace(
        go.Bar(x=metrics_df['Agent'], y=metrics_df['RMSE'], name='RMSE', marker_color='lightsalmon'),
        row=1, col=2
    )

    fig_metrics.update_layout(
        title='Performance Metrics: Agent Alignment with Reference',
        height=500,
        showlegend=False
    )

    fig_metrics.update_yaxes(title_text="MAE", row=1, col=1)
    fig_metrics.update_yaxes(title_text="RMSE", row=1, col=2)

    filename = f"{output_prefix}metrics_summary.png"
    fig_metrics.write_image(filename, width=1400, height=500)
    print(f"Saved metrics summary to {filename}")

    # Print metrics table
    print("\n=== Performance Metrics ===")
    print(metrics_df.to_string(index=False))
    print("\nNote: Lower MAE/RMSE = better alignment.")

    # 4. Accuracy Metrics
    accuracy_data = {'Agent': [], 'Exact Match %': [], 'Within ±1 %': []}

    print("\n=== Accuracy Metrics ===")
    for col in agent_rating_columns:
        agent_name = col.replace('_condition_rating', '')
        display_name = get_display_name(agent_name, agent_name_mapping)

        # Filter out NaN values
        mask = df[col].notna() & df['reference_condition_rating'].notna()
        ref_values = df.loc[mask, 'reference_condition_rating']
        agent_values = df.loc[mask, col]

        if len(ref_values) > 0:
            # Calculate exact matches
            exact_matches = (agent_values == ref_values).sum()
            exact_match_pct = (exact_matches / len(ref_values)) * 100

            # Calculate within ±1
            errors = abs(agent_values - ref_values)
            within_one = (errors <= 1).sum()
            within_one_pct = (within_one / len(ref_values)) * 100

            accuracy_data['Agent'].append(display_name)
            accuracy_data['Exact Match %'].append(exact_match_pct)
            accuracy_data['Within ±1 %'].append(within_one_pct)

            print(f"\n{display_name}:")
            print(f"  Exact Match: {exact_matches}/{len(ref_values)} ({exact_match_pct:.1f}%)")
            print(f"  Within ±1:   {within_one}/{len(ref_values)} ({within_one_pct:.1f}%)")

    accuracy_df = pd.DataFrame(accuracy_data)

    # Create accuracy visualization
    fig_accuracy = go.Figure()

    fig_accuracy.add_trace(go.Bar(
        name='Exact Match',
        x=accuracy_df['Agent'],
        y=accuracy_df['Exact Match %'],
        marker_color='steelblue'
    ))

    fig_accuracy.add_trace(go.Bar(
        name='Within ±1',
        x=accuracy_df['Agent'],
        y=accuracy_df['Within ±1 %'],
        marker_color='lightblue'
    ))

    fig_accuracy.update_layout(
        title='Prediction Accuracy: Exact Match vs Within ±1 Rating',
        xaxis_title='Agent',
        yaxis_title='Accuracy (%)',
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    filename = f"{output_prefix}accuracy_comparison.png"
    fig_accuracy.write_image(filename, width=1200, height=500)
    print(f"\nSaved accuracy comparison to {filename}")

    # 5. Rating Distribution Analysis
    print("\n=== Rating Distribution Analysis ===")

    # Reference distribution
    print("\nReference Condition Ratings:")
    ref_dist = df['reference_condition_rating'].value_counts().sort_index()
    print(ref_dist)
    print(f"Mean: {df['reference_condition_rating'].mean():.2f}, Std Dev: {df['reference_condition_rating'].std():.2f}")

    # Agent distributions
    for col in agent_rating_columns:
        agent_name = col.replace('_condition_rating', '')
        display_name = get_display_name(agent_name, agent_name_mapping)
        mask = df[col].notna()
        agent_vals = df.loc[mask, col]

        print(f"\n{display_name}:")
        agent_dist = agent_vals.value_counts().sort_index()
        print(agent_dist)
        print(f"Mean: {agent_vals.mean():.2f}, Std Dev: {agent_vals.std():.2f}")


def analyze_answer_correctness(
    df: pd.DataFrame,
    output_prefix: str = "",
    agent_name_mapping: Dict[str, str] | None = None
):
    """
    Analyze and visualize answer correctness scores from LLM judge and/or VLM judge.

    Generates comprehensive visualizations including:
    - Average score comparisons
    - Statistical summary tables
    - Separate visualizations for text-based and VLM-based scores

    Args:
        df: DataFrame with agent results and correctness scores
        output_prefix: Prefix for output file names (e.g., "grok-4-fast_")
        agent_name_mapping: Optional mapping of agent names to display names
    """
    # Get text-based correctness score columns (LLM judge)
    text_correctness_columns = [
        col for col in df.columns
        if col.endswith('_correctness_score') and not col.endswith('_vlm_correctness_score')
    ]

    # Get VLM-based correctness score columns
    vlm_correctness_columns = [
        col for col in df.columns
        if col.endswith('_vlm_correctness_score')
    ]

    if not text_correctness_columns and not vlm_correctness_columns:
        print("\n=== No Correctness Scores Found ===")
        print("Run with run_llm_judge=True or run_vlm_judge=True to generate answer correctness scores.\n")
        return

    # Process text-based LLM judge scores
    if text_correctness_columns:
        _analyze_correctness_scores(df, text_correctness_columns, output_prefix, agent_name_mapping, score_type="text")

    # Process VLM judge scores
    if vlm_correctness_columns:
        _analyze_correctness_scores(df, vlm_correctness_columns, output_prefix, agent_name_mapping, score_type="vlm")


def _analyze_correctness_scores(
    df: pd.DataFrame,
    correctness_columns: List[str],
    output_prefix: str,
    agent_name_mapping: Dict[str, str] | None,
    score_type: str = "text"
):
    """
    Helper function to analyze and visualize a specific type of correctness scores.

    Args:
        df: DataFrame with agent results
        correctness_columns: List of correctness score column names
        output_prefix: Prefix for output files
        agent_name_mapping: Optional agent name mapping
        score_type: "text" for LLM judge or "vlm" for VLM judge
    """
    score_label = "Text-based LLM" if score_type == "text" else "VLM (Image-grounded)"
    suffix = "" if score_type == "text" else "_vlm"

    print("\n" + "="*80)
    print(f"=== {score_label} Answer Correctness Analysis ===")
    print("="*80)

    # 1. Average Correctness Score Bar Chart
    print(f"Generating {score_type} average correctness scores bar chart...")
    stats_data = {'Agent': [], 'Mean Score': [], 'Median Score': []}

    # Strip the appropriate suffix based on score type
    suffix_to_remove = '_vlm_correctness_score' if score_type == 'vlm' else '_correctness_score'

    for col in correctness_columns:
        agent_name = col.replace(suffix_to_remove, '')
        display_name = get_display_name(agent_name, agent_name_mapping)
        mask = df[col].notna()
        correctness_values = df.loc[mask, col]

        if len(correctness_values) > 0:
            stats_data['Agent'].append(display_name)
            stats_data['Mean Score'].append(correctness_values.mean())
            stats_data['Median Score'].append(correctness_values.median())

    stats_df = pd.DataFrame(stats_data)

    fig_avg = go.Figure()

    fig_avg.add_trace(go.Bar(
        name='Mean Correctness Score',
        x=stats_df['Agent'],
        y=stats_df['Mean Score'],
        marker_color='steelblue',
        text=stats_df['Mean Score'].round(3),
        textposition='outside'
    ))

    fig_avg.add_trace(go.Bar(
        name='Median Correctness Score',
        x=stats_df['Agent'],
        y=stats_df['Median Score'],
        marker_color='lightblue',
        text=stats_df['Median Score'].round(3),
        textposition='outside'
    ))

    fig_avg.update_layout(
        title=f'{score_label} Judge: Average Answer Correctness Scores by Agent (Higher is Better)',
        xaxis_title='Agent',
        yaxis_title='Correctness Score',
        yaxis_range=[0, 1.1],
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    filename = f"{output_prefix}average_correctness_scores{suffix}.png"
    fig_avg.write_image(filename, width=1200, height=500)
    print(f"Saved to {filename}")

    # 2. Statistical Summary Table
    print(f"\n=== {score_label} Answer Correctness Statistics ===")
    print("\nDetailed statistics for each agent:\n")

    detailed_stats = {
        'Agent': [],
        'Count': [],
        'Mean': [],
        'Median': [],
        'Std Dev': [],
        'Min': [],
        'Max': [],
        'High Quality (>=0.7)': [],
        'High Quality %': [],
        'Moderate+ (>=0.5)': [],
        'Moderate+ %': []
    }

    for col in correctness_columns:
        agent_name = col.replace(suffix_to_remove, '')
        display_name = get_display_name(agent_name, agent_name_mapping)
        mask = df[col].notna()
        correctness_values = df.loc[mask, col]

        if len(correctness_values) > 0:
            high_quality_count = (correctness_values >= 0.7).sum()
            high_quality_pct = (high_quality_count / len(correctness_values)) * 100

            moderate_count = (correctness_values >= 0.5).sum()
            moderate_pct = (moderate_count / len(correctness_values)) * 100

            detailed_stats['Agent'].append(display_name)
            detailed_stats['Count'].append(len(correctness_values))
            detailed_stats['Mean'].append(correctness_values.mean())
            detailed_stats['Median'].append(correctness_values.median())
            detailed_stats['Std Dev'].append(correctness_values.std())
            detailed_stats['Min'].append(correctness_values.min())
            detailed_stats['Max'].append(correctness_values.max())
            detailed_stats['High Quality (>=0.7)'].append(high_quality_count)
            detailed_stats['High Quality %'].append(high_quality_pct)
            detailed_stats['Moderate+ (>=0.5)'].append(moderate_count)
            detailed_stats['Moderate+ %'].append(moderate_pct)

    detailed_stats_df = pd.DataFrame(detailed_stats)

    # Format the output
    print(detailed_stats_df.to_string(index=False))

    print("\n" + "="*80)
    print("Note: Correctness scores range from 0.0 (completely wrong) to 1.0 (perfect)")
    print("      Scores >= 0.7 are considered high-quality answers")
    print("      Scores >= 0.5 are considered moderate or better")
    print("="*80 + "\n")

    # Save detailed statistics to CSV
    csv_filename = f"{output_prefix}answer_correctness_statistics{suffix}.csv"
    detailed_stats_df.to_csv(csv_filename, index=False)
    print(f"Saved detailed statistics to {csv_filename}\n")


def analyze_image_relevance(
    df: pd.DataFrame,
    output_prefix: str = "",
    agent_name_mapping: Dict[str, str] | None = None
):
    """
    Analyze and visualize image relevance statistics.

    For each agent response, checks if the agent used any of the reference images
    (images that were used in the ground truth answer).

    Args:
        df: DataFrame with agent results and reference images
        output_prefix: Prefix for output file names (e.g., "grok-4-fast_")
        agent_name_mapping: Optional mapping of agent names to display names
    """
    # Get agent reference image columns
    agent_image_columns = [
        col for col in df.columns
        if col.endswith('_reference_images') and col != 'reference_images'
    ]

    if not agent_image_columns:
        print("\n=== No Agent Reference Images Found ===")
        print("Skipping image relevance analysis.\n")
        return

    print("\n" + "="*80)
    print("=== Image Relevance Analysis ===")
    print("="*80)

    # Calculate image relevance for each agent
    relevance_stats = {
        'Agent': [],
        'Total Responses': [],
        'Avg Jaccard Similarity': [],
        'Avg Images Used': [],
        'Avg Reference Images': []
    }

    for col in agent_image_columns:
        agent_name = col.replace('_reference_images', '')
        display_name = get_display_name(agent_name, agent_name_mapping)

        # Filter rows where both agent and reference images exist
        mask = (
            df['reference_images'].notna() &
            df[col].apply(lambda x: isinstance(x, list) and len(x) >= 0) &
            df['reference_images'].apply(lambda x: isinstance(x, list) and len(x) >= 0)
        )

        filtered_df = df[mask]
        total_responses = len(filtered_df)

        if total_responses == 0:
            continue

        # Calculate relevance for each response
        jaccard_scores = []
        total_images_used = 0
        total_reference_images = 0
        valid_counts = 0

        for idx, row in filtered_df.iterrows():
            agent_images = row[col]
            reference_images = row['reference_images']
            total_images_used += len(agent_images)
            total_reference_images += len(reference_images)
            if len(reference_images) > 0:
                valid_counts += 1
                jaccard_score = calculate_jaccard_similarity(agent_images, reference_images)
                jaccard_scores.append(jaccard_score)

        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
        avg_images = total_images_used / valid_counts if valid_counts > 0 else 0
        avg_ref_images = total_reference_images / valid_counts if valid_counts > 0 else 0

        relevance_stats['Agent'].append(display_name)
        relevance_stats['Total Responses'].append(total_responses)
        relevance_stats['Avg Jaccard Similarity'].append(avg_jaccard)
        relevance_stats['Avg Images Used'].append(avg_images)
        relevance_stats['Avg Reference Images'].append(avg_ref_images)

    if not relevance_stats['Agent']:
        print("No valid image data found for analysis.\n")
        return

    relevance_df = pd.DataFrame(relevance_stats)

    # 1. Bar Chart: Average Jaccard Similarity
    print("\nGenerating image relevance bar chart...")
    fig_relevance = go.Figure()

    fig_relevance.add_trace(go.Bar(
        x=relevance_df['Agent'],
        y=relevance_df['Avg Jaccard Similarity'],
        text=relevance_df['Avg Jaccard Similarity'].round(3),
        textposition='outside',
        marker_color='mediumseagreen',
        texttemplate='%{text:.3f}'
    ))

    fig_relevance.update_layout(
        title='Average Jaccard Similarity: Agent Image Selection vs Ground Truth',
        xaxis_title='Agent',
        yaxis_title='Jaccard Similarity (0.0-1.0)',
        yaxis_range=[0, 1.1],
        height=500
    )

    filename = f"{output_prefix}image_jaccard_similarity.png"
    fig_relevance.write_image(filename, width=1200, height=500)
    print(f"Saved to {filename}")

    # 2. Bar Chart: Average Images Used per Response
    print("Generating average images used bar chart...")
    fig_avg_images = go.Figure()

    fig_avg_images.add_trace(go.Bar(
        x=relevance_df['Agent'],
        y=relevance_df['Avg Images Used'],
        text=relevance_df['Avg Images Used'].round(2),
        textposition='outside',
        marker_color='steelblue',
        texttemplate='%{text:.2f}'
    ))

    fig_avg_images.update_layout(
        title='Average Number of Images Referenced per Response',
        xaxis_title='Agent',
        yaxis_title='Average Images Used',
        height=500
    )

    filename = f"{output_prefix}average_images_used.png"
    fig_avg_images.write_image(filename, width=1200, height=500)
    print(f"Saved to {filename}")

    # 3. Print Statistics Table
    print("\n=== Image Relevance Statistics ===")
    print("\nDetailed statistics for each agent:\n")
    print(relevance_df.to_string(index=False))

    print("\n" + "="*80)
    print("Jaccard Similarity Interpretation:")
    print("  1.0 = Perfect overlap (agent selected exactly the same images as ground truth)")
    print("  0.5 = Moderate overlap (50% of combined image set is shared)")
    print("  0.0 = No overlap (agent selected completely different images)")
    print()
    print("Image Usage:")
    print("  Avg Images Used: Average number of images the agent referenced per response")
    print("  Avg Reference Images: Average number of images in ground truth per response")
    print("  Higher Jaccard + Similar counts = Efficient and accurate image selection")
    print("="*80 + "\n")

    # Save detailed statistics to CSV
    csv_filename = f"{output_prefix}image_relevance_statistics.csv"
    relevance_df.to_csv(csv_filename, index=False)
    print(f"Saved image relevance statistics to {csv_filename}\n")
