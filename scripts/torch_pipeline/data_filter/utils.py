import scanpy as sc
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from IPython.display import HTML, display
from IPython.display import FileLink

from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d # For Gaussian smoothing
from statsmodels.nonparametric.smoothers_lowess import lowess # For LOWESS

from pathlib import Path
from itertools import chain, repeat

####### Utils #######

def gmt_to_decoupler(pth: Path) -> pd.DataFrame:
    """Parse a gmt file to a decoupler pathway dataframe."""

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    return pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )

def gmt_to_decoupler_multiple_pathways(gmt_paths, geneset_name=None, genesymbol_name=None):
    """Parse multiple gmt files and return a combined decoupler pathway dataframe."""
    all_records = []
    for pth in gmt_paths:
        with Path(pth).open("r") as f:
            for line in f:
                name, _, *genes = line.strip().split("\t")
                all_records.extend(zip(repeat(name), genes))
    return pd.DataFrame.from_records(all_records, columns=[geneset_name, genesymbol_name])

####### Plotting functions #######

def plot_age_vs_pseudotime_by_annotation(
    adata: AnnData,
    annotation_col: str = 'Annotation', # Column to use for grouping into subplots
    point_size: int = 120,
    show_plot: bool = True,
    n_subplot_cols: int = 2, # Number of columns for subplots
    figsize_per_subplot: tuple = (7, 7) # Figure size for each individual subplot
):
    """
    Plot scaled sample age vs scaled mean pseudotime, with separate subplots for each annotation.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing obs with 'sample', 'age', annotation_col, 'Pseudotime' columns.
    annotation_col : str, default 'Annotation'
        The column name in adata.obs to use for creating subplots (e.g., 'cell_type', 'Annotation').
    point_size : int, default 120
        Size of scatter plot points.
    show_plot : bool, default True
        Whether to display the plot.
    n_subplot_cols : int, default 2
        Number of columns in the subplot grid.
    figsize_per_subplot : tuple, default (7, 7)
        Figure size for each individual subplot.
    
    Returns:
    --------
    matplotlib.figure.Figure or None
        The generated figure object, or None if plotting fails.
    pandas.DataFrame or None
        Summary dataframe with scaled values for all annotations, or None.
    """
    
    # 1. Select relevant columns and create a working copy
    required_cols = ['sample', 'age', annotation_col, 'Pseudotime']
    for col in required_cols:
        if col not in adata.obs.columns:
            print(f"Error: Required column '{col}' not found in adata.obs.")
            return None, None
            
    obs_df = adata.obs[required_cols].copy()
    
    # 2. Filter out cells where Pseudotime is NaN
    obs_df_valid_pt = obs_df.dropna(subset=['Pseudotime'])
    
    if obs_df_valid_pt.empty:
        print("No cells with valid pseudotime values found. Cannot proceed.")
        return None, None
    
    # 3. Group by 'sample' AND the specified annotation_col to get mean pseudotime per sample *within each annotation*
    # This is crucial: if a sample has cells of Type I and Type II, they will get separate mean pseudotimes.
    sample_summary_df_all_annotations = obs_df_valid_pt.groupby(['sample', annotation_col]).agg(
        mean_pseudotime=('Pseudotime', 'mean'),
        age=('age', 'first') # Assuming age is consistent per sample
    ).reset_index()
    
    if sample_summary_df_all_annotations.empty:
        print("Summary DataFrame is empty after grouping. Cannot plot.")
        return None, None
        
    # 4. Scale 'mean_pseudotime' and 'age' globally across all samples and annotations
    # This ensures the 0-1 scale is consistent across subplots for comparison.
    scaler_pt = MinMaxScaler()
    scaler_age = MinMaxScaler()
    
    sample_summary_df_all_annotations['scaled_mean_pseudotime'] = scaler_pt.fit_transform(
        sample_summary_df_all_annotations[['mean_pseudotime']]
    )
    sample_summary_df_all_annotations['scaled_age'] = scaler_age.fit_transform(
        sample_summary_df_all_annotations[['age']]
    )
    
    # 5. Create subplots for each unique annotation value
    unique_annotations = sorted(sample_summary_df_all_annotations[annotation_col].unique())
    if not unique_annotations:
        print(f"No unique values found in '{annotation_col}' column after processing. Cannot create subplots.")
        return None, sample_summary_df_all_annotations

    num_annotations = len(unique_annotations)
    n_subplot_rows = (num_annotations + n_subplot_cols - 1) // n_subplot_cols
    
    fig, axes = plt.subplots(
        n_subplot_rows, 
        n_subplot_cols, 
        figsize=(figsize_per_subplot[0] * n_subplot_cols, figsize_per_subplot[1] * n_subplot_rows),
        squeeze=False # Ensure axes is always 2D
    )
    axes = axes.flatten() # Easier to iterate
    
    plot_successful = False
    for i, current_annotation_val in enumerate(unique_annotations):
        ax = axes[i]
        
        # Filter the summary dataframe for the current annotation
        df_subset_annotation = sample_summary_df_all_annotations[
            sample_summary_df_all_annotations[annotation_col] == current_annotation_val
        ]
        
        if df_subset_annotation.empty:
            ax.set_title(f"Annotation: {current_annotation_val}\n(No data for this annotation)")
            ax.axis('off')
            continue

        plot_successful = True # At least one subplot will be generated
        
        sns.scatterplot(
            data=df_subset_annotation,
            x='scaled_mean_pseudotime',
            y='scaled_age',
            hue='sample', # Color points by sample ID
            s=point_size,
            legend=False, # Turn off individual scatterplot legends; a figure legend might be better if needed
            ax=ax
        )
        
        # Add labels for each point (sample ID)
        for _, row in df_subset_annotation.iterrows():
            ax.text(
                row['scaled_mean_pseudotime'] + 0.01,
                row['scaled_age'] + 0.01,
                row['sample'],
                fontsize=9
            )
        
        # Add a y=x line for reference
        min_val, max_val = 0, 1 # Data is scaled 0-1
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7) # No label for subplot y=x line to avoid clutter
        
        # Formatting for each subplot
        ax.set_xlabel("Scaled Mean Pseudotime (0-1)")
        ax.set_ylabel("Scaled Chronological Age (0-1)")
        ax.set_title(f"Annotation: {current_annotation_val}")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Turn off any unused subplots
    for j in range(num_annotations, len(axes)):
        axes[j].axis('off')
        
    # Add a main title for the entire figure
    fig.suptitle("Scaled Sample Age vs. Scaled Mean Pseudotime by Annotation", fontsize=16, y=1.02 if n_subplot_rows > 1 else 1.05)
    
    # Optional: Add a single legend for the y=x line if desired, though it's quite standard
    # handles, labels = ax.get_legend_handles_labels() # Get from last populated ax
    # if handles:
    #     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96 if n_subplot_rows > 1 else 0.93]) # Adjust rect to make space for suptitle
    
    if show_plot:
        plt.show()
        
    if not plot_successful:
        print("No data was plotted for any annotation.")
        return None, sample_summary_df_all_annotations

    return fig, sample_summary_df_all_annotations

def plot_pathway_dotplot_by_age(
    df_cell_level,
    score_cols, # List of pathway score column names
    age_col='age',
    sample_col='sample',
    annotation_col='Annotation',
    target_annotation=None,
    figsize=(10, 6),
    min_dot_size=20,
    max_dot_size=500,
    dot_size_scale_factor=1.0, # Adjust if sizes are too uniform or too varied
    cmap_name="coolwarm", # Or "RdYlBu_r" to mimic your example more closely
    value_legend_title="Mean Score",
    size_legend_title="# Cells"
):
    """
    Plots a dot plot where dot color is mean pathway score and dot size is number of cells.
    Samples are on x-axis ordered by age, pathways on y-axis.

    Parameters:
        df_cell_level (pd.DataFrame): Cell-level data with scores, age, sample, annotation.
        score_cols (list): List of score column names (pathways).
        age_col (str): Column name for age.
        sample_col (str): Column name for sample IDs.
        annotation_col (str): Column name for annotations.
        target_annotation (str or None): If provided, subset to this annotation.
        figsize (tuple): Figure size.
        min_dot_size (int): Minimum size for dots.
        max_dot_size (int): Maximum size for dots.
        dot_size_scale_factor (float): Multiplier for raw cell counts before scaling to dot size.
        cmap_name (str): Colormap for the scores.
        value_legend_title (str): Title for the colorbar.
        size_legend_title (str): Title for the size legend.
    """
    plot_df = df_cell_level.copy()

    # 1. Data Preparation
    if target_annotation is not None:
        if annotation_col not in plot_df.columns:
            print(f"Warning: Annotation column '{annotation_col}' not found. Cannot filter by '{target_annotation}'.")
            return
        plot_df = plot_df[plot_df[annotation_col] == target_annotation]
        if plot_df.empty:
            print(f"No cells found for annotation '{target_annotation}'.")
            return

    required_cols = [age_col, sample_col, annotation_col] + score_cols
    for col in required_cols:
        if col not in plot_df.columns:
            print(f"Warning: Required column '{col}' not found. Aborting.")
            return

    plot_df = plot_df.dropna(subset=[age_col, sample_col], how='any')
    if plot_df.empty:
        print("No data to plot after initial NaN filtering.")
        return

    # Aggregate: mean scores and cell counts
    grouped = plot_df.groupby(sample_col)
    mean_scores_df = grouped[score_cols].mean()
    cell_counts_series = grouped.size()
    sample_ages_series = grouped[age_col].mean() # Assuming age is consistent per sample

    # Order samples by age
    ordered_samples = sample_ages_series.sort_values().index.tolist()
    if not ordered_samples:
        print("Could not determine sample order.")
        return

    # Prepare data for plotting (long format)
    plot_data_list = []
    for sample_id in ordered_samples:
        for pathway in score_cols:
            mean_score = mean_scores_df.loc[sample_id, pathway] if sample_id in mean_scores_df.index else np.nan
            cell_count = cell_counts_series.loc[sample_id] if sample_id in cell_counts_series.index else 0
            age = sample_ages_series.loc[sample_id] if sample_id in sample_ages_series.index else np.nan
            plot_data_list.append({
                'sample': sample_id,
                'pathway': pathway,
                'mean_score': mean_score,
                'cell_count': cell_count,
                'age': age
            })
    plot_data_df = pd.DataFrame(plot_data_list)
    plot_data_df = plot_data_df.dropna(subset=['mean_score']) # Drop if score couldn't be calculated

    if plot_data_df.empty:
        print("No data to plot after aggregation.")
        return

    # Scale cell counts for dot sizes
    min_count = plot_data_df['cell_count'].min()
    max_count = plot_data_df['cell_count'].max()
    if max_count == min_count : # Avoid division by zero if all counts are the same
         plot_data_df['dot_size'] = min_dot_size if max_count == 0 else (min_dot_size + max_dot_size) / 2
    else:
        # Apply scale factor first
        scaled_counts = plot_data_df['cell_count'] * dot_size_scale_factor
        # Then normalize and scale to dot size range
        # Handle cases where scaled_counts might still be uniform after scaling factor
        min_s_count = scaled_counts.min()
        max_s_count = scaled_counts.max()
        if max_s_count == min_s_count:
             plot_data_df['dot_size'] = min_dot_size if max_s_count == 0 else (min_dot_size + max_dot_size) / 2
        else:
            plot_data_df['dot_size'] = min_dot_size + \
                (scaled_counts - min_s_count) / (max_s_count - min_s_count) * (max_dot_size - min_dot_size)


    # 2. Plotting
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique pathway names for y-axis (preserve order from score_cols)
    pathway_names = score_cols
    pathway_y_coords = {name: i for i, name in enumerate(pathway_names)}

    # Get unique sample names for x-axis (already ordered by age)
    sample_x_coords = {name: i for i, name in enumerate(ordered_samples)}

    scatter = ax.scatter(
        x=plot_data_df['sample'].map(sample_x_coords),
        y=plot_data_df['pathway'].map(pathway_y_coords),
        s=plot_data_df['dot_size'],
        c=plot_data_df['mean_score'],
        cmap=cmap_name,
        edgecolors='gray', # Optional: add edge to dots
        linewidths=0.5    # Optional: edge width
    )

    # --- X-axis (Samples) ---
    ax.set_xticks(list(sample_x_coords.values()))
    ax.set_xticklabels(ordered_samples, rotation=45, ha="right")
    ax.set_xlabel("Sample (Ordered by Age)")

    # --- Age annotations on top ---
    ax2 = ax.twiny() # Create a second x-axis sharing the y-axis
    ax2.set_xlim(ax.get_xlim()) # Ensure limits match
    ax2.set_xticks(list(sample_x_coords.values()))
    ax2.set_xticklabels([f"{sample_ages_series[s]:.0f}" for s in ordered_samples], rotation=45, ha="left")
    ax2.set_xlabel("Age")

    # --- Y-axis (Pathways) ---
    ax.set_yticks(list(pathway_y_coords.values()))
    # Clean up pathway names for display
    #clean_pathway_names = [p.replace('KEGG_', '').replace('GOBP_', '').replace('REACTOME_', '').replace('_', ' ').title() for p in pathway_names]
    ax.set_yticklabels(pathway_names)
    ax.set_ylabel("DNA Damage and Repair Pathway")

    # --- Colorbar for Mean Score ---
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.15) # Adjust fraction and pad
    cbar.set_label(value_legend_title)

    # --- Legend for Dot Size (# Cells) ---
    # Create some proxy artists for the legend
    # Define some representative cell counts for the legend
    if max_count > 0 :
        legend_counts_raw = np.linspace(min_count, max_count, num=4, dtype=int)
        # Ensure 0 is included if min_count is 0 and it's meaningful
        if min_count == 0 and 0 not in legend_counts_raw and len(legend_counts_raw) > 1:
            legend_counts_raw[0] = 0
        legend_counts_raw = np.unique(legend_counts_raw) # Ensure unique values
    else: # Handle case where all counts are 0 or only one value
        legend_counts_raw = np.array([min_count]) if min_count > 0 else np.array([0])


    legend_dots = []
    legend_labels = []
    for count_val in legend_counts_raw:
        if max_count == min_count:
            size_val = min_dot_size if max_count == 0 else (min_dot_size + max_dot_size) / 2
        else:
            scaled_c = count_val * dot_size_scale_factor
            size_val = min_dot_size + \
                       (scaled_c - (min_count*dot_size_scale_factor)) / ((max_count*dot_size_scale_factor) - (min_count*dot_size_scale_factor)) * (max_dot_size - min_dot_size)
        size_val = max(min_dot_size, min(max_dot_size, size_val)) # Ensure it's within bounds

        legend_dots.append(plt.scatter([], [], s=size_val, c='gray', label=f"{int(count_val)}")) # Use gray for legend dots
        #legend_labels.append(f"{int(count_val)}") # No need, label is in scatter call

    # Position the size legend to the right, below the colorbar
    # The legend function needs handles and labels. We just pass the handles.
    size_leg = ax.legend(handles=legend_dots, title=size_legend_title,
                         bbox_to_anchor=(1.18, 0.4), loc='center left', # Adjust bbox_to_anchor
                         labelspacing=1.5, borderpad=1, frameon=True,
                         handletextpad=1.5, # Increase spacing between dot and text
                         scatterpoints=1) # Ensure only one dot per legend item

    # Adjust layout
    fig_title = f'Pathway Activity Dot Plot (Annotation: {target_annotation})' if target_annotation else 'Pathway Activity Dot Plot'
    plt.suptitle(fig_title, fontsize=16, y=1.05)
    plt.subplots_adjust(top=0.85, bottom=0.15, right=0.8) # Adjust subplot to make space for legends/title
    plt.grid(True, linestyle='--', alpha=0.5, axis='y') # Add y-axis grid
    ax.tick_params(axis='both', which='major', pad=7)
    ax2.tick_params(axis='x', which='major', pad=7)

    plt.show()

def plot_pathway_box_distributions_by_age(
    df_cell_level,
    score_cols,
    geneset_sizes,
    age_col='age',
    sample_col='sample',
    annotation_col='Annotation',
    target_annotation=None,
    filter_quantile_to_keep_top=None,
    n_subplot_cols=2,
    figsize_per_subplot=(8, 6),
    box_width=0.8, # Control width of boxes
    sample_label_fontsize=8,
    sample_label_y_offset_factor=0.05
):
    """
    Plots box plots for multiple pathway scores, with samples positioned by age on x-axis.
    Sample IDs are annotated above the boxes.
    Optionally filters cells to keep only the most active ones per sample per pathway.

    Parameters:
        df_cell_level (pd.DataFrame): Cell-level data.
        score_cols (list): List of score column names to plot.
        geneset_sizes (pd.Series or dict): Maps score_col names to their sizes.
        age_col (str): Column name for age.
        sample_col (str): Column name for sample IDs.
        annotation_col (str): Column name for annotations.
        target_annotation (str or None): If provided, subset to this annotation.
        filter_quantile_to_keep_top (float or None): Quantile of top scores to keep.
        n_subplot_cols (int): Number of columns in the subplot grid.
        figsize_per_subplot (tuple): Figure size for each individual subplot.
        box_width (float): Width of the box plots.
        sample_label_fontsize (int): Fontsize for sample ID labels.
        sample_label_y_offset_factor (float): Factor to adjust y-position of sample labels.
    """
    plot_df = df_cell_level.copy()

    if target_annotation is not None:
        if annotation_col not in plot_df.columns:
            print(f"Warning: Annotation column '{annotation_col}' not found. Cannot filter by '{target_annotation}'.")
            return
        plot_df = plot_df[plot_df[annotation_col] == target_annotation]
        if plot_df.empty:
            print(f"No cells found for annotation '{target_annotation}'.")
            return

    required_cols = [age_col, sample_col] + score_cols
    for col in required_cols:
        if col not in plot_df.columns:
            print(f"Warning: Required column '{col}' not found. Aborting.")
            return

    plot_df = plot_df.dropna(subset=[age_col, sample_col], how='any')
    plot_df = plot_df.dropna(subset=score_cols, how='all')

    if plot_df.empty:
        print("No data to plot after initial NaN filtering.")
        return

    unique_ages_sorted = sorted(plot_df[age_col].unique())
    age_to_xpos = {age: i for i, age in enumerate(unique_ages_sorted)}
    plot_df['x_position'] = plot_df[age_col].map(age_to_xpos)

    xpos_to_samples = plot_df.groupby('x_position')[sample_col].unique().apply(lambda x: '/'.join(sorted(x)))


    num_scores = len(score_cols)
    n_subplot_rows = (num_scores + n_subplot_cols - 1) // n_subplot_cols

    fig, axes = plt.subplots(
        n_subplot_rows,
        n_subplot_cols,
        figsize=(figsize_per_subplot[0] * n_subplot_cols, figsize_per_subplot[1] * n_subplot_rows),
        squeeze=False
    )
    axes = axes.flatten()

    for i, score in enumerate(score_cols):
        ax = axes[i]
        current_score_df = plot_df[['x_position', sample_col, age_col, score]].copy().dropna(subset=[score])

        if current_score_df.empty:
            ax.set_title(f"{score.replace('_', ' ')}\n(No data)")
            ax.axis('off')
            continue

        df_for_plot = current_score_df.copy()

        if filter_quantile_to_keep_top is not None and 0 < filter_quantile_to_keep_top < 1:
            quantile_for_threshold = 1.0 - filter_quantile_to_keep_top
            def filter_by_quantile(group):
                if group.empty or len(group) < 2:
                    return group
                original_sample_group = plot_df[
                    (plot_df[sample_col] == group[sample_col].iloc[0]) &
                    (plot_df[age_col] == group[age_col].iloc[0])
                ][score]
                if original_sample_group.empty or len(original_sample_group) < 2:
                    return group
                threshold_val = original_sample_group.quantile(quantile_for_threshold)
                return group[group[score] >= threshold_val]
            df_for_plot = df_for_plot.groupby([sample_col, age_col], group_keys=False).apply(filter_by_quantile)

            if df_for_plot.empty:
                ax.set_title(f"{score.replace('_', ' ')}\n(No data after filtering)")
                ax.axis('off')
                continue
        
        # --- Changed from violinplot to boxplot ---
        sns.boxplot(x='x_position', y=score, data=df_for_plot, ax=ax, palette="tab10", width=box_width, fliersize=2) # fliersize controls outlier point size

        ax.set_xticks(list(age_to_xpos.values()))
        ax.set_xticklabels([str(int(ua)) if ua==int(ua) else str(round(ua,1)) for ua in unique_ages_sorted], rotation=45, ha="right")
        ax.set_xlabel('Age')

        # Annotate sample IDs above boxes
        # Determine y-position for labels based on existing data in the current plot
        if not df_for_plot.empty:
            # Consider whisker ends for max y to place labels above.
            # Boxplot calculations for whiskers can be complex, so we take a simpler approach:
            # Use the max data point within 1.5*IQR or the absolute max if no outliers,
            # or simply the max of the data points plotted.
            # A robust way is to find the top of the whiskers if possible, or just use max data for simplicity.
            y_values_in_plot = []
            for x_pos_val in df_for_plot['x_position'].unique():
                y_values_in_plot.extend(df_for_plot[df_for_plot['x_position'] == x_pos_val][score])
            
            if y_values_in_plot:
                max_y_val_for_labeling = np.percentile(y_values_in_plot, 99) # Use 99th percentile to avoid extreme outliers influencing label position too much
                min_y_val_for_labeling = np.percentile(y_values_in_plot, 1)
                y_range = max_y_val_for_labeling - min_y_val_for_labeling
                y_offset = max_y_val_for_labeling + (y_range * sample_label_y_offset_factor) if y_range > 0 else max_y_val_for_labeling * (1 + sample_label_y_offset_factor*2)
            else: # Fallback if no data somehow
                y_offset = ax.get_ylim()[1] * 0.05 # Default offset if no data
        else:
            y_offset = ax.get_ylim()[1] * 0.05


        current_x_positions = sorted(df_for_plot['x_position'].unique())
        for x_pos in current_x_positions:
            sample_names_at_xpos = xpos_to_samples.get(x_pos, "")
            if sample_names_at_xpos:
                 ax.text(x_pos, y_offset, sample_names_at_xpos,
                        ha='center', va='bottom', fontsize=sample_label_fontsize, rotation=90)

        title_text = score.replace('_', ' ')
        size = geneset_sizes.get(score)
        if size is not None:
            title_text += f"\n(n={int(size)})"
        if filter_quantile_to_keep_top is not None:
            title_text += f"\n(Top {filter_quantile_to_keep_top*100:.0f}% cells per sample)"

        ax.set_title(title_text)
        ax.set_ylabel('Score Distribution')
        ax.grid(True, alpha=0.3, axis='y')

    for j in range(num_scores, len(axes)):
        axes[j].axis('off')

    fig_title = 'Pathway Score Distributions by Age'
    if target_annotation:
        fig_title += f' (Annotation: {target_annotation})'
    plt.suptitle(fig_title, fontsize=16, y=1.02 if n_subplot_rows > 1 else 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.98 if n_subplot_rows > 1 else 0.95])
    plt.show()

def plot_multi_score_lines_by_age(
    df, 
    score_cols, 
    age_col='age', 
    sample_col='sample', 
    annotation=None, 
    figsize=(14, 7)
):
    """
    Plots multiple scores as colored lines with sample dots, ordered by age.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        score_cols (list): List of score column names to plot.
        age_col (str): Column name for age (x-axis).
        sample_col (str): Column name for sample IDs (for labeling).
        annotation (str or None): If provided, subset to this annotation.
        figsize (tuple): Figure size.
    """
    # Optionally subset by annotation
    if annotation is not None:
        df = df[df['Annotation'] == annotation]
    
    # Drop rows with missing age or all scores missing
    df = df.dropna(subset=[age_col] + score_cols)
    # Sort by age
    df = df.sort_values(age_col)
    
    plt.figure(figsize=figsize)
    colors = plt.cm.tab10.colors  # Up to 10 distinct colors
    
    for i, score in enumerate(score_cols):
        y = df[score]
        x = df[age_col]
        plt.plot(x, y, label=score.replace('_', ' '), color=colors[i % len(colors)], marker='o')
        # Optionally, annotate with sample names
        for xi, yi, sample in zip(x, y, df[sample_col]):
            plt.annotate(
                sample, 
                (xi, yi), 
                color=colors[i % len(colors)], 
                fontsize=8, 
                xytext=(0, 5), 
                textcoords="offset points", 
                ha="center"
            )
    
    plt.xlabel('Age')
    plt.ylabel('Score')
    plt.title('Multiple DNA Damage/Repair Scores by Age' + (f' (Annotation: {annotation})' if annotation else ''))
    plt.legend(title='Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_multi_score_lines_by_pseudotime(
    df_with_scores_and_pseudotime, # This df should have scores AND mean_pseudotime per sample
    score_cols,
    pseudotime_col='mean_pseudotime', # Column for mean pseudotime of the sample
    sample_col='sample',
    age_col_for_labeling='age', # Optional: still use original age for point labels if desired
    annotation=None,
    figsize=(14, 7)
):
    """
    Plots multiple scores as colored lines with sample dots, ordered by mean sample pseudotime.

    Parameters:
        df_with_scores_and_pseudotime (pd.DataFrame): Input dataframe with scores and mean_pseudotime.
        score_cols (list): List of score column names to plot.
        pseudotime_col (str): Column name for mean pseudotime (x-axis).
        sample_col (str): Column name for sample IDs (for labeling points).
        age_col_for_labeling (str or None): Original age column, if you want to include age in point labels.
        annotation (str or None): If provided, subset to this annotation.
        figsize (tuple): Figure size.
    """
    plot_df = df_with_scores_and_pseudotime.copy()

    # Optionally subset by annotation (if 'Annotation' column exists in this df)
    if annotation is not None and 'Annotation' in plot_df.columns:
        plot_df = plot_df[plot_df['Annotation'] == annotation]
    elif annotation is not None and 'Annotation' not in plot_df.columns:
        print(f"Warning: Annotation column not found in DataFrame, cannot filter by '{annotation}'.")


    # Drop rows with missing pseudotime or all scores missing
    # Ensure pseudotime_col is present
    if pseudotime_col not in plot_df.columns:
        print(f"Error: Pseudotime column '{pseudotime_col}' not found in DataFrame.")
        return
    plot_df = plot_df.dropna(subset=[pseudotime_col] + score_cols, how='any') # Drop if pseudotime or any score is NA

    if plot_df.empty:
        print("No data to plot after dropping NaNs.")
        return

    # Sort by the pseudotime column
    plot_df = plot_df.sort_values(pseudotime_col)

    plt.figure(figsize=figsize)
    colors = plt.cm.tab10.colors

    for i, score in enumerate(score_cols):
        if score not in plot_df.columns:
            print(f"Warning: Score column '{score}' not found. Skipping.")
            continue
        y = plot_df[score]
        x = plot_df[pseudotime_col] # Use pseudotime for x-axis

        plt.plot(x, y, label=score.replace('_', ' '), color=colors[i % len(colors)], marker='o')

        # Annotate with sample names (and optionally age)
        for idx, row in plot_df.iterrows():
            xi = row[pseudotime_col]
            yi = row[score]
            sample_name = row[sample_col]
            label_text = sample_name
            if age_col_for_labeling and age_col_for_labeling in row:
                age_val = row[age_col_for_labeling]
                if pd.notna(age_val):
                    label_text += f" (Age:{int(age_val)})" # Add age to the label

            plt.annotate(
                label_text,
                (xi, yi),
                color=colors[i % len(colors)],
                fontsize=8,
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom"
            )

    plt.xlabel(f'Mean Sample {pseudotime_col.replace("_", " ").title()}') # Updated x-axis label
    plt.ylabel('Pathway Score')
    title = 'Pathway Scores by Mean Sample Pseudotime'
    if annotation:
        title += f' (Annotation: {annotation})'
    plt.title(title)
    plt.legend(title='Pathway Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def _df_plot_smoothed_group_lines_on_ax(
    ax,
    df_cell_level: pd.DataFrame,
    pathway_score_key: str,
    pseudotime_key: str,
    group_by_key: str,
    smoothing_method: str = None, # 'gaussian', 'lowess', or None/'' for raw
    smoothing_strength: float = None, # sigma for gaussian, frac for lowess
    geneset_size: int = None,
    groups_to_plot: list = None,
    colors_dict: dict = None,
    xlabel: str = "Pseudotime",
    ylabel: str = "Score",
    title_override: str = None,
    legend_labels: dict = None,
    legend_title: str = None,
    legend_loc: str = 'best',
    legend_frameon: bool = False,
    line_kwargs: dict = None,
    ylim: tuple = None,
    xticks_list: list = None,
    yticks_list: list = None,
    despine: bool = True,
    show_legend: bool = True,
    debug_prints: bool = False
):
    # --- Initial checks and data prep ---
    required_cols_for_plot = [pathway_score_key, pseudotime_key, group_by_key]
    for col in required_cols_for_plot:
        if col not in df_cell_level.columns:
            ax.set_title(f"{pathway_score_key.replace('_', ' ')}\n(Error: Column '{col}' missing)")
            ax.axis('off'); return False
    obs_df = df_cell_level[required_cols_for_plot].copy()
    obs_df.dropna(subset=required_cols_for_plot, inplace=True)
    subplot_title_text = title_override if title_override else pathway_score_key.replace('_', ' ')
    if geneset_size is not None: subplot_title_text += f"\n(n={int(geneset_size)})"
    if obs_df.empty:
        ax.set_title(f"{subplot_title_text}\n(No valid data after NA removal)")
        ax.axis('off'); return False
    obs_df[group_by_key] = obs_df[group_by_key].astype(str)
    if groups_to_plot is None: unique_groups_to_plot = sorted(obs_df[group_by_key].unique())
    else:
        groups_to_plot_str = [str(g) for g in groups_to_plot]
        available_groups_in_df = obs_df[group_by_key].unique()
        unique_groups_to_plot = [g for g in groups_to_plot_str if g in available_groups_in_df]
        if not unique_groups_to_plot:
            ax.set_title(f"{subplot_title_text}\n(Specified groups not found or no data)")
            ax.axis('off'); return False
    _default_colors = plt.cm.get_cmap('tab10').colors
    actual_line_kwargs = {'linewidth': 1.5};
    if line_kwargs: actual_line_kwargs.update(line_kwargs)
    plotted_anything_on_this_ax = False

    # --- Loop through groups and plot ---
    for i, group_name_str in enumerate(unique_groups_to_plot):
        group_df = obs_df[obs_df[group_by_key] == group_name_str]
        if group_df.empty:
            if debug_prints: print(f"D: Group '{group_name_str}' empty for '{pathway_score_key}'.")
            continue

        group_df_sorted = group_df.sort_values(by=pseudotime_key)
        x_vals = group_df_sorted[pseudotime_key].values.astype(float)
        y_vals_raw = group_df_sorted[pathway_score_key].values.astype(float)

        if len(x_vals) < 2:
            if debug_prints: print(f"D: <2 points for group '{group_name_str}', score '{pathway_score_key}'.")
            continue
        
        y_vals_to_plot = y_vals_raw
        x_vals_to_plot = x_vals

        # Apply smoothing if requested
        if smoothing_method and smoothing_strength is not None:
            if smoothing_method.lower() == 'gaussian':
                if len(y_vals_raw) > int(4 * smoothing_strength) and smoothing_strength > 0:
                    try: y_vals_to_plot = gaussian_filter1d(y_vals_raw, sigma=smoothing_strength)
                    except Exception as e:
                        if debug_prints: print(f"D: Gaussian smooth error for '{group_name_str}', '{pathway_score_key}': {e}. Using raw.")
                elif debug_prints: print(f"D: Not enough points for Gaussian (sigma={smoothing_strength}) for '{group_name_str}', '{pathway_score_key}'. Using raw.")
            
            elif smoothing_method.lower() == 'lowess':
                if len(x_vals) >= 3 and len(np.unique(x_vals)) >= 2 and 0 < smoothing_strength < 1:
                    try:
                        smoothed_points = lowess(y_vals_raw, x_vals, frac=smoothing_strength, is_sorted=True, return_sorted=True)
                        x_vals_to_plot = smoothed_points[:, 0]
                        y_vals_to_plot = smoothed_points[:, 1]
                    except Exception as e:
                        if debug_prints: print(f"D: LOWESS smooth error (frac={smoothing_strength}) for '{group_name_str}', '{pathway_score_key}': {e}. Using raw.")
                elif debug_prints: print(f"D: Not enough/suitable points for LOWESS (frac={smoothing_strength}) for '{group_name_str}', '{pathway_score_key}'. Using raw.")
            
            elif debug_prints: print(f"D: Unknown smoothing_method '{smoothing_method}' or invalid strength. Using raw.")
        elif smoothing_method and debug_prints : print(f"D: Smoothing method '{smoothing_method}' requested but strength is None. Using raw.")


        if debug_prints:
            print(f"D: Group '{group_name_str}', Score '{pathway_score_key}' (Method: {smoothing_method or 'raw'}, Strength: {smoothing_strength}):")
            print(f"  Plotting with x_vals count: {len(x_vals_to_plot)}")
            print(f"  Plotting with y_vals min/max: {np.nanmin(y_vals_to_plot):.4f}/{np.nanmax(y_vals_to_plot):.4f}")
            
        color_to_use = colors_dict.get(group_name_str, _default_colors[i % len(_default_colors)]) if colors_dict else _default_colors[i % len(_default_colors)]
        label_for_legend = legend_labels.get(group_name_str, group_name_str) if legend_labels else group_name_str
            
        ax.plot(x_vals_to_plot, y_vals_to_plot, label=label_for_legend, color=color_to_use, **actual_line_kwargs)
        plotted_anything_on_this_ax = True

    # --- Finalize subplot ---
    if not plotted_anything_on_this_ax:
        ax.set_title(f"{subplot_title_text}\n(No lines plotted)"); return False
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(subplot_title_text)
    if show_legend and unique_groups_to_plot:
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(handles, labels, title=legend_title, loc=legend_loc, frameon=legend_frameon, fontsize='small')
    if ylim: ax.set_ylim(ylim)
    if xticks_list is not None: ax.set_xticks(xticks_list if xticks_list else [])
    if yticks_list is not None: ax.set_yticks(yticks_list if yticks_list else [])
    if despine:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5); ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(width=1.5)
    return True


def plot_multi_smoothed_lines_from_df(
    df_cell_level: pd.DataFrame,
    score_cols: list,
    pseudotime_key: str,
    group_by_key: str,
    smoothing_method: str = None, # 'gaussian', 'lowess', or None
    smoothing_strength: float = None, # sigma or frac
    geneset_sizes: pd.Series = None,
    groups_to_plot: list = None, colors_dict: dict = None, legend_labels_map: dict = None,
    n_subplot_cols: int = 2, figsize_per_subplot: tuple = (7, 4),
    common_xlabel: str = r"Pseudotime $\rightarrow$", common_ylabel: str = "Score",
    common_ylim: tuple = None, common_xticks_list: list = None, common_yticks_list: list = None,
    common_legend_loc: str = (0.02, 0.85), common_legend_frameon: bool = False,
    common_line_kwargs: dict = None, despine_subplots: bool = True,
    main_figure_title: str = None, show_legend_in_each_subplot: bool = True,
    debug_prints_for_subp_plot_func: bool = False
):
    # --- Initial checks for main function ---
    num_scores = len(score_cols)
    if num_scores == 0: print("No score columns"); return
    required_df_cols = score_cols + [pseudotime_key, group_by_key]
    missing_cols = [col for col in required_df_cols if col not in df_cell_level.columns]
    if missing_cols: print(f"Missing columns: {missing_cols}"); return
    if smoothing_method and smoothing_method.lower() not in ['gaussian', 'lowess']:
        print(f"Warning: Unknown smoothing_method '{smoothing_method}'. Plotting raw lines.")
        smoothing_method = None # Default to raw if unknown

    # --- Setup subplots ---
    n_subplot_rows = (num_scores + n_subplot_cols - 1) // n_subplot_cols
    fig, axes = plt.subplots(n_subplot_rows, n_subplot_cols,
        figsize=(figsize_per_subplot[0] * n_subplot_cols, figsize_per_subplot[1] * n_subplot_rows),
        squeeze=False)
    axes = axes.flatten()

    # --- Loop and call helper ---
    for i, score_key_current in enumerate(score_cols):
        ax = axes[i]
        current_geneset_size = geneset_sizes.get(score_key_current) if geneset_sizes is not None else None
        _df_plot_smoothed_group_lines_on_ax(
            ax=ax, df_cell_level=df_cell_level, pathway_score_key=score_key_current,
            pseudotime_key=pseudotime_key, group_by_key=group_by_key,
            smoothing_method=smoothing_method, smoothing_strength=smoothing_strength,
            geneset_size=current_geneset_size, groups_to_plot=groups_to_plot,
            colors_dict=colors_dict, xlabel=common_xlabel, ylabel=common_ylabel,
            legend_labels=legend_labels_map, legend_loc=common_legend_loc,
            legend_frameon=common_legend_frameon, line_kwargs=common_line_kwargs,
            ylim=common_ylim, xticks_list=common_xticks_list, yticks_list=common_yticks_list,
            despine=despine_subplots, show_legend=show_legend_in_each_subplot,
            debug_prints=debug_prints_for_subp_plot_func
        )

    # --- Finalize figure ---
    for j in range(num_scores, len(axes)): axes[j].axis('off')
    if main_figure_title: fig.suptitle(main_figure_title, fontsize=16, y=1.02 if n_subplot_rows > 1 else 1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.97 if main_figure_title and n_subplot_rows > 1 else 0.98])
    plt.show()

def create_animated_pathway_plot(
    df_cell_level: pd.DataFrame,
    score_cols: list,
    pseudotime_key: str = 'Pseudotime',
    group_by_key: str = 'Annotation',
    smoothing_method: str = 'gaussian',
    smoothing_strength: float = 120.0,
    geneset_sizes: pd.Series = None,
    groups_to_plot: list = None,
    colors_dict: dict = None,
    legend_labels_map: dict = None,
    n_subplot_cols: int = 2,
    figsize_per_subplot: tuple = (7, 4),
    # Animation parameters
    nframes: int = 200,
    fps: int = 30,
    dot_size: int = 150,
    output_file: str = 'pathway_animation.mp4',
    use_gif: bool = False  # Set True if ffmpeg doesn't work
):
    """
    Creates animated pathway plots with moving dots.
    """
    
    print("Starting animation creation...")
    
    # Validate inputs
    required_cols = score_cols + [pseudotime_key, group_by_key]
    missing = [c for c in required_cols if c not in df_cell_level.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return None
    
    # Setup figure
    num_scores = len(score_cols)
    n_subplot_rows = (num_scores + n_subplot_cols - 1) // n_subplot_cols
    
    fig, axes = plt.subplots(
        n_subplot_rows, n_subplot_cols,
        figsize=(figsize_per_subplot[0] * n_subplot_cols, 
                figsize_per_subplot[1] * n_subplot_rows),
        squeeze=False
    )
    axes = axes.flatten()
    
    # Store animation data
    all_lines_data = []
    
    # Plot each pathway
    for subplot_idx, score_col in enumerate(score_cols):
        ax = axes[subplot_idx]
        
        # Get data for this pathway
        plot_df = df_cell_level[[score_col, pseudotime_key, group_by_key]].copy()
        plot_df = plot_df.dropna()
        
        if plot_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Determine groups
        if groups_to_plot:
            groups = [g for g in groups_to_plot if g in plot_df[group_by_key].unique()]
        else:
            groups = sorted(plot_df[group_by_key].unique())
        
        # Plot each group
        for group_idx, group in enumerate(groups):
            group_data = plot_df[plot_df[group_by_key] == group].copy()
            group_data = group_data.sort_values(pseudotime_key)
            
            if len(group_data) < 2:
                continue
            
            x_vals = group_data[pseudotime_key].values
            y_vals = group_data[score_col].values
            
            # Apply smoothing
            if smoothing_method == 'gaussian' and len(y_vals) > 10:
                try:
                    y_vals_smooth = gaussian_filter1d(y_vals, sigma=smoothing_strength)
                except:
                    y_vals_smooth = y_vals
                    print(f"Warning: Smoothing failed for {group} in {score_col}")
            else:
                y_vals_smooth = y_vals
            
            # Get color
            color = colors_dict.get(group, f'C{group_idx}') if colors_dict else f'C{group_idx}'
            label = legend_labels_map.get(group, group) if legend_labels_map else group
            
            # Plot line
            line, = ax.plot(x_vals, y_vals_smooth, color=color, linewidth=1.5, 
                           label=label, alpha=0.8)
            
            # Create dot for animation (initially invisible)
            dot = ax.scatter([], [], s=dot_size, color=color, 
                           zorder=100, edgecolors='white', linewidths=2)
            
            # Store data for animation
            all_lines_data.append({
                'ax': ax,
                'x': x_vals,
                'y': y_vals_smooth,
                'dot': dot,
                'color': color
            })
        
        # Format subplot
        title = score_col.replace('_', ' ')
        if geneset_sizes is not None and score_col in geneset_sizes.index:
            title += f"\n(n={int(geneset_sizes[score_col])})"
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"Pseudotime $\rightarrow$", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_scores, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    print(f"Figure created with {len(all_lines_data)} animated lines")
    
    # Create animation
    def init():
        """Initialize animation"""
        for line_data in all_lines_data:
            line_data['dot'].set_offsets(np.empty((0, 2)))
        return [ld['dot'] for ld in all_lines_data]
    
    def animate(frame):
        """Update function for each frame"""
        progress = frame / (nframes - 1)
        
        for line_data in all_lines_data:
            x = line_data['x']
            y = line_data['y']
            
            if len(x) > 0:
                # Find position along line
                idx = int(progress * (len(x) - 1))
                idx = min(idx, len(x) - 1)
                
                # Update dot position
                line_data['dot'].set_offsets([[x[idx], y[idx]]])
        
        return [ld['dot'] for ld in all_lines_data]
    
    # Create the animation
    print(f"Creating animation with {nframes} frames at {fps} fps...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=nframes, interval=1000/fps,
        blit=True, repeat=True
    )
    
    # Save animation
    try:
        if use_gif:
            # Use GIF if ffmpeg doesn't work
            output_file = output_file.replace('.mp4', '.gif')
            print(f"Saving as GIF to {output_file}...")
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=100)
            print(f"✓ GIF saved successfully!")
        else:
            # Try MP4
            print(f"Saving as MP4 to {output_file}...")
            writer = FFMpegWriter(fps=fps, codec='mpeg4', bitrate=1800)
            anim.save(output_file, writer=writer, dpi=100)
            print(f"✓ MP4 saved successfully!")
        
        # Create download link
        display(FileLink(output_file))
        print(f"\nAnimation saved! Click the link above to download.")
        
        return anim, fig
        
    except Exception as e:
        print(f"\n❌ Error saving animation: {e}")
        print("\nTrying alternative methods...")
        
        # Try GIF as fallback
        try:
            output_file_gif = output_file.replace('.mp4', '.gif')
            print(f"Attempting GIF format: {output_file_gif}")
            writer = PillowWriter(fps=fps)
            anim.save(output_file_gif, writer=writer, dpi=100)
            print(f"✓ GIF saved successfully!")
            display(FileLink(output_file_gif))
            return anim, fig
        except Exception as e2:
            print(f"❌ GIF also failed: {e2}")
            print("\n🔧 TROUBLESHOOTING:")
            print("1. Check if ffmpeg is installed: !which ffmpeg")
            print("2. Install ffmpeg: conda install ffmpeg")
            print("3. Or use: use_gif=True in the function call")
            print("\n📺 Displaying animation in notebook instead...")
            
            # Show in notebook as last resort
            return HTML(anim.to_jshtml()), fig

def calculate_pairwise_significance(data, groups, x_var, y_var):
    """
    Calculate pairwise significance between all groups
    Returns a dictionary of p-values and significance levels
    """
    from scipy import stats
    results = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = data[data[x_var] == groups[i]][y_var]  # Changed from 'category' and 'senescence_score'
            group2 = data[data[x_var] == groups[j]][y_var]  # Changed from 'category' and 'senescence_score'
            
            # Perform Mann-Whitney U test
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Add significance stars
            if pvalue < 0.001:
                sig = '***'
            elif pvalue < 0.01:
                sig = '**'
            elif pvalue < 0.05:
                sig = '*'
            else:
                sig = 'ns'
                
            results[(i, j)] = {'p-value': pvalue, 'significance': sig}
    
    return results
    
def plot_violin_box_combo(data, x_var, y_var, title=None, x_ticks=None, palette=None, rotation=45, show_scatter=True):
    """
    Create a combined violin-box plot with optional scatter points
    
    Parameters:
    -----------
    show_scatter : bool, default=True
        If True, shows individual data points as scatter. If False, shows only violin and box.
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 6))
    
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9)

    # Calculate y-axis limits based on data
    y_min = data[y_var].min()
    y_max = data[y_var].max()
    y_range = y_max - y_min
    
    # Add padding proportional to the data range (10% on each side)
    padding = y_range * 0.1
    y_min_plot = y_min - padding
    y_max_plot = y_max + padding
    
    # Only use floor/ceil if the range is large enough
    if y_range > 1.0:
        y_min_plot = np.floor(y_min_plot * 2) / 2
        y_max_plot = np.ceil(y_max_plot * 2) / 2
    else:
        y_min_plot = max(0, y_min_plot)
    
    # Set initial y-axis limits
    ax.set_ylim(y_min_plot, y_max_plot)
    
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

    # Determine order
    if x_ticks is not None:
        categories = x_ticks
    else:
        categories = sorted(data[x_var].unique(), key=lambda x: float(x) if x.replace('.','').isdigit() else x)

    # Create violin plot with explicit order
    violin = sns.violinplot(
        data=data, x=x_var, y=y_var,
        order=categories,
        palette=palette, inner=None,
        linewidth=0, saturation=1.0,
        alpha=0.3, width=0.4, cut=0
    )

    # Create box plot with explicit order
    box_plot = sns.boxplot(
        data=data, x=x_var, y=y_var,
        order=categories,
        width=0.4, linewidth=1.2,
        flierprops={'marker': ' '},
        showmeans=False,
        boxprops={
            'facecolor': 'none',
            'edgecolor': 'none'
        },
        whiskerprops={'color': 'none'},
        medianprops={'color': 'none'},
        showcaps=False,
        ax=ax
    )

    # Count number of boxes and lines per box
    num_boxes = len(categories)
    lines_per_box = len(ax.lines) // num_boxes

    # Update box plot colors after creation
    for i, (name, box) in enumerate(zip(categories, ax.patches)):
        color = palette[name]
        
        # Create filled box with transparency
        box.set_facecolor(color)
        box.set_edgecolor('none')
        box.set_alpha(0.3)
        box.set_zorder(1)
        
        # Create box edges with full opacity
        import matplotlib.patches as mpatches
        path = box.get_path()
        edges = mpatches.PathPatch(
            path,
            facecolor='none',
            edgecolor=color,
            linewidth=1.2,
            alpha=1.0,
            zorder=2
        )
        ax.add_patch(edges)
        
        # Get and color all lines for this box
        box_lines = ax.lines[i * lines_per_box : (i + 1) * lines_per_box]
        for line in box_lines:
            line.set_color(color)
            line.set_alpha(1.0)
            line.set_linewidth(1.2)
            line.set_zorder(2)

    # ========== CONDITIONAL SCATTER POINTS ==========
    if show_scatter:
        # Add individual points on top with explicit order
        sns.stripplot(
            data=data, x=x_var, y=y_var,
            order=categories,
            palette=palette, size=6,
            alpha=1.0, linewidth=0,
            jitter=0.2, zorder=3
        )
    # ================================================
    
    # Calculate significance using the ordered categories
    significance_info = calculate_pairwise_significance(data, categories, x_var, y_var)

    # Get current y limits before adding bars
    current_ymin, current_ymax = ax.get_ylim()
    y_range_plot = current_ymax - current_ymin
    
    # Make bar spacing relative to the data range
    bar_spacing = y_range_plot * 0.08
    bar_tips = y_range_plot * 0.02
    bar_height = current_ymax + bar_spacing * 0.5

    # Add significance bar function
    def add_significance_bar(start, end, height, p_value, sig_symbol):
        # Draw the bar
        ax.plot([start, start, end, end], 
                [height, height + bar_tips, height + bar_tips, height],
                color='black', linewidth=0.8)
        
        # If p-value rounds to 0.0000 (very small), show only asterisks
        if p_value < 0.00005:  # This rounds to 0.0000 with 4 decimals
            text = sig_symbol  # Just "***"
        else:
            text = f'p = {p_value:.4f} {sig_symbol}'  # "p = 0.0123 **"
        ax.text((start + end) * 0.5, height + bar_tips, 
                text, ha='center', va='bottom', fontsize=8)

    # Add significant bars (p < 0.05 only)
    for (group1_idx, group2_idx), sig_data in significance_info.items():
        if sig_data['significance'] != 'ns':
            add_significance_bar(
                group1_idx, 
                group2_idx, 
                bar_height,
                sig_data['p-value'],
                sig_data['significance']
            )
            bar_height += bar_spacing

    # Adjust y-axis limits to accommodate bars
    ax.set_ylim(current_ymin, bar_height + bar_spacing * 0.5)

    if title:
        plt.title(title, pad=20)

    if x_ticks is None:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
    else:
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, rotation=rotation, ha='right')
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
        ax.spines['bottom'].set_visible(True)

    # Configure ticks and spines with thinner lines
    ax.minorticks_off()
    ax.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False)
    ax.tick_params(axis='x', which='major', top=False)
    ax.tick_params(axis='y', which='major', right=False, width=0.8)
    
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_tick_params(width=0.8)
    
    plt.setp(ax.get_yticklabels(), weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.yaxis.grid(False)
    
    sns.despine(offset=5, trim=True, bottom=(x_ticks is None), right=True)
    
    # Force rotation of x-tick labels
    if x_ticks is not None:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
    
    plt.close()
    
    return fig