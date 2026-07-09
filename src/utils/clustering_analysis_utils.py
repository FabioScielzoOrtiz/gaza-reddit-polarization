#########################################################################################################################################################

import os, sys
import polars as pl
import numpy as np
from db_robust_clust.models import SampleDistClustering
from db_robust_clust.plots import clustering_MDS_plot_one_method
from kmedoids import KMedoids
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from robust_mixed_dist.quantitative import euclidean_dist_matrix
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from dataclasses import dataclass, field
from typing import Literal, Sequence
import pandas as pd
from scipy import stats

sns.set_style('whitegrid')

script_path = os.getcwd()
project_path = os.path.join(script_path, '..', '..')

sys.path.append(project_path)

#########################################################################################################################################################

def group_stats(df, cols, group_by):
    stats = [
        expr
        for col in cols
        for expr in [
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).quantile(0.25).alias(f"{col}_q25"),
            pl.col(col).quantile(0.50).alias(f"{col}_median"),
            pl.col(col).quantile(0.75).alias(f"{col}_q75"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).max().alias(f"{col}_max"),
        ]
    ]
    return df.group_by(group_by).agg(stats)

#########################################################################################################################################################

def clustering_MDS_plot_one_method(X_mds, y_pred, y_true=None, title='', clustering_method=None, accuracy=None, time=None, 
                                   outliers_boolean=None, figsize=(8, 5), bbox_to_anchor=(1.2, 1), 
                                   title_size=13, title_weight='bold', points_size=40, title_height=0.98, 
                                   subtitles_size=12, subtitle_weight='bold', hspace=0.8, wspace=0.4, 
                                   save=False, file_name=None, format='jpg', dpi=250, legend_size=9, palette='bright'):
    """
    Computes and display the MDS plot for a considered clustering configuration, 
    differentiating the cluster labels and the real groups, if they are known.

    Parameters (inputs)
    ----------
    X_mds: a numpy array with the MDS matrix.
    y_pred: predicted cluster labels.
    y_true: true labels (if available).
    outliers_boolean: array-like boolean (0 or 1) indicating outliers (if available).
    ...

    Returns
    -------
    The described plot.
    """
    X_mds_df = pl.DataFrame(X_mds, schema=["Z1", "Z2"])
    labels_df = pl.DataFrame(y_pred, schema=["cluster_labels"])
    
    # Extraer y ordenar los clústeres para forzar a Seaborn a mostrarlos en orden
    cluster_order = sorted(list(np.unique(y_pred)))

    if outliers_boolean is not None:
        outliers_df = pl.DataFrame(outliers_boolean, schema=["outliers"])
        MDS_cluster_df = pl.concat((X_mds_df, labels_df, outliers_df), how='horizontal')
    else:
        MDS_cluster_df = pl.concat((X_mds_df, labels_df), how='horizontal')

    if y_true is not None:
        Y_df = pl.DataFrame(y_true, schema=["Y"])
        # Extraer y ordenar las etiquetas reales también por si acaso
        y_order = sorted(list(np.unique(y_true)))

        if outliers_boolean is not None:
            MDS_true_df = pl.concat((X_mds_df, Y_df, outliers_df), how='horizontal')
        else:
            MDS_true_df = pl.concat((X_mds_df, Y_df), how='horizontal')

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = axes.flatten()

        if outliers_boolean is not None:
            sns.scatterplot(x='Z1', y='Z2', hue='Y', style='outliers', data=MDS_true_df, ax=axes[0],
                            s=points_size, palette=palette, markers={0: 'o', 1: '^'}, hue_order=y_order)
        else:
            sns.scatterplot(x='Z1', y='Z2', hue='Y', data=MDS_true_df, ax=axes[0], s=points_size, 
                            palette=palette, hue_order=y_order)

        if outliers_boolean is not None:
            sns.scatterplot(x='Z1', y='Z2', hue='cluster_labels', style='outliers', data=MDS_cluster_df, ax=axes[1],
                            s=points_size, palette=palette, markers={0: 'o', 1: '^'}, hue_order=cluster_order)
        else:
            sns.scatterplot(x='Z1', y='Z2', hue='cluster_labels', data=MDS_cluster_df, ax=axes[1], s=points_size, 
                            palette=palette, hue_order=cluster_order)

        axes[0].set_title('Real groups', fontsize=subtitles_size, weight=subtitle_weight)

        if accuracy is not None and time is not None:
            axes[1].set_title(f'Predicted groups by\n{clustering_method}\nAcc:{np.round(accuracy,3)}, Time:{np.round(time,1)} secs', 
                              fontsize=subtitles_size, weight=subtitle_weight)
        elif accuracy is not None:
            axes[1].set_title(f'Predicted groups by\n{clustering_method}\nAcc:{np.round(accuracy,3)}', 
                              fontsize=subtitles_size, weight=subtitle_weight)
        elif time is not None:
            axes[1].set_title(f'Predicted groups by\n{clustering_method}\nTime:{np.round(time,1)} secs', 
                              fontsize=subtitles_size, weight=subtitle_weight)
        else:
            axes[1].set_title('Predicted groups', fontsize=subtitles_size, weight=subtitle_weight)

        axes[0].legend(title='Y', bbox_to_anchor=bbox_to_anchor, loc='upper right', fontsize=legend_size, title_fontsize=legend_size)
        axes[1].legend(title='Cluster labels', bbox_to_anchor=bbox_to_anchor, loc='upper right', fontsize=legend_size, title_fontsize=legend_size)

        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.suptitle(title, fontsize=title_size, y=title_height, weight=title_weight, color='black')

    else:
        fig, ax = plt.subplots(figsize=figsize)

        if outliers_boolean is not None:
            sns.scatterplot(x='Z1', y='Z2', hue='cluster_labels', style='outliers', data=MDS_cluster_df, 
                            s=points_size, palette=palette, markers={0: 'o', 1: '^'}, hue_order=cluster_order)
        else:
            sns.scatterplot(x='Z1', y='Z2', hue='cluster_labels', data=MDS_cluster_df, s=points_size, 
                            palette=palette, hue_order=cluster_order)

        ax.set_title(title, fontsize=title_size, y=title_height, weight=title_weight, color='black')
        ax.legend(title='Cluster labels', bbox_to_anchor=bbox_to_anchor, loc='upper right', fontsize=legend_size)

    if save:
        fig.savefig(file_name, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.show()

def plot_cat_distribution(df, cat_cols, order=None, max_cols=3, palette="Set2", x_rotation=30, orient="v", save_path=None):

    n_vars = len(cat_cols)
    if n_vars == 0:
        print("Aviso: La lista de columnas está vacía.")
        return

    n_cols_fig = min(n_vars, max_cols)
    n_rows_fig = math.ceil(n_vars / n_cols_fig)

    fig, axes = plt.subplots(nrows=n_rows_fig, ncols=n_cols_fig, figsize=(6 * n_cols_fig, 5 * n_rows_fig))

    if n_rows_fig == 1 and n_cols_fig == 1:
        axes = np.array([[axes]])
    elif n_rows_fig == 1:
        axes = axes[None, :]
    elif n_cols_fig == 1:
        axes = axes[:, None]

    for i, col in enumerate(cat_cols):

        r = i // n_cols_fig
        c = i % n_cols_fig
        ax = axes[r, c]

        serie_str = df[col].fill_null("Nulo").cast(pl.String).to_pandas()

        if order is True:
            col_order = serie_str.value_counts().index.tolist()
        elif order is False:
            col_order = serie_str.value_counts().index.tolist()[::-1]
        else:
            col_order = order

        if orient == "h":
            sns.countplot(
                y=serie_str,
                ax=ax,
                palette=palette,
                hue=serie_str,
                legend=False,
                stat='proportion',
                order=col_order
            )
            ax.set_xlabel('Proportion')
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelsize=12)
        else:
            sns.countplot(
                x=serie_str,
                ax=ax,
                palette=palette,
                hue=serie_str,
                legend=False,
                stat='proportion',
                order=col_order
            )
            ax.set_xlabel('')
            ax.set_ylabel('Proportion')
            ax.tick_params(axis='x', rotation=x_rotation, labelsize=12)

        ax.set_title(col.upper(), fontsize=12, fontweight='bold')

    total_blocks = n_rows_fig * n_cols_fig
    for i in range(n_vars, total_blocks):
        r = i // n_cols_fig
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    
    plt.show()
    
#########################################################################################################################################################

def plot_quant_distribution(df, quant_cols, max_cols=3, box_color="skyblue", hist_color="salmon", save_path=None):

    n_vars = len(quant_cols)
    
    # Validación por si la lista viene vacía
    if n_vars == 0:
        print("Aviso: La lista de columnas cuantitativas está vacía.")
        return

    # --- 1. CONFIGURACIÓN ADAPTATIVA ---
    n_cols_fig = min(n_vars, max_cols)
    
    # Calculamos cuántas "filas de variables" necesitamos
    n_rows_vars = math.ceil(n_vars / n_cols_fig) 
    
    # Como cada variable usa 2 filas reales (boxplot + hist), multiplicamos por 2
    n_rows_fig = n_rows_vars * 2

    # Ajustamos el tamaño de la figura dinámicamente
    fig, axes = plt.subplots(nrows=n_rows_fig, ncols=n_cols_fig, figsize=(6 * n_cols_fig, 3 * n_rows_fig))

    # Forzamos a que axes sea siempre un array 2D
    if n_rows_fig == 2 and n_cols_fig == 1:
        axes = axes.reshape(2, 1)

    # --- 2. DIBUJAR LOS GRÁFICOS ---
    for i, col in enumerate(quant_cols):
        
        # Mágia matemática para encontrar la coordenada exacta
        r = (i // n_cols_fig) * 2  
        c = i % n_cols_fig         
        
        ax_box = axes[r, c]
        ax_hist = axes[r + 1, c] 
        
        # Compatibilidad Polars -> Pandas (limpiamos nulos para que el KDE no falle)
        serie_num = df[col].drop_nulls().to_pandas()
        
        # --- BOXPLOT ---
        sns.boxplot(x=serie_num, ax=ax_box, color=box_color)
        ax_box.set_title(col.upper(), fontsize=12, fontweight='bold')
        ax_box.set_xlabel('')
        ax_box.tick_params(axis='x', labelsize=12)
        
        # --- HISTOGRAMA ---
        sns.histplot(x=serie_num, kde=True, ax=ax_hist, color=hist_color, edgecolor="black", stat='proportion')
        ax_hist.set_title('', fontsize=12, fontweight='bold')
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel('Proportion')
        ax_hist.tick_params(axis='x', labelsize=12)

    # --- 3. LIMPIEZA DE ESPACIOS VACÍOS ---
    total_blocks = n_rows_vars * n_cols_fig
    for i in range(n_vars, total_blocks):
        r = (i // n_cols_fig) * 2
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])
        fig.delaxes(axes[r + 1, c])

    # --- 4. RENDERIZADO ---
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)

    plt.show()

#########################################################################################################################################################

def plot_quant_comparison(df, comparisons, group_by=None, figsize=None, showfliers=True, order=None, labelbottom=True, 
                          xlabel_rotation=30, max_cols=3, title=None, palette="Set2", bbox_to_anchor=(0.5, -0.03),
                          save_path=None):
    
    n_blocks = len(comparisons)

    if n_blocks == 0:
        print("Aviso: La lista de comparaciones está vacía.")
        return

    # --- 1. CONFIGURACIÓN ADAPTATIVA ---
    n_cols_fig = min(n_blocks, max_cols)
    n_rows_blocks = math.ceil(n_blocks / n_cols_fig)
    n_rows_fig = n_rows_blocks

    fig, axes = plt.subplots(
        nrows=n_rows_fig, ncols=n_cols_fig,
        figsize=(6 * n_cols_fig, 4 * n_rows_fig) if not figsize else figsize
    )

    if n_rows_fig == 1 and n_cols_fig == 1:
        axes = np.array([[axes]])
    elif n_rows_fig == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_fig == 1:
        axes = axes.reshape(-1, 1)

    if group_by:
        group_vals = df[group_by].drop_nulls().unique().sort().to_list()
        hue_order = group_vals
        x_order_grouped = order if order else group_vals
        group_colors = sns.color_palette(palette, len(x_order_grouped))
        group_color_map = dict(zip(x_order_grouped, group_colors))

    # --- 2. DIBUJAR LOS BLOQUES ---
    for i, col_group in enumerate(comparisons):

        r = i // n_cols_fig
        c = i % n_cols_fig
        ax_box = axes[r, c]

        # --- CONSTRUCCIÓN DEL TIDY ---
        if group_by:
            pdf = df.select(col_group + [group_by]).to_pandas()
            tidy = pdf.melt(
                id_vars=group_by,
                value_vars=col_group,
                var_name="variable",
                value_name="valor"
            ).dropna(subset=["valor", group_by])
        else:
            pdf = df.select(col_group).to_pandas()
            tidy = pdf.melt(
                var_name="variable",
                value_name="valor"
            ).dropna()

        # --- TÍTULO ---
        if not title:
            block_title = " vs ".join(col.upper() for col in col_group)
            if group_by:
                block_title += f"\n(por {group_by})"
            if not showfliers:
                block_title += "\n(Outliers Hidden)"
            ax_box.set_title(block_title, fontsize=11, fontweight="bold", y=1.05)
        else:
            ax_box.set_title(title, fontsize=11, fontweight="bold", y=1.05)

        # --- BOXPLOT ---
        if group_by and len(col_group) == 1:
            x_order_filtered = [g for g in x_order_grouped if g in tidy[group_by].values]

            for j, group_val in enumerate(x_order_filtered):
                mask = tidy[group_by] == group_val
                subset = tidy.loc[mask, "valor"].dropna()
                if len(subset) == 0:
                    continue

                ax_box.boxplot(
                    subset,
                    positions=[j],
                    widths=0.5,
                    showfliers=showfliers,
                    patch_artist=True,
                    boxprops=dict(facecolor=group_color_map[group_val], alpha=0.7),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    flierprops=dict(marker='o', markerfacecolor=group_color_map[group_val], markersize=4, alpha=0.5)
                )

                mean_val = subset.mean()
                ax_box.plot(
                    j, mean_val, marker="D",
                    color=group_color_map[group_val],
                    markersize=6,
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=5
                )

            ax_box.set_xticks(range(len(x_order_filtered)))
            ax_box.set_xticklabels(x_order_filtered)

        elif group_by and len(col_group) > 1:
            sns.boxplot(
                data=tidy, x="variable", y="valor",
                hue=group_by, showfliers=showfliers,
                palette=palette, ax=ax_box,
                width=0.5, linewidth=1.5,
                legend=False, hue_order=hue_order,
                order=col_group
            )
            for j, col in enumerate(col_group):
                for g, group_val in enumerate(hue_order):
                    mask = (tidy["variable"] == col) & (tidy[group_by] == group_val)
                    if mask.sum() == 0:
                        continue
                    mean_val = tidy.loc[mask, "valor"].mean()
                    n_groups = len(hue_order)
                    offset = (g - (n_groups - 1) / 2) * (0.5 / n_groups)
                    ax_box.plot(
                        j + offset, mean_val, marker="D",
                        color=group_color_map[group_val],
                        markersize=6,
                        markeredgecolor="black", markeredgewidth=0.8,
                        zorder=5
                    )

        else:
            x_order = order if order else col_group
            var_colors = sns.color_palette(palette, len(col_group))
            sns.boxplot(
                data=tidy, x="variable", y="valor",
                hue="variable", showfliers=showfliers,
                palette=palette, ax=ax_box,
                width=0.5, linewidth=1.5,
                legend=False,
                order=x_order
            )
            for j, col in enumerate(x_order):
                mean_val = tidy[tidy["variable"] == col]["valor"].mean()
                ax_box.plot(
                    j, mean_val, marker="D",
                    color=var_colors[col_group.index(col)],
                    markersize=7,
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=5
                )

        ax_box.set_xlabel("")
        ax_box.set_ylabel("")
        ax_box.tick_params(axis="x", labelbottom=labelbottom, rotation=xlabel_rotation)

    # --- 3. LEYENDA GLOBAL MANUAL ---
    if group_by:
        legend_keys = x_order_grouped if len(comparisons[0]) == 1 else hue_order
        handles = [
            mpatches.Patch(color=group_color_map[g], alpha=0.7, label=str(g))
            for g in legend_keys if g in group_color_map
        ]
        legend_ncol = len(handles)
    else:
        all_vars = [col for col_group in comparisons for col in col_group]
        unique_vars = list(dict.fromkeys(all_vars))
        legend_colors = sns.color_palette(palette, len(unique_vars))
        handles = [
            mpatches.Patch(color=legend_colors[j], alpha=0.7, label=var)
            for j, var in enumerate(unique_vars)
        ]
        legend_ncol = len(unique_vars)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=legend_ncol,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=bbox_to_anchor
    )

    # --- 4. LIMPIEZA DE ESPACIOS VACÍOS ---
    total_blocks = n_rows_blocks * n_cols_fig
    for i in range(n_blocks, total_blocks):
        r = i // n_cols_fig
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)

    plt.show()

#########################################################################################################################################################

def plot_cat_comparison(df, comparisons, group_by=None, max_cols=3, title=None, subplots_title=True, order=None,
                        hue_order=None, palette="Set2", cat_palette=None, 
                        bbox_to_anchor=(0.5, -0.03), sharey=False,
                        x_rotation=30, orientation="v", save_path=None):

    n_blocks = len(comparisons)

    if n_blocks == 0:
        print("Aviso: La lista de comparaciones está vacía.")
        return

    # --- 1. PALETA FIJA POR CATEGORÍA ---
    if group_by:
        group_vals = df[group_by].drop_nulls().cast(pl.String).unique().sort().to_list()
        if cat_palette:
            fixed_palette = cat_palette
        else:
            fixed_palette = dict(zip(group_vals, sns.color_palette(palette, len(group_vals))))
        legend_keys = group_vals
    else:
        all_cats = (
            df.select([pl.col(col).cast(pl.String) for col_group in comparisons for col in col_group])
            .to_pandas().stack().unique()
        )
        if cat_palette:
            fixed_palette = cat_palette
        else:
            fixed_palette = dict(zip(all_cats, sns.color_palette(palette, len(all_cats))))
        legend_keys = list(fixed_palette.keys())

    # --- 2. CONFIGURACIÓN ADAPTATIVA ---
    n_cols_fig = min(n_blocks, max_cols)
    n_rows_blocks = math.ceil(n_blocks / n_cols_fig)
    n_rows_fig = n_rows_blocks

    fig, axes = plt.subplots(
        nrows=n_rows_fig, ncols=n_cols_fig,
        figsize=(6 * n_cols_fig, 5 * n_rows_fig),
        sharey=sharey
    )

    if n_rows_fig == 1 and n_cols_fig == 1:
        axes = np.array([[axes]])
    elif n_rows_fig == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_fig == 1:
        axes = axes.reshape(-1, 1)

    # --- 3. DIBUJAR LOS BLOQUES ---
    for i, col_group in enumerate(comparisons):

        r = i // n_cols_fig
        c = i % n_cols_fig
        ax = axes[r, c]

        if group_by:
            for col in col_group:
                pdf = df.select([col, group_by]).to_pandas()
                pdf[col] = pdf[col].fillna("Nulo").astype(str)

                # Proporción condicional por grupo: n(cat, grupo) / n(grupo)
                prop = (
                    pdf.groupby([group_by, col])
                    .size()
                    .reset_index(name="n")
                )
                prop["proporcion"] = prop.groupby(group_by)["n"].transform(lambda x: x / x.sum())

                # order → eje X (valores de col) manejando diccionario o lista
                fallback_order = pdf[col].value_counts().sort_values(ascending=False).index.tolist()
                if isinstance(order, dict):
                    col_order = order.get(col, fallback_order)
                else:
                    col_order = order if order else fallback_order

                # hue_order → orden de los grupos (valores de group_by)
                resolved_hue_order = hue_order if hue_order else group_vals
                local_palette = {k: fixed_palette[k] for k in group_vals if k in fixed_palette}

                if orientation == "h":
                    sns.barplot(
                        data=prop, x="proporcion", y=col, hue=group_by,
                        palette=local_palette, order=col_order,
                        hue_order=resolved_hue_order, ax=ax, legend=False
                    )
                else:
                    sns.barplot(
                        data=prop, x=col, y="proporcion", hue=group_by,
                        palette=local_palette, order=col_order,
                        hue_order=resolved_hue_order, ax=ax, legend=False
                    )

        else:
            frames = []
            for col in col_group:
                serie = df[col].fill_null("Nulo").cast(pl.String).to_pandas()
                frames.append(pd.DataFrame({"valor": serie, "variable": col}))
            tidy = pd.concat(frames, ignore_index=True)

            # Proporción condicional por variable: n(cat, variable) / n(variable)
            prop = (
                tidy.groupby(["variable", "valor"])
                .size()
                .reset_index(name="n")
            )
            prop["proporcion"] = prop.groupby("variable")["n"].transform(lambda x: x / x.sum())

            # Para agrupar, buscar el orden de la primera columna que lo tenga definido
            fallback_order = tidy["valor"].value_counts().sort_values(ascending=False).index.tolist()
            if isinstance(order, dict):
                col_order = fallback_order
                for c_name in col_group:
                    if c_name in order:
                        col_order = order[c_name]
                        break
            else:
                col_order = order if order else fallback_order

            if orientation == "h":
                sns.barplot(
                    data=prop, x="proporcion", y="valor", hue="variable",
                    palette=palette, order=col_order, ax=ax, legend=False
                )
            else:
                sns.barplot(
                    data=prop, x="valor", y="proporcion", hue="variable",
                    palette=palette, order=col_order, ax=ax, legend=False
                )

        if subplots_title:
            block_title = " vs ".join(col.upper() for col in col_group)
            if group_by:
                block_title += f"\n(por {group_by})"
            ax.set_title(block_title, fontsize=11, fontweight="bold")

        # Ajuste de etiquetas de ejes según la orientación
        if orientation == "h":
            ax.set_xlabel("Conditional proportion")
            ax.set_ylabel("")
        else:
            ax.set_xlabel("")
            ax.set_ylabel("Conditional proportion")
            
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=10)

    # --- 4. LEYENDA GLOBAL MANUAL ---
    if group_by:
        legend_order = hue_order if hue_order else legend_keys
        handles = [
            mpatches.Patch(color=fixed_palette[g], alpha=0.8, label=str(g))
            for g in legend_order if g in fixed_palette
        ]
    else:
        all_vars = [col for col_group in comparisons for col in col_group]
        unique_vars = list(dict.fromkeys(all_vars))
        var_colors = sns.color_palette(palette, len(unique_vars))
        handles = [
            mpatches.Patch(color=var_colors[j], alpha=0.8, label=var)
            for j, var in enumerate(unique_vars)
        ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=bbox_to_anchor
    )

    # --- 5. LIMPIEZA DE ESPACIOS VACÍOS ---
    total_blocks = n_rows_blocks * n_cols_fig
    for i in range(n_blocks, total_blocks):
        r = i // n_cols_fig
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)

    plt.show()

#########################################################################################################################################################

def plot_quant_scatter(df, comparisons, group_by=None, figsize=None,
                       order=None, max_cols=3, title=False,
                       palette="Set2", alpha=0.6, show_regression=True, corr_annotation=True,
                       bbox_to_anchor=(0.5, -0.03), save_path=None):

    n_blocks = len(comparisons)
    if n_blocks == 0:
        print("Aviso: La lista de comparaciones está vacía.")
        return

    # --- 1. LAYOUT ---
    n_cols_fig = min(n_blocks, max_cols)
    n_rows_fig = math.ceil(n_blocks / n_cols_fig)

    fig, axes = plt.subplots(
        nrows=n_rows_fig, ncols=n_cols_fig,
        figsize=(5 * n_cols_fig, 4.5 * n_rows_fig) if not figsize else figsize
    )

    if n_rows_fig == 1 and n_cols_fig == 1:
        axes = np.array([[axes]])
    elif n_rows_fig == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_fig == 1:
        axes = axes.reshape(-1, 1)

    # --- 2. GRUPOS Y COLORES ---
    if group_by:
        group_vals = df[group_by].drop_nulls().unique().sort().to_list()
        hue_order  = order if order else group_vals
        group_colors = sns.color_palette(palette, len(hue_order))
        group_color_map = dict(zip(hue_order, group_colors))
    else:
        pair_colors = sns.color_palette(palette, n_blocks)

    # --- 3. DIBUJAR BLOQUES ---
    for i, (x_col, y_col) in enumerate(comparisons):
        r = i // n_cols_fig
        c = i % n_cols_fig
        ax = axes[r, c]

        # Construir pandas subset
        cols = [x_col, y_col] + ([group_by] if group_by else [])
        pdf = df.select(cols).to_pandas().dropna(subset=[x_col, y_col])

        # --- TÍTULO ---
        if title:
            block_title = f"{x_col.upper()} vs {y_col.upper()}"
            if group_by:
                block_title += f"\n(por {group_by})"
            ax.set_title(block_title, fontsize=11, fontweight="bold", y=1.02)

        # --- SCATTER + REGRESIÓN ---
        if group_by:
            for g in hue_order:
                mask = pdf[group_by] == g
                sub  = pdf[mask]
                if sub.empty:
                    continue
                color = group_color_map[g]
                ax.scatter(sub[x_col], sub[y_col],
                           color=color, alpha=alpha, s=25,
                           linewidths=0, label=str(g), zorder=3)
                if show_regression and len(sub) > 2:
                    m, b, r_val, *_ = stats.linregress(sub[x_col], sub[y_col])
                    x_line = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
                    ax.plot(x_line, m * x_line + b,
                            color=color, linewidth=1.6, zorder=4)
            # r global (todos los grupos)
            r_val, p_val = stats.pearsonr(pdf[x_col], pdf[y_col])
        else:
            color = pair_colors[i]
            ax.scatter(pdf[x_col], pdf[y_col],
                       color=color, alpha=alpha, s=25,
                       linewidths=0, zorder=3)
            if show_regression and len(pdf) > 2:
                m, b, r_val, p_val, _ = stats.linregress(pdf[x_col], pdf[y_col])
                x_line = np.linspace(pdf[x_col].min(), pdf[x_col].max(), 100)
                ax.plot(x_line, m * x_line + b,
                        color=color, linewidth=1.8, zorder=4)
            r_val, p_val = stats.pearsonr(pdf[x_col], pdf[y_col])

        # --- LÍNEAS DE MEDIA ---
        ax.axvline(pdf[x_col].mean(), color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=2)
        ax.axhline(pdf[y_col].mean(), color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=2)

        # --- ANOTACIÓN r ---
        if corr_annotation:
            p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            ax.annotate(f"r = {r_val:.2f}  ({p_str})",
                        xy=(0.04, 0.96), xycoords="axes fraction",
                        fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                fc="white", ec="lightgray", alpha=0.8))

        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.tick_params(labelsize=8)

    # --- 4. LEYENDA GLOBAL ---
    if group_by:
        handles = [
            mpatches.Patch(color=group_color_map[g], alpha=0.8, label=str(g))
            for g in hue_order
        ]
    else:
        handles = [
            mpatches.Patch(color=pair_colors[j], alpha=0.8,
                           label=f"{x} vs {y}")
            for j, (x, y) in enumerate(comparisons)
        ]

    fig.legend(handles=handles,
               loc="lower center",
               ncol=len(handles),
               fontsize=9,
               frameon=True,
               bbox_to_anchor=bbox_to_anchor)

    # --- 5. LIMPIAR SUBPLOTS VACÍOS ---
    total = n_rows_fig * n_cols_fig
    for i in range(n_blocks, total):
        fig.delaxes(axes[i // n_cols_fig, i % n_cols_fig])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)

    plt.show()

#########################################################################################################################################################

def get_medoid_and_knn_by_cluster(clust_object, dist_matrix, knn_k=10):

    medoid_and_knn_by_cluster = {}

    for cluster_label in np.unique(clust_object.labels_):
        # Máscara e índices del cluster en espacio D
        cluster_mask = clust_object.labels_[clust_object.sample_idx] == cluster_label
        cluster_indices = np.where(cluster_mask)[0]

        # Medoide del cluster en espacio D
        medoid_idx_dist_matrix = clust_object.medoid_indices_[cluster_label]

        # Distancias del medoide a todos los puntos del cluster
        distances_to_medoid = dist_matrix[medoid_idx_dist_matrix, cluster_indices]

        # Ordenar por distancia e incluir medoide (posición 0)
        sorted_positions = np.argsort(distances_to_medoid)
        knn_indices_dist_matrix = cluster_indices[sorted_positions[0:knn_k+1]]
        knn_indices_input_data = clust_object.sample_idx[knn_indices_dist_matrix]

        medoid_and_knn_by_cluster[cluster_label] = knn_indices_input_data

    return medoid_and_knn_by_cluster

#########################################################################################################################################################

def generate_cluster_tfidf_visualizations(
    data: pl.DataFrame,
    clust_label_col,
    clust_object,
    cluster_labels,
    dist_matrix,
    results_dir: str,
    knn_k: int = 10,
    top_n: int = 10,
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 1000,
    bar_plot_name: str = 'optimal_clust_tf_idf_bar_plot.png',
    wordcloud_name: str = 'optimal_clust_tf_idf_wordcloud_plot.png',
    custom_stopwords: list[str] | None = None   
):
    """
    Filtra los datos por medoides y KNN, procesa el texto (NLP), 
    calcula c-TF-IDF por cluster y genera gráficos de barras y nubes de palabras.

    Args:
        ...
        custom_stopwords: Lista opcional de palabras adicionales a excluir
                          del análisis TF-IDF (e.g. ["mean", "yes", "thing"]).
    """
    
    # 1. Obtención de índices
    medoid_and_knn_by_cluster = get_medoid_and_knn_by_cluster(
        clust_object, 
        dist_matrix=dist_matrix, 
        knn_k=knn_k
    )

    all_target_indices = [idx for indices in medoid_and_knn_by_cluster.values() for idx in indices]

    # Normalizar custom_stopwords a un set de términos en minúscula
    extra_stopwords = {w.lower() for w in custom_stopwords} if custom_stopwords else set()  # ← NUEVO

    # Cargar modelo NLP
    nlp = spacy.load(spacy_model, disable=["ner", "parser"]) 

    # ==========================================
    # FASE 1: FILTRADO Y PREPROCESAMIENTO
    # ==========================================
    print("Filtrando dataset mediante índices de medoids y KNN...")

    df = data.with_row_index("row_idx").filter(pl.col("row_idx").is_in(all_target_indices))
    df = df.with_columns(pl.col("comment_body").fill_null("").alias("target_text"))

    def clean_text_batch(texts):
        for doc in nlp.pipe(texts, batch_size=batch_size):
            tokens = [
                token.lemma_.lower() for token in doc
                if not token.is_stop
                and token.is_alpha
                and token.lemma_.lower() not in extra_stopwords  # ← NUEVO FILTRO
            ]
            yield " ".join(tokens)

    print("Lematizando exclusivamente los comentarios filtrados...")
    raw_comments = df["target_text"].to_list()
    df = df.with_columns(pl.Series(name="cleaned_comment", values=list(clean_text_batch(raw_comments))))

    # ==========================================
    # FASE 2: C-TF-IDF (SIN INFLACIÓN DE CONTEXTO)
    # ==========================================
    print("Agrupando comentarios por cluster...")
    cluster_df = (
        df.group_by(clust_label_col)
        .agg(pl.col("cleaned_comment").str.join(" ")) 
        .sort(clust_label_col)
    )

    cluster_corpus = cluster_df["cleaned_comment"].to_list()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(cluster_corpus)
    feature_names = vectorizer.get_feature_names_out()

    # ==========================================
    # FASE 3: PREPARACIÓN DE DATOS
    # ==========================================
    plot_data = []
    cluster_term_frequencies = {} 

    for i, label in enumerate(cluster_labels):
        row = tfidf_matrix.getrow(i).toarray()[0]
        
        # Extraer Top N para el gráfico de barras
        top_indices = row.argsort()[-top_n:][::-1]
        for idx in top_indices:
            plot_data.append({
                "Cluster": label,
                "Term": feature_names[idx],
                "c-TF-IDF Score": row[idx]
            })
            
        # Extraer todas las palabras con score > 0 para la nube
        cluster_term_frequencies[label] = {
            feature_names[idx]: row[idx] for idx in np.where(row > 0)[0]
        }

    df_plot = pl.DataFrame(plot_data)

    # ==========================================
    # FASE 4: VISUALIZACIONES SOTA Y WORDCLOUDS
    # ==========================================

    # --- 1. GRÁFICO DE BARRAS FACETEADO ---
    print("Generando Gráfico de Barras SOTA...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    g = sns.FacetGrid(
        df_plot.to_pandas(), # Convertido a pandas por compatibilidad con seaborn
        col="Cluster", 
        col_wrap=2, 
        sharex=False, 
        sharey=False, 
        height=4, 
        aspect=1.2
    )

    def barplot_wrapper(x, y, **kwargs):
        ax = plt.gca()
        sns.barplot(x=x, y=y, ax=ax, palette="viridis", hue=y, legend=False)
        ax.set_xlabel("c-TF-IDF Score")
        ax.set_ylabel("")

    g.map(barplot_wrapper, "c-TF-IDF Score", "Term")
    g.set_titles(col_template="Cluster {col_name}", fontweight='bold')
    plt.suptitle('TF-IDF Barplot', y=1.02)
    plt.tight_layout()

    # Guardar gráfico de barras
    os.makedirs(results_dir, exist_ok=True) # Aseguramos que el directorio exista
    bar_plot_path = os.path.join(results_dir, bar_plot_name)
    plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    # --- 2. NUBES DE PALABRAS ---
    print("Generando Nubes de Palabras basadas en pesos c-TF-IDF...")
    num_clusters = len(cluster_labels)
    
    # Cálculo dinámico de filas para los subplots de la nube de palabras
    cols = 2
    rows = math.ceil(num_clusters / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    
    # Si solo hay 1 fila o 1 cluster, homogeneizamos axes a un array plano
    if num_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, label in enumerate(cluster_labels):
        wc = WordCloud(
            width=800, 
            height=800, 
            background_color="white",
            colormap="viridis",
            max_words=100
        ).generate_from_frequencies(cluster_term_frequencies[label])
        
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Cluster {label}", fontweight='bold', fontsize=16)
        axes[i].axis("off")

    # Ocultar ejes vacíos si num_clusters es impar
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle('TF-IDF WordCloud', fontsize=20, y=1.02)
    plt.tight_layout()

    # Guardar nube de palabras
    wordcloud_path = os.path.join(results_dir, wordcloud_name)
    plt.savefig(wordcloud_path, dpi=300, bbox_inches="tight")
    plt.show()


#########################################################################################################################################################

def _cramers_v(x, y):
    """
    Calcula la V de Cramer entre dos variables categóricas (nominales),
    con corrección de sesgo (Bergsma, 2013).
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    # Corrección de sesgo
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan

    return np.sqrt(phi2corr / denom)


def _pairwise_association(pdf, var_types):
    """
    Construye una matriz de asociación entre variables mixtas, eligiendo el
    coeficiente según el tipo de cada par de variables:

      - continua-continua   -> Pearson
      - continua-ordinal    -> Spearman
      - ordinal-ordinal     -> Spearman
      - nominal-nominal     -> V de Cramer
      - nominal-otro tipo   -> V de Cramer

    var_types: dict {nombre_columna: "continuous" | "ordinal" | "nominal"}
    """
    cols = list(pdf.columns)
    n = len(cols)
    corr = pd.DataFrame(np.eye(n), index=cols, columns=cols)

    for i in range(n):
        for j in range(i + 1, n):
            col_i, col_j = cols[i], cols[j]
            type_i, type_j = var_types[col_i], var_types[col_j]

            pair = pdf[[col_i, col_j]].dropna()

            if len(pair) < 2:
                value = np.nan
            elif type_i == "nominal" or type_j == "nominal":
                # nominal-nominal o nominal-cualquier otro tipo -> V de Cramer
                value = _cramers_v(pair[col_i], pair[col_j])
            elif type_i == "continuous" and type_j == "continuous":
                # continua-continua -> Pearson
                value = stats.pearsonr(pair[col_i], pair[col_j])[0]
            else:
                # continua-ordinal u ordinal-ordinal -> Spearman
                value = stats.spearmanr(pair[col_i], pair[col_j])[0]

            corr.loc[col_i, col_j] = value
            corr.loc[col_j, col_i] = value

    return corr

def plot_mixed_association_heatmap(df, var_types, group_by=None, figsize=(8, 6), cmap="coolwarm",
                                   annot=True, fmt=".2f", title=None, save_path=None):
    """
    Dibuja un heatmap de asociación entre un conjunto de variables de tipos mixtos,
    eligiendo automáticamente el coeficiente adecuado para cada par de variables.
    """

    cols = list(var_types.keys())

    if len(cols) == 0:
        print("Aviso: La lista de columnas está vacía.")
        return

    valid_types = {"continuous", "ordinal", "nominal"}
    invalid = {c: t for c, t in var_types.items() if t not in valid_types}
    if invalid:
        raise ValueError(f"Tipos de variable no válidos: {invalid}. Deben ser 'continuous', 'ordinal' o 'nominal'.")

    if group_by is None:

        pdf = df[cols].to_pandas()
        corr = _pairwise_association(pdf, var_types)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr, annot=annot, fmt=fmt, cmap=cmap,
            vmin=-1, vmax=1, square=True, ax=ax,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 11}  # <-- Aumenta el tamaño de los números internos
        )
        
        # Aumentar tamaño de variables y rotar el eje X
        ax.tick_params(axis='y', labelsize=11)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=11)
        
        ax.set_title(title if title else "Association Heatmap (Pearson / Spearman / Cramer's V)", fontsize=13, fontweight='bold')
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)

        plt.show()

    else:

        group_vals = df[group_by].drop_nulls().unique().sort().to_list()
        n_groups = len(group_vals)

        n_cols_fig = min(n_groups, 2)
        n_rows_fig = math.ceil(n_groups / n_cols_fig)  

        fig, axes = plt.subplots(nrows=n_rows_fig, ncols=n_cols_fig, figsize=(figsize[0]*n_cols_fig, figsize[1]*n_rows_fig))

        if n_rows_fig == 1 and n_cols_fig == 1:
            axes = np.array([[axes]])
        elif n_rows_fig == 1:
            axes = axes.reshape(1, -1)
        elif n_cols_fig == 1:
            axes = axes.reshape(-1, 1)

        for i, gval in enumerate(group_vals):
            r, c = divmod(i, n_cols_fig)
            ax = axes[r, c]

            pdf_group = df.filter(pl.col(group_by) == gval)[cols].to_pandas()
            corr = _pairwise_association(pdf_group, var_types)

            sns.heatmap(
                corr, annot=annot, fmt=fmt, cmap=cmap,
                vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10}  # <-- Aumenta el tamaño de los números internos en subfiguras
            )
            
            # Aumentar tamaño de variables y rotar eje X en cada subfigura
            ax.tick_params(axis='y', labelsize=10)
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=10)
            
            ax.set_title(f"{group_by} = {gval}", fontsize=12, fontweight='bold')

        total_blocks = n_rows_fig * n_cols_fig
        for i in range(n_groups, total_blocks):
            r, c = divmod(i, n_cols_fig)
            fig.delaxes(axes[r, c])

        plt.suptitle(title if title else "Association Heatmap by Cluster (Pearson / Spearman / Cramer's V)", fontsize=15, y=0.98)
        
        # Reducción de espacio entre subfiguras
        fig.tight_layout(h_pad=1.0, w_pad=1.0) 
        fig.subplots_adjust(wspace=0.15, hspace=0.32) # <-- Control fino del espacio horizontal (wspace) y vertical (hspace)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)

        plt.show()

#########################################################################################################################################################

VariableType = Literal["ordinal", "nominal"]

@dataclass
class VariableSeparability:
    """Separability result for a single variable."""
    variable: str
    var_type: VariableType
    metric_name: str          # "epsilon_squared" or "cramers_v"
    value: float               # in [0, 1]
    stat: float                 # underlying test statistic (H or chi2)
    p_value: float
    n_obs: int
    n_groups: int


@dataclass
class ConfigSeparability:
    """Aggregated separability result for a full clustering configuration."""
    config_name: str
    per_variable: list[VariableSeparability] = field(default_factory=list)

    @property
    def separability_index(self) -> float:
        """Unweighted mean of per-variable separability values (in [0, 1])."""
        if not self.per_variable:
            return float("nan")
        return float(np.mean([v.value for v in self.per_variable]))

    def weighted_separability_index(self, weights: dict[str, float]) -> float:
        """
        Weighted mean, e.g. if some variables matter more substantively.
        `weights` maps variable name -> weight (need not sum to 1; normalized
        internally). Variables absent from `weights` get weight 0.
        """
        vals, ws = [], []
        for v in self.per_variable:
            w = weights.get(v.variable, 0.0)
            if w > 0:
                vals.append(v.value)
                ws.append(w)
        if not ws:
            return float("nan")
        ws = np.array(ws) / np.sum(ws)
        return float(np.dot(vals, ws))

    def to_frame(self) -> pd.DataFrame:
        """Tidy dataframe, one row per variable, for reporting/plotting."""
        rows = [
            {
                "config": self.config_name,
                "variable": v.variable,
                "type": v.var_type,
                "metric": v.metric_name,
                "value": v.value,
                "stat": v.stat,
                "p_value": v.p_value,
                "n_obs": v.n_obs,
                "n_groups": v.n_groups,
            }
            for v in self.per_variable
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _effect_size_label(value: float) -> str:
        """Cohen-style qualitative label for eta-squared/Cramer's-V-scale effect sizes."""
        if value >= 0.14:
            return "large"
        elif value >= 0.06:
            return "medium"
        elif value >= 0.01:
            return "small"
        return "negligible"

    @staticmethod
    def _p_stars(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    def summary(self) -> str:
        """
        Plain-text, notebook-friendly summary table: one row per variable
        (metric, value, effect-size label, significance stars) plus the
        aggregated separability index at the bottom. Use inside `print()`.
        """
        if not self.per_variable:
            return f"ConfigSeparability('{self.config_name}'): no variables computed."

        rows = sorted(self.per_variable, key=lambda v: v.value, reverse=True)

        name_w = max(len(v.variable) for v in rows) + 2
        header = (
            f"{'Variable':<{name_w}} {'Metric':<16} {'Value':>7} "
            f"{'Effect':<11} {'p-value':>10} {'n':>7} {'k':>3}"
        )
        sep = "-" * len(header)

        lines = [
            f"Separability report — {self.config_name}",
            sep,
            header,
            sep,
        ]
        for v in rows:
            stars = self._p_stars(v.p_value)
            p_str = f"{v.p_value:.2e}{stars}" if v.p_value < 0.0001 else f"{v.p_value:.4f}{stars}"
            lines.append(
                f"{v.variable:<{name_w}} {v.metric_name:<16} {v.value:>7.3f} "
                f"{self._effect_size_label(v.value):<11} {p_str:>10} {v.n_obs:>7} {v.n_groups:>3}"
            )
        lines.append(sep)
        lines.append(
            f"{'SEPARABILITY INDEX (mean)':<{name_w + 16}} {self.separability_index:>7.3f} "
            f"{self._effect_size_label(self.separability_index):<11}"
        )
        lines.append("(significance: * p<.05  ** p<.01  *** p<.001)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        """
        Rich HTML rendering for Jupyter: automatically shown when the
        object is the last expression in a cell (e.g. just `result_I`),
        no print() needed. Falls back gracefully if pandas Styler is
        unavailable for any reason.
        """
        if not self.per_variable:
            return f"<b>ConfigSeparability('{self.config_name}')</b>: no variables computed."

        df = self.to_frame().sort_values("value", ascending=False).reset_index(drop=True)
        df["effect_size"] = df["value"].apply(self._effect_size_label)
        df["sig"] = df["p_value"].apply(self._p_stars)
        df = df[["variable", "type", "metric", "value", "effect_size", "p_value", "sig", "n_obs", "n_groups"]]

        idx_val = self.separability_index
        idx_label = self._effect_size_label(idx_val)

        styler = (
            df.style
            .background_gradient(subset=["value"], cmap="Greens", vmin=0, vmax=1)
            .format({"value": "{:.3f}", "p_value": "{:.2e}"})
            .hide(axis="index")
            .set_caption(
                f"<b>Separability report — {self.config_name}</b> "
                f"&nbsp;|&nbsp; Separability Index = <b>{idx_val:.3f}</b> ({idx_label})"
            )
            .set_table_styles([
                {"selector": "caption", "props": [("caption-side", "top"),
                                                    ("font-size", "1.05em"),
                                                    ("padding-bottom", "8px")]},
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ])
        )
        return styler.to_html()


def epsilon_squared_kruskal(values: np.ndarray, groups: np.ndarray) -> VariableSeparability:
    """
    Rank-based effect size (epsilon-squared) for an ordinal/quantitative
    variable across k groups (cluster labels), derived from the
    Kruskal-Wallis H statistic:

        epsilon^2 = (H - k + 1) / (n - k)

    This is the non-parametric analogue of eta-squared from a one-way
    ANOVA, and is preferred over eta-squared here because the discursive
    variables (e.g. political_stance_score, argument_quality_score) are
    ordinal/discrete Likert-type scales rather than continuous normal
    variables.

    Negative values (which can occur when H is close to its null-expected
    value under small samples) are clipped to 0, since epsilon-squared has
    no meaningful negative interpretation.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)

    mask = ~pd.isna(values) & ~pd.isna(groups)
    values, groups = values[mask], groups[mask]

    unique_groups = np.unique(groups)
    k = len(unique_groups)
    n = len(values)

    samples = [values[groups == g] for g in unique_groups]
    H, p = stats.kruskal(*samples)

    eps2 = (H - k + 1) / (n - k)
    eps2 = float(np.clip(eps2, 0.0, 1.0))

    return VariableSeparability(
        variable="",  # filled by caller
        var_type="ordinal",
        metric_name="epsilon_squared",
        value=eps2,
        stat=float(H),
        p_value=float(p),
        n_obs=n,
        n_groups=k,
    )


def cramers_v(categories: np.ndarray, groups: np.ndarray) -> VariableSeparability:
    """
    Cramer's V for the association between a nominal categorical variable
    and cluster membership, derived from a chi-square test of independence
    on the contingency table (cluster label x category).

    V = sqrt( (chi2 / n) / min(k - 1, r - 1) )

    where k = number of clusters, r = number of categories.

    Uses the bias correction of Bergsma (2013), which is preferred over the
    raw Cramer's V because the raw statistic is known to be upward-biased,
    especially with small samples or many categories/clusters.
    """
    categories = np.asarray(categories)
    groups = np.asarray(groups)

    mask = ~pd.isna(categories) & ~pd.isna(groups)
    categories, groups = categories[mask], groups[mask]

    contingency = pd.crosstab(groups, categories)
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    r, k = contingency.shape

    # Bias-corrected Cramer's V (Bergsma, 2013)
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)

    v = float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0
    v = float(np.clip(v, 0.0, 1.0))

    return VariableSeparability(
        variable="",  # filled by caller
        var_type="nominal",
        metric_name="cramers_v",
        value=v,
        stat=float(chi2),
        p_value=float(p),
        n_obs=int(n),
        n_groups=r,
    )


def compute_config_separability(
    data: pd.DataFrame,
    cluster_col: str,
    numeric_vars: Sequence[str] = (),
    nominal_vars: Sequence[str] = (),
    config_name: str = "",
) -> ConfigSeparability:
    """
    Main entry point: computes per-variable separability (rank-based
    epsilon-squared for numeric vars -- continuous OR ordinal-discrete --
    and Cramer's V for nominal vars) and returns a ConfigSeparability object
    exposing an aggregated `.separability_index` comparable across
    configurations, analogous to how silhouette scores are already
    compared.

    IMPORTANT: pass continuous variables (e.g. sentiment_score) in their
    native continuous form, NOT a categorized/binned version. Kruskal-Wallis
    -based epsilon-squared works directly on continuous data (it only needs
    to rank observations) and does not require discretizing first.
    Categorizing a continuous variable before this test throws away
    information and tends to *understate* true separation. It would also be
    inconsistent with the clustering pipeline itself, where sentiment_score
    enters the d1 (Mahalanobis) subspace as a continuous variable in both
    Config I and Config II.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain `cluster_col` and all variables listed in
        `numeric_vars` / `nominal_vars`.
    cluster_col : str
        Column with the cluster label for this configuration
        (e.g. "clust_label_I", "clust_label_III").
    numeric_vars : sequence of str
        Continuous or ordinal-discrete discursive variables, e.g.
        ["sentiment_score", "political_stance_score", "argument_quality_score"].
        Use each variable's native scale; do not pre-bin continuous ones.
    nominal_vars : sequence of str
        Nominal categorical discursive variables, e.g.
        ["discourse_tone_score", "dominant_frame_score"] (move a variable
        here only if it is genuinely unordered; otherwise keep it under
        `numeric_vars`, since epsilon-squared also handles ordinal-discrete
        data correctly).
    config_name : str
        Label for the configuration, used in the returned object and in
        `.to_frame()` for easy concatenation across configs.

    Returns
    -------
    ConfigSeparability
        .separability_index -> single aggregated number in [0, 1]
        .per_variable        -> list of per-variable results
        .to_frame()           -> tidy DataFrame for reporting/plotting
        .summary()            -> pretty printable text summary
        (also renders as a styled HTML table automatically in Jupyter)

    Example
    -------
    >>> result_I = compute_config_separability(
    ...     data=df,
    ...     cluster_col="clust_label_I",
    ...     numeric_vars=["sentiment_score", "political_stance_score", "argument_quality_score"],
    ...     nominal_vars=["discourse_tone_score", "dominant_frame_score"],
    ...     config_name="Config I",
    ... )
    >>> result_I  # in a notebook cell: renders a styled HTML table
    >>> print(result_I.summary())  # plain-text version
    """
    groups = data[cluster_col].to_numpy()
    per_variable: list[VariableSeparability] = []

    for var in numeric_vars:
        res = epsilon_squared_kruskal(data[var].to_numpy(), groups)
        res.variable = var
        per_variable.append(res)

    for var in nominal_vars:
        res = cramers_v(data[var].to_numpy(), groups)
        res.variable = var
        per_variable.append(res)

    return ConfigSeparability(config_name=config_name, per_variable=per_variable)


def style_separability_summary(summary: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Applies a green color gradient to a separability summary table
    (config vs separability_index), matching the same visual style used
    by ConfigSeparability._repr_html_. Returns a pandas Styler, which
    renders automatically as the last expression in a Jupyter cell.
    """
    vmax = max(float(summary["separability_index"].max()), 0.3)

    styler = (
        summary.style
        .format({"separability_index": "{:.3f}"})
        .background_gradient(subset=["separability_index"], cmap="Greens", vmin=0, vmax=vmax)
        .hide(axis="index")
        .set_caption("<b>Separability Index comparison across configurations</b>")
        .set_table_styles([
            {"selector": "caption", "props": [("caption-side", "top"),
                                                ("font-size", "1.05em"),
                                                ("padding-bottom", "8px")]},
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ])
    )
    return styler


def compare_configs_separability(
    results: Sequence[ConfigSeparability],
) -> tuple[pd.DataFrame, pd.DataFrame, "pd.io.formats.style.Styler"]:
    """
    Convenience function: stacks several ConfigSeparability results (one per
    configuration) into a single tidy comparison table, a summary table
    with the aggregated separability_index per configuration, and a
    color-styled version of that summary ready to display in a notebook
    cell (green gradient, same visual style as ConfigSeparability's own
    HTML repr).

    Intended to sit next to your existing silhouette comparison table, so
    both can be reported side by side in the same section of the paper.

    Returns
    -------
    long_df : pd.DataFrame
        One row per (config, variable) -- the full detail table.
    summary : pd.DataFrame
        One row per config -- plain DataFrame, sorted descending by
        separability_index (use this for further computation/export).
    summary_styled : pandas Styler
        Same data as `summary`, with a green color gradient applied.
        Just put it as the last line of a notebook cell to render it:
            long_df, summary, summary_styled = compare_configs_separability([...])
            summary_styled
    """
    long_df = pd.concat([r.to_frame() for r in results], ignore_index=True)

    summary = (
        pd.DataFrame(
            {
                "config": [r.config_name for r in results],
                "separability_index": [r.separability_index for r in results],
            }
        )
        .sort_values("separability_index", ascending=False)
        .reset_index(drop=True)
    )

    summary_styled = style_separability_summary(summary)

    return long_df, summary, summary_styled

#########################################################################################################################################################

def show_silhouette_table(silhouette_dict: dict[str, float]) -> None:
    """
    silhouette_dict: {'Config I': 0.22, 'Config II': 0.071, 'Config III': 0.073, 'Config III-b': 0.031}
    """
    df = (
        pd.DataFrame({
            'config': list(silhouette_dict.keys()),
            'silhouette': list(silhouette_dict.values()),
        })
        .sort_values('silhouette', ascending=False)
        .reset_index(drop=True)
    )

    styler = (
        df.style
        .format({'silhouette': '{:.3f}'})
        .background_gradient(subset=['silhouette'], cmap='Greens', vmin=0, vmax=max(df['silhouette'].max(), 0.3))
        .hide(axis='index')
        .set_caption('<b>Silhouette score comparison</b>')
        .set_table_styles([
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '1.05em'), ('padding-bottom', '8px')]},
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ])
    )
    return styler  # última línea de celda -> se renderiza sola

#########################################################################################################################################################

def get_SampleDistClustering_results(n_clusters, data, quant_cols, binary_cols, multiclass_cols, QUANT_COMPARISON_COLS, CAT_COMPARISON_COLS, CAT_COMPARISON_ORDER, ASSOCIATION_VAR_TYPES=None):
    
    case_title = f'SampleDistClustering-KMedoids-PAM - k={n_clusters}'

    print('='*100)
    print(case_title)
    print('='*100)


    KMEDOIDS_METHOD = 'pam'
    METRIC = 'ggower'
    D1 = 'robust_mahalanobis'
    D2 = 'sokal'
    D3 = 'hamming'
    ROBUST_METHOD = 'trimmed'
    ALPHA = 0.05
    FRAC_SAMPLE_SIZE = 0.15
    RANDOM_STATE = 123

    p1 = len(quant_cols)
    p2 = len(binary_cols)
    p3 = len(multiclass_cols)

    X = data[quant_cols + binary_cols + multiclass_cols]

    clustering_method = KMedoids(
        n_clusters=n_clusters, 
        metric='precomputed', 
        method=KMEDOIDS_METHOD, 
        init='build', 
        max_iter=100, 
        random_state=RANDOM_STATE
    )

    clust_object = SampleDistClustering(
        clustering_method = clustering_method,
        metric = METRIC,
        frac_sample_size=FRAC_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=False,
        p1=p1, p2=p2, p3=p3,
        d1=D1, d2=D2, d3=D3, 
        robust_method=ROBUST_METHOD, alpha=ALPHA
    )

    clust_object.fit(X)

    clust_labels = clust_object.labels_

    ##################################################################################################
    
    D, D1, D2, D3 = clust_object.dist_output
    
    sample_idx = clust_object.sample_idx

    n_reduced = 2000

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=123) 

    X_mds = mds.fit_transform(D[:n_reduced, :n_reduced])

    clustering_MDS_plot_one_method(X_mds=X_mds, 
                                y_pred=clust_labels[sample_idx[:n_reduced]], 
                                y_true=None, title="MDS visualization of clustering results", 
                                accuracy=None, time=None, 
                                figsize=(8,7), bbox_to_anchor=(1,1), 
                                title_size=13, title_weight='bold', 
                                points_size=45, title_height=1, 
                                save=False, legend_size=9)

    ##################################################################################################

    data = data.with_columns(
        pl.lit(clust_labels).cast(pl.String).alias('clust_labels')
    )

    ##################################################################################################

    plot_quant_comparison(
        figsize=(12, 4),
        df=data,
        comparisons=[
            [col] for col in QUANT_COMPARISON_COLS
        ],
        group_by="clust_labels",
        showfliers=True,
        title='',
        labelbottom=True,
        xlabel_rotation=0,
        bbox_to_anchor=(0.5, -0.12)
    )
    
    ##################################################################################################

    plot_cat_comparison(
        df=data, 
        comparisons=[
            [col] for col in CAT_COMPARISON_COLS 
        ], 
        title="",
        bbox_to_anchor=(0.5,-0.07),
        group_by="clust_labels",
        max_cols=3,
        x_rotation=0,
        orientation='h',
        order=CAT_COMPARISON_ORDER
    )

    ##################################################################################################

    if ASSOCIATION_VAR_TYPES:
        
        plot_mixed_association_heatmap(
            df=data,
            var_types=ASSOCIATION_VAR_TYPES,
            group_by="clust_labels",
            title="Association Heatmap (Pearson / Spearman / Cramer's V) by Cluster\n",
        )

    ##################################################################################################

    medoid_and_knn_by_cluster = get_medoid_and_knn_by_cluster(clust_object,dist_matrix=D, knn_k=10)

    for cluster_label in np.unique(clust_object.labels_):
         
        print('-'*50)
        print(f'Medoid and KNN for Cluster {cluster_label}:\n')
        print('-'*50)
        print(data[medoid_and_knn_by_cluster[cluster_label], ['post_title', 'post_body', 'comment_body'] + CAT_COMPARISON_COLS])

####################################################################################################################################################################################################

def get_KMeans_results(n_clusters, data, X, QUANT_COMPARISON_COLS, CAT_COMPARISON_COLS, CAT_COMPARISON_ORDER, ASSOCIATION_VAR_TYPES=None):
    
    case_title = f'KMeans - k={n_clusters}'

    print('='*100)
    print(case_title)
    print('='*100)

    RANDOM_STATE = 123

    clust_object = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE
    )

    clust_object.fit(X)

    clust_labels = clust_object.labels_

    ##################################################################################################
    
    n_reduced = 2000
    
    D = euclidean_dist_matrix(X[:n_reduced,:])

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=123) 

    X_mds = mds.fit_transform(D[:n_reduced, :n_reduced])

    clustering_MDS_plot_one_method(X_mds=X_mds, 
                                y_pred=clust_labels[:n_reduced], 
                                y_true=None, title="MDS visualization of clustering results", 
                                accuracy=None, time=None, 
                                figsize=(8,7), bbox_to_anchor=(1,1), 
                                title_size=13, title_weight='bold', 
                                points_size=45, title_height=1, 
                                save=False, legend_size=9)
    
    ##################################################################################################

    data = data.with_columns(
        pl.lit(clust_labels).cast(pl.String).alias('clust_labels')
    )

    ##################################################################################################

    plot_quant_comparison(
        figsize=(12, 4),
        df=data,
        comparisons=[
            [col] for col in QUANT_COMPARISON_COLS
        ],
        group_by="clust_labels",
        showfliers=True,
        title='',
        labelbottom=True,
        xlabel_rotation=0,
        bbox_to_anchor=(0.5, -0.12)
    )

    ##################################################################################################

    plot_cat_comparison(
        df=data, 
        comparisons=[
            [col] for col in CAT_COMPARISON_COLS 
        ], 
        title="",
        bbox_to_anchor=(0.5,-0.07),
        group_by="clust_labels",
        max_cols=3,
        x_rotation=0,
        orientation='h',
        order=CAT_COMPARISON_ORDER
    )

    ##################################################################################################

    if ASSOCIATION_VAR_TYPES:
        
        plot_mixed_association_heatmap(
            df=data,
            var_types=ASSOCIATION_VAR_TYPES,
            group_by="clust_labels",
            title="Association Heatmap (Pearson / Spearman / Cramer's V) by Cluster\n",
        )

####################################################################################################################################################################################################