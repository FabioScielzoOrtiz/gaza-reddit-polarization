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

def plot_cat_distribution(df, cat_cols, order=None, max_cols=3, palette="Set2", x_rotation=30, orient="v"):

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
            ax.set_xlabel('Proporción')
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
            ax.set_ylabel('Proporción')
            ax.tick_params(axis='x', rotation=x_rotation, labelsize=12)

        ax.set_title(col.upper(), fontsize=12, fontweight='bold')

    total_blocks = n_rows_fig * n_cols_fig
    for i in range(n_vars, total_blocks):
        r = i // n_cols_fig
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    plt.show()
    
#########################################################################################################################################################

def plot_quant_distribution(df, quant_cols, max_cols=3, box_color="skyblue", hist_color="salmon"):

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
        ax_hist.set_ylabel('Proporción')
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
    plt.show()

#########################################################################################################################################################

def plot_quant_comparison(df, comparisons, group_by=None, figsize=None, showfliers=True, order=None, labelbottom=True, xlabel_rotation=30, max_cols=3, title=None, palette="Set2", bbox_to_anchor=(0.5, -0.03)):
    
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
    plt.show()

#########################################################################################################################################################

def plot_cat_comparison(df, comparisons, group_by=None, max_cols=3, title=None, subplots_title=True, order=None,
                        hue_order=None, palette="Set2", cat_palette=None, 
                        bbox_to_anchor=(0.5, -0.03), sharey=False,
                        x_rotation=30, orientation="v"):

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
            ax.set_xlabel("Proporción condicional")
            ax.set_ylabel("")
        else:
            ax.set_xlabel("")
            ax.set_ylabel("Proporción condicional")
            
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
    plt.show()

#########################################################################################################################################################

def plot_quant_scatter(df, comparisons, group_by=None, figsize=None,
                       order=None, max_cols=3, title=False,
                       palette="Set2", alpha=0.6, show_regression=True, corr_annotation=True,
                       bbox_to_anchor=(0.5, -0.03)):

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

def get_SampleDistClustering_results(n_clusters, data, quant_cols, binary_cols, multiclass_cols, QUANT_COMPARISON_COLS, CAT_COMPARISON_COLS, CAT_COMPARISON_ORDER):
    
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
        random_state=123
    )

    clust_object = SampleDistClustering(
        clustering_method = clustering_method,
        metric = METRIC,
        frac_sample_size=FRAC_SAMPLE_SIZE,
        random_state=123,
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

    medoid_and_knn_by_cluster = get_medoid_and_knn_by_cluster(clust_object, knn_k=10)

    for cluster_label in np.unique(clust_object.labels_):
         
        print('-'*50)
        print(f'Medoid and KNN for Cluster {cluster_label}:\n')
        print('-'*50)
        display(data[medoid_and_knn_by_cluster[cluster_label], ['post_title', 'post_body', 'comment_body'] + CAT_COMPARISON_COLS])

####################################################################################################################################################################################################

def get_KMeans_results(n_clusters, data, X, QUANT_COMPARISON_COLS, CAT_COMPARISON_COLS, CAT_COMPARISON_ORDER):
    
    case_title = f'KMeans - k={n_clusters}'

    print('='*100)
    print(case_title)
    print('='*100)

    clust_object = KMeans(
        n_clusters=n_clusters
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

####################################################################################################################################################################################################