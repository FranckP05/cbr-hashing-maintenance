import os
from bokeh.plotting import figure
from bokeh.io import export_png
from bokeh.models import ColumnDataSource
from bokeh.transform import dodge

BACKGROUND_COLOR = "#f5f5f5"
TITLE_FONT = "Space Grotesk, sans-serif"
LABEL_FONT = "Manrope, sans-serif"

# 5 New Premium Color Palettes for 16:9 Ratio
PALETTES_16_9 = {
    "palette1_midnight_gold": {
        "name": "1. Midnight Luxe & Amber Gold",
        "description": "Contraste moderne entre ardoise profonde, cyan électrique et ambre chaud.",
        "colors": {
            "Baseline": "#94A3B8",  # Muted Slate
            "KD-Tree": "#06B6D4",   # Electric Cyan
            "LSH": "#F59E0B",       # Amber Gold
            "HeCBR": "#1E293B"      # Deep Slate Midnight
        }
    },
    "palette2_nordic_minimalist": {
        "name": "2. Nordic Minimalist",
        "description": "Style scandinave ultra-épuration : bleu nordique, orange brûlé et cendre.",
        "colors": {
            "Baseline": "#CBD5E1",  # Cool Ash
            "KD-Tree": "#0284C7",   # Nordic Sapphire
            "LSH": "#EA580C",       # Burnt Orange
            "HeCBR": "#0F172A"      # Obsidian Navy
        }
    },
    "palette3_emerald_indigo": {
        "name": "3. Emerald & Royal Indigo",
        "description": "Luxe académique : indigo profond, vert émeraude et ocre chaud.",
        "colors": {
            "Baseline": "#94A3B8",  # Cool Grey
            "KD-Tree": "#D97706",   # Warm Ochre
            "LSH": "#059669",       # Emerald Green
            "HeCBR": "#1E1B4B"      # Deep Royal Indigo
        }
    },
    "palette4_amethyst_teal": {
        "name": "4. Amethyst & Deep Teal (Navy Anchor)",
        "description": "Conservant le bleu marine d'origine, marié à un fuchsia améthyste et teal.",
        "colors": {
            "Baseline": "#64748B",  # Steel Grey
            "KD-Tree": "#059669",   # Emerald Teal
            "LSH": "#D946EF",       # Fuchsia Amethyst
            "HeCBR": "#041C6B"      # Deep Navy Blue
        }
    },
    "palette5_cobalt_crimson": {
        "name": "5. Cobalt & Vivid Crimson",
        "description": "Haute visibilité : bleu cobalt, rouge cramoisi vif et cyan sombre.",
        "colors": {
            "Baseline": "#94A3B8",  # Slate Neutral
            "KD-Tree": "#0D9488",   # Teal Cyan
            "LSH": "#E11D48",       # Vivid Crimson
            "HeCBR": "#1E3A8A"      # Cobalt Royal
        }
    }
}


def create_16_9_plot(palette_info):
    categories = ["k=1", "k=5", "k=10"]
    source = ColumnDataSource(data={
        "k": categories,
        "Baseline": [12.4, 15.1, 18.3],
        "KD-Tree": [4.2, 5.8, 7.1],
        "LSH": [1.8, 2.3, 2.9],
        "HeCBR": [0.9, 1.2, 1.5],
    })

    methods = ["Baseline", "KD-Tree", "LSH", "HeCBR"]
    offsets = [-0.3, -0.1, 0.1, 0.3]
    width = 0.18

    # 16:9 Aspect Ratio: width 800, height 450 (ratio 16/9)
    plot = figure(
        x_range=categories,
        title=f"Benchmark de récupération (16:9) — {palette_info['name']}",
        x_axis_label="Taille du voisinage (k)",
        y_axis_label="Temps de requête moyen (ms)",
        width=800,
        height=450,
        toolbar_location=None,
    )

    plot.background_fill_color = BACKGROUND_COLOR
    plot.border_fill_color = BACKGROUND_COLOR

    plot.title.text_font = TITLE_FONT
    plot.title.text_font_size = "14pt"
    plot.title.text_color = palette_info["colors"]["HeCBR"]
    plot.title.align = "center"

    plot.xaxis.axis_label_text_font = LABEL_FONT
    plot.yaxis.axis_label_text_font = LABEL_FONT
    plot.xaxis.major_label_text_font = LABEL_FONT
    plot.yaxis.major_label_text_font = LABEL_FONT
    plot.xaxis.axis_label_text_font_size = "11pt"
    plot.yaxis.axis_label_text_font_size = "11pt"

    plot.xgrid.grid_line_color = "#e2e8f0"
    plot.ygrid.grid_line_color = "#e2e8f0"

    for method, offset in zip(methods, offsets):
        color = palette_info["colors"][method]
        plot.vbar(
            x=dodge("k", offset, range=plot.x_range),
            top=method,
            width=width,
            source=source,
            color=color,
            legend_label=method,
        )

    plot.y_range.start = 0
    plot.legend.location = "top_left"
    plot.legend.orientation = "horizontal"
    plot.legend.background_fill_alpha = 0.85
    plot.legend.label_text_font = LABEL_FONT
    plot.legend.label_text_font_size = "10pt"

    return plot


def main():
    output_dir = "results/figures/palettes_16_9"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating 5 high-resolution 16:9 PNG figures (2400x1350 px)...")
    
    for key, palette_info in PALETTES_16_9.items():
        plot = create_16_9_plot(palette_info)
        filename = os.path.join(output_dir, f"{key}.png")
        # scale_factor=3 on 800x450 canvas produces exact 2400x1350 resolution (16:9 ratio)
        export_png(plot, filename=filename, scale_factor=3)
        print(f"Saved: {filename}")

    print(f"\nAll 5 16:9 PNG figures successfully saved in {output_dir}/ !")


if __name__ == "__main__":
    main()
