import os
from bokeh.plotting import figure
from bokeh.io import export_png
from bokeh.models import ColumnDataSource
from bokeh.transform import dodge

BACKGROUND_COLOR = "#f5f5f5"

# 5 Font Combos
FONT_COMBOS = {
    "combo1_deeptech": {
        "name": "Combo 1: DeepTech Research (Space Grotesk / Manrope)",
        "title_font": "Space Grotesk, sans-serif",
        "label_font": "Manrope, sans-serif"
    },
    "combo2_swiss": {
        "name": "Combo 2: Swiss Studio Edge (Studio Feixen Sans / Onest)",
        "title_font": "Studio Feixen Sans TRIAL, sans-serif",
        "label_font": "Onest, sans-serif"
    },
    "combo3_compact": {
        "name": "Combo 3: Variable Compact (Georama / Rethink Sans)",
        "title_font": "Georama, sans-serif",
        "label_font": "Rethink Sans, sans-serif"
    },
    "combo4_neogrotesque": {
        "name": "Combo 4: Neogrotesque Clean (Noi Grotesk / Rubik)",
        "title_font": "Noi Grotesk Trial, sans-serif",
        "label_font": "Rubik, sans-serif"
    },
    "combo5_academic": {
        "name": "Combo 5: Academic Slab (Zilla Slab / Roboto Mono)",
        "title_font": "Zilla Slab, serif",
        "label_font": "Roboto Mono, monospace"
    }
}

# 3 Color Palettes (All 4 methods, HeCBR = #041C6B)
COLOR_PALETTES = {
    "palette_A": {
        "name": "Palette A (Terracotta & Teal)",
        "colors": {
            "Baseline": "#94A3B8",  # Slate Grey
            "KD-Tree": "#0D9488",   # Deep Teal
            "LSH": "#E65F2B",       # Terracotta / Coral
            "HeCBR": "#041C6B"      # Deep Navy Blue (Anchor)
        }
    },
    "palette_B": {
        "name": "Palette B (Amethyst & Amber)",
        "colors": {
            "Baseline": "#64748B",  # Cool Grey
            "KD-Tree": "#D97706",   # Amber / Golden
            "LSH": "#7C3AED",       # Violet Amethyst
            "HeCBR": "#041C6B"      # Deep Navy Blue (Anchor)
        }
    },
    "palette_C": {
        "name": "Palette C (Nordic Copper & Cyan)",
        "colors": {
            "Baseline": "#CBD5E0",  # Light Slate
            "KD-Tree": "#319795",   # Nordic Cyan
            "LSH": "#C05621",       # Copper Ochre
            "HeCBR": "#041C6B"      # Deep Navy Blue (Anchor)
        }
    }
}


def create_preview_plot(font_combo, palette, title_text):
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

    plot = figure(
        x_range=categories,
        title=title_text,
        x_axis_label="Taille du voisinage (k)",
        y_axis_label="Temps de requête moyen (ms)",
        height=450,
        width=700,
        toolbar_location=None,
    )

    plot.background_fill_color = BACKGROUND_COLOR
    plot.border_fill_color = BACKGROUND_COLOR

    plot.title.text_font = font_combo["title_font"]
    plot.title.text_font_size = "13pt"
    plot.title.text_color = "#041C6B"
    plot.title.align = "center"

    plot.xaxis.axis_label_text_font = font_combo["label_font"]
    plot.yaxis.axis_label_text_font = font_combo["label_font"]
    plot.xaxis.major_label_text_font = font_combo["label_font"]
    plot.yaxis.major_label_text_font = font_combo["label_font"]

    plot.xgrid.grid_line_color = "#dddddd"
    plot.ygrid.grid_line_color = "#dddddd"

    for method, offset in zip(methods, offsets):
        color = palette["colors"][method]
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
    plot.legend.background_fill_alpha = 0.8
    plot.legend.label_text_font = font_combo["label_font"]

    return plot


def main():
    output_dir = "results/figures/theme_previews"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating theme preview PNGs for all combinations...")
    count = 0

    for combo_key, font_combo in FONT_COMBOS.items():
        for palette_key, palette in COLOR_PALETTES.items():
            title = f"Temps de récupération : 4 Méthodes ({palette['name'].split(' ')[0]} - {combo_key.split('_')[0].upper()})"
            plot = create_preview_plot(font_combo, palette, title)

            filename = os.path.join(output_dir, f"{combo_key}_{palette_key}.png")
            export_png(plot, filename=filename, scale_factor=3)
            count += 1
            print(f"[{count}/15] Saved: {filename}")

    print(f"\nSuccessfully generated {count} theme preview PNGs in {output_dir}/ !")


if __name__ == "__main__":
    main()
