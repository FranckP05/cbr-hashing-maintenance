/* Global Chart.js lifecycle for HTMX-rendered demo fragments. */
(function () {
  "use strict";

  const instances = new Map();

  function formatTime(milliseconds) {
    if (milliseconds >= 1000) return `${(milliseconds / 1000).toFixed(2)} s`;
    if (milliseconds >= 1) return `${milliseconds.toFixed(2)} ms`;
    return `${Math.round(milliseconds * 1000)} µs`;
  }

  function showFallback(container, message) {
    container.classList.add("chart-unavailable");
    const fallback = container.querySelector(".chart-fallback");
    if (fallback) {
      fallback.hidden = false;
      fallback.textContent = message;
    }
  }

  function destroyBenchmark(container) {
    const canvas = container.querySelector("canvas[data-chart-canvas='benchmark']");
    if (!canvas) return;

    const instance = instances.get(canvas);
    if (instance) {
      instance.destroy();
      instances.delete(canvas);
    }
  }

  function destroyCharts(root) {
    if (!root || !root.querySelectorAll) return;
    root.querySelectorAll("[data-chart='benchmark']").forEach(destroyBenchmark);
  }

  function readData(container) {
    const dataElement = container.querySelector("script[data-chart-data='benchmark']");
    if (!dataElement) return null;

    try {
      return JSON.parse(dataElement.textContent);
    } catch (_) {
      return null;
    }
  }

  function initBenchmark(container) {
    const canvas = container.querySelector("canvas[data-chart-canvas='benchmark']");
    if (!canvas || instances.has(canvas)) return;

    if (typeof window.Chart === "undefined") {
      showFallback(container, "Graphique indisponible ; les temps restent indiqués dans le tableau.");
      return;
    }

    const data = readData(container);
    if (!data || !Array.isArray(data.labels) || !Array.isArray(data.values) || !data.labels.length) {
      showFallback(container, "Aucune donnée de benchmark exploitable.");
      return;
    }

    try {
      const fallback = container.querySelector(".chart-fallback");
      if (fallback) fallback.hidden = true;
      container.classList.remove("chart-unavailable");

      const chart = new window.Chart(canvas, {
        type: "bar",
        data: {
          labels: data.labels,
          datasets: [{
            data: data.values,
            backgroundColor: data.colors,
            borderRadius: 4,
            borderSkipped: false,
            barPercentage: 0.75,
            categoryPercentage: 0.85,
          }],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          resizeDelay: 100,
          animation: false,
          layout: { padding: { right: 18, top: 10, bottom: 10 } },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: "#18181B",
              titleFont: { family: "Space Grotesk", size: 13 },
              bodyFont: { family: "Space Grotesk", size: 12 },
              padding: 10,
              callbacks: {
                label(context) { return ` ${formatTime(context.raw)}`; },
              },
            },
          },
          scales: {
            x: {
              beginAtZero: true,
              grid: { color: "rgba(0, 0, 0, 0.05)", drawBorder: false },
              ticks: {
                font: { family: "Space Grotesk", size: 11 },
                color: "#6B7280",
                callback(value) { return formatTime(value); },
              },
            },
            y: {
              grid: { display: false, drawBorder: false },
              ticks: {
                font: { family: "Space Grotesk", weight: "600", size: 12 },
                color: "#18181B",
              },
            },
          },
        },
      });
      instances.set(canvas, chart);
    } catch (_) {
      showFallback(container, "Le graphique n’a pas pu être initialisé ; les temps restent indiqués dans le tableau.");
    }
  }

  function initCharts(root) {
    if (!root || !root.querySelectorAll) return;
    root.querySelectorAll("[data-chart='benchmark']").forEach(initBenchmark);
  }

  document.addEventListener("DOMContentLoaded", () => initCharts(document));
  document.body.addEventListener("htmx:beforeSwap", (event) => destroyCharts(event.detail.target));
  document.body.addEventListener("htmx:afterSettle", (event) => {
    // HTMX 1.9 emits this event on the settled target; its detail has no `elt`.
    initCharts(event.detail.target || event.target);
  });
})();
