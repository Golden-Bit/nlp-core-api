import os
import time
from typing import List, Optional

import matplotlib
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from threading import Lock
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # Imposta il backend non interattivo
# Lock globale per Matplotlib
matplotlib_lock = Lock()
# Modelli Pydantic per i punti dati
#matplotlib.use("Agg")  # Imposta il backend non interattivo

class LinePlotDataPoint(BaseModel):
    x: float = Field(..., description="Coordinata X del punto.")
    y: float = Field(..., description="Coordinata Y del punto.")

class BarChartDataPoint(BaseModel):
    category: str = Field(..., description="Nome della categoria.")
    value: float = Field(..., description="Valore associato alla categoria.")

class PieChartDataPoint(BaseModel):
    label: str = Field(..., description="Etichetta per la fetta.")
    size: float = Field(..., description="Dimensione della fetta.")

# Modelli Pydantic per le funzioni di generazione grafici

class GenerateLinePlotModel(BaseModel):
    data_points: List[LinePlotDataPoint] = Field(
        ...,
        title="Data Points",
        description="Lista di punti dati con valori 'x' e 'y'."
    )
    x_label: str = Field(
        ...,
        title="X-Axis Label",
        description="Etichetta per l'asse X."
    )
    y_label: str = Field(
        ...,
        title="Y-Axis Label",
        description="Etichetta per l'asse Y."
    )
    plot_title: str = Field(
        ...,
        title="Plot Title",
        description="Titolo del grafico."
    )
    filename: str = Field(
        ...,
        title="Filename",
        description="Nome del file PNG da salvare (inclusa l'estensione .png)."
    )
    root_dir: str = Field(
        ...,
        title="Root Directory",
        description="Directory principale dove salvare il grafico."
    )
    color: Optional[str] = Field(
        None,
        title="Line Color",
        description="Colore della linea (es. 'blue', '#FF0000')."
    )
    linestyle: Optional[str] = Field(
        None,
        title="Line Style",
        description="Stile della linea (es. '-', '--', '-.', ':')."
    )
    marker: Optional[str] = Field(
        None,
        title="Marker",
        description="Simbolo del marker (es. 'o', 's', '^')."
    )

class GenerateBarChartModel(BaseModel):
    data_points: List[BarChartDataPoint] = Field(
        ...,
        title="Data Points",
        description="Lista di categorie e i loro valori associati."
    )
    x_label: str = Field(
        ...,
        title="X-Axis Label",
        description="Etichetta per l'asse X."
    )
    y_label: str = Field(
        ...,
        title="Y-Axis Label",
        description="Etichetta per l'asse Y."
    )
    plot_title: str = Field(
        ...,
        title="Plot Title",
        description="Titolo del grafico."
    )
    filename: str = Field(
        ...,
        title="Filename",
        description="Nome del file PNG da salvare (inclusa l'estensione .png)."
    )
    root_dir: str = Field(
        ...,
        title="Root Directory",
        description="Directory principale dove salvare il grafico."
    )
    color: Optional[str] = Field(
        None,
        title="Bar Color",
        description="Colore delle barre (es. 'skyblue', '#00FF00')."
    )
    edgecolor: Optional[str] = Field(
        None,
        title="Edge Color",
        description="Colore dei bordi delle barre."
    )

class GeneratePieChartModel(BaseModel):
    data_points: List[PieChartDataPoint] = Field(
        ...,
        title="Data Points",
        description="Lista di etichette e dimensioni per le fette della torta."
    )
    plot_title: str = Field(
        ...,
        title="Plot Title",
        description="Titolo del grafico."
    )
    filename: str = Field(
        ...,
        title="Filename",
        description="Nome del file PNG da salvare (inclusa l'estensione .png)."
    )
    root_dir: str = Field(
        ...,
        title="Root Directory",
        description="Directory principale dove salvare il grafico."
    )
    colors: Optional[List[str]] = Field(
        None,
        title="Colors",
        description="Lista di colori per le fette (es. ['gold', 'lightcoral', 'lightskyblue'])."
    )
    autopct: Optional[str] = Field(
        None,
        title="Autopct",
        description="Formato per visualizzare le percentuali sulle fette (es. '%1.1f%%')."
    )
    startangle: Optional[float] = Field(
        None,
        title="Start Angle",
        description="Angolo di inizio del grafico (es. 90)."
    )
    explode: Optional[List[float]] = Field(
        None,
        title="Explode",
        description="Lista di valori per 'esplodere' le fette (es. [0, 0.1, 0])."
    )
    shadow: Optional[bool] = Field(
        False,
        title="Shadow",
        description="Se True, aggiunge un'ombra al grafico."
    )

# Classe GraphManager con metodi per generare grafici e restituire gli strumenti

class GraphManager:
    def __init__(self):
        """Inizializza il GraphManager."""
        pass

    # Metodo per generare un grafico a linee
    def generate_line_plot(self, **kwargs):
        """Genera un grafico a linee e lo salva come file PNG."""
        with matplotlib_lock:  # Sincronizza l'accesso a Matplotlib
            try:
                args = GenerateLinePlotModel(**kwargs)
                x_values = [point.x for point in args.data_points]
                y_values = [point.y for point in args.data_points]
                plt.figure()
                plt.plot(
                    x_values,
                    y_values,
                    color=args.color,
                    linestyle=args.linestyle,
                    marker=args.marker
                )
                plt.xlabel(args.x_label)
                plt.ylabel(args.y_label)
                plt.title(args.plot_title)
                os.makedirs(args.root_dir, exist_ok=True)
                full_path = os.path.join(args.root_dir, args.filename)
                plt.savefig(full_path)
                plt.close()
            finally:
                plt.close()
            time.sleep(5)
        return f"Line plot saved successfully at {full_path}"

    # Metodo per generare un grafico a barre
    def generate_bar_chart(self, **kwargs):

        """Genera un grafico a barre e lo salva come file PNG."""
        with matplotlib_lock:  # Sincronizza l'accesso a Matplotlib
            try:
                args = GenerateBarChartModel(**kwargs)
                categories = [point.category for point in args.data_points]
                values = [point.value for point in args.data_points]
                plt.figure()
                plt.bar(
                    categories,
                    values,
                    color=args.color,
                    edgecolor=args.edgecolor
                )
                plt.xlabel(args.x_label)
                plt.ylabel(args.y_label)
                plt.title(args.plot_title)
                os.makedirs(args.root_dir, exist_ok=True)
                full_path = os.path.join(args.root_dir, args.filename)
                plt.savefig(full_path)
            finally:
                plt.close()
            time.sleep(5)
        return f"Bar chart saved successfully at {full_path}"

    # Metodo per generare un grafico a torta
    def generate_pie_chart(self, **kwargs):

        """Genera un grafico a torta e lo salva come file PNG."""
        with matplotlib_lock:  # Sincronizza l'accesso a Matplotlib
            try:
                args = GeneratePieChartModel(**kwargs)
                labels = [point.label for point in args.data_points]
                sizes = [point.size for point in args.data_points]
                plt.figure()
                plt.pie(
                    sizes,
                    labels=labels,
                    colors=args.colors,
                    autopct=args.autopct,
                    startangle=args.startangle,
                    explode=args.explode,
                    shadow=args.shadow
                )
                plt.title(args.plot_title)
                os.makedirs(args.root_dir, exist_ok=True)
                full_path = os.path.join(args.root_dir, args.filename)
                plt.savefig(full_path)
            finally:
                plt.close()
            time.sleep(5)
        return f"Pie chart saved successfully at {full_path}"

    # Metodo per restituire gli strumenti strutturati
    def get_tools(self):
        """Restituisce una lista degli strumenti configurati usando StructuredTool."""
        return [
            StructuredTool(
                name="generate_line_plot",
                func=self.generate_line_plot,
                description=(
                    "Usa questo strumento per generare un grafico a linee. "
                    "Richiede punti dati, etichette degli assi, titolo, nome del file e directory di salvataggio."
                ),
                args_schema=GenerateLinePlotModel
            ),
            StructuredTool(
                name="generate_bar_chart",
                func=self.generate_bar_chart,
                description=(
                    "Usa questo strumento per generare un grafico a barre. "
                    "Richiede punti dati (categorie e valori), etichette degli assi, titolo, nome del file e directory di salvataggio."
                ),
                args_schema=GenerateBarChartModel
            ),
            StructuredTool(
                name="generate_pie_chart",
                func=self.generate_pie_chart,
                description=(
                    "Usa questo strumento per generare un grafico a torta. "
                    "Richiede punti dati (etichette e dimensioni), titolo, nome del file e directory di salvataggio."
                ),
                args_schema=GeneratePieChartModel
            ),
        ]

# Esempi dettagliati di utilizzo

if __name__ == "__main__":
    # Inizializza il GraphManager
    graph_manager = GraphManager()

    # Esempio: Genera un grafico a linee
    line_plot_args = {
        "data_points": [
            {"x": 1, "y": 2},
            {"x": 2, "y": 3},
            {"x": 3, "y": 5},
            {"x": 4, "y": 7}
        ],
        "x_label": "X-Axis",
        "y_label": "Y-Axis",
        "plot_title": "Sample Line Plot",
        "filename": "line_plot.png",
        "root_dir": "data/plots",
        "color": "blue",
        "linestyle": "--",
        "marker": "o"
    }
    message_line = graph_manager.generate_line_plot(**line_plot_args)
    print(message_line)

    # Esempio: Genera un grafico a barre
    bar_chart_args = {
        "data_points": [
            {"category": "Category A", "value": 4},
            {"category": "Category B", "value": 7},
            {"category": "Category C", "value": 1}
        ],
        "x_label": "Categories",
        "y_label": "Values",
        "plot_title": "Sample Bar Chart",
        "filename": "bar_chart.png",
        "root_dir": "data/plots",
        "color": "skyblue",
        "edgecolor": "black"
    }
    message_bar = graph_manager.generate_bar_chart(**bar_chart_args)
    print(message_bar)

    # Esempio: Genera un grafico a torta
    pie_chart_args = {
        "data_points": [
            {"label": "Slice A", "size": 30},
            {"label": "Slice B", "size": 45},
            {"label": "Slice C", "size": 25}
        ],
        "plot_title": "Sample Pie Chart",
        "filename": "pie_chart.png",
        "root_dir": "data/plots",
        "colors": ["gold", "lightcoral", "lightskyblue"],
        "autopct": "%1.1f%%",
        "startangle": 90,
        "explode": [0, 0.1, 0],
        "shadow": True
    }
    message_pie = graph_manager.generate_pie_chart(**pie_chart_args)
    print(message_pie)

    # Ottenere gli strumenti strutturati
    tools = graph_manager.get_tools()
    for tool in tools:
        print(f"Tool Name: {tool.name}")
        print(f"Description: {tool.description}")
        print(f"Arguments Schema: {tool.args_schema.schema()}\n")
