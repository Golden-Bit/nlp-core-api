import os
import re
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


# Pydantic model for the input parameters
class ReplacePlaceholderModel(BaseModel):
    template_root: str = Field(..., description="The root directory of the template HTML file.")
    company_name: str = Field(..., description="The name of the company.")
    key: str = Field(..., description="The placeholder key to replace.")
    content: str = Field(..., description="The content to insert in place of the placeholder.")


class TemplateManager:
    def __init__(self):
        """Initialize the TemplateManager."""
        pass

    def generate_report(self, company_name: str, template_path: str):
        """
        Generates the HTML report by replacing the company name placeholder.

        Args:
            company_name (str): The name of the company.
            template_path (str): The path to the HTML template file.
        """
        # Read the template HTML content
        with open(template_path, 'r', encoding='utf-8') as file:
            html_template = file.read()

        # Replace the {company_name} placeholder with the actual company name
        html_output = html_template.replace('{company_name}', company_name)

        # Write the updated content back to the file
        with open(template_path, 'w', encoding='utf-8') as file:
            file.write(html_output)

    def check_and_generate_template(self, root_directory: str, company_name: str):
        """
        Checks if the template exists; if not, generates it and replaces the company name placeholder.

        Args:
            root_directory (str): The root directory where the template is stored.
            company_name (str): The name of the company.
        """
        os.makedirs(root_directory, exist_ok=True)
        template_filename = 'bilancio_sostenibilita.html'
        template_path = os.path.join(root_directory, template_filename)

        # Check if the template exists
        if os.path.exists(template_path):
            print(f"The template already exists at {template_path}.")
        else:
            # If not, create the template with a basic structure
            basic_template = '''<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>Bilancio di Sostenibilità 2023</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            color: #333;
        }}
        header, footer {{
            background-color: #0F9D58;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        nav {{
            background-color: #F1F1F1;
            padding: 10px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        nav a {{
            margin: 0 15px;
            color: #0F9D58;
            text-decoration: none;
            font-weight: bold;
        }}
        main {{
            padding: 20px;
        }}
        h1, h2 {{
            color: #0F9D58;
            margin-top: 40px;
        }}
        h3 {{
            color: #333;
            margin-top: 30px;
        }}
        .section-banner {{
            width: 100%;
            height: 200px;
            background-color: #E0E0E0;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #757575;
            font-size: 24px;
            margin-bottom: 30px;
        }}
        .content-placeholder {{
            background-color: #F9F9F9;
            padding: 20px;
            border-left: 5px solid #0F9D58;
            margin-bottom: 20px;
            line-height: 1.6;
        }}
        .image-placeholder {{
            width: 100%;
            height: 300px;
            background-color: #D3D3D3;
            margin: 20px 0;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #666;
            font-size: 18px;
        }}
        .table-placeholder {{
            width: 100%;
            background-color: #F0F0F0;
            padding: 15px;
            margin: 20px 0;
            text-align: center;
            color: #666;
            font-size: 18px;
        }}
        footer {{
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <!-- Header -->
    <header>
        <h1>Bilancio di Sostenibilità 2023</h1>
    </header>

    <!-- Navigation -->
    <nav>
        <a href="#introduzione">Introduzione</a>
        <a href="#capitolo1">Capitolo 1</a>
        <a href="#capitolo2">Capitolo 2</a>
        <a href="#capitolo3">Capitolo 3</a>
        <a href="#capitolo4">Capitolo 4</a>
        <a href="#conclusione">Conclusione</a>
    </nav>

    <!-- Main Content -->
    <main>
        <!-- Introduzione -->
        <section id="introduzione">
            <h2>Introduzione</h2>
            <div class="content-placeholder">
                <p>&lt;Introduzione| Inserire un'introduzione generale al bilancio di sostenibilità, presentando l'azienda, la sua missione, i valori e l'importanza della sostenibilità per l'organizzazione. Fornire una panoramica degli obiettivi principali e delle aree chiave trattate nel rapporto. |Introduzione&gt;</p>
            </div>
        </section>

        <!-- Capitolo 1 -->
        <section id="capitolo1">
            <h2>Capitolo 1: The Future is {company_name}</h2>
            <div class="section-banner">
                <p>Immagine rappresentativa del Capitolo 1</p>
            </div>

            <!-- 1.1 La nostra storia -->
            <h3>1.1 La nostra storia</h3>
            <div class="content-placeholder">
                <p>&lt;1.1 La nostra storia| Fornire una descrizione della storia dell'azienda, inclusi gli eventi chiave, le tappe fondamentali, la crescita nel tempo, e come l'azienda si è evoluta fino ad oggi, enfatizzando gli aspetti legati alla sostenibilità. |1.1 La nostra storia&gt;</p>
            </div>

            <!-- 1.2 Chi è {company_name} -->
            <h3>1.2 Chi è {company_name}</h3>
            <div class="content-placeholder">
                <p>&lt;1.2 Chi siamo| Fornire una descrizione dell'azienda, inclusa la missione, la visione, i valori fondamentali, i settori di attività e i principali risultati ottenuti. |1.2 Chi siamo&gt;</p>
            </div>
            <!-- Esempio di grafico o tabella -->
            <div class="image-placeholder">
                <p>Grafico o immagine relativa ai risultati aziendali</p>
            </div>

            <!-- 1.3 La nostra identità -->
            <h3>1.3 La nostra identità</h3>
            <div class="content-placeholder">
                <p>&lt;1.3 La nostra identità| Descrivere l'identità aziendale, come l'azienda si posiziona sul mercato, cosa la distingue dai concorrenti, e come la sostenibilità è integrata nel modello di business. |1.3 La nostra identità&gt;</p>
            </div>

            <!-- 1.4 La cultura aziendale -->
            <h3>1.4 La cultura aziendale</h3>
            <div class="content-placeholder">
                <p>&lt;1.4 La cultura aziendale| Presentare i valori e i principi che guidano la cultura aziendale, le pratiche interne che promuovono l'innovazione, la collaborazione, l'inclusione, e come questi elementi contribuiscono al raggiungimento degli obiettivi di sostenibilità. |1.4 La cultura aziendale&gt;</p>
            </div>
            <!-- Esempio di box informativo -->
            <div class="table-placeholder">
                <p>Box informativo sui valori aziendali</p>
            </div>
        </section>

        <!-- Capitolo 2 -->
        <section id="capitolo2">
            <h2>Capitolo 2: Up for Our People</h2>
            <div class="section-banner">
                <p>Immagine rappresentativa del Capitolo 2</p>
            </div>
            <div class="content-placeholder">
                <p>&lt;Descrizione Cap.2| Fornire una descrizione generale del Capitolo 2, focalizzato sulle persone dell'azienda, le iniziative intraprese per il loro benessere, sviluppo professionale e personale. Evidenziare l'importanza del capitale umano per l'organizzazione. |Descrizione Cap.2&gt;</p>
            </div>

            <!-- 2.1 Il Green Team -->
            <h3>2.1 Il Green Team</h3>
            <div class="content-placeholder">
                <p>&lt;2.1 Il Green Team| Descrivere il team dedicato alle iniziative di sostenibilità all'interno dell'azienda, includendo la composizione del team, i ruoli chiave, le responsabilità e gli obiettivi raggiunti durante il periodo di rendicontazione. |2.1 Il Green Team&gt;</p>
            </div>

            <!-- ... Altre sezioni del Capitolo 2 ... -->

        </section>

        <!-- Capitolo 3 -->
        <section id="capitolo3">
            <h2>Capitolo 3: Up for Our Planet</h2>
            <div class="section-banner">
                <p>Immagine rappresentativa del Capitolo 3</p>
            </div>
            <div class="content-placeholder">
                <p>&lt;Descrizione Cap.3| Fornire una descrizione generale del Capitolo 3, focalizzato sull'impegno dell'azienda verso la sostenibilità ambientale, le iniziative intraprese per ridurre l'impatto ambientale e contribuire alla lotta contro il cambiamento climatico. |Descrizione Cap.3&gt;</p>
            </div>

            <!-- ... Sezioni del Capitolo 3 ... -->

        </section>

        <!-- Capitolo 4 -->
        <section id="capitolo4">
            <h2>Capitolo 4: Up for Our Growth</h2>
            <div class="section-banner">
                <p>Immagine rappresentativa del Capitolo 4</p>
            </div>
            <div class="content-placeholder">
                <p>&lt;Descrizione Cap.4| Fornire una descrizione generale del Capitolo 4, focalizzato sulla crescita sostenibile dell'azienda, le pratiche di governance, l'etica nei rapporti commerciali e l'innovazione tecnologica a supporto dell'azione climatica. |Descrizione Cap.4&gt;</p>
            </div>

            <!-- ... Sezioni del Capitolo 4 ... -->

        </section>

        <!-- Conclusione -->
        <section id="conclusione">
            <h2>Conclusione</h2>
            <div class="content-placeholder">
                <p>&lt;Conclusione| Fornire una conclusione che riassuma i punti chiave del bilancio di sostenibilità, esprima l'impegno continuo dell'azienda verso la sostenibilità, delinei le prospettive future e ringrazi i lettori per l'attenzione. |Conclusione&gt;</p>
            </div>
        </section>

        <!-- Nota metodologica -->
        <section id="nota-metodologica">
            <h2>Nota metodologica</h2>
            <div class="content-placeholder">
                <p>&lt;Nota metodologica| Descrivere la metodologia adottata per la redazione del bilancio di sostenibilità, specificando gli standard e le linee guida seguite (ad esempio, ESRS), il perimetro di rendicontazione, le fonti dei dati utilizzati, eventuali cambiamenti rispetto all'anno precedente e informazioni sulla verifica dei dati. |Nota metodologica&gt;</p>
            </div>
        </section>

        <!-- Indice dei contenuti ESRS -->
        <section id="indice-esrs">
            <h2>Indice dei contenuti ESRS</h2>
            <div class="content-placeholder">
                <p>&lt;Indice ESRS| Fornire una tabella o un elenco che mappi i contenuti del bilancio di sostenibilità con i requisiti specifici degli standard ESRS, facilitando la consultazione e la verifica della conformità da parte dei lettori interessati. |Indice ESRS&gt;</p>
            </div>
            <!-- Esempio di tabella di mappatura -->
            <div class="table-placeholder">
                <p>Tabella di mappatura dei contenuti ESRS</p>
            </div>
        </section>

        <!-- Informazioni di contatto -->
        <section id="contatti">
            <h2>Informazioni di contatto</h2>
            <div class="content-placeholder">
                <p>&lt;Informazioni di contatto| Inserire le informazioni di contatto dell'azienda per domande, commenti o richieste di ulteriori informazioni riguardanti il bilancio di sostenibilità, includendo indirizzi email, sito web e indirizzi fisici delle sedi principali. |Informazioni di contatto&gt;</p>
            </div>
            <!-- Esempio di dettagli di contatto -->
            <div class="table-placeholder">
                <p>Dettagli di contatto dell'azienda</p>
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer>
        <p>&copy; 2023 {company_name} - Tutti i diritti riservati</p>
    </footer>
</body>
</html>
'''
            # Replace the {company_name} placeholder
            html_output = basic_template.replace('{company_name}', company_name)

            # Write the template to the file
            with open(template_path, 'w', encoding='utf-8') as file:
                file.write(html_output)
            print(f"Template generated and saved at {template_path}.")

    def replace_placeholder_in_html(self, template_root: str, company_name: str, key: str, content: str):
        """
        Replaces a placeholder in the HTML template with the specified content.

        Args:
            template_root (str): The root directory of the template.
            company_name (str): The name of the company.
            key (str): The placeholder key to replace.
            content (str): The content to insert.
        """
        template_path = os.path.join(template_root, 'bilancio_sostenibilita.html')

        # Ensure the template exists
        self.check_and_generate_template(template_root, company_name)

        # Read the HTML content
        with open(template_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Build the regex pattern to find the placeholder
        # The pattern looks for <Key| any text |Key>
        pattern = r'&lt;%s\|.*?\|%s&gt;' % (re.escape(key), re.escape(key))

        # Replace the placeholder with the provided content
        new_html_content = re.sub(pattern, content, html_content, flags=re.DOTALL)

        # Write the updated content back to the HTML file
        with open(template_path, 'w', encoding='utf-8') as file:
            file.write(new_html_content)

        print(f"Placeholder '{key}' replaced in {template_path}.")

    def get_tools(self):
        """
        Returns a list of tools configured using StructuredTool.

        Returns:
            list: A list containing the StructuredTool for replace_placeholder_in_html.
        """
        return [
            StructuredTool(
                name="replace_placeholder_in_html",
                func=self.replace_placeholder_in_html,
                description=(
                    "Use this tool to replace a placeholder in an HTML template with specified content. "
                    "Requires the template root directory, company name, placeholder key, and the new content."
                ),
                args_schema=ReplacePlaceholderModel
            )
        ]
