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
            basic_template = f'''<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>Bilancio di Sostenibilità 2024</title>
    <style>
    body {{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        color: #000000;
    }}
    header, footer {{
        background-color: #FFFFFF;
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
        color: #000000;
        text-decoration: none;
        font-weight: bold;
    }}
    main {{
        padding: 20px;
    }}
    h1, h2 {{
        color: #000000;
        margin-top: 40px;
    }}
    h3 {{
        color: #000000;
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
        border-left: 5px solid #000000;
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
    <img src="https://static.wixstatic.com/media/63b1fb_77b5c06b09e64c49803ab947527dbaac~mv2.png" 
         alt="Intestazione" style="width: 100%; height: auto;">
</header>

    <!-- Main Content -->
    <main>
        <!-- Introduzione -->
        <section id="introduzione">
            <h2>Introduzione</h2>
            <div class="content-placeholder">
                <Introduzione| Inserire un'introduzione generale al bilancio di sostenibilità, presentando l'azienda, la sua missione, i valori e l'importanza della sostenibilità per l'organizzazione. Fornire una panoramica degli obiettivi principali e delle aree chiave trattate nel rapporto. |Introduzione>
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
                <1.1 La nostra storia| Fornire una descrizione della storia dell'azienda, inclusi gli eventi chiave, le tappe fondamentali, la crescita nel tempo, e come l'azienda si è evoluta fino ad oggi, enfatizzando gli aspetti legati alla sostenibilità. |1.1 La nostra storia>
            </div>

            <!-- 1.2 Chi è {company_name} -->
            <h3>1.2 Chi è {company_name}</h3>
            <div class="content-placeholder">
                <1.2 Chi siamo| Fornire una descrizione dell'azienda, inclusa la missione, la visione, i valori fondamentali, i settori di attività e i principali risultati ottenuti. |1.2 Chi siamo>
            </div>

            <!-- 1.3 La nostra identità -->
            <h3>1.3 La nostra identità</h3>
            <div class="content-placeholder">
                <1.3 La nostra identità| Descrivere l'identità aziendale, come l'azienda si posiziona sul mercato, cosa la distingue dai concorrenti, e come la sostenibilità è integrata nel modello di business. |1.3 La nostra identità>
            </div>

            <!-- 1.4 La cultura aziendale -->
            <h3>1.4 La cultura aziendale</h3>
            <div class="content-placeholder">
                <1.4 La cultura aziendale| Presentare i valori e i principi che guidano la cultura aziendale, le pratiche interne che promuovono l'innovazione, la collaborazione, l'inclusione, e come questi elementi contribuiscono al raggiungimento degli obiettivi di sostenibilità. |1.4 La cultura aziendale>
            </div>
            
            <!-- 1.5 La cultura aziendale -->
            <h3>1.5 Il processo di doppia materialità</h3>
            <div class="content-placeholder">
                <1.5 Il processo di doppia materialità| La sezione 1.5 Il processo di doppia materialità illustra come l'azienda adotta gli European Sustainability Reporting Standards (ESRS) per analizzare i temi materiali, considerando sia l'impatto dell'azienda su ambiente e società, sia l'influenza dei fattori esterni sulle performance aziendali. Attraverso un'analisi di contesto e il coinvolgimento degli stakeholder, sono stati identificati i temi chiave per la sostenibilità, sintetizzati in una matrice di doppia materialità. La classificazione degli stakeholder e l'uso di questionari dedicati hanno guidato il processo, garantendo un approccio bilaterale e trasparente. Questa metodologia ha permesso di costruire un bilancio di sostenibilità orientato agli obiettivi aziendali e agli SDG delle Nazioni Unite. |1.5 Il processo di doppia materialità>
            </div>
        </section>

        <!-- Capitolo 2 -->
        <section id="capitolo2">
            <h2>Capitolo 2: Our People</h2>
            <div class="section-banner">
                <p>Immagine rappresentativa del Capitolo 2</p>
            </div>
            <div class="content-placeholder">
                <Descrizione Cap.2| Fornire una descrizione generale del Capitolo 2, focalizzato sulle persone dell'azienda, le iniziative intraprese per il loro benessere, sviluppo professionale e personale. Evidenziare l'importanza del capitale umano per l'organizzazione. |Descrizione Cap.2>
            </div>

            <!-- 2.1 Il Green Team -->
            <h3>2.1 Il Green Team</h3>
            <div class="content-placeholder">
                <2.1 Il Green Team| Descrivere il team dedicato alle iniziative di sostenibilità all'interno dell'azienda, includendo la composizione del team, i ruoli chiave, le responsabilità e gli obiettivi raggiunti durante il periodo di rendicontazione. |2.1 Il Green Team>
            </div>

            <!-- 2.2 Attrazione e conservazione dei talenti -->
            <h3>2.2 Attrazione e conservazione dei talenti</h3>
            <div class="content-placeholder">
                <2.2 Attrazione e conservazione dei talenti| Dettagliare le strategie e le politiche adottate per attrarre nuovi talenti e mantenere quelli esistenti, inclusi programmi di formazione, piani di carriera, benefit offerti, ambiente di lavoro flessibile e risultati ottenuti (es. tassi di retention, nuove assunzioni). |2.2 Attrazione e conservazione dei talenti>
            </div>

            <!-- 2.3 Crescita e sviluppo del personale -->
            <h3>2.3 Crescita e sviluppo del personale</h3>
            <div class="content-placeholder">
                <2.3 Crescita e sviluppo del personale| Illustrare le opportunità di formazione e sviluppo professionale offerte ai dipendenti, come corsi di aggiornamento, workshop, programmi di mentoring e come queste iniziative contribuiscono alla crescita delle competenze e al raggiungimento degli obiettivi aziendali. |2.3 Crescita e sviluppo del personale>
            </div>

            <!-- 2.4 Salute mentale e fisica delle persone -->
            <h3>2.4 Salute mentale e fisica delle persone</h3>
            <div class="content-placeholder">
                <2.4 Salute mentale e fisica delle persone| Descrivere le misure adottate per garantire la salute e il benessere fisico e mentale dei dipendenti, come programmi di supporto psicologico, promozione dell'equilibrio vita-lavoro, attività di wellness, e presentare eventuali risultati o feedback raccolti. |2.4 Salute mentale e fisica delle persone>
            </div>

            <!-- 2.5 Valutazione delle performance -->
            <h3>2.5 Valutazione delle performance</h3>
            <div class="content-placeholder">
                <2.5 Valutazione delle performance| Spiegare il processo di valutazione delle performance utilizzato dall'azienda, inclusi i criteri di valutazione, la frequenza delle valutazioni, come vengono utilizzati i feedback per migliorare le performance individuali e organizzative, e l'impatto sullo sviluppo dei dipendenti. |2.5 Valutazione delle performance>
            </div>

            <!-- 2.6 Condivisione, retreat e team building -->
            <h3>2.6 Condivisione, retreat e team building</h3>
            <div class="content-placeholder">
                <2.6 Condivisione, retreat e team building| Presentare le attività organizzate per promuovere la coesione del team, come ritiri aziendali, eventi di team building, workshop collaborativi, e spiegare come queste iniziative contribuiscono a migliorare la comunicazione e le relazioni tra i dipendenti. |2.6 Condivisione, retreat e team building>
            </div>
        </section>

        <!-- Capitolo 3 -->
        <section id="capitolo3">
            <h2>Capitolo 3: Planet</h2>
            <div class="section-banner">
                <p>Immagine rappresentativa del Capitolo 3</p>
            </div>
            <div class="content-placeholder">
                <Descrizione Cap.3| Fornire una descrizione generale del Capitolo 3, focalizzato sull'impegno dell'azienda verso la sostenibilità ambientale, le iniziative intraprese per ridurre l'impatto ambientale e contribuire alla lotta contro il cambiamento climatico. |Descrizione Cap.3>
            </div>

            <!-- 3.1 Attenzione al cambiamento climatico -->
            <h3>3.1 Attenzione al cambiamento climatico</h3>
            <div class="content-placeholder">
                <3.1 Attenzione al cambiamento climatico| Descrivere le strategie adottate dall'azienda per affrontare il cambiamento climatico, inclusi gli obiettivi di riduzione delle emissioni di CO2, le azioni intraprese per migliorare l'efficienza energetica, l'utilizzo di energie rinnovabili e i progressi compiuti verso questi obiettivi. |3.1 Attenzione al cambiamento climatico>
            </div>

            <!-- 3.2 Impatti ambientali legati a prodotti e servizi offerti -->
            <h3>3.2 Impatti ambientali legati a prodotti e servizi offerti</h3>
            <div class="content-placeholder">
                <3.2 Impatti ambientali| Analizzare gli impatti ambientali associati ai prodotti e servizi dell'azienda, come l'impronta di carbonio dei prodotti, l'uso di risorse naturali, e le iniziative per migliorare la sostenibilità lungo il ciclo di vita dei prodotti. |3.2 Impatti ambientali>
            </div>

            <!-- 3.3 Supporto al Voluntary Carbon Market -->
            <h3>3.3 Supporto al Voluntary Carbon Market</h3>
            <div class="content-placeholder">
                <3.3 Supporto al Voluntary Carbon Market| Spiegare come l'azienda partecipa e supporta il Mercato Volontario del Carbonio, includendo dettagli sui progetti di compensazione delle emissioni sostenuti, i criteri di selezione dei progetti, e l'impatto ambientale e sociale generato da tali iniziative. |3.3 Supporto al Voluntary Carbon Market>
            </div>
        </section>

        <!-- Capitolo 4 -->
        <section id="capitolo4">
            <h2>Capitolo 4:Growth</h2>
            <div class="section-banner">
                <p>Immagine rappresentativa del Capitolo 4</p>
            </div>
            <div class="content-placeholder">
                <Descrizione Cap.4| Fornire una descrizione generale del Capitolo 4, focalizzato sulla crescita sostenibile dell'azienda, le pratiche di governance, l'etica nei rapporti commerciali e l'innovazione tecnologica a supporto dell'azione climatica. |Descrizione Cap.4>
            </div>

            <!-- 4.1 La governance in {company_name} -->
            <h3>4.1 La governance in {company_name}</h3>
            <div class="content-placeholder">
                <4.1 La governance in Up2You| Descrivere la struttura di governance dell'azienda, inclusa la composizione del Consiglio di Amministrazione, i ruoli chiave, le politiche di governance adottate per garantire trasparenza e responsabilità, e come queste pratiche supportano gli obiettivi di sostenibilità. |4.1 La governance in Up2You>
            </div>

            <!-- 4.2 Rapporti commerciali etici -->
            <h3>4.2 Rapporti commerciali etici</h3>
            <div class="content-placeholder">
                <4.2 Rapporti commerciali etici| Presentare le politiche e le pratiche etiche adottate nei rapporti commerciali, come il rispetto delle normative, la prevenzione della corruzione, l'integrità nelle vendite, e come l'azienda garantisce trasparenza e correttezza nelle transazioni commerciali. |4.2 Rapporti commerciali etici>
            </div>

            <!-- 4.3 Tecnologia al servizio dell’azione climatica -->
            <h3>4.3 Tecnologia al servizio dell’azione climatica</h3>
            <div class="content-placeholder">
                <4.3 Tecnologia al servizio dell’azione climatica| Illustrare come l'azienda utilizza la tecnologia per promuovere l'azione climatica, descrivendo le soluzioni digitali sviluppate, come la piattaforma CliMax e PlaNet, gli aggiornamenti implementati nel periodo di rendicontazione, e l'impatto positivo generato. |4.3 Tecnologia al servizio dell’azione climatica>
            </div>

        </section>

        <!-- Conclusione -->
        <section id="conclusione">
            <h2>Conclusione</h2>
            <div class="content-placeholder">
                <Conclusione| Fornire una conclusione che riassuma i punti chiave del bilancio di sostenibilità, esprima l'impegno continuo dell'azienda verso la sostenibilità, delinei le prospettive future e ringrazi i lettori per l'attenzione. |Conclusione>
            </div>
        </section>

        <!-- Nota metodologica -->
        <section id="nota-metodologica">
            <h2>Nota metodologica</h2>
            <div class="content-placeholder">
                <Nota metodologica| Descrivere la metodologia adottata per la redazione del bilancio di sostenibilità, specificando gli standard e le linee guida seguite (ad esempio, ESRS), il perimetro di rendicontazione, le fonti dei dati utilizzati, eventuali cambiamenti rispetto all'anno precedente e informazioni sulla verifica dei dati. |Nota metodologica>
            </div>
        </section>

        <!-- Indice dei contenuti ESRS -->
        <section id="indice-esrs">
            <h2>Indice dei contenuti ESRS</h2>
            <div class="content-placeholder">
                <Indice ESRS| Fornire una tabella o un elenco che mappi i contenuti del bilancio di sostenibilità con i requisiti specifici degli standard ESRS, facilitando la consultazione e la verifica della conformità da parte dei lettori interessati. |Indice ESRS>
            </div>
        </section>

        <!-- Informazioni di contatto -->
        <section id="contatti">
            <h2>Informazioni di contatto</h2>
            <div class="content-placeholder">
                <Informazioni di contatto| Inserire le informazioni di contatto dell'azienda per domande, commenti o richieste di ulteriori informazioni riguardanti il bilancio di sostenibilità, includendo indirizzi email, sito web e indirizzi fisici delle sedi principali. |Informazioni di contatto>
            </div>
        </section>
    </main>

    <!-- Footer -->
        <footer>
            <img src="https://static.wixstatic.com/media/63b1fb_b560355ef4a848f7a7ca24148db4cc62~mv2.png"
                 alt="Intestazione" style="width: 100%; height: auto;">
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
        pattern = r'<%s\|.*?\|%s>' % (re.escape(key), re.escape(key))

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
