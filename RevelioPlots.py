import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import Polypeptide, is_aa, protein_letters_3to1
import io
import tempfile
import shutil
import atexit
from streamlit_molstar import st_molstar
from fpdf import FPDF

# --- Setup a temporary directory for uploaded files ---
TEMP_DIR = tempfile.mkdtemp()
atexit.register(lambda: shutil.rmtree(TEMP_DIR))

# --- Helper Functions ---

def get_color_for_plddt(plddt):
    """Returns a hex color code based on pLDDT score using AlphaFold's scheme."""
    if plddt > 90:
        return "#0053D6"
    elif plddt > 70:
        return "#65CBF3"
    elif plddt > 50:
        return "#FFDB13"
    else:
        return "#FF7D45"

def get_legend_html():
    """Generates HTML for a color legend."""
    legend_html = """
    <b>pLDDT Confidence Legend:</b>
    <div style="display: flex; flex-wrap: wrap; align-items: center; margin-bottom: 10px; font-size: 14px;">
        <div style="display: flex; align-items: center; margin-right: 15px; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #0053D6; margin-right: 5px; border: 1px solid #ddd;"></div><span>Very high (&gt; 90)</span>
        </div>
        <div style="display: flex; align-items: center; margin-right: 15px; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #65CBF3; margin-right: 5px; border: 1px solid #ddd;"></div><span>Confident (70-90)</span>
        </div>
        <div style="display: flex; align-items: center; margin-right: 15px; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #FFDB13; margin-right: 5px; border: 1px solid #ddd;"></div><span>Low (50-70)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #FF7D45; margin-right: 5px; border: 1px solid #ddd;"></div><span>Very low (&lt; 50)</span>
        </div>
    </div>
    """
    return legend_html

def generate_sequence_figure_html(df):
    """Generates an HTML string for the colored amino acid sequence with a position scale."""
    html = "<div style='font-family: monospace; word-wrap: break-word;'>"
    residues = list(df.to_dict('records'))
    for i in range(0, len(residues), 10):
        group = residues[i:i+10]
        start_res_num = group[0]['Residue']
        html += f"<div style='display: inline-block; vertical-align: top; margin-right: 10px; margin-bottom: 10px;'>"
        html += f"<div style='font-size: 10px; color: grey; padding-left: 2px;'>{start_res_num}</div>"
        html += "<div>"
        for res_data in group:
            aa = res_data['AA']
            plddt = res_data['pLDDT']
            color = get_color_for_plddt(plddt)
            text_color = "white" if plddt > 90 or plddt < 50 else "black"
            tooltip = f"Residue: {res_data['Residue']} | pLDDT: {plddt:.2f}"
            html += f"<span style='background-color: {color}; color: {text_color}; font-size: 16px; padding: 3px 1px; margin: 1px; border-radius: 3px;' title='{tooltip}'>{aa}</span>"
        html += "</div></div>"
    html += "</div>"
    return html

def add_dihedral_angles_to_df(structure, df):
    """Calculates Phi/Psi angles for the structure and merges them into the DataFrame."""
    phi_psi_list = []
    for model in structure:
        for chain in model:
            residues = [res for res in chain if is_aa(res, standard=True)]
            if len(residues) < 2:
                continue
            poly_chain = Polypeptide(residues)
            phi_psi_tuples = poly_chain.get_phi_psi_list()
            for i, res in enumerate(poly_chain):
                res_id = res.get_id()[1]
                phi, psi = phi_psi_tuples[i]
                phi_psi_list.append({
                    'Residue': res_id,
                    'Phi': np.degrees(phi) if phi is not None else None,
                    'Psi': np.degrees(psi) if psi is not None else None,
                })
    if not phi_psi_list:
        df[['Phi', 'Psi']] = None
        return df
    phi_psi_df = pd.DataFrame(phi_psi_list)
    return pd.merge(df, phi_psi_df, on='Residue', how='left')

def generate_ramachandran_plot(df, file_name):
    """Generates an interactive Ramachandran plot."""
    plot_df = df.dropna(subset=['Phi', 'Psi']).copy()
    if plot_df.empty:
        return None
    plot_df['Color'] = plot_df['pLDDT'].apply(get_color_for_plddt)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df['Phi'], y=plot_df['Psi'], mode='markers',
        marker=dict(color=plot_df['Color'], size=8, line=dict(width=1, color='DarkSlateGrey'), showscale=False),
        text=plot_df.apply(lambda r: f"Residue: {r['AA']}{r['Residue']}<br>pLDDT: {r['pLDDT']:.2f}<br>Phi: {r['Phi']:.2f}<br>Psi: {r['Psi']:.2f}", axis=1),
        hoverinfo='text', name='Residues'
    ))
    shapes = [
        dict(type="rect", xref="x", yref="y", x0=-180, y0=100, x1=-40, y1=180, fillcolor="rgba(173, 216, 230, 0.2)", layer="below", line_width=0),
        dict(type="rect", xref="x", yref="y", x0=-160, y0=-70, x1=-30, y1=50, fillcolor="rgba(144, 238, 144, 0.2)", layer="below", line_width=0),
        dict(type="rect", xref="x", yref="y", x0=30, y0=0, x1=100, y1=100, fillcolor="rgba(255, 182, 193, 0.2)", layer="below", line_width=0)
    ]
    fig.update_layout(
        title=f"Ramachandran Plot for {file_name}",
        xaxis_title="Phi (Φ) degrees", yaxis_title="Psi (Ψ) degrees",
        xaxis=dict(range=[-180, 180], tickvals=list(range(-180, 181, 60)), zeroline=True, zerolinecolor='black', zerolinewidth=1, showgrid=True, gridcolor='LightGray'),
        yaxis=dict(range=[-180, 180], tickvals=list(range(-180, 181, 60)), zeroline=True, zerolinecolor='black', zerolinewidth=1, showgrid=True, gridcolor='LightGray'),
        width=600, height=600, showlegend=False, shapes=shapes,
        template='plotly_white'
    )
    return fig

def calculate_protein_data(file_or_buffer, file_name_for_error_logging=""):
    """
    Parses a CIF file once to extract pLDDT, residue info, and dihedral angles.
    """
    try:
        if hasattr(file_or_buffer, 'seek'):
            file_or_buffer.seek(0)
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", file_or_buffer)
        data = []
        if '_ma_qa_metric_local' in structure.header and structure.header['_ma_qa_metric_local']:
            for row in structure.header['_ma_qa_metric_local']:
                data.append({
                    'Residue': int(row['label_seq_id']), 'pLDDT': float(row['metric_value']),
                    'AA': protein_letters_3to1.get(row.get('label_comp_id', '').upper(), 'X')
                })
        else:
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':
                            for atom in residue:
                                if atom.get_name() == 'CA':
                                    data.append({
                                        'Residue': residue.id[1], 'pLDDT': atom.get_bfactor(),
                                        'AA': protein_letters_3to1.get(residue.get_resname().upper(), 'X')
                                    })
                                    break
        if not data:
            st.warning(f"Could not extract pLDDT values for **{file_name_for_error_logging}**.")
            return None
        df = pd.DataFrame(data).sort_values(by='Residue').reset_index(drop=True)
        return add_dihedral_angles_to_df(structure, df)
    except Exception as e:
        st.error(f"An error occurred while parsing **{file_name_for_error_logging}**: {e}")
        return None

@st.cache_data
def create_pdf_report(file_name, plddt_stats, rama_fig, dist_fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Analysis Report: {file_name}", ln=True, align="C")

    pdf.set_font("Helvetica", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Mean pLDDT: {plddt_stats['mean']:.2f}", ln=True)
    pdf.cell(0, 10, f"Median pLDDT: {plddt_stats['median']:.2f}", ln=True)
    pdf.cell(0, 10, f"Std. Deviation: {plddt_stats['std']:.2f}", ln=True)

    # Save plots as temp images
    rama_path = os.path.join(TEMP_DIR, f"rama_{file_name}.png")
    dist_path = os.path.join(TEMP_DIR, f"dist_{file_name}.png")

    if rama_fig: rama_fig.write_image(rama_path, width=800, height=800, scale=2)
    if dist_fig: dist_fig.write_image(dist_path, width=800, height=600, scale=2)

    if os.path.exists(dist_path):
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "pLDDT Distribution", ln=True)
        pdf.image(dist_path, x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)

    if os.path.exists(rama_path):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Ramachandran Plot", ln=True)
        pdf.image(rama_path, x=10, y=pdf.get_y(), w=190)

    return bytes(pdf.output())

# --- UI Tabs ---

def single_structure_tab():
    """Handles the UI and logic for the 'Single Structure' tab."""
    st.header("Analyze a Single Protein Structure")
    source_option = st.radio("Choose structure source:", ("Upload a file", "Use an example"), horizontal=True)
    file_path_for_viewer, cif_file_source, file_name = None, None, None

    if source_option == "Upload a file":
        uploaded_file = st.file_uploader("Upload a CIF file (.cif or .mmcif)", type=['cif', 'mmcif'])
        if uploaded_file:
            file_name = uploaded_file.name
            temp_path = os.path.join(TEMP_DIR, file_name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            file_path_for_viewer = temp_path
            cif_file_source = temp_path
    else:
        try:
            example_files = [f for f in os.listdir("examples") if f.endswith(('.cif', '.mmcif'))]
            if not example_files:
                st.warning("No example files found in the 'examples' folder."); return
            selected_file = st.selectbox("Choose an example structure:", [""] + example_files)
            if selected_file:
                file_name = selected_file
                file_path_for_viewer = os.path.join("examples", selected_file)
                cif_file_source = file_path_for_viewer
        except FileNotFoundError:
            st.error("The 'examples' directory was not found."); return

    if cif_file_source and file_name:
        st.subheader("3D Interactive View")
        st_molstar(file_path_for_viewer, height='700px', key=f"molstar_single_{file_name}")

        st.info(f"Processing analysis for: **{file_name}**")
        protein_df = calculate_protein_data(cif_file_source, file_name)
        if protein_df is not None and not protein_df.empty:
            st.subheader("pLDDT Statistics")
            plddt_values = protein_df['pLDDT']
            col1, col2, col3 = st.columns(3); col1.metric("Mean pLDDT", f"{plddt_values.mean():.2f}"); col2.metric("Median pLDDT", f"{plddt_values.median():.2f}"); col3.metric("Std. Deviation", f"{plddt_values.std():.2f}")
            st.subheader("pLDDT Distribution Plot"); st.plotly_chart(px.box(protein_df, y="pLDDT", points="all", title=f"Distribution of pLDDT Values for {file_name}", labels={"pLDDT": "pLDDT Score"}, template="plotly_white", hover_data=['Residue', 'pLDDT']).update_layout(yaxis_range=[0,100], title_x=0.5), width='stretch')
            st.subheader("Confidence-Colored Sequence"); st.markdown(get_legend_html(), unsafe_allow_html=True); st.markdown(generate_sequence_figure_html(protein_df), unsafe_allow_html=True)
            st.subheader("Ramachandran Plot"); st.markdown(get_legend_html(), unsafe_allow_html=True)
            rama_fig = generate_ramachandran_plot(protein_df, file_name)
            if rama_fig:
                st.plotly_chart(rama_fig, width='stretch', config={'toImageButtonOptions': {'format': 'png', 'filename': f'Ramachandran_{file_name}', 'height': 1080, 'width': 1080, 'scale': 3}})
                png_bytes = rama_fig.to_image(format="png", width=1080, height=1080, scale=3)
                st.download_button(label="Download Ramachandran Plot (PNG)", data=png_bytes, file_name=f"Ramachandran_{file_name}.png", mime="image/png")
            else: st.warning("Could not generate Ramachandran plot.")

def multi_structure_tab():
    """Handles the UI and logic for the 'Multi-Structure' tab."""
    st.header("Compare Multiple Protein Structures")
    source_option = st.radio("Choose structure source:", ("Upload files", "Use examples"), horizontal=True, key="multi_source")
    
    files_to_process = []
    if source_option == "Upload files":
        uploaded_files = st.file_uploader("Upload CIF files (.cif or .mmcif)", type=['cif', 'mmcif'], accept_multiple_files=True)
        if uploaded_files:
            for up_file in uploaded_files:
                temp_path = os.path.join(TEMP_DIR, up_file.name)
                with open(temp_path, "wb") as f:
                    f.write(up_file.getvalue())
                files_to_process.append({'path': temp_path, 'name': up_file.name})
    else:
        try:
            example_files = [f for f in os.listdir("examples") if f.endswith(('.cif', '.mmcif'))];
            if not example_files: st.warning("No example files found in the 'examples' folder."); return
            selected_files = st.multiselect("Choose example structures to compare:", example_files)
            if selected_files:
                for sel_file in selected_files:
                    files_to_process.append({'path': os.path.join("examples", sel_file), 'name': sel_file})
        except FileNotFoundError: st.error("The 'examples' directory was not found."); return

    if files_to_process:
        all_data = []
        with st.spinner("Processing analysis..."):
            for file_info in files_to_process:
                protein_df = calculate_protein_data(file_info['path'], file_info['name'])
                if protein_df is not None:
                    all_data.append({'df': protein_df, 'name': file_info['name'], 'path': file_info['path']})
        
        if not all_data: st.error("Could not process analysis for any of the selected files."); return

        all_dfs = [d['df'].assign(File=d['name']) for d in all_data]
        stats_data = {d['name']: {'Mean': d['df']['pLDDT'].mean(),'Median': d['df']['pLDDT'].median(),'Std Dev': d['df']['pLDDT'].std()} for d in all_data}
        st.subheader("pLDDT Statistics Summary"); st.dataframe(pd.DataFrame.from_dict(stats_data, orient='index').style.format("{:.2f}"), width='stretch')
        
        st.subheader("Comparative pLDDT Distribution Plot"); combined_df = pd.concat(all_dfs, ignore_index=True); st.plotly_chart(px.box(combined_df, x="File", y="pLDDT", color="File", points="all", title="Distribution of pLDDT Values Across Multiple Structures", labels={"pLDDT": "pLDDT Score", "File": "Structure File"}, template="plotly_white", hover_data=['Residue', 'pLDDT']).update_layout(yaxis_range=[0,100], xaxis_title=None, title_x=0.5, showlegend=False), width='stretch')

        st.subheader("Per-Structure Analysis")
        st.markdown(get_legend_html(), unsafe_allow_html=True)
        for data_dict in all_data:
            file_name, protein_df, file_path = data_dict['name'], data_dict['df'], data_dict['path']
            with st.expander(f"Analysis for: {file_name}"):
                st.markdown("##### 3D Interactive View")
                # --- THIS IS THE CORRECTED BLOCK ---
                st_molstar(file_path, height='500px', key=f"molstar_multi_{file_name}")
                st.markdown("---")
                st.markdown("##### Confidence-Colored Sequence"); st.markdown(generate_sequence_figure_html(protein_df), unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("##### Ramachandran Plot")
                rama_fig = generate_ramachandran_plot(protein_df, file_name)
                if rama_fig:
                    st.plotly_chart(rama_fig, width='stretch', key=f"rama_plot_{file_name}", config={'toImageButtonOptions': {'format': 'png', 'filename': f'Ramachandran_{file_name}', 'height': 1080, 'width': 1080, 'scale': 3}})
                    png_bytes = rama_fig.to_image(format="png", width=1080, height=1080, scale=3)
                    st.download_button(label="Download Ramachandran Plot (PNG)", data=png_bytes, file_name=f"Ramachandran_{file_name}.png", mime="image/png", key=f"dl_rama_{file_name}")
                else: st.warning("Could not generate Ramachandran plot for this structure.", key=f"rama_warning_{file_name}")

def documentation_tab():
    """Handles the UI for the Documentation tab."""
    st.header("RevelioPlots Documentation")
    doc_path = "readme.md"
    try:
        with open(doc_path, "r") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Documentation file not found: `{doc_path}`")
        st.info("Please ensure the `readme.md` file is in the same directory as the application script.")

# --- Main App ---
st.set_page_config(page_title="RevelioPlots", page_icon="🪄", layout="wide")
st.sidebar.title("RevelioPlots")
if os.path.exists('RevelioPlots-logo.png'):
    st.sidebar.image('RevelioPlots-logo.png', use_column_width=True)
else:
    st.sidebar.markdown("### pLDDT Visualization Tool")
st.sidebar.markdown("---"); st.sidebar.info("This application analyzes pLDDT scores from protein structure files (.cif) to visualize model confidence.")
if not os.path.exists("examples"):
    os.makedirs("examples")
    st.info("An 'examples' folder has been created. Add some .cif files to it to use the example feature.")

tab1, tab2, tab3 = st.tabs(["Single Structure Analysis", "Multi-Structure Comparison", "Documentation"])

with tab1: single_structure_tab()
with tab2: multi_structure_tab()
with tab3: documentation_tab()