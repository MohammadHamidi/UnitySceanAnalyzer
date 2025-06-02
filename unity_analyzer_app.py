import streamlit as st
import os
import json
import yaml
import re
import networkx as nx
from pathlib import Path
import tempfile
import zipfile
from typing import Dict, Set, List, Tuple
import pandas as pd
import base64
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Unity Dependency Analyzer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

class UnityDependencyAnalyzer:
    def __init__(self):
        self.project_root = None
        self.guid_index = {}
        self.scene_dependencies = {}
        self.script_dependencies = {}
        self.dependency_graph = None
        self.node_types = {}
    
    def build_guid_index(self, project_root: str) -> Dict[str, str]:
        """Build GUID to asset path mapping from .meta files"""
        guid_map = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            meta_files = list(Path(project_root).rglob("*.meta"))
            total_files = len(meta_files)
            
            st.info(f"Found {total_files} .meta files to process")
            
            if total_files == 0:
                st.warning("No .meta files found. Make sure you're pointing to the correct Unity project root directory.")
                return guid_map
            
            for idx, meta_path in enumerate(meta_files):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Try YAML parsing first
                        try:
                            data = yaml.safe_load(content)
                            if isinstance(data, dict) and 'guid' in data:
                                guid = data['guid']
                            else:
                                # Fallback to regex
                                match = re.search(r'guid:\s*([0-9a-f]{32})', content)
                                if match:
                                    guid = match.group(1)
                                else:
                                    continue
                        except yaml.YAMLError:
                            # Fallback to regex
                            match = re.search(r'guid:\s*([0-9a-f]{32})', content)
                            if match:
                                guid = match.group(1)
                            else:
                                continue
                        
                        # Get asset path
                        asset_filename = meta_path.stem
                        rel_dir = os.path.relpath(meta_path.parent, project_root)
                        asset_relpath = os.path.join(rel_dir, asset_filename).replace("\\", "/")
                        guid_map[guid] = asset_relpath
                        
                except Exception as e:
                    st.warning(f"Error processing {meta_path}: {str(e)}")
                
                # Update progress
                progress = (idx + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"Processing meta files: {idx + 1}/{total_files}")
                
        except Exception as e:
            st.error(f"Error scanning for .meta files: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()
        
        st.success(f"Built GUID index with {len(guid_map)} entries")
        return guid_map
    
    def extract_guid_references(self, filepath: str) -> Set[str]:
        """Extract GUID references from Unity YAML files"""
        guids = set()
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                # Find all 32-character hex GUIDs
                matches = re.findall(r"guid:\s*([0-9a-f]{32})", text)
                guids.update(matches)
        except Exception as e:
            # Log error but don't stop processing
            st.warning(f"Error reading {filepath}: {str(e)}")
        
        return guids
    
    def scan_scenes_and_prefabs(self, project_root: str) -> Dict[str, Set[str]]:
        """Scan scenes and prefabs for dependencies"""
        dependencies = {}
        
        try:
            # Find all scene and prefab files with detailed logging
            st.info("Searching for Unity scene and prefab files...")
            
            unity_files = list(Path(project_root).rglob("*.unity"))
            prefab_files = list(Path(project_root).rglob("*.prefab"))
            
            st.info(f"Found {len(unity_files)} scene files (.unity)")
            st.info(f"Found {len(prefab_files)} prefab files (.prefab)")
            
            # Debug: Show some example files found
            if unity_files:
                st.success(f"Example scene files found:")
                for scene in unity_files[:3]:  # Show first 3
                    rel_path = os.path.relpath(scene, project_root)
                    st.text(f"  - {rel_path}")
                if len(unity_files) > 3:
                    st.text(f"  ... and {len(unity_files) - 3} more")
            else:
                st.warning("❌ No scene files (.unity) found!")
                st.info("Make sure you're pointing to the Unity project root directory that contains the Assets folder.")
            
            if prefab_files:
                st.success(f"Example prefab files found:")
                for prefab in prefab_files[:3]:  # Show first 3
                    rel_path = os.path.relpath(prefab, project_root)
                    st.text(f"  - {rel_path}")
                if len(prefab_files) > 3:
                    st.text(f"  ... and {len(prefab_files) - 3} more")
            
            all_files = unity_files + prefab_files
            
            if not all_files:
                st.error("No Unity scene or prefab files found in the specified directory!")
                st.info("""
                **Troubleshooting:**
                1. Make sure you're pointing to the Unity project root directory
                2. The directory should contain an 'Assets' folder
                3. Scene files should have .unity extension
                4. Check if the directory path is correct
                """)
                return dependencies
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file_path in enumerate(all_files):
                try:
                    rel_path = os.path.relpath(file_path, project_root).replace("\\", "/")
                    guids = self.extract_guid_references(str(file_path))
                    
                    # Convert GUIDs to asset paths
                    asset_paths = set()
                    for guid in guids:
                        if guid in self.guid_index:
                            asset_paths.add(self.guid_index[guid])
                        else:
                            asset_paths.add(f"<UNKNOWN GUID {guid}>")
                    
                    dependencies[rel_path] = asset_paths
                    
                    # Update progress
                    progress = (idx + 1) / len(all_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Scanning Unity files: {idx + 1}/{len(all_files)} - {os.path.basename(file_path)}")
                    
                except Exception as e:
                    st.warning(f"Error processing {file_path}: {str(e)}")
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Error during scene/prefab scanning: {str(e)}")
            return {}
        
        st.success(f"Successfully scanned {len(dependencies)} Unity files")
        return dependencies
    
    def extract_script_dependencies(self, cs_filepath: str) -> Set[str]:
        """Extract dependencies from C# scripts using regex patterns"""
        deps = set()
        try:
            with open(cs_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
                # Pattern matching for common Unity script patterns
                patterns = [
                    r"GetComponent<\s*([\w]+)\s*>",  # GetComponent<ScriptName>
                    r"FindObjectOfType<\s*([\w]+)\s*>",  # FindObjectOfType<ScriptName>
                    r"new\s+([\w]+)\s*\(",  # new ScriptName(
                    r"class\s+\w+\s*:\s*([\w]+)",  # Inheritance
                    r"using\s+([\w\.]+)\s*;",  # Using statements
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if pattern == r"using\s+([\w\.]+)\s*;":
                            # For using statements, take the last part
                            last_token = match.split('.')[-1]
                            deps.add(last_token + ".cs")
                        else:
                            deps.add(match + ".cs")
                            
        except Exception as e:
            st.warning(f"Error processing script {cs_filepath}: {str(e)}")
        
        return deps
    
    def build_dependency_graph(self, project_root: str) -> Tuple[nx.DiGraph, Dict[str, str]]:
        """Build the complete dependency graph"""
        G = nx.DiGraph()
        node_types = {}
        
        st.info("Building dependency graph...")
        
        # Add scene/prefab dependencies
        for source, deps in self.scene_dependencies.items():
            G.add_node(source)
            if source.endswith(".unity"):
                node_types[source] = 'scene'
            elif source.endswith(".prefab"):
                node_types[source] = 'prefab'
            else:
                node_types[source] = 'asset'
            
            for dep in deps:
                if not dep.startswith("<UNKNOWN"):
                    G.add_node(dep)
                    if dep.endswith(".prefab"):
                        node_types[dep] = 'prefab'
                    elif dep.endswith(".cs"):
                        node_types[dep] = 'script'
                    else:
                        node_types[dep] = 'asset'
                    
                    G.add_edge(source, dep, type="uses")
        
        # Add script dependencies
        cs_files = list(Path(project_root).rglob("*.cs"))
        all_cs_names = {f.name: str(f) for f in cs_files}
        
        st.info(f"Processing {len(cs_files)} C# script files...")
        
        for cs_file in cs_files:
            rel_path = os.path.relpath(cs_file, project_root).replace("\\", "/")
            deps = self.extract_script_dependencies(str(cs_file))
            
            # Filter dependencies to only include existing scripts
            valid_deps = set()
            for dep in deps:
                if dep in all_cs_names:
                    dep_rel_path = os.path.relpath(all_cs_names[dep], project_root).replace("\\", "/")
                    valid_deps.add(dep_rel_path)
            
            if valid_deps:
                if not G.has_node(rel_path):
                    G.add_node(rel_path)
                    node_types[rel_path] = 'script'
                
                for dep_path in valid_deps:
                    if not G.has_node(dep_path):
                        G.add_node(dep_path)
                        node_types[dep_path] = 'script'
                    
                    G.add_edge(rel_path, dep_path, type="script_uses")
        
        st.success(f"Dependency graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G, node_types
    
    def generate_mermaid_diagram(self, subgraph: nx.DiGraph, title: str = "") -> str:
        """Generate Mermaid diagram from NetworkX graph"""
        lines = ["graph TD"]
        if title:
            lines.insert(0, f"---\ntitle: {title}\n---")
        
        # Create sanitized node IDs and labels
        node_map = {}
        for node in subgraph.nodes():
            sanitized = re.sub(r'[^\w]', '_', node)
            label = os.path.basename(node)
            node_map[node] = (sanitized, label)
        
        # Add edges
        for source, target in subgraph.edges():
            source_id, source_label = node_map[source]
            target_id, target_label = node_map[target]
            
            # Add styling based on node type
            source_type = self.node_types.get(source, 'unknown')
            target_type = self.node_types.get(target, 'unknown')
            
            # Create styled nodes
            if source_type == 'scene':
                lines.append(f'    {source_id}["{source_label}"]:::scene')
            elif source_type == 'prefab':
                lines.append(f'    {source_id}["{source_label}"]:::prefab')
            elif source_type == 'script':
                lines.append(f'    {source_id}["{source_label}"]:::script')
            else:
                lines.append(f'    {source_id}["{source_label}"]:::asset')
            
            if target_type == 'scene':
                lines.append(f'    {target_id}["{target_label}"]:::scene')
            elif target_type == 'prefab':
                lines.append(f'    {target_id}["{target_label}"]:::prefab')
            elif target_type == 'script':
                lines.append(f'    {target_id}["{target_label}"]:::script')
            else:
                lines.append(f'    {target_id}["{target_label}"]:::asset')
            
            # Add edge
            lines.append(f'    {source_id} --> {target_id}')
        
        # Add styling
        lines.extend([
            "",
            "    classDef scene fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000",
            "    classDef prefab fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000",
            "    classDef script fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#000",
            "    classDef asset fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000"
        ])
        
        return "\n".join(lines)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = UnityDependencyAnalyzer()

# Main UI
st.title("🎮 Unity Project Dependency Analyzer")
st.markdown("Analyze Unity project dependencies, generate dependency graphs, and visualize relationships between scenes, prefabs, and scripts.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Project directory input
    project_path = st.text_input(
        "Unity Project Directory",
        placeholder="C:/Projects/MyUnityProject",
        help="Enter the full path to your Unity project root directory (should contain Assets folder)"
    )
    
    # Add directory validation helper
    if project_path:
        if os.path.exists(project_path):
            if os.path.exists(os.path.join(project_path, "Assets")):
                st.success("✅ Valid Unity project directory detected")
            else:
                st.warning("⚠️ Directory exists but no 'Assets' folder found. Make sure this is the Unity project root.")
        else:
            st.error("❌ Directory does not exist")
    
    if st.button("🔍 Scan Project", type="primary"):
        if project_path and os.path.exists(project_path):
            with st.spinner("Scanning Unity project..."):
                analyzer = st.session_state.analyzer
                analyzer.project_root = project_path
                
                # Step 1: Build GUID index
                st.info("Step 1: Building GUID index from .meta files...")
                analyzer.guid_index = analyzer.build_guid_index(project_path)
                
                # Step 2: Scan scenes and prefabs
                st.info("Step 2: Scanning scenes and prefabs...")
                analyzer.scene_dependencies = analyzer.scan_scenes_and_prefabs(project_path)
                
                # Step 3: Build dependency graph
                st.info("Step 3: Building dependency graph...")
                analyzer.dependency_graph, analyzer.node_types = analyzer.build_dependency_graph(project_path)
                
                # Final results
                scenes_found = len([k for k in analyzer.scene_dependencies.keys() if k.endswith('.unity')])
                prefabs_found = len([k for k in analyzer.scene_dependencies.keys() if k.endswith('.prefab')])
                
                if scenes_found > 0 or prefabs_found > 0:
                    st.success(f"✅ Project scanned successfully!")
                    st.markdown(f"**Analysis Results:**")
                    st.markdown(f"- 📄 {len(analyzer.guid_index)} assets with GUIDs")
                    st.markdown(f"- 🎬 {scenes_found} scenes")
                    st.markdown(f"- 🧩 {prefabs_found} prefabs")
                    st.markdown(f"- 🔗 {analyzer.dependency_graph.number_of_nodes()} total nodes in dependency graph")
                    st.markdown(f"- ➡️ {analyzer.dependency_graph.number_of_edges()} dependency relationships")
                else:
                    st.error("❌ No scenes or prefabs found!")
                    st.info("""
                    **Troubleshooting Tips:**
                    1. Make sure the path points to your Unity project root directory
                    2. The directory should contain an 'Assets' folder with your scenes
                    3. Scene files should have the .unity extension
                    4. Try using forward slashes (/) in the path instead of backslashes
                    """)
        else:
            st.error("Please enter a valid Unity project directory path.")

# Main content area
if st.session_state.analyzer.dependency_graph is not None:
    analyzer = st.session_state.analyzer
    
    # Get available scenes
    scenes = [k for k in analyzer.scene_dependencies.keys() if k.endswith('.unity')]
    
    if scenes:
        st.header("📋 Scene Analysis")
        
        # Show found scenes for debugging
        with st.expander("🔍 Debug: Found Scenes"):
            st.write("Scene files detected:")
            for scene in scenes:
                st.text(f"  - {scene}")
        
        # Scene selection
        selected_scenes = st.multiselect(
            "Select scenes to analyze:",
            scenes,
            default=[],
            help="Choose one or more scenes to generate dependency analysis"
        )
        
        if selected_scenes:
            # Analysis options
            col1, col2 = st.columns(2)
            with col1:
                include_scripts = st.checkbox("Include script dependencies", value=True)
                include_assets = st.checkbox("Include asset dependencies", value=True)
            with col2:
                max_depth = st.slider("Maximum dependency depth", 1, 5, 2)
                show_statistics = st.checkbox("Show dependency statistics", value=True)
            
            if st.button("🔬 Analyze Selected Scenes", type="primary"):
                for scene in selected_scenes:
                    st.subheader(f"📄 Analysis for: {os.path.basename(scene)}")
                    
                    # Get subgraph for this scene
                    if scene in analyzer.dependency_graph:
                        # Get all descendants up to max_depth
                        descendants = set()
                        current_level = {scene}
                        
                        for depth in range(max_depth):
                            next_level = set()
                            for node in current_level:
                                successors = set(analyzer.dependency_graph.successors(node))
                                next_level.update(successors)
                                descendants.update(successors)
                            current_level = next_level
                            if not current_level:
                                break
                        
                        # Filter by type if requested
                        nodes_to_include = {scene} | descendants
                        if not include_scripts:
                            nodes_to_include = {n for n in nodes_to_include 
                                              if analyzer.node_types.get(n) != 'script'}
                        if not include_assets:
                            nodes_to_include = {n for n in nodes_to_include 
                                              if analyzer.node_types.get(n) != 'asset'}
                        
                        scene_subgraph = analyzer.dependency_graph.subgraph(nodes_to_include).copy()
                        
                        # Statistics
                        if show_statistics:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                scripts = len([n for n in scene_subgraph.nodes() 
                                             if analyzer.node_types.get(n) == 'script'])
                                st.metric("Scripts", scripts)
                            with col2:
                                prefabs = len([n for n in scene_subgraph.nodes() 
                                             if analyzer.node_types.get(n) == 'prefab'])
                                st.metric("Prefabs", prefabs)
                            with col3:
                                assets = len([n for n in scene_subgraph.nodes() 
                                            if analyzer.node_types.get(n) == 'asset'])
                                st.metric("Assets", assets)
                            with col4:
                                st.metric("Dependencies", scene_subgraph.number_of_edges())
                        
                        # Generate Mermaid diagram
                        mermaid_diagram = analyzer.generate_mermaid_diagram(
                            scene_subgraph, 
                            f"Dependencies for {os.path.basename(scene)}"
                        )
                        
                        # Display diagram
                        st.markdown("### 📊 Dependency Diagram")
                        st.markdown("```mermaid")
                        st.markdown(mermaid_diagram)
                        st.markdown("```")
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Download Mermaid Diagram",
                                mermaid_diagram,
                                file_name=f"{os.path.splitext(os.path.basename(scene))[0]}_dependencies.mmd",
                                mime="text/plain"
                            )
                        
                        with col2:
                            # Create dependency table
                            dependencies_data = []
                            for source, target in scene_subgraph.edges():
                                dependencies_data.append({
                                    "Source": os.path.basename(source),
                                    "Target": os.path.basename(target),
                                    "Source Type": analyzer.node_types.get(source, 'unknown'),
                                    "Target Type": analyzer.node_types.get(target, 'unknown'),
                                    "Source Path": source,
                                    "Target Path": target
                                })
                            
                            if dependencies_data:
                                df = pd.DataFrame(dependencies_data)
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Dependency Table (CSV)",
                                    csv,
                                    file_name=f"{os.path.splitext(os.path.basename(scene))[0]}_dependencies.csv",
                                    mime="text/csv"
                                )
                        
                        # Detailed dependency table
                        if dependencies_data:
                            st.markdown("### 📊 Detailed Dependencies")
                            st.dataframe(
                                df[["Source", "Target", "Source Type", "Target Type"]], 
                                use_container_width=True
                            )
                        
                        st.markdown("---")
                    else:
                        st.warning(f"Scene {scene} not found in dependency graph.")
    else:
        st.info("No scenes found in the project. Please scan a valid Unity project.")
        
        # Enhanced troubleshooting
        st.markdown("### 🔧 Troubleshooting")
        st.error("""
        **Scene files not found!** This usually happens when:
        
        1. **Wrong Directory**: Make sure you're pointing to the Unity project root (the folder that contains the 'Assets' directory)
        2. **Path Format**: Try using forward slashes (/) instead of backslashes (\\)
        3. **File Extensions**: Scene files must have the .unity extension
        4. **Permissions**: Make sure the directory is readable
        
        **Expected Structure:**
        ```
        YourUnityProject/
        ├── Assets/
        │   ├── Scenes/
        │   │   ├── MainMenu.unity
        │   │   └── GameLevel.unity
        │   └── Scripts/
        └── ProjectSettings/
        ```
        """)
        
else:
    st.info("👆 Please select a Unity project directory and scan it to begin analysis.")
    
    # Example usage section
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Steps to Analyze Your Unity Project:
        
        1. **Set Project Path**: Enter the full path to your Unity project root directory in the sidebar
        2. **Verify Path**: Make sure the path points to the folder containing the 'Assets' directory
        3. **Scan Project**: Click the "Scan Project" button to analyze all files
        4. **Check Results**: Verify that scenes and prefabs were found
        5. **Select Scenes**: Choose which scenes you want to analyze from the dropdown
        6. **Configure Options**: Adjust analysis settings like dependency depth and included types
        7. **Generate Analysis**: Click "Analyze Selected Scenes" to generate dependency graphs
        8. **Download Results**: Save Mermaid diagrams and CSV dependency tables
        
        ### What This Tool Analyzes:
        
        - **Scene Dependencies**: Which prefabs, scripts, and assets each scene uses
        - **Script Dependencies**: How C# scripts reference each other
        - **Prefab Relationships**: Dependencies between prefabs and their components
        - **Asset Usage**: Which textures, materials, and other assets are referenced
        
        ### Supported File Types:
        
        - `.unity` scene files
        - `.prefab` prefab files  
        - `.cs` C# script files
        - `.meta` metadata files for GUID resolution
        
        ### Common Issues:
        
        - **No scenes found**: Make sure you're pointing to the Unity project root directory
        - **Path errors**: Use forward slashes (/) or double backslashes (\\\\)
        - **Missing Assets folder**: The directory should contain an 'Assets' folder
        """)

# Footer
st.markdown("---")
st.markdown("**Unity Dependency Analyzer** - Built for Unity developers to understand project structure and dependencies.")