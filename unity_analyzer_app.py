import streamlit as st
import streamlit_mermaid as stmd
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
    
    def extract_project_from_zip(self, uploaded_file) -> str:
        """Extract uploaded ZIP file to temporary directory"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(temp_dir)
                
                # Find the actual project root (look for Assets folder)
                for root, dirs, files in os.walk(temp_dir):
                    if 'Assets' in dirs:
                        return root
                
                # If no Assets folder found, return temp_dir
                return temp_dir
                
        except Exception as e:
            st.error(f"Error extracting ZIP file: {str(e)}")
            return None
    
    def build_guid_index(self, project_root: str) -> Dict[str, str]:
        """Build GUID to asset path mapping from .meta files"""
        guid_map = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            meta_files = list(Path(project_root).rglob("*.meta"))
            total_files = len(meta_files)
            
            if total_files == 0:
                st.warning("No .meta files found. Make sure you uploaded a complete Unity project.")
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
            st.error(f"Error scanning meta files: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()
        
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
            # Silently skip problematic files
            pass
        
        return guids
    
    def scan_scenes_and_prefabs(self, project_root: str) -> Dict[str, Set[str]]:
        """Scan scenes and prefabs for dependencies"""
        dependencies = {}
        
        # Find all scene and prefab files
        unity_files = list(Path(project_root).rglob("*.unity"))
        prefab_files = list(Path(project_root).rglob("*.prefab"))
        all_files = unity_files + prefab_files
        
        if not all_files:
            st.warning("No Unity scene or prefab files found.")
            return dependencies
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for idx, file_path in enumerate(all_files):
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
                status_text.text(f"Scanning Unity files: {idx + 1}/{len(all_files)}")
        
        except Exception as e:
            st.error(f"Error scanning Unity files: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()
        
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
            # Silently skip problematic files
            pass
        
        return deps
    
    def build_dependency_graph(self, project_root: str) -> Tuple[nx.DiGraph, Dict[str, str]]:
        """Build the complete dependency graph"""
        G = nx.DiGraph()
        node_types = {}
        
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
    
    # File upload instead of directory input
    st.markdown("### 📁 Upload Unity Project")
    uploaded_file = st.file_uploader(
        "Upload Unity Project (ZIP)",
        type=['zip'],
        help="Upload a ZIP file containing your Unity project. Make sure to include the Assets folder and .meta files."
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.info("💡 **Tip**: Make sure your ZIP includes the Assets folder and all .meta files for accurate analysis.")
        
        if st.button("🔍 Analyze Project", type="primary"):
            with st.spinner("Extracting and analyzing Unity project..."):
                analyzer = st.session_state.analyzer
                
                # Extract ZIP file
                st.info("Extracting project files...")
                project_root = analyzer.extract_project_from_zip(uploaded_file)
                
                if project_root:
                    analyzer.project_root = project_root
                    
                    # Step 1: Build GUID index
                    st.info("Building GUID index from .meta files...")
                    analyzer.guid_index = analyzer.build_guid_index(project_root)
                    
                    # Step 2: Scan scenes and prefabs
                    st.info("Scanning scenes and prefabs...")
                    analyzer.scene_dependencies = analyzer.scan_scenes_and_prefabs(project_root)
                    
                    # Step 3: Build dependency graph
                    st.info("Building dependency graph...")
                    analyzer.dependency_graph, analyzer.node_types = analyzer.build_dependency_graph(project_root)
                    
                    st.success(f"✅ Project analyzed successfully!")
                    st.markdown(f"**Found:**")
                    st.markdown(f"- {len(analyzer.guid_index)} assets with GUIDs")
                    st.markdown(f"- {len([k for k in analyzer.scene_dependencies.keys() if k.endswith('.unity')])} scenes")
                    st.markdown(f"- {len([k for k in analyzer.scene_dependencies.keys() if k.endswith('.prefab')])} prefabs")
                    st.markdown(f"- {analyzer.dependency_graph.number_of_nodes()} total nodes in dependency graph")
                    st.markdown(f"- {analyzer.dependency_graph.number_of_edges()} dependency relationships")
                else:
                    st.error("Failed to extract project. Please ensure you uploaded a valid Unity project ZIP file.")
    
    else:
        st.info("👆 Please upload a Unity project ZIP file to begin analysis.")

# Instructions for creating ZIP file
with st.sidebar:
    with st.expander("📦 How to Create Project ZIP"):
        st.markdown("""
        ### Creating a Unity Project ZIP:
        
        1. **Include Essential Folders:**
           - `Assets/` (required)
           - `ProjectSettings/` (recommended)
           - `Packages/` (if using custom packages)
        
        2. **Include .meta Files:**
           - Make sure to include all `.meta` files
           - These contain the GUIDs needed for dependency resolution
        
        3. **What to Exclude:**
           - `Library/` folder (large, auto-generated)
           - `Temp/` folder
           - `obj/` and `bin/` folders
           - `.git/` if using version control
        
        4. **ZIP the Project:**
           - Select your project folder contents
           - Create a ZIP archive
           - Upload here for analysis
        
        **💡 Pro Tip:** Use Unity's built-in Package Manager export or create the ZIP from your project's root directory.
        """)

# Main content area
if st.session_state.analyzer.dependency_graph is not None:
    analyzer = st.session_state.analyzer
    
    # Get available scenes
    scenes = [k for k in analyzer.scene_dependencies.keys() if k.endswith('.unity')]
    
    if scenes:
        st.header("📋 Scene Analysis")
        
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
                        
                        # Additional analysis options
                        with st.expander("🔧 Advanced Analysis Options"):
                            col1, col2 = st.columns(2)
                            with col1:
                                show_orphaned = st.checkbox("Show orphaned assets", value=False)
                                show_circular = st.checkbox("Detect circular dependencies", value=False)
                            with col2:
                                filter_by_type = st.selectbox(
                                    "Filter by asset type:",
                                    ["All", "Scripts only", "Prefabs only", "Assets only"]
                                )
                        
                        # Circular dependency detection
                        if show_circular:
                            try:
                                cycles = list(nx.simple_cycles(scene_subgraph))
                                if cycles:
                                    st.warning(f"⚠️ Found {len(cycles)} circular dependencies:")
                                    for i, cycle in enumerate(cycles[:5]):  # Show first 5
                                        cycle_str = " → ".join([os.path.basename(node) for node in cycle])
                                        st.text(f"{i+1}. {cycle_str}")
                                    if len(cycles) > 5:
                                        st.text(f"... and {len(cycles) - 5} more")
                                else:
                                    st.success("✅ No circular dependencies found")
                            except Exception as e:
                                st.error(f"Error detecting cycles: {str(e)}")
                        
                        # Orphaned assets detection
                        if show_orphaned:
                            all_referenced = set()
                            for deps in analyzer.scene_dependencies.values():
                                all_referenced.update(deps)
                            
                            all_assets = set(analyzer.scene_dependencies.keys())
                            orphaned = all_assets - all_referenced
                            orphaned = {asset for asset in orphaned if not asset.endswith('.unity')}
                            
                            if orphaned:
                                st.warning(f"⚠️ Found {len(orphaned)} potentially orphaned assets:")
                                for asset in list(orphaned)[:10]:  # Show first 10
                                    st.text(f"• {os.path.basename(asset)}")
                                if len(orphaned) > 10:
                                    st.text(f"... and {len(orphaned) - 10} more")
                            else:
                                st.success("✅ No orphaned assets found")
                        
                        # Generate Mermaid diagram
                        mermaid_diagram = analyzer.generate_mermaid_diagram(
                            scene_subgraph, 
                            f"Dependencies for {os.path.basename(scene)}"
                        )
                        
                        # Display diagram with user-configurable height
                        st.markdown("### 📊 Dependency Diagram")
                        
                        # Add diagram size controls
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            diagram_height = st.slider(
                                "Diagram Height (pixels)", 
                                min_value=400, 
                                max_value=1200, 
                                value=800,
                                step=50,
                                help="Adjust the height of the dependency diagram for better visibility"
                            )
                        with col2:
                            fullscreen_mode = st.checkbox("Fullscreen Mode", value=False, help="Use maximum width for the diagram")
                        with col3:
                            zoom_level = st.selectbox("Zoom Level", ["Small", "Medium", "Large"], index=1)
                        
                        # Apply zoom level to height
                        zoom_multipliers = {"Small": 0.8, "Medium": 1.0, "Large": 1.3}
                        adjusted_height = int(diagram_height * zoom_multipliers[zoom_level])
                        
                        try:
                            # Use container width based on fullscreen mode
                            if fullscreen_mode:
                                stmd.st_mermaid(mermaid_diagram, height=adjusted_height, width="100%")
                            else:
                                stmd.st_mermaid(mermaid_diagram, height=adjusted_height)
                                
                            # Add tips for better diagram viewing
                            with st.expander("💡 Diagram Viewing Tips"):
                                st.markdown("""
                                **For Better Visibility:**
                                - **Increase Height**: Use the slider above to make the diagram taller
                                - **Fullscreen Mode**: Enable for wider diagrams with many nodes
                                - **Zoom Level**: Choose 'Large' for complex diagrams with many dependencies
                                - **Browser Zoom**: Use Ctrl+/Cmd+ to zoom in your browser
                                - **Download**: Save the Mermaid file to view in external tools like Mermaid Live Editor
                                """)
                                
                        except Exception as e:
                            st.warning(f"Could not render interactive diagram: {str(e)}")
                            st.markdown("**Mermaid Code (copy to external viewer if needed):**")
                            st.code(mermaid_diagram, language="mermaid")
                            st.info("💡 **Tip**: Copy the code above and paste it into [Mermaid Live Editor](https://mermaid.live/) for interactive viewing.")
                        
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
                            
                            # Performance insights
                            if dependencies_data:
                                st.markdown("### 🎯 Performance Insights")
                                
                                # Most referenced assets
                                target_counts = {}
                                for dep in dependencies_data:
                                    target = dep["Target"]
                                    target_counts[target] = target_counts.get(target, 0) + 1
                                
                                top_referenced = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Most Referenced Assets:**")
                                    for asset, count in top_referenced:
                                        st.text(f"• {asset} ({count} references)")
                                
                                with col2:
                                    # Dependency complexity score
                                    script_complexity = len([d for d in dependencies_data if d["Source Type"] == "script"])
                                    total_deps = len(dependencies_data)
                                    complexity_score = min(100, (script_complexity / max(total_deps, 1)) * 100)
                                    
                                    st.metric(
                                        "Script Coupling Score", 
                                        f"{complexity_score:.1f}%",
                                        help="Lower is better. High coupling may indicate refactoring opportunities."
                                    )
                            
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
        st.info("No scenes found in the uploaded project. Please ensure you uploaded a complete Unity project with scene files.")
else:
    st.info("👆 Please upload a Unity project ZIP file and analyze it to begin dependency analysis.")
    
    # Example usage section
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Steps to Analyze Your Unity Project:
        
        1. **Prepare Project**: Create a ZIP file of your Unity project (see sidebar for instructions)
        2. **Upload Project**: Use the file uploader to upload your Unity project ZIP
        3. **Analyze Project**: Click "Analyze Project" to scan all files and build dependency graph
        4. **Select Scenes**: Choose which scenes you want to analyze from the dropdown
        5. **Configure Options**: Adjust analysis settings like dependency depth and included types
        6. **Generate Analysis**: Click "Analyze Selected Scenes" to generate dependency graphs
        7. **Download Results**: Save Mermaid diagrams and CSV dependency tables
        
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
        
        ### Cloud Deployment Benefits:
        
        - **No Local Installation**: Access from any device with a web browser
        - **Secure Processing**: Files are processed temporarily and not stored
        - **Cross-Platform**: Works on Windows, Mac, and Linux
        - **Team Sharing**: Share analysis results with your development team
        """)

# Footer
st.markdown("---")
st.markdown("**Unity Dependency Analyzer** - Built for Unity developers to understand project structure and dependencies. Now cloud-ready! 🚀")