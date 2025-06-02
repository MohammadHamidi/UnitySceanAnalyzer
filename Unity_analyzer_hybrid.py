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
import requests
import time

# Set page configuration
st.set_page_config(
    page_title="Unity Dependency Analyzer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AIEnhancedAnalysis:
    """AI-powered analysis using AvalAI API for advanced Unity project insights"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.avalai.ir/v1"
        self.model = "gpt-4o-mini"  # Using available model from AvalAI
    
    def make_api_request(self, messages: List[dict], max_tokens: int = 2000) -> str:
        """Make request to AvalAI API"""
        if not self.api_key:
            return "API key not provided"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"API Error: {str(e)}"
    
    def analyze_scene_execution_flow(self, scene_data: dict, dependencies: dict) -> str:
        """Analyze scene execution flow and identify potential issues"""
        scene_info = {
            "scene_name": scene_data.get("name", "Unknown"),
            "dependencies": list(dependencies.keys())[:20],  # Limit for API
            "script_count": len([d for d in dependencies.keys() if d.endswith('.cs')]),
            "prefab_count": len([d for d in dependencies.keys() if d.endswith('.prefab')])
        }
        
        messages = [
            {
                "role": "system",
                "content": """You are a Unity development expert. Analyze the scene execution flow and identify potential issues, bottlenecks, and optimization opportunities. Focus on:
1. Script execution order issues
2. Potential performance bottlenecks
3. Memory management concerns
4. Loading sequence problems
5. Dependency conflicts"""
            },
            {
                "role": "user",
                "content": f"""Analyze this Unity scene execution flow:

Scene: {scene_info['scene_name']}
Scripts: {scene_info['script_count']}
Prefabs: {scene_info['prefab_count']}

Key Dependencies:
{chr(10).join(f"- {dep}" for dep in scene_info['dependencies'][:10])}

Provide analysis of execution flow, potential issues, and recommendations."""
            }
        ]
        
        return self.make_api_request(messages)
    
    def detect_potential_bugs(self, script_dependencies: dict, scene_dependencies: dict) -> str:
        """Detect potential bugs and issues in the project structure"""
        
        # Analyze dependency patterns
        circular_deps = []
        missing_deps = []
        
        for script, deps in script_dependencies.items():
            for dep in deps:
                if dep.startswith("<UNKNOWN"):
                    missing_deps.append((script, dep))
        
        analysis_data = {
            "total_scripts": len(script_dependencies),
            "missing_dependencies": len(missing_deps),
            "circular_potential": len(circular_deps),
            "sample_missing": missing_deps[:5]
        }
        
        messages = [
            {
                "role": "system", 
                "content": """You are a Unity debugging expert. Analyze project structure for potential bugs, issues, and problems. Focus on:
1. Missing reference bugs
2. Null reference potential
3. Initialization order issues  
4. Resource loading problems
5. Performance anti-patterns
6. Memory leaks potential"""
            },
            {
                "role": "user",
                "content": f"""Analyze this Unity project for potential bugs:

Project Stats:
- Total Scripts: {analysis_data['total_scripts']}
- Missing Dependencies: {analysis_data['missing_dependencies']}

Sample Missing Dependencies:
{chr(10).join(f"- {script} missing {dep}" for script, dep in analysis_data['sample_missing'])}

Identify potential bugs, risks, and provide debugging recommendations."""
            }
        ]
        
        return self.make_api_request(messages)
    
    def identify_corner_cases(self, dependency_graph: nx.DiGraph, node_types: dict) -> str:
        """Identify corner cases and edge scenarios"""
        
        # Analyze graph structure for edge cases
        isolated_nodes = list(nx.isolates(dependency_graph))
        high_degree_nodes = [node for node in dependency_graph.nodes() 
                           if dependency_graph.degree(node) > 10]
        
        corner_case_data = {
            "isolated_assets": len(isolated_nodes),
            "highly_connected": len(high_degree_nodes),
            "graph_density": nx.density(dependency_graph),
            "sample_isolated": isolated_nodes[:5],
            "sample_connected": high_degree_nodes[:5]
        }
        
        messages = [
            {
                "role": "system",
                "content": """You are a Unity architecture expert. Identify corner cases, edge scenarios, and potential problems in project structure. Focus on:
1. Orphaned assets and components
2. Over-coupling issues
3. Unusual dependency patterns
4. Scalability concerns
5. Maintenance challenges
6. Edge case scenarios in loading/runtime"""
            },
            {
                "role": "user", 
                "content": f"""Analyze this Unity project structure for corner cases:

Project Structure:
- Isolated Assets: {corner_case_data['isolated_assets']}
- Highly Connected Assets: {corner_case_data['highly_connected']}
- Dependency Density: {corner_case_data['graph_density']:.3f}

Sample Isolated Assets:
{chr(10).join(f"- {asset}" for asset in corner_case_data['sample_isolated'])}

Sample Highly Connected Assets:
{chr(10).join(f"- {asset}" for asset in corner_case_data['sample_connected'])}

Identify corner cases, edge scenarios, and architectural concerns."""
            }
        ]
        
        return self.make_api_request(messages)
    
    def generate_optimization_suggestions(self, analysis_results: dict) -> str:
        """Generate comprehensive optimization suggestions"""
        
        messages = [
            {
                "role": "system",
                "content": """You are a Unity performance optimization expert. Based on project analysis, provide specific, actionable optimization recommendations. Focus on:
1. Performance improvements
2. Memory optimization
3. Build size reduction
4. Loading time improvements
5. Runtime efficiency
6. Code refactoring suggestions"""
            },
            {
                "role": "user",
                "content": f"""Based on Unity project analysis, provide optimization recommendations:

Project Analysis Summary:
{json.dumps(analysis_results, indent=2)}

Provide specific, actionable optimization suggestions with implementation steps."""
            }
        ]
        
        return self.make_api_request(messages, max_tokens=3000)

class UnityDependencyAnalyzer:
    def __init__(self):
        self.project_root = None
        self.guid_index = {}
        self.scene_dependencies = {}
        self.script_dependencies = {}
        self.dependency_graph = None
        self.node_types = {}
    
    def download_from_url(self, url: str) -> tempfile.NamedTemporaryFile:
        """Download file from URL for analysis"""
        try:
            # Handle Google Drive URLs
            if "drive.google.com" in url:
                if "/file/d/" in url:
                    # Convert sharing URL to direct download
                    file_id = url.split("/file/d/")[1].split("/")[0]
                    url = f"https://drive.google.com/uc?id={file_id}"
                elif "uc?id=" not in url:
                    raise ValueError("Invalid Google Drive URL format")
            
            # Handle Dropbox URLs
            elif "dropbox.com" in url and "?dl=0" in url:
                url = url.replace("?dl=0", "?dl=1")
            
            # Download with progress tracking
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            # Download with progress bar
            if total_size > 0:
                progress_bar = st.progress(0)
                status_text = st.empty()
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB")
                
                progress_bar.empty()
                status_text.empty()
            else:
                # Download without progress tracking
                st.info("Downloading file... (size unknown)")
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
            
            temp_file.close()
            return temp_file
            
        except Exception as e:
            st.error(f"Failed to download from URL: {str(e)}")
            return None
    
    def extract_project_from_zip(self, uploaded_file_or_path) -> str:
        """Extract uploaded ZIP file or downloaded file to temporary directory"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Handle both uploaded files and file paths
            if isinstance(uploaded_file_or_path, str):
                # It's a file path from URL download
                zip_path = uploaded_file_or_path
            else:
                # It's an uploaded file object
                zip_path = uploaded_file_or_path
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
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
            
            st.info(f"Found {total_files} .meta files to process")
            
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
                st.info("Make sure your ZIP contains Unity scene files with .unity extension.")
            
            if prefab_files:
                st.success(f"Example prefab files found:")
                for prefab in prefab_files[:3]:  # Show first 3
                    rel_path = os.path.relpath(prefab, project_root)
                    st.text(f"  - {rel_path}")
                if len(prefab_files) > 3:
                    st.text(f"  ... and {len(prefab_files) - 3} more")
            
            all_files = unity_files + prefab_files
            
            if not all_files:
                st.error("No Unity scene or prefab files found in the uploaded project!")
                st.info("""
                **Troubleshooting:**
                1. Make sure your ZIP contains an 'Assets' folder
                2. Scene files should have .unity extension
                3. Prefab files should have .prefab extension
                4. Check if you uploaded the correct Unity project ZIP
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
            # Silently skip problematic files
            pass
        
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
        
        # Initialize script_dependencies if not already done
        if not hasattr(self, 'script_dependencies'):
            self.script_dependencies = {}
        
        for cs_file in cs_files:
            rel_path = os.path.relpath(cs_file, project_root).replace("\\", "/")
            deps = self.extract_script_dependencies(str(cs_file))
            
            # Store script dependencies for AI analysis
            self.script_dependencies[rel_path] = deps
            
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
    
    # AI Enhancement section
    st.markdown("### 🤖 AI Enhanced Analysis")
    ai_enhanced = st.checkbox(
        "Enable AI Enhancement", 
        value=False,
        help="Use AvalAI API for advanced analysis including execution flow, bug detection, and corner cases"
    )
    
    avalai_api_key = None
    if ai_enhanced:
        avalai_api_key = st.text_input(
            "AvalAI API Key",
            type="password", 
            placeholder="aa-***********",
            help="Enter your AvalAI API key. Get one from https://avalai.ir"
        )
        
        if avalai_api_key:
            st.success("🔑 API key configured")
            st.info("💡 AI analysis will include: execution flow, bug detection, corner cases, and optimization suggestions")
        else:
            st.warning("⚠️ Please enter your AvalAI API key to use AI enhancement")
    
    st.markdown("---")
    
    # File upload section
    st.markdown("### 📁 Upload Unity Project")
    
    # File size warning and options
    st.warning("⚠️ **File Size Limits**: Streamlit has a 200MB upload limit. For larger projects, use the options below.")
    
    upload_method = st.radio(
        "Choose upload method:",
        ["Direct Upload (< 200MB)", "Essential Files Only", "URL Upload (External Storage)"],
        help="Select the best method based on your project size"
    )
    
    uploaded_file = None
    project_url = None
    
    if upload_method == "Direct Upload (< 200MB)":
        uploaded_file = st.file_uploader(
            "Upload Unity Project (ZIP)",
            type=['zip'],
            help="Upload a complete Unity project ZIP file under 200MB"
        )
        
    elif upload_method == "Essential Files Only":
        st.info("""
        **📦 Create a lightweight ZIP with only essential files:**
        - `Assets/` folder (scripts, scenes, prefabs only)
        - All `.meta` files (critical for GUID resolution)
        - `ProjectSettings/` folder
        - **Exclude**: Textures, audio, videos, large 3D models
        """)
        
        uploaded_file = st.file_uploader(
            "Upload Essential Files (ZIP)",
            type=['zip'],
            help="Upload only scripts, scenes, prefabs, and .meta files"
        )
        
        with st.expander("🛠️ How to Create Essential Files ZIP"):
            st.code("""
# PowerShell/CMD (Windows)
# Navigate to your Unity project folder, then:
tar -czf essential_project.zip Assets/*.cs Assets/*.unity Assets/*.prefab Assets/*.meta ProjectSettings/

# Linux/Mac Terminal
# Navigate to your Unity project folder, then:
find Assets -name "*.cs" -o -name "*.unity" -o -name "*.prefab" -o -name "*.meta" | zip -@ essential_project.zip
zip -r essential_project.zip ProjectSettings/
            """, language="bash")
            
    else:  # URL Upload
        st.info("""
        **🌐 Upload your project to external storage first:**
        - Google Drive (make publicly accessible)
        - Dropbox (share public link)
        - GitHub releases
        - Any direct download URL
        """)
        
        project_url = st.text_input(
            "Project ZIP URL",
            placeholder="https://drive.google.com/uc?id=YOUR_FILE_ID",
            help="Direct download link to your Unity project ZIP"
        )
        
        if project_url:
            st.success("🔗 URL provided - will download when analyzing")
            
            # Validate URL format
            if "drive.google.com" in project_url and "uc?id=" not in project_url:
                st.warning("💡 **Google Drive Tip**: Make sure to use the direct download format: `https://drive.google.com/uc?id=YOUR_FILE_ID`")
                
        with st.expander("📋 External Storage Setup Guide"):
            st.markdown("""
            ### Google Drive Setup:
            1. Upload your Unity project ZIP to Google Drive
            2. Right-click → Share → Change to "Anyone with the link"
            3. Copy the sharing link (looks like: `https://drive.google.com/file/d/FILE_ID/view`)
            4. Convert to direct download: `https://drive.google.com/uc?id=FILE_ID`
            
            ### Dropbox Setup:
            1. Upload ZIP to Dropbox
            2. Create a shared link
            3. Change `?dl=0` to `?dl=1` at the end of the URL
            
            ### GitHub Releases:
            1. Create a release in your repository
            2. Attach the ZIP file as an asset
            3. Use the direct download URL from the release
            """)
    
    # File size optimization tips
    with st.expander("💡 Project Size Optimization Tips"):
        st.markdown("""
        ### For Dependency Analysis, You Can Safely Exclude:
        
        **Large Assets (Not Needed):**
        - `.png`, `.jpg`, `.tga` texture files
        - `.wav`, `.mp3`, `.ogg` audio files  
        - `.mp4`, `.mov` video files
        - `.fbx`, `.obj` large 3D models
        - `Packages/` cached packages
        
        **Essential Files (Keep These):**
        - `.cs` script files
        - `.unity` scene files
        - `.prefab` prefab files
        - `.meta` metadata files (crucial!)
        - `ProjectSettings/` folder
        - `.asmdef` assembly definition files
        
        ### Typical Size Reductions:
        - **Full Project**: 500MB - 2GB
        - **Essential Only**: 5MB - 50MB  
        - **Compression Savings**: 60-90% size reduction
        
        ### Command Line Helper:
        ```bash
        # Create essential-only ZIP (Windows PowerShell)
        Get-ChildItem -Recurse | Where-Object {$_.Extension -in @('.cs','.unity','.prefab','.meta','.asmdef')} | Compress-Archive -DestinationPath essential.zip
        ```
        """)
    
    # Add size estimation
    st.markdown("### 📊 Size Guidelines")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Full Project", "500MB - 2GB", "❌ Too Large")
    with col2:
        st.metric("Essential Files", "5MB - 50MB", "✅ Perfect")
    with col3:
        st.metric("Upload Limit", "200MB", "⚠️ Hard Limit")

with st.sidebar:
    with st.expander("📦 Quick ZIP Creation"):
        st.markdown("""
        ### For Large Projects (Essential Files):
        
        **Windows PowerShell:**
        ```powershell
        # In your Unity project folder:
        Get-ChildItem -Recurse -Include *.cs,*.unity,*.prefab,*.meta | Compress-Archive -DestinationPath essential.zip
        ```
        
        **Mac/Linux Terminal:**
        ```bash  
        # In your Unity project folder:
        find . -name "*.cs" -o -name "*.unity" -o -name "*.prefab" -o -name "*.meta" | zip essential.zip -@
        ```
        
        💡 **Result**: 90% smaller file, same analysis quality!
        """)

if uploaded_file is not None or project_url:
    # Show file information
    if uploaded_file:
        file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
        uploaded_file.seek(0)  # Reset file pointer
        
        if file_size_mb > 200:
            st.error(f"❌ File too large: {file_size_mb:.1f}MB (limit: 200MB)")
            st.info("💡 Consider using 'Essential Files Only' or 'URL Upload' method")
        else:
            st.success(f"✅ Uploaded: {uploaded_file.name} ({file_size_mb:.1f}MB)")
            
    elif project_url:
        st.success(f"🔗 URL provided: {project_url}")
    
    # Analysis methods info
    if upload_method == "Essential Files Only":
        st.info("🎯 **Essential Files Mode**: Analysis will focus on code structure and dependencies")
    elif upload_method == "URL Upload (External Storage)":
        st.info("🌐 **URL Mode**: Will download and analyze the project from external storage")
    
    # Show what will be analyzed
    with st.expander("🔍 What Will Be Analyzed"):
        if upload_method == "Essential Files Only":
            st.markdown("""
            **Focused Analysis:**
            - ✅ Script dependencies and references
            - ✅ Scene structure and prefab relationships  
            - ✅ Assembly definitions and namespaces
            - ✅ Code architecture and patterns
            - ⚠️ Asset dependencies may be limited
            - ⚠️ Texture/audio references may be missing
            """)
        else:
            st.markdown("""
            **Complete Analysis:**
            - ✅ Full dependency mapping
            - ✅ All asset relationships
            - ✅ Script and prefab dependencies
            - ✅ Texture and audio references
            - ✅ Complete project structure
            """)
    
    if st.button("🔍 Analyze Project", type="primary"):
        with st.spinner("Extracting and analyzing Unity project..."):
            analyzer = st.session_state.analyzer
            
            # Handle different upload methods
            project_root = None
            
            if uploaded_file:
                # Direct upload
                st.info("Extracting uploaded project files...")
                project_root = analyzer.extract_project_from_zip(uploaded_file)
                
            elif project_url:
                # URL download
                st.info("Downloading project from URL...")
                temp_file = analyzer.download_from_url(project_url)
                
                if temp_file:
                    st.info("Extracting downloaded project files...")
                    project_root = analyzer.extract_project_from_zip(temp_file.name)
                    # Clean up downloaded file
                    os.unlink(temp_file.name)
                else:
                    st.error("Failed to download project from URL")
            
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
                
                # Show results with method-specific messaging
                if upload_method == "Essential Files Only" and len(analyzer.guid_index) < 100:
                    st.warning("⚠️ Limited assets detected. This is normal for essential-only uploads.")
                
                # Final results with better feedback
                scenes_found = len([k for k in analyzer.scene_dependencies.keys() if k.endswith('.unity')])
                prefabs_found = len([k for k in analyzer.scene_dependencies.keys() if k.endswith('.prefab')])
                
                if scenes_found > 0 or prefabs_found > 0:
                    st.success(f"✅ Project analyzed successfully!")
                    
                    # Enhanced statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Analysis Results:**")
                        st.markdown(f"- 📄 {len(analyzer.guid_index)} assets with GUIDs")
                        st.markdown(f"- 🎬 {scenes_found} scenes")
                        st.markdown(f"- 🧩 {prefabs_found} prefabs")
                    with col2:
                        st.markdown(f"**Dependency Graph:**")
                        st.markdown(f"- 🔗 {analyzer.dependency_graph.number_of_nodes()} total nodes")
                        st.markdown(f"- ➡️ {analyzer.dependency_graph.number_of_edges()} relationships")
                        st.markdown(f"- 📊 Method: {upload_method}")
                    
                    # Analysis quality indicator
                    if upload_method == "Essential Files Only":
                        st.info("🎯 **Analysis Quality**: Focused on code structure (recommended for large projects)")
                    else:
                        st.info("🔍 **Analysis Quality**: Complete project analysis")
                else:
                    st.error("❌ No scenes or prefabs found!")
                    st.info("""
                    **Troubleshooting Tips:**
                    1. Make sure your ZIP contains Unity scene files (.unity extension)
                    2. Check that the Assets folder is included in your ZIP
                    3. Verify the project structure is correct
                    4. Try uploading a different Unity project for testing
                    """)
                    
            else:
                st.error("Failed to extract project. Please ensure you uploaded a valid Unity project ZIP file.")

else:
    st.info("👆 Please upload a Unity project or provide a URL to begin analysis.")
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        ### Choose Your Method:
        
        **🟢 Direct Upload (< 200MB)**
        - Best for: Small to medium projects
        - Upload: Complete project ZIP
        - Analysis: Full dependency mapping
        
        **🟡 Essential Files Only**  
        - Best for: Large projects (> 200MB)
        - Upload: Scripts, scenes, prefabs + .meta files only
        - Analysis: Code structure and core dependencies
        
        **🟠 URL Upload**
        - Best for: Very large projects or team sharing
        - Upload: Project to Google Drive/Dropbox first
        - Analysis: Full dependency mapping
        
        ### File Size Estimates:
        - **Mobile Game**: 50-200MB → Direct Upload ✅
        - **Indie Game**: 200MB-1GB → Essential Files ✅  
        - **AAA Project**: 1GB+ → URL Upload ✅
        """)

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
                        
                        # AI Enhanced Analysis Section
                        if ai_enhanced and avalai_api_key and dependencies_data:
                            st.markdown("---")
                            st.markdown("### 🤖 AI Enhanced Analysis")
                            st.info("🔄 Performing AI-powered analysis using AvalAI...")
                            
                            # Initialize AI analyzer
                            ai_analyzer = AIEnhancedAnalysis(avalai_api_key)
                            
                            # Create tabs for different AI analyses
                            ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
                                "🔄 Execution Flow", 
                                "🐛 Bug Detection", 
                                "⚠️ Corner Cases", 
                                "⚡ Optimization"
                            ])
                            
                            with ai_tab1:
                                st.markdown("#### Scene Execution Flow Analysis")
                                with st.spinner("Analyzing execution flow..."):
                                    scene_data = {"name": os.path.basename(scene)}
                                    flow_analysis = ai_analyzer.analyze_scene_execution_flow(
                                        scene_data, 
                                        {k: v for k, v in analyzer.scene_dependencies.items() if k == scene}
                                    )
                                
                                if "API Error" not in flow_analysis:
                                    st.markdown(flow_analysis)
                                    
                                    # Download flow analysis
                                    st.download_button(
                                        "📥 Download Flow Analysis",
                                        flow_analysis,
                                        file_name=f"{os.path.splitext(os.path.basename(scene))[0]}_flow_analysis.md",
                                        mime="text/markdown"
                                    )
                                else:
                                    st.error(f"Flow analysis failed: {flow_analysis}")
                            
                            with ai_tab2:
                                st.markdown("#### Potential Bug Detection")
                                with st.spinner("Detecting potential bugs..."):
                                    bug_analysis = ai_analyzer.detect_potential_bugs(
                                        analyzer.script_dependencies,
                                        analyzer.scene_dependencies
                                    )
                                
                                if "API Error" not in bug_analysis:
                                    st.markdown(bug_analysis)
                                    
                                    # Highlight critical issues
                                    if any(keyword in bug_analysis.lower() for keyword in ['critical', 'severe', 'high risk']):
                                        st.error("🚨 Critical issues detected! Please review the analysis above.")
                                    elif any(keyword in bug_analysis.lower() for keyword in ['warning', 'potential', 'moderate']):
                                        st.warning("⚠️ Potential issues found. Consider reviewing the recommendations.")
                                    else:
                                        st.success("✅ No major issues detected in the analysis.")
                                    
                                    st.download_button(
                                        "📥 Download Bug Analysis",
                                        bug_analysis,
                                        file_name=f"{os.path.splitext(os.path.basename(scene))[0]}_bug_analysis.md",
                                        mime="text/markdown"
                                    )
                                else:
                                    st.error(f"Bug detection failed: {bug_analysis}")
                            
                            with ai_tab3:
                                st.markdown("#### Corner Cases & Edge Scenarios")
                                with st.spinner("Identifying corner cases..."):
                                    corner_analysis = ai_analyzer.identify_corner_cases(
                                        scene_subgraph,
                                        analyzer.node_types
                                    )
                                
                                if "API Error" not in corner_analysis:
                                    st.markdown(corner_analysis)
                                    
                                    st.download_button(
                                        "📥 Download Corner Case Analysis",
                                        corner_analysis,
                                        file_name=f"{os.path.splitext(os.path.basename(scene))[0]}_corner_cases.md",
                                        mime="text/markdown"
                                    )
                                else:
                                    st.error(f"Corner case analysis failed: {corner_analysis}")
                            
                            with ai_tab4:
                                st.markdown("#### Optimization Suggestions")
                                with st.spinner("Generating optimization suggestions..."):
                                    # Compile analysis results
                                    optimization_data = {
                                        "scene": os.path.basename(scene),
                                        "total_dependencies": len(dependencies_data),
                                        "script_count": len([d for d in dependencies_data if d["Source Type"] == "script"]),
                                        "prefab_count": len([d for d in dependencies_data if d["Source Type"] == "prefab"]),
                                        "complexity_score": complexity_score if 'complexity_score' in locals() else 0,
                                        "circular_deps": len(cycles) if 'cycles' in locals() else 0
                                    }
                                    
                                    optimization_suggestions = ai_analyzer.generate_optimization_suggestions(optimization_data)
                                
                                if "API Error" not in optimization_suggestions:
                                    st.markdown(optimization_suggestions)
                                    
                                    # Add implementation checklist
                                    with st.expander("📋 Implementation Checklist"):
                                        st.markdown("""
                                        **Before implementing optimizations:**
                                        - [ ] Backup your project
                                        - [ ] Test in a separate branch
                                        - [ ] Profile performance before changes
                                        - [ ] Implement changes incrementally
                                        - [ ] Test thoroughly after each change
                                        - [ ] Monitor performance improvements
                                        """)
                                    
                                    st.download_button(
                                        "📥 Download Optimization Guide",
                                        optimization_suggestions,
                                        file_name=f"{os.path.splitext(os.path.basename(scene))[0]}_optimization.md",
                                        mime="text/markdown"
                                    )
                                else:
                                    st.error(f"Optimization analysis failed: {optimization_suggestions}")
                            
                            # AI Analysis Summary
                            st.markdown("#### 🎯 AI Analysis Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Analysis Depth", 
                                    "Advanced",
                                    help="AI-powered deep analysis completed"
                                )
                            with col2:
                                st.metric(
                                    "Issues Scanned", 
                                    "Multi-layer",
                                    help="Flow, bugs, corner cases, and optimization analyzed"
                                )
                            with col3:
                                st.metric(
                                    "Recommendations", 
                                    "Actionable",
                                    help="Specific implementation suggestions provided"
                                )
                        
                        st.markdown("---")
                    else:
                        st.warning(f"Scene {scene} not found in dependency graph.")
    else:
        st.info("No scenes found in the uploaded project. Please ensure you uploaded a complete Unity project with scene files.")
        
        # Enhanced troubleshooting
        st.markdown("### 🔧 Troubleshooting")
        st.error("""
        **Scene files not found!** This usually happens when:
        
        1. **Wrong ZIP Contents**: Make sure your ZIP contains Unity scene files with .unity extension
        2. **Missing Assets Folder**: The ZIP should include the 'Assets' directory structure
        3. **File Extensions**: Scene files must have the .unity extension
        4. **Upload Method**: Try different upload methods if one doesn't work
        
        **Expected ZIP Structure:**
        ```
        YourUnityProject.zip
        ├── Assets/
        │   ├── Scenes/
        │   │   ├── MainMenu.unity
        │   │   └── GameLevel.unity
        │   └── Scripts/
        └── ProjectSettings/ (optional)
        ```
        
        **Quick Fixes:**
        - ✅ Verify your ZIP contains .unity files
        - ✅ Try the 'Essential Files Only' upload method
        - ✅ Check that scene files aren't in a nested subfolder
        - ✅ Ensure the ZIP extraction was successful
        """)
        
        # Add a reanalysis option
        if st.button("🔄 Re-analyze Current Project"):
            if st.session_state.analyzer.project_root:
                with st.spinner("Re-analyzing project..."):
                    analyzer = st.session_state.analyzer
                    
                    # Re-run the analysis steps
                    st.info("Re-scanning for scenes and prefabs...")
                    analyzer.scene_dependencies = analyzer.scan_scenes_and_prefabs(analyzer.project_root)
                    
                    st.info("Rebuilding dependency graph...")
                    analyzer.dependency_graph, analyzer.node_types = analyzer.build_dependency_graph(analyzer.project_root)
                    
                    st.success("Re-analysis complete! Check above for results.")
                    st.experimental_rerun()
            else:
                st.warning("No project loaded. Please upload a project first.")
else:
    st.info("👆 Please upload a Unity project ZIP file and analyze it to begin dependency analysis.")
    
    # Example usage section
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Steps to Analyze Your Unity Project:
        
        1. **Enable AI Enhancement** (Optional): Toggle AI analysis and enter your AvalAI API key
        2. **Prepare Project**: Create a ZIP file of your Unity project (see sidebar for instructions)
        3. **Upload Project**: Use the file uploader to upload your Unity project ZIP
        4. **Analyze Project**: Click "Analyze Project" to scan all files and build dependency graph
        5. **Select Scenes**: Choose which scenes you want to analyze from the dropdown
        6. **Configure Options**: Adjust analysis settings like dependency depth and included types
        7. **Generate Analysis**: Click "Analyze Selected Scenes" to generate dependency graphs
        8. **AI Enhancement**: If enabled, get advanced AI-powered insights
        9. **Download Results**: Save Mermaid diagrams, CSV tables, and AI analysis reports
        
        ### What This Tool Analyzes:
        
        **Standard Analysis:**
        - **Scene Dependencies**: Which prefabs, scripts, and assets each scene uses
        - **Script Dependencies**: How C# scripts reference each other
        - **Prefab Relationships**: Dependencies between prefabs and their components
        - **Asset Usage**: Which textures, materials, and other assets are referenced
        
        **AI Enhanced Analysis** (with AvalAI API):
        - **🔄 Execution Flow**: Scene loading order and script execution analysis
        - **🐛 Bug Detection**: Potential null references, missing dependencies, and runtime issues
        - **⚠️ Corner Cases**: Edge scenarios, orphaned assets, and unusual patterns
        - **⚡ Optimization**: Performance improvements and architectural suggestions
        
        ### Supported File Types:
        
        - `.unity` scene files
        - `.prefab` prefab files  
        - `.cs` C# script files
        - `.meta` metadata files for GUID resolution
        
        ### AI Enhancement Benefits:
        
        - **Smart Analysis**: AI-powered insights beyond basic dependency mapping
        - **Bug Prevention**: Early detection of potential runtime issues
        - **Performance Focus**: Targeted optimization recommendations
        - **Best Practices**: Architectural guidance from Unity expertise
        - **Actionable Reports**: Specific implementation steps and checklists
        
        ### Getting AvalAI API Key:
        
        1. Visit [AvalAI.ir](https://avalai.ir)
        2. Create a developer account
        3. Generate an API key from the dashboard
        4. Use competitive pricing with no hidden fees
        5. Access multiple AI models including GPT-4, Claude, and Gemini
        
        ### Cloud Deployment Benefits:
        
        - **No Local Installation**: Access from any device with a web browser
        - **Secure Processing**: Files are processed temporarily and not stored
        - **Cross-Platform**: Works on Windows, Mac, and Linux
        - **Team Sharing**: Share analysis results with your development team
        - **AI-Powered**: Advanced insights with cutting-edge AI models
        """)

# Footer
st.markdown("---")
st.markdown("**Unity Dependency Analyzer** - Built for Unity developers to understand project structure and dependencies. Now with AI-powered insights! 🚀🤖")