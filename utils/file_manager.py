"""
MÓDULO: GESTIÓN DE ARCHIVOS Y DIRECTORIOS
=========================================
Centraliza toda la lógica de manejo de archivos del sistema
"""

import os
import shutil
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ProjectPaths:
    """Rutas de un proyecto organizadas"""
    root: Path
    original: Path
    audio: Path
    audio_separated: Path
    transcription: Path
    translation: Path
    audio_spanish: Path
    final: Path
    
    def to_dict(self) -> Dict[str, str]:
        """Convierte rutas a diccionario"""
        return {
            'root': str(self.root),
            'original': str(self.original),
            'audio': str(self.audio),
            'audio_separated': str(self.audio_separated),
            'transcription': str(self.transcription),
            'translation': str(self.translation),
            'audio_spanish': str(self.audio_spanish),
            'final': str(self.final)
        }

@dataclass
class FileInfo:
    """Información detallada de un archivo"""
    path: Path
    exists: bool
    size_bytes: int = 0
    size_mb: float = 0.0
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    extension: str = ""
    name: str = ""
    stem: str = ""
    
    @staticmethod
    def from_path(file_path: Union[str, Path]) -> 'FileInfo':
        """Crea FileInfo desde una ruta"""
        path = Path(file_path)
        
        if not path.exists():
            return FileInfo(
                path=path,
                exists=False,
                name=path.name,
                stem=path.stem,
                extension=path.suffix
            )
        
        stat = path.stat()
        
        return FileInfo(
            path=path,
            exists=True,
            size_bytes=stat.st_size,
            size_mb=round(stat.st_size / 1024 / 1024, 2),
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            extension=path.suffix,
            name=path.name,
            stem=path.stem
        )

class FileManager:
    """Gestor de archivos del sistema"""
    
    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 50) -> str:
        """Limpia nombre de archivo para uso seguro"""
        # Remover caracteres problemáticos
        clean = re.sub(r'[<>:"/\\|?*\[\]{}()]', '_', filename)
        clean = re.sub(r'[^\w\s-]', '', clean)
        clean = ' '.join(clean.split())  # Eliminar espacios múltiples
        clean = clean.replace(' ', '_')
        
        # Limitar longitud
        if len(clean) > max_length:
            clean = clean[:max_length]
        
        # Asegurar que no esté vacío
        return clean if clean else "unnamed_file"
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> FileInfo:
        """Obtiene información detallada de un archivo"""
        return FileInfo.from_path(file_path)
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> bool:
        """Asegura que un directorio exista"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    @staticmethod
    def safe_remove(file_path: Union[str, Path]) -> bool:
        """Elimina archivo de forma segura"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
            return True
        except Exception:
            return False
    
    @staticmethod
    def safe_move(source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Mueve archivo de forma segura"""
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            # Crear directorio destino si no existe
            FileManager.ensure_directory(dest_path.parent)
            
            shutil.move(str(source_path), str(dest_path))
            return True
        except Exception:
            return False
    
    @staticmethod
    def copy_file(source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Copia archivo de forma segura"""
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            # Crear directorio destino si no existe
            FileManager.ensure_directory(dest_path.parent)
            
            shutil.copy2(str(source_path), str(dest_path))
            return True
        except Exception:
            return False
    
    @staticmethod
    def cleanup_temp_files(directory: Union[str, Path], patterns: List[str] = None) -> int:
        """Limpia archivos temporales de un directorio"""
        if patterns is None:
            patterns = ["*_temp.*", "*.tmp", "temp_*", "*_temporal.*"]
        
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        cleaned_count = 0
        for pattern in patterns:
            for temp_file in directory.rglob(pattern):
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                except Exception:
                    pass
        
        return cleaned_count
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """Guarda datos en archivo JSON"""
        try:
            path = Path(file_path)
            FileManager.ensure_directory(path.parent)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Carga datos desde archivo JSON"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

class DirectoryManager:
    """Gestor de directorios del proyecto"""
    
    def __init__(self, base_path: Union[str, Path] = "downloads"):
        self.base_path = Path(base_path)
    
    def ensure_base_directory(self) -> bool:
        """Asegura que el directorio base exista"""
        return FileManager.ensure_directory(self.base_path)
    
    def get_next_project_number(self) -> int:
        """Obtiene el siguiente número de proyecto"""
        if not self.base_path.exists():
            return 1
        
        existing_numbers = []
        for item in self.base_path.iterdir():
            if item.is_dir() and len(item.name) >= 3 and item.name[:3].isdigit():
                try:
                    existing_numbers.append(int(item.name[:3]))
                except ValueError:
                    continue
        
        return max(existing_numbers, default=0) + 1
    
    def generate_project_name(self, title: str, video_id: str = "") -> str:
        """Genera nombre de proyecto único"""
        # Limpiar título
        clean_title = FileManager.sanitize_filename(title, max_length=25)
        
        # Obtener número de proyecto
        project_number = self.get_next_project_number()
        
        # Crear nombre final
        if video_id:
            return f"{project_number:03d}_{clean_title}_{video_id[:8]}"
        else:
            return f"{project_number:03d}_{clean_title}"
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """Lista todos los proyectos disponibles"""
        if not self.base_path.exists():
            return []
        
        projects = []
        for item in self.base_path.iterdir():
            if not item.is_dir():
                continue
            
            project_info = {
                'name': item.name,
                'path': str(item),
                'created': datetime.fromtimestamp(item.stat().st_ctime),
                'size_mb': self._get_directory_size(item)
            }
            
            # Buscar metadata del proyecto
            metadata_file = item / "project_metadata.json"
            if metadata_file.exists():
                metadata = FileManager.load_json(metadata_file)
                if metadata:
                    project_info.update(metadata)
            
            projects.append(project_info)
        
        # Ordenar por fecha de creación (más recientes primero)
        projects.sort(key=lambda x: x['created'], reverse=True)
        return projects
    
    def _get_directory_size(self, directory: Path) -> float:
        """Calcula tamaño de directorio en MB"""
        try:
            total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
            return round(total_size / 1024 / 1024, 2)
        except Exception:
            return 0.0
    
    def cleanup_old_projects(self, keep_count: int = 10) -> int:
        """Limpia proyectos antiguos manteniendo solo los más recientes"""
        projects = self.list_projects()
        
        if len(projects) <= keep_count:
            return 0
        
        projects_to_remove = projects[keep_count:]
        removed_count = 0
        
        for project in projects_to_remove:
            try:
                shutil.rmtree(project['path'])
                removed_count += 1
            except Exception:
                pass
        
        return removed_count

class ProjectStructureManager:
    """Gestor de estructura de proyectos"""
    
    @staticmethod
    def create_project_structure(project_path: Union[str, Path]) -> ProjectPaths:
        """Crea estructura completa del proyecto"""
        project_path = Path(project_path)
        
        # Definir todas las rutas
        paths = ProjectPaths(
            root=project_path,
            original=project_path / "1_original",
            audio=project_path / "2_audio",
            audio_separated=project_path / "audio_separado",
            transcription=project_path / "3_transcripcion",
            translation=project_path / "4_traduccion",
            audio_spanish=project_path / "5_audio_es",
            final=project_path / "6_final"
        )
        
        # Crear todos los directorios
        for path in [paths.root, paths.original, paths.audio, paths.audio_separated, paths.transcription, paths.translation, paths.audio_spanish, paths.final]:
            FileManager.ensure_directory(path)
        
        return paths
    
    @staticmethod
    def validate_project_structure(project_path: Union[str, Path]) -> Dict[str, bool]:
        """Valida que la estructura del proyecto esté completa"""
        project_path = Path(project_path)
        
        expected_dirs = [
            "1_original", "2_audio", "audio_separado",
            "3_transcripcion", "4_traduccion", "5_audio_es", "6_final"
        ]
        
        validation = {'project_exists': project_path.exists()}
        
        for dir_name in expected_dirs:
            dir_path = project_path / dir_name
            validation[dir_name] = dir_path.exists()
        
        validation['structure_complete'] = all(validation.values())
        
        return validation
    
    @staticmethod
    def get_project_assets(project_path: Union[str, Path]) -> Dict[str, Optional[FileInfo]]:
        """Obtiene información de los assets del proyecto"""
        project_path = Path(project_path)
        assets = {}
        
        # Video original
        original_dir = project_path / "1_original"
        assets['original_video'] = ProjectStructureManager._find_file_in_dir(
            original_dir, ['.mp4', '.avi', '.mkv', '.mov', '.webm']
        )
        
        # Audio extraído
        audio_dir = project_path / "2_audio"
        assets['extracted_audio'] = ProjectStructureManager._find_file_in_dir(
            audio_dir, ['.wav', '.mp3', '.m4a']
        )
        
        # Audio separado
        separated_dir = project_path / "audio_separado"
        assets['vocals'] = ProjectStructureManager._find_file_in_dir(
            separated_dir, ['.wav'], pattern="*vocals*"
        )
        assets['accompaniment'] = ProjectStructureManager._find_file_in_dir(
            separated_dir, ['.wav'], pattern="*accompaniment*"
        )
        
        # Transcripción
        transcription_dir = project_path / "3_transcripcion"
        assets['transcription_text'] = ProjectStructureManager._find_file_in_dir(
            transcription_dir, ['.txt'], pattern="*texto*"
        )
        assets['subtitles'] = ProjectStructureManager._find_file_in_dir(
            transcription_dir, ['.srt']
        )
        
        # Traducción
        translation_dir = project_path / "4_traduccion"
        assets['translation_text'] = ProjectStructureManager._find_file_in_dir(
            translation_dir, ['.txt'], pattern="*_es*"
        )
        
        # Audio en español
        audio_es_dir = project_path / "5_audio_es"
        assets['spanish_audio'] = ProjectStructureManager._find_file_in_dir(
            audio_es_dir, ['.wav', '.mp3']
        )
        
        # Video final
        final_dir = project_path / "6_final"
        assets['final_video'] = ProjectStructureManager._find_file_in_dir(
            final_dir, ['.mp4', '.avi', '.mkv'], pattern="*final*"
        )
        
        return assets
    
    @staticmethod
    def _find_file_in_dir(directory: Path, extensions: List[str], pattern: str = "*") -> Optional[FileInfo]:
        """Busca archivo en directorio con extensiones específicas"""
        if not directory.exists():
            return None
        
        for ext in extensions:
            files = list(directory.glob(f"{pattern}{ext}"))
            if files:
                return FileInfo.from_path(files[0])
        
        return None
    
    @staticmethod
    def save_project_metadata(project_path: Union[str, Path], metadata: Dict[str, Any]) -> bool:
        """Guarda metadata del proyecto"""
        project_path = Path(project_path)
        metadata_file = project_path / "project_metadata.json"
        
        # Agregar timestamp
        metadata['last_updated'] = datetime.now().isoformat()
        
        return FileManager.save_json(metadata, metadata_file)
    
    @staticmethod
    def load_project_metadata(project_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Carga metadata del proyecto"""
        project_path = Path(project_path)
        metadata_file = project_path / "project_metadata.json"
        
        return FileManager.load_json(metadata_file)
    
    @staticmethod
    def get_project_progress(project_path: Union[str, Path]) -> Dict[str, Any]:
        """Analiza el progreso del proyecto"""
        assets = ProjectStructureManager.get_project_assets(project_path)
        
        steps = {
            'video_downloaded': assets['original_video'] is not None,
            'audio_extracted': assets['extracted_audio'] is not None,
            'audio_separated': assets['vocals'] is not None and assets['accompaniment'] is not None,
            'transcribed': assets['transcription_text'] is not None,
            'translated': assets['translation_text'] is not None,
            'spanish_audio': assets['spanish_audio'] is not None,
            'final_video': assets['final_video'] is not None
        }
        
        completed_steps = sum(steps.values())
        total_steps = len(steps)
        progress_percentage = (completed_steps / total_steps) * 100
        
        return {
            'steps': steps,
            'completed_steps': completed_steps,
            'total_steps': total_steps,
            'progress_percentage': progress_percentage,
            'is_complete': completed_steps == total_steps,
            'next_step': ProjectStructureManager._get_next_step(steps)
        }
    
    @staticmethod
    def _get_next_step(steps: Dict[str, bool]) -> Optional[str]:
        """Determina el siguiente paso a realizar"""
        step_order = [
            ('video_downloaded', 'Descargar video'),
            ('audio_extracted', 'Extraer audio'),
            ('audio_separated', 'Separar audio'),
            ('transcribed', 'Transcribir audio'),
            ('translated', 'Traducir texto'),
            ('spanish_audio', 'Generar audio en español'),
            ('final_video', 'Componer video final')
        ]
        
        for step_key, step_name in step_order:
            if not steps.get(step_key, False):
                return step_name
        
        return None