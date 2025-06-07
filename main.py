#!/usr/bin/env python3
"""
TRADUCTOR DE VIDEOS AL ESPAÑOL - Aplicación de Consola
=====================================================
Sistema completo para traducir videos usando IA

Uso:
    python main.py translate "https://youtube.com/watch?v=ABC123"
    python main.py translate video.mp4 --voice Female --quality balanced
    python main.py setup
    python main.py status
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional

# Agregar el directorio actual al path para imports
sys.path.insert(0, str(Path(__file__).parent))

# Imports del sistema
from utils.error_handler import ErrorHandler, handle_critical_error
from utils.config_manager import get_default_config_manager, validate_environment_setup
from utils.validators import quick_validate_for_translation
from utils.progress_tracker import create_pipeline_tracker

from video_processing.factories import (
    CompleteVideoTranslationFactory,
    create_video_translator,
    create_complete_video_processor
)


class VideoTranslatorApp:
    """Aplicación principal de consola para traducir videos"""
    
    def __init__(self):
        self.config_manager = None
        self.error_handler = ErrorHandler()
        self.version = "1.0.0"
        
    def main(self):
        """Punto de entrada principal"""
        try:
            # Mostrar banner
            self._show_banner()
            
            # Configurar argumentos
            parser = self._create_argument_parser()
            args = parser.parse_args()
            
            # Inicializar configuración
            self._initialize_config()
            
            # Ejecutar comando
            if args.command == 'translate':
                self._handle_translate_command(args)
            elif args.command == 'setup':
                self._handle_setup_command(args)
            elif args.command == 'status':
                self._handle_status_command(args)
            elif args.command == 'voices':
                self._handle_voices_command(args)
            elif args.command == 'validate':
                self._handle_validate_command(args)
            else:
                parser.print_help()
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Proceso cancelado por el usuario")
            sys.exit(130)
        except Exception as e:
            error_info = self.error_handler.handle_exception(e)
            print(f"\n❌ Error crítico: {error_info.user_message}")
            if error_info.suggested_actions:
                print("💡 Sugerencias:")
                for action in error_info.suggested_actions:
                    print(f"   • {action}")
            sys.exit(1)
    
    def _show_banner(self):
        """Muestra banner de la aplicación"""
        print("🎬" + "=" * 58 + "🎬")
        print("🎥     TRADUCTOR DE VIDEOS AL ESPAÑOL v{}     🎥".format(self.version))
        print("🎬" + "=" * 58 + "🎬")
        print("🤖 Powered by: Whisper + OpenAI/Google + Edge TTS + FFmpeg")
        print("🎯 Traduce cualquier video a español con IA")
        print()
    
    def _create_argument_parser(self):
        """Crea parser de argumentos de línea de comandos"""
        parser = argparse.ArgumentParser(
            description='🎬 Traductor completo de videos al español usando IA',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
🎯 EJEMPLOS DE USO:

  📥 Traducir desde YouTube:
    python main.py translate "https://youtube.com/watch?v=dQw4w9WgXcQ"
    
  📁 Traducir archivo local:
    python main.py translate mi_video.mp4 --voice Female --quality balanced
    
  ⚙️  Configuración inicial:
    python main.py setup
    
  📊 Ver estado del sistema:
    python main.py status
    
  🎤 Listar voces disponibles:
    python main.py voices
    
  ✅ Validar entorno:
    python main.py validate

🔧 CONFIGURACIÓN:
  • Sin API keys: Usa Google Translate (gratis) + Edge TTS
  • Con OpenAI: Traducción premium de mayor calidad
  • Con DeepL: Traducción profesional especializada

📁 ESTRUCTURA DE SALIDA:
  downloads/
  └── 001_nombre_video_ID/
      ├── 1_original/        # Video descargado
      ├── 2_audio/          # Audio extraído  
      ├── 3_transcripcion/  # Transcripción y subtítulos
      ├── 4_traduccion/     # Texto traducido
      ├── 5_audio_es/       # Audio en español (TTS)
      └── 6_final/          # Video final traducido

💡 TIPS:
  • Usa --quality fast para videos largos
  • Usa --quality quality para máxima calidad
  • El proceso puede tardar 2-5x la duración del video
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
        
        # Comando TRANSLATE
        translate_parser = subparsers.add_parser(
            'translate', 
            help='🎬 Traducir video completo',
            description='Traduce un video completo: descarga → audio → transcripción → traducción → TTS → video final'
        )
        translate_parser.add_argument(
            'input', 
            help='URL de YouTube o ruta de archivo de video'
        )
        translate_parser.add_argument(
            '--voice', 
            choices=['Male', 'Female'], 
            default='Female',
            help='Género de voz para TTS (default: Female)'
        )
        translate_parser.add_argument(
            '--quality', 
            choices=['fast', 'balanced', 'quality'], 
            default='balanced',
            help='Nivel de calidad vs velocidad (default: balanced)'
        )
        translate_parser.add_argument(
            '--output-dir', 
            default='downloads',
            help='Directorio de salida (default: downloads)'
        )
        translate_parser.add_argument(
            '--project-name',
            help='Nombre del proyecto (se genera automáticamente si no se especifica)'
        )
        translate_parser.add_argument(
            '--force-language',
            help='Forzar idioma de origen (ej: en, fr, de). Auto-detección si no se especifica'
        )
        translate_parser.add_argument(
            '--transcription-model',
            choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3'],
            default='base',
            help='Modelo de Whisper para transcripción (default: base)'
        )
        
        # Comando SETUP
        setup_parser = subparsers.add_parser(
            'setup', 
            help='⚙️ Configurar API keys y entorno'
        )
        setup_parser.add_argument(
            '--create-configs',
            action='store_true',
            help='Crear archivos de configuración por defecto'
        )
        
        # Comando STATUS
        status_parser = subparsers.add_parser(
            'status', 
            help='📊 Ver estado del sistema y configuración'
        )
        status_parser.add_argument(
            '--detailed',
            action='store_true',
            help='Mostrar información detallada'
        )
        
        # Comando VOICES
        voices_parser = subparsers.add_parser(
            'voices', 
            help='🎤 Listar voces TTS disponibles'
        )
        voices_parser.add_argument(
            '--filter-gender',
            choices=['Male', 'Female'],
            help='Filtrar por género de voz'
        )
        voices_parser.add_argument(
            '--filter-locale',
            help='Filtrar por locale (ej: es-ES, es-MX)'
        )
        
        # Comando VALIDATE
        validate_parser = subparsers.add_parser(
            'validate', 
            help='✅ Validar entorno y dependencias'
        )
        validate_parser.add_argument(
            '--fix',
            action='store_true',
            help='Intentar corregir problemas automáticamente'
        )
        
        return parser
    
    def _initialize_config(self):
        """Inicializa configuración del sistema"""
        try:
            self.config_manager = get_default_config_manager()
            print("✅ Configuración inicializada correctamente")
        except Exception as e:
            print(f"⚠️  Advertencia en configuración: {e}")
            print("🔧 Ejecuta 'python main.py setup' para configurar")
    
    def _handle_translate_command(self, args):
        """Maneja comando de traducción"""
        print(f"\n🎯 INICIANDO TRADUCCIÓN DE VIDEO")
        print("=" * 50)
        
        # Validar entrada
        print("🔍 Validando entrada...")
        is_valid, errors = quick_validate_for_translation(args.input, args.output_dir)
        
        if not is_valid:
            print("❌ Errores de validación encontrados:")
            for error in errors:
                print(f"   • {error}")
            print("\n💡 Ejecuta 'python main.py validate --fix' para corregir problemas")
            sys.exit(1)
        
        print("✅ Validación exitosa")
        
        # Crear servicios según calidad
        print(f"🔧 Configurando pipeline (calidad: {args.quality})...")
        
        try:
            # Crear servicios completos
            if args.quality == 'fast':
                services = CompleteVideoTranslationFactory.create_fast_translator()
            elif args.quality == 'quality':
                services = CompleteVideoTranslationFactory.create_quality_focused_translator()
            else:  # balanced
                services = CompleteVideoTranslationFactory.create_complete_translator_pipeline()
            
            # Obtener servicio de pipeline completo
            complete_processor = services['full_pipeline_service']
            
            print("✅ Pipeline configurado correctamente")
            print(f"🎤 Voz seleccionada: {args.voice}")
            print(f"📁 Directorio de salida: {args.output_dir}")
            
            # Mostrar configuración de traductores
            translation_service = services['translation_service']
            if hasattr(translation_service, 'available_translators'):
                translators = [t.get_translator_name() for t in translation_service.available_translators]
                print(f"🌐 Traductores disponibles: {', '.join(translators)}")
            
            # Iniciar procesamiento
            print(f"\n🚀 INICIANDO PROCESAMIENTO COMPLETO")
            print("=" * 50)
            
            start_time = time.time()
            
            # Ejecutar pipeline completo
            result = complete_processor.process_video_url_to_final_spanish_video(
                video_url=args.input,
                voice_preference=args.voice
            )
            
            total_time = time.time() - start_time
            
            # Mostrar resultados
            if result['success']:
                print(f"\n🎉 ¡TRADUCCIÓN COMPLETADA EXITOSAMENTE!")
                print("=" * 60)
                print(f"⏱️  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} minutos)")
                print(f"🎬 Video final: {result.get('final_video_path', 'Ver directorio de salida')}")
                print(f"📁 Proyecto en: {args.output_dir}")
                
                # Estadísticas adicionales si están disponibles
                if 'download_and_processing' in result:
                    processing_data = result['download_and_processing']
                    if 'transcription_translation' in processing_data:
                        trans_data = processing_data['transcription_translation']
                        if trans_data['success'] and 'transcription' in trans_data:
                            transcription = trans_data['transcription']['transcription']
                            translation = trans_data['translation']['translation'] if trans_data['translation'] else None
                            
                            print(f"\n📊 ESTADÍSTICAS:")
                            print(f"🌍 Idioma detectado: {transcription.language.name}")
                            print(f"📝 Palabras transcritas: {transcription.word_count:,}")
                            if translation:
                                print(f"🌐 Traductor usado: {translation.translator_used}")
                                if translation.cost_estimate > 0:
                                    print(f"💰 Costo traducción: ${translation.cost_estimate:.4f}")
                                else:
                                    print(f"💰 Costo traducción: Gratis")
                
                print(f"\n🎊 ¡LISTO! Tu video está traducido al español")
                
            else:
                print(f"\n❌ ERROR EN TRADUCCIÓN")
                print("=" * 40)
                print(f"⏱️  Tiempo transcurrido: {total_time:.1f}s")
                
                # Intentar mostrar detalles del error
                if 'download_and_processing' in result:
                    processing_data = result['download_and_processing']
                    if not processing_data.get('success'):
                        print("🔍 El error ocurrió durante el procesamiento inicial")
                
                if 'video_composition' in result:
                    composition_data = result['video_composition']
                    if not composition_data.get('success'):
                        print("🔍 El error ocurrió durante la composición final")
                
                print("💡 Revisa los logs anteriores para más detalles")
                sys.exit(1)
                
        except Exception as e:
            error_info = self.error_handler.handle_exception(e)
            print(f"\n❌ Error durante traducción: {error_info.user_message}")
            if error_info.suggested_actions:
                print("💡 Sugerencias:")
                for action in error_info.suggested_actions:
                    print(f"   • {action}")
            sys.exit(1)
    
    def _handle_setup_command(self, args):
        """Maneja comando de configuración"""
        print(f"\n⚙️  CONFIGURACIÓN DEL SISTEMA")
        print("=" * 40)
        
        if args.create_configs:
            # Crear archivos de configuración
            try:
                self.config_manager.create_default_config_files()
                print("✅ Archivos de configuración creados")
            except Exception as e:
                print(f"❌ Error creando configuraciones: {e}")
                return
        
        # Configuración interactiva de API keys
        print("🔑 CONFIGURACIÓN DE API KEYS")
        print("-" * 30)
        print("Para mejorar la calidad de traducción puedes configurar:")
        print("• OpenAI API Key → Traducción premium con GPT")
        print("• DeepL API Key → Traducción profesional")
        print("\n⚠️  Sin API keys usaremos Google Translate (gratuito)")
        
        # Mostrar estado actual
        api_keys = self.config_manager.get_api_keys() if self.config_manager else {}
        
        print(f"\n📊 Estado actual:")
        print(f"• OpenAI: {'✅ Configurado' if api_keys.get('openai') else '❌ No configurado'}")
        print(f"• DeepL: {'✅ Configurado' if api_keys.get('deepl') else '❌ No configurado'}")
        print(f"• Google: ✅ Siempre disponible (gratuito)")
        
        # Configuración interactiva
        configure = input(f"\n¿Quieres configurar API keys? (s/n): ").lower().strip()
        
        if configure in ['s', 'si', 'sí', 'y', 'yes']:
            self._interactive_api_setup()
        
        # Validar entorno
        print(f"\n🔧 VALIDANDO ENTORNO")
        print("-" * 25)
        validation_result = validate_environment_setup()
        
        if validation_result.is_valid:
            print("✅ Entorno configurado correctamente")
        else:
            print("⚠️  Problemas encontrados:")
            for error in validation_result.errors:
                print(f"   • {error}")
            
            if validation_result.warnings:
                print("💡 Advertencias:")
                for warning in validation_result.warnings:
                    print(f"   • {warning}")
        
        print(f"\n🎉 Configuración completada")
        print("💡 Ya puedes usar: python main.py translate <video>")
    
    def _interactive_api_setup(self):
        """Configuración interactiva de API keys"""
        print(f"\n🔐 Configuración de API Keys")
        print("(Presiona Enter para omitir)")
        
        # OpenAI
        current_openai = "***configurado***" if self.config_manager.get_config('openai_api_key') else "no configurado"
        print(f"\nOpenAI API Key (actual: {current_openai})")
        print("📖 Obtener en: https://platform.openai.com/api-keys")
        openai_key = input("Nueva OpenAI API Key: ").strip()
        
        if openai_key:
            try:
                self.config_manager.set_config('openai_api_key', openai_key)
                print("✅ OpenAI API Key guardada")
            except Exception as e:
                print(f"❌ Error guardando OpenAI key: {e}")
        
        # DeepL
        current_deepl = "***configurado***" if self.config_manager.get_config('deepl_api_key') else "no configurado"
        print(f"\nDeepL API Key (actual: {current_deepl})")
        print("📖 Obtener en: https://www.deepl.com/pro-api")
        deepl_key = input("Nueva DeepL API Key: ").strip()
        
        if deepl_key:
            try:
                self.config_manager.set_config('deepl_api_key', deepl_key)
                print("✅ DeepL API Key guardada")
            except Exception as e:
                print(f"❌ Error guardando DeepL key: {e}")
        
        # Configuraciones adicionales
        print(f"\n⚙️  CONFIGURACIONES ADICIONALES")
        
        voice_pref = input("Género de voz preferido (Male/Female) [Female]: ").strip()
        if voice_pref:
            self.config_manager.set_config('default_voice_gender', voice_pref)
        
        quality_pref = input("Calidad por defecto (fast/balanced/quality) [balanced]: ").strip()
        if quality_pref:
            self.config_manager.set_config('default_quality_preset', quality_pref)
    
    def _handle_status_command(self, args):
        """Maneja comando de estado"""
        print(f"\n📊 ESTADO DEL SISTEMA")
        print("=" * 40)
        
        # Estado de configuración
        if self.config_manager:
            summary = self.config_manager.get_config_summary()
            
            print("🔧 CONFIGURACIÓN:")
            config_status = summary['config_status']
            print(f"• Configuración cargada: {'✅' if config_status['environment_loaded'] else '❌'}")
            print(f"• Items configurados: {config_status['user_config_items']}")
            print(f"• Validación: {'✅ Válida' if config_status['validation_status'] else '❌ Inválida'}")
            
            print(f"\n🔑 API KEYS:")
            api_status = summary['api_keys_configured']
            print(f"• OpenAI: {'✅ Configurado' if api_status['openai'] else '❌ No configurado'}")
            print(f"• DeepL: {'✅ Configurado' if api_status['deepl'] else '❌ No configurado'}")
            print(f"• Google: ✅ Siempre disponible")
            
            print(f"\n📁 DIRECTORIOS:")
            dirs = summary['directories']
            for dir_type, path in dirs.items():
                exists = "✅" if os.path.exists(path) else "❌"
                print(f"• {dir_type.title()}: {exists} {path}")
            
            print(f"\n🎛️ CARACTERÍSTICAS:")
            features = summary['enabled_features']
            for feature, enabled in features.items():
                status = "✅" if enabled else "❌"
                print(f"• {feature.replace('_', ' ').title()}: {status}")
        
        # Estado de dependencias
        print(f"\n📦 DEPENDENCIAS:")
        deps = self._check_dependencies()
        for dep, status in deps.items():
            icon = "✅" if status else "❌"
            print(f"• {dep}: {icon}")
        
        # Estado de voces TTS
        print(f"\n🎤 VOCES TTS:")
        try:
            from video_processing.tts_generators import get_available_spanish_voices
            voices = get_available_spanish_voices()
            print(f"• Voces disponibles: {len(voices)}")
            
            # Contar por género
            male_count = len([v for v in voices if v.gender == 'Male'])
            female_count = len([v for v in voices if v.gender == 'Female'])
            print(f"• Masculinas: {male_count}")
            print(f"• Femeninas: {female_count}")
            
            # Proveedores
            providers = list(set(v.provider for v in voices))
            print(f"• Proveedores: {', '.join(providers)}")
            
        except Exception as e:
            print(f"• ❌ Error obteniendo voces: {e}")
        
        if args.detailed:
            self._show_detailed_status()
    
    def _check_dependencies(self):
        """Verifica estado de dependencias"""
        deps = {}
        
        # FFmpeg
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            deps['FFmpeg'] = True
        except:
            deps['FFmpeg'] = False
        
        # Python packages
        packages = [
            ('faster-whisper', 'faster_whisper'),
            ('edge-tts', 'edge_tts'),
            ('googletrans', 'googletrans'),
            ('yt-dlp', 'yt_dlp'),
            ('moviepy', 'moviepy')
        ]
        
        for name, module in packages:
            try:
                __import__(module)
                deps[name] = True
            except ImportError:
                deps[name] = False
        
        return deps
    
    def _show_detailed_status(self):
        """Muestra estado detallado"""
        print(f"\n🔍 INFORMACIÓN DETALLADA")
        print("=" * 40)
        
        # Información del sistema
        print(f"🖥️  Sistema: {os.name}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print(f"📁 Directorio de trabajo: {os.getcwd()}")
        
        # Información de memoria y disco
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            print(f"💾 RAM disponible: {memory.available / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB")
            print(f"💿 Disco disponible: {disk.free / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB")
        except ImportError:
            print("💾 psutil no disponible para información de sistema")
    
    def _handle_voices_command(self, args):
        """Maneja comando de voces"""
        print(f"\n🎤 VOCES TTS DISPONIBLES")
        print("=" * 40)
        
        try:
            from video_processing.tts_generators import get_available_spanish_voices
            from video_processing.models import SpanishVoiceFilter
            
            voices = get_available_spanish_voices()
            
            # Aplicar filtros
            if args.filter_gender or args.filter_locale:
                filter_criteria = SpanishVoiceFilter(
                    gender=args.filter_gender,
                    locale=args.filter_locale
                )
                voices = [v for v in voices if filter_criteria.matches(v)]
            
            if not voices:
                print("❌ No se encontraron voces con los filtros especificados")
                return
            
            # Agrupar por locale
            voices_by_locale = {}
            for voice in voices:
                if voice.locale not in voices_by_locale:
                    voices_by_locale[voice.locale] = []
                voices_by_locale[voice.locale].append(voice)
            
            # Mostrar voces agrupadas
            for locale, locale_voices in sorted(voices_by_locale.items()):
                print(f"\n🌍 {locale}:")
                
                # Separar por género
                male_voices = [v for v in locale_voices if v.gender == 'Male']
                female_voices = [v for v in locale_voices if v.gender == 'Female']
                
                if male_voices:
                    print("  👨 Masculinas:")
                    for voice in male_voices:
                        quality = "🔥 Neural" if voice.is_neural else "⚡ Estándar"
                        print(f"    • {voice.name} ({voice.id}) - {quality}")
                
                if female_voices:
                    print("  👩 Femeninas:")
                    for voice in female_voices:
                        quality = "🔥 Neural" if voice.is_neural else "⚡ Estándar"
                        print(f"    • {voice.name} ({voice.id}) - {quality}")
            
            print(f"\n📊 RESUMEN:")
            print(f"• Total de voces: {len(voices)}")
            print(f"• Locales disponibles: {len(voices_by_locale)}")
            print(f"• Proveedores: {', '.join(set(v.provider for v in voices))}")
            
        except Exception as e:
            print(f"❌ Error obteniendo voces: {e}")
    
    def _handle_validate_command(self, args):
        """Maneja comando de validación"""
        print(f"\n✅ VALIDACIÓN DEL ENTORNO")
        print("=" * 40)
        
        validation_result = validate_environment_setup()
        
        if validation_result.is_valid:
            print("🎉 ¡Entorno completamente válido!")
            print("✅ Todas las dependencias están correctamente instaladas")
            print("✅ Configuración es válida")
            print("✅ Directorios tienen permisos correctos")
        else:
            print("⚠️  Se encontraron problemas:")
            
            for error in validation_result.errors:
                print(f"❌ {error}")
            
            for warning in validation_result.warnings:
                print(f"⚠️  {warning}")
            
            if args.fix:
                print(f"\n🔧 Intentando corregir problemas...")
                self._attempt_fixes()
            else:
                print(f"\n💡 Para corregir automáticamente: python main.py validate --fix")
        
        # Validación adicional específica
        print(f"\n🔍 VALIDACIÓN DETALLADA:")
        
        # Validar que podemos crear servicios
        try:
            services = CompleteVideoTranslationFactory.create_complete_translator_pipeline()
            print("✅ Factory de servicios funciona correctamente")
        except Exception as e:
            print(f"❌ Error creando servicios: {e}")
        
        # Validar transcripción
        try:
            from video_processing.factories import TranscriptionFactory
            transcription_service = TranscriptionFactory.create_whisper_transcription_service()
            print("✅ Servicio de transcripción disponible")
        except Exception as e:
            print(f"❌ Error en transcripción: {e}")
        
        # Validar TTS
        try:
            from video_processing.tts_generators import get_available_spanish_voices
            voices = get_available_spanish_voices()
            if voices:
                print(f"✅ TTS disponible ({len(voices)} voces)")
            else:
                print("❌ No hay voces TTS disponibles")
        except Exception as e:
            print(f"❌ Error en TTS: {e}")
    
    def _attempt_fixes(self):
        """Intenta corregir problemas automáticamente"""
        print("🔧 Implementando correcciones automáticas...")
        
        # Crear directorios faltantes
        try:
            if self.config_manager:
                dirs = self.config_manager.get_directory_config()
                for dir_type, path in dirs.items():
                    os.makedirs(path, exist_ok=True)
                print("✅ Directorios creados/verificados")
        except Exception as e:
            print(f"❌ Error creando directorios: {e}")
        
        # Crear archivos de configuración
        try:
            if self.config_manager:
                self.config_manager.create_default_config_files()
                print("✅ Archivos de configuración creados")
        except Exception as e:
            print(f"❌ Error creando configuraciones: {e}")
        
        print("🎉 Correcciones aplicadas")


def main():
    """Función principal de entrada"""
    app = VideoTranslatorApp()
    app.main()


if __name__ == "__main__":
    main()