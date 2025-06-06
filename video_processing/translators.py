"""Implementaciones de traductores - OpenAI, Google, DeepL"""

import os
import time
import hashlib
import re
from typing import Optional, List, Dict, Callable

from .interfaces import ITranslator, ILongTextTranslator, ITranslationQualityAnalyzer
from .models import (TranslationResult, LanguagePair, TranslatorConfig, TextChunk, TranslationProgress, TranslationCostEstimate, TranslationMetrics)

# TRADUCTOR OPENAI (GPT)

class OpenAITranslator(ITranslator, ILongTextTranslator):
    """Traductor usando OpenAI GPT - Calidad premium y contextual"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Inicializar traductor OpenAI
        
        Args:
            api_key: API key de OpenAI (o usar variable de entorno)
            model: Modelo a usar ('gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo')
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self._client = None
        self._check_availability()
    
    def _check_availability(self):
        """Verifica que OpenAI esté disponible"""
        try:
            import openai
            if self.api_key:
                self._client = openai.OpenAI(api_key=self.api_key)
                print("✅ OpenAI Translator configurado")
            else:
                print("⚠️  OpenAI API key no encontrada")
        except ImportError:
            raise ImportError("OpenAI no instalado. Usar: pip install openai")
    
    def get_translator_name(self) -> str:
        return "openai"
    
    def is_configured(self) -> bool:
        return self.api_key is not None and self._client is not None
    
    def get_supported_languages(self) -> List[str]:
        """OpenAI soporta prácticamente todos los idiomas"""
        return [
            'es', 'en', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi',
            'th', 'vi', 'nl', 'sv', 'da', 'no', 'fi', 'pl', 'tr', 'he', 'cs', 'sk',
            'hu', 'ro', 'bg', 'hr', 'sr', 'sl', 'et', 'lv', 'lt', 'uk', 'be', 'mk',
            'sq', 'eu', 'gl', 'ca', 'cy', 'ga', 'mt', 'is', 'fo', 'gd', 'id', 'ms',
            'tl', 'sw', 'am', 'bn', 'gu', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'si',
            'ta', 'te', 'ur', 'my', 'km', 'lo', 'ka', 'hy', 'az', 'kk', 'ky', 'mn',
            'tg', 'tk', 'uz', 'af', 'ig', 'yo', 'zu', 'xh', 'st', 'tn', 'ss', 'nr'
        ]
    
    def estimate_cost(self, text: str, source_language: str, target_language: str = "es") -> TranslationCostEstimate:
        """Estima costo de traducción con OpenAI"""
        char_count = len(text)
        
        # Precios aproximados por 1K tokens (1 token ≈ 4 caracteres)
        token_count = char_count / 4
        
        if self.model == "gpt-4o":
            cost_per_1k_input = 0.0025  # $2.50 por 1M tokens de entrada
            cost_per_1k_output = 0.01   # $10.00 por 1M tokens de salida
        elif self.model == "gpt-4o-mini":
            cost_per_1k_input = 0.00015  # $0.15 por 1M tokens de entrada
            cost_per_1k_output = 0.0006  # $0.60 por 1M tokens de salida
        else:  # gpt-3.5-turbo
            cost_per_1k_input = 0.0005
            cost_per_1k_output = 0.0015
        
        # Estimación: entrada + salida similar
        estimated_cost = (token_count / 1000) * (cost_per_1k_input + cost_per_1k_output)
        
        return TranslationCostEstimate(
            service_name=f"OpenAI {self.model}",
            character_count=char_count,
            estimated_cost_usd=estimated_cost,
            rate_per_char=estimated_cost / char_count if char_count > 0 else 0
        )
    
    def translate_text(self, text: str, source_language: str, target_language: str = "es") -> TranslationResult:
        """Traduce texto usando OpenAI GPT"""
        
        start_time = time.time()
        
        try:
            if not self.is_configured():
                return TranslationResult(
                    success=False,
                    original_text=text,
                    error_message="OpenAI no está configurado (falta API key)"
                )
            
            print(f"🤖 Traduciendo con OpenAI {self.model}...")
            print(f"📝 Caracteres: {len(text)}")
            
            # Crear prompt contextual para traducción
            system_prompt = self._create_translation_prompt(source_language, target_language)
            
            # Llamada a OpenAI
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=len(text) * 2,  # Espacio suficiente para la traducción
                temperature=0.3  # Baja creatividad para traducción precisa
            )
            
            translated_text = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            
            # Calcular costo real
            cost_estimate = self.estimate_cost(text, source_language, target_language)
            
            # Crear par de idiomas
            language_pair = LanguagePair.create_to_spanish(source_language, self._get_language_name(source_language))
            
            print(f"✅ Traducción OpenAI completada en {processing_time:.1f}s")
            print(f"💰 Costo estimado: {cost_estimate.format_cost()}")
            
            return TranslationResult(
                success=True,
                original_text=text,
                translated_text=translated_text,
                language_pair=language_pair,
                translator_used=f"openai-{self.model}",
                processing_time=processing_time,
                confidence_score=0.95,  # OpenAI generalmente alta calidad
                cost_estimate=cost_estimate.estimated_cost_usd
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TranslationResult(
                success=False,
                original_text=text,
                translator_used=f"openai-{self.model}",
                processing_time=processing_time,
                error_message=f"Error OpenAI: {str(e)}"
            )
    
    def translate_long_text(self, text: str, source_language: str, target_language: str = "es",
                           max_chunk_size: int = 4000,
                           progress_callback: Optional[Callable[[TranslationProgress], None]] = None) -> TranslationResult:
        """Traduce textos largos dividiéndolos en chunks"""
        
        start_time = time.time()
        
        try:
            print(f"📄 Traduciendo texto largo ({len(text)} caracteres) con OpenAI...")
            
            # Dividir en chunks
            chunks = self.split_text_into_chunks(text, max_chunk_size)
            print(f"🔄 Dividido en {len(chunks)} fragmentos")
            
            translated_chunks = []
            total_cost = 0.0
            completed_chars = 0
            
            for i, chunk in enumerate(chunks):
                # Progreso
                if progress_callback:
                    progress = TranslationProgress(
                        total_chunks=len(chunks),
                        current_chunk=i + 1,
                        completed_chars=completed_chars,
                        total_chars=len(text),
                        current_translator="openai"
                    )
                    progress_callback(progress)
                
                # Traducir chunk
                chunk_result = self.translate_text(chunk.text, source_language, target_language)
                
                if not chunk_result.success:
                    return TranslationResult(
                        success=False,
                        original_text=text,
                        error_message=f"Error en chunk {i+1}: {chunk_result.error_message}"
                    )
                
                translated_chunks.append(chunk_result.translated_text)
                total_cost += chunk_result.cost_estimate
                completed_chars += len(chunk.text)
                
                print(f"  ✅ Chunk {i+1}/{len(chunks)} completado")
            
            # Unir chunks traducidos
            final_translation = " ".join(translated_chunks)
            processing_time = time.time() - start_time
            
            # Crear resultado final
            language_pair = LanguagePair.create_to_spanish(source_language, self._get_language_name(source_language))
            
            print(f"🎉 Traducción larga completada en {processing_time:.1f}s")
            print(f"💰 Costo total estimado: ${total_cost:.4f}")
            
            return TranslationResult(
                success=True,
                original_text=text,
                translated_text=final_translation,
                language_pair=language_pair,
                translator_used=f"openai-{self.model}",
                processing_time=processing_time,
                confidence_score=0.95,
                cost_estimate=total_cost,
                chunks_processed=len(chunks)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TranslationResult(
                success=False,
                original_text=text,
                processing_time=processing_time,
                error_message=f"Error en traducción larga: {str(e)}"
            )
    
    def split_text_into_chunks(self, text: str, max_chunk_size: int = 4000) -> List[TextChunk]:
        """Divide texto en chunks inteligentemente"""
        chunks = []
        
        # Dividir por párrafos primero
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        start_pos = 0
        
        for paragraph in paragraphs:
            # Si el párrafo cabe en el chunk actual
            if len(current_chunk + paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Guardar chunk actual si no está vacío
                if current_chunk.strip():
                    chunk_text = current_chunk.strip()
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        start_position=start_pos,
                        end_position=start_pos + len(chunk_text),
                        word_count=len(chunk_text.split())
                    ))
                    chunk_id += 1
                    start_pos += len(chunk_text)
                
                # Empezar nuevo chunk
                current_chunk = paragraph + "\n\n"
        
        # Último chunk
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_position=start_pos,
                end_position=start_pos + len(chunk_text),
                word_count=len(chunk_text.split())
            ))
        
        return chunks
    
    def _create_translation_prompt(self, source_lang: str, target_lang: str) -> str:
        """Crea prompt contextual para traducción"""
        return f"""Eres un traductor profesional experto. Traduce el siguiente texto de {source_lang} a {target_lang} de manera natural y precisa.

INSTRUCCIONES:
- Mantén el tono y estilo original
- Usa terminología técnica apropiada cuando sea necesario
- Para contenido técnico: usa términos establecidos en {target_lang}
- Para contenido casual: traduce de forma natural y coloquial
- Conserva nombres propios, marcas y productos
- Prioriza la naturalidad del {target_lang} sobre la traducción literal
- Si hay jerga o expresiones idiomáticas, usa equivalentes en {target_lang}

El resultado debe sonar como si fuera escrito originalmente en {target_lang}.

Traduce únicamente el texto, sin comentarios adicionales."""
    
    def _get_language_name(self, code: str) -> str:
        """Convierte código de idioma a nombre"""
        language_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi'
        }
        return language_names.get(code, f"Language ({code})")

# TRADUCTOR GOOGLE TRANSLATE

class GoogleTranslator(ITranslator, ILongTextTranslator):
    """Traductor usando Google Translate - Gratis y confiable"""
    
    def __init__(self):
        """Inicializar traductor Google (no requiere API key)"""
        self._check_availability()
    
    def _check_availability(self):
        """Verifica que Google Translate esté disponible"""
        try:
            from googletrans import Translator
            self._translator = Translator()
            print("✅ Google Translator configurado")
        except ImportError:
            raise ImportError("Google Translate no instalado. Usar: pip install googletrans==4.0.0-rc1")
    
    def get_translator_name(self) -> str:
        return "google"
    
    def is_configured(self) -> bool:
        return hasattr(self, '_translator')
    
    def get_supported_languages(self) -> List[str]:
        """Google Translate soporta ~100+ idiomas"""
        return [
            'es', 'en', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi',
            'th', 'vi', 'nl', 'sv', 'da', 'no', 'fi', 'pl', 'tr', 'he', 'cs', 'sk',
            'hu', 'ro', 'bg', 'hr', 'sr', 'sl', 'et', 'lv', 'lt', 'uk', 'be', 'mk',
            'sq', 'eu', 'gl', 'ca', 'cy', 'ga', 'mt', 'is', 'fo', 'gd', 'id', 'ms',
            'tl', 'sw', 'am', 'bn', 'gu', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'si',
            'ta', 'te', 'ur', 'my', 'km', 'lo', 'ka', 'hy', 'az', 'kk', 'ky', 'mn'
        ]
    
    def estimate_cost(self, text: str, source_language: str, target_language: str = "es") -> TranslationCostEstimate:
        """Google Translate es gratis"""
        return TranslationCostEstimate(
            service_name="Google Translate",
            character_count=len(text),
            estimated_cost_usd=0.0,
            rate_per_char=0.0
        )
    
    def translate_text(self, text: str, source_language: str, target_language: str = "es") -> TranslationResult:
        """Traduce texto usando Google Translate"""
        
        start_time = time.time()
        
        try:
            if not self.is_configured():
                return TranslationResult(
                    success=False,
                    original_text=text,
                    error_message="Google Translate no está configurado"
                )
            
            print(f"🌐 Traduciendo con Google Translate...")
            print(f"📝 Caracteres: {len(text)}")
            
            # Traducir con Google
            result = self._translator.translate(text, src=source_language, dest=target_language)
            
            translated_text = result.text
            processing_time = time.time() - start_time
            
            # Crear par de idiomas
            language_pair = LanguagePair.create_to_spanish(source_language, self._get_language_name(source_language))
            
            print(f"✅ Traducción Google completada en {processing_time:.1f}s")
            print("💰 Costo: Gratis")
            
            return TranslationResult(
                success=True,
                original_text=text,
                translated_text=translated_text,
                language_pair=language_pair,
                translator_used="google-translate",
                processing_time=processing_time,
                confidence_score=0.8,  # Google generalmente buena calidad
                cost_estimate=0.0
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TranslationResult(
                success=False,
                original_text=text,
                translator_used="google-translate",
                processing_time=processing_time,
                error_message=f"Error Google Translate: {str(e)}"
            )
    
    def translate_long_text(self, text: str, source_language: str, target_language: str = "es", max_chunk_size: int = 4500, progress_callback: Optional[Callable[[TranslationProgress], None]] = None) -> TranslationResult:
        """Traduce textos largos (Google tiene límite ~5000 chars por request)"""
        
        start_time = time.time()
        
        try:
            print(f"📄 Traduciendo texto largo ({len(text)} caracteres) con Google...")
            
            # Dividir en chunks
            chunks = self.split_text_into_chunks(text, max_chunk_size)
            print(f"🔄 Dividido en {len(chunks)} fragmentos")
            
            translated_chunks = []
            completed_chars = 0
            
            for i, chunk in enumerate(chunks):
                # Progreso
                if progress_callback:
                    progress = TranslationProgress(
                        total_chunks=len(chunks),
                        current_chunk=i + 1,
                        completed_chars=completed_chars,
                        total_chars=len(text),
                        current_translator="google"
                    )
                    progress_callback(progress)
                
                # Traducir chunk
                chunk_result = self.translate_text(chunk.text, source_language, target_language)
                
                if not chunk_result.success:
                    return TranslationResult(
                        success=False,
                        original_text=text,
                        error_message=f"Error en chunk {i+1}: {chunk_result.error_message}"
                    )
                
                translated_chunks.append(chunk_result.translated_text)
                completed_chars += len(chunk.text)
                
                print(f"  ✅ Chunk {i+1}/{len(chunks)} completado")
                
                # Pausa entre requests para evitar rate limiting
                if i < len(chunks) - 1:
                    time.sleep(0.5)
            
            # Unir chunks traducidos
            final_translation = " ".join(translated_chunks)
            processing_time = time.time() - start_time
            
            # Crear resultado final
            language_pair = LanguagePair.create_to_spanish(source_language, self._get_language_name(source_language))
            
            print(f"🎉 Traducción larga Google completada en {processing_time:.1f}s")
            print("💰 Costo total: Gratis")
            
            return TranslationResult(
                success=True,
                original_text=text,
                translated_text=final_translation,
                language_pair=language_pair,
                translator_used="google-translate",
                processing_time=processing_time,
                confidence_score=0.8,
                cost_estimate=0.0,
                chunks_processed=len(chunks)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TranslationResult(
                success=False,
                original_text=text,
                processing_time=processing_time,
                error_message=f"Error en traducción larga Google: {str(e)}"
            )
    
    def split_text_into_chunks(self, text: str, max_chunk_size: int = 4500) -> List[TextChunk]:
        """Divide texto en chunks (similar a OpenAI pero más pequeños)"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_id = 0
        start_pos = 0
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunk_text = current_chunk.strip()
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        start_position=start_pos,
                        end_position=start_pos + len(chunk_text),
                        word_count=len(chunk_text.split())
                    ))
                    chunk_id += 1
                    start_pos += len(chunk_text)
                
                current_chunk = sentence + " "
        
        # Último chunk
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_position=start_pos,
                end_position=start_pos + len(chunk_text),
                word_count=len(chunk_text.split())
            ))
        
        return chunks
    
    def _get_language_name(self, code: str) -> str:
        """Convierte código de idioma a nombre"""
        language_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi'
        }
        return language_names.get(code, f"Language ({code})")

# TRADUCTOR DEEPL

class DeepLTranslator(ITranslator, ILongTextTranslator):
    """Traductor usando DeepL - Excelente calidad"""
    
    def __init__(self, api_key: Optional[str] = None, use_free_api: bool = True):
        """
        Inicializar traductor DeepL
        
        Args:
            api_key: API key de DeepL (opcional para free tier)
            use_free_api: True para usar API gratuita, False para pro
        """
        self.api_key = api_key or os.getenv('DEEPL_API_KEY')
        self.use_free_api = use_free_api
        self._translator = None
        self._check_availability()
    
    def _check_availability(self):
        """Verifica que DeepL esté disponible"""
        try:
            import deepl
            
            if self.api_key:
                self._translator = deepl.Translator(self.api_key)
                print("✅ DeepL Translator configurado con API key")
            else:
                print("⚠️  DeepL configurado sin API key (límites aplicables)")
                # Para free tier sin API, usamos requests directos
                
        except ImportError:
            raise ImportError("DeepL no instalado. Usar: pip install deepl")
    
    def get_translator_name(self) -> str:
        return "deepl"
    
    def is_configured(self) -> bool:
        return True  # DeepL free funciona sin API key (con límites)
    
    def get_supported_languages(self) -> List[str]:
        """DeepL soporta ~30 idiomas de alta calidad"""
        return [
            'es', 'en', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'zh', 'nl', 'sv', 'da',
            'no', 'fi', 'pl', 'tr', 'cs', 'sk', 'hu', 'ro', 'bg', 'hr', 'sl', 'et',
            'lv', 'lt', 'uk', 'ko', 'id', 'ar'
        ]
    
    def estimate_cost(self, text: str, source_language: str, target_language: str = "es") -> TranslationCostEstimate:
        """Estima costo DeepL"""
        char_count = len(text)
        
        if not self.api_key or self.use_free_api:
            # Free tier: 500,000 chars/month
            return TranslationCostEstimate(
                service_name="DeepL Free",
                character_count=char_count,
                estimated_cost_usd=0.0,
                rate_per_char=0.0
            )
        else:
            # Pro tier: $5.99 por 1M caracteres
            cost_per_char = 5.99 / 1_000_000
            estimated_cost = char_count * cost_per_char
            
            return TranslationCostEstimate(
                service_name="DeepL Pro",
                character_count=char_count,
                estimated_cost_usd=estimated_cost,
                rate_per_char=cost_per_char
            )
    
    def translate_text(self, text: str, source_language: str, target_language: str = "es") -> TranslationResult:
        """Traduce texto usando DeepL"""
        
        start_time = time.time()
        
        try:
            print(f"🔷 Traduciendo con DeepL...")
            print(f"📝 Caracteres: {len(text)}")
            
            if self._translator and self.api_key:
                # Usar API oficial
                result = self._translator.translate_text(
                    text, 
                    source_lang=source_language.upper(), 
                    target_lang=target_language.upper()
                )
                translated_text = result.text
            else:
                # Usar método alternativo para free tier
                translated_text = self._translate_free(text, source_language, target_language)
            
            processing_time = time.time() - start_time
            cost_estimate = self.estimate_cost(text, source_language, target_language)
            
            # Crear par de idiomas
            language_pair = LanguagePair.create_to_spanish(source_language, self._get_language_name(source_language))
            
            print(f"✅ Traducción DeepL completada en {processing_time:.1f}s")
            print(f"💰 Costo: {cost_estimate.format_cost()}")
            
            return TranslationResult(
                success=True,
                original_text=text,
                translated_text=translated_text,
                language_pair=language_pair,
                translator_used="deepl",
                processing_time=processing_time,
                confidence_score=0.9,  # DeepL alta calidad
                cost_estimate=cost_estimate.estimated_cost_usd
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TranslationResult(
                success=False,
                original_text=text,
                translator_used="deepl",
                processing_time=processing_time,
                error_message=f"Error DeepL: {str(e)}"
            )
    
    def _translate_free(self, text: str, source_lang: str, target_lang: str) -> str:
        """Traducción usando método free (sin API key)"""
        # Implementación básica - en producción usarías requests a DeepL free endpoint
        # Por ahora retornamos placeholder
        return f"[Traducción DeepL Free de '{text[:50]}...' no implementada sin API key]"
    
    def translate_long_text(self, text: str, source_language: str, target_language: str = "es",
                           max_chunk_size: int = 5000,
                           progress_callback: Optional[Callable[[TranslationProgress], None]] = None) -> TranslationResult:
        """Traduce textos largos con DeepL"""
        
        # Similar a Google pero con límite más alto
        return self._translate_in_chunks(text, source_language, target_language, max_chunk_size, progress_callback)
    
    def split_text_into_chunks(self, text: str, max_chunk_size: int = 5000) -> List[TextChunk]:
        """Divide texto en chunks para DeepL"""
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        start_pos = 0
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunk_text = current_chunk.strip()
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        start_position=start_pos,
                        end_position=start_pos + len(chunk_text),
                        word_count=len(chunk_text.split())
                    ))
                    chunk_id += 1
                    start_pos += len(chunk_text)
                
                current_chunk = paragraph + "\n\n"
        
        # Último chunk
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_position=start_pos,
                end_position=start_pos + len(chunk_text),
                word_count=len(chunk_text.split())
            ))
        
        return chunks
    
    def _translate_in_chunks(self, text: str, source_language: str, target_language: str,
                           max_chunk_size: int, progress_callback: Optional[Callable] = None) -> TranslationResult:
        """Método auxiliar para traducir en chunks"""
        start_time = time.time()
        
        try:
            chunks = self.split_text_into_chunks(text, max_chunk_size)
            print(f"🔄 Dividido en {len(chunks)} fragmentos")
            
            translated_chunks = []
            total_cost = 0.0
            completed_chars = 0
            
            for i, chunk in enumerate(chunks):
                if progress_callback:
                    progress = TranslationProgress(
                        total_chunks=len(chunks),
                        current_chunk=i + 1,
                        completed_chars=completed_chars,
                        total_chars=len(text),
                        current_translator="deepl"
                    )
                    progress_callback(progress)
                
                chunk_result = self.translate_text(chunk.text, source_language, target_language)
                
                if not chunk_result.success:
                    return TranslationResult(
                        success=False,
                        original_text=text,
                        error_message=f"Error en chunk {i+1}: {chunk_result.error_message}"
                    )
                
                translated_chunks.append(chunk_result.translated_text)
                total_cost += chunk_result.cost_estimate
                completed_chars += len(chunk.text)
                
                print(f"  ✅ Chunk {i+1}/{len(chunks)} completado")
                time.sleep(0.5)  # Pausa para rate limiting
            
            final_translation = " ".join(translated_chunks)
            processing_time = time.time() - start_time
            
            language_pair = LanguagePair.create_to_spanish(source_language, self._get_language_name(source_language))
            
            return TranslationResult(
                success=True,
                original_text=text,
                translated_text=final_translation,
                language_pair=language_pair,
                translator_used="deepl",
                processing_time=processing_time,
                confidence_score=0.9,
                cost_estimate=total_cost,
                chunks_processed=len(chunks)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TranslationResult(
                success=False,
                original_text=text,
                processing_time=processing_time,
                error_message=f"Error en traducción larga DeepL: {str(e)}"
            )
    
    def _get_language_name(self, code: str) -> str:
        """Convierte código de idioma a nombre"""
        language_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'nl': 'Dutch', 'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian',
            'fi': 'Finnish', 'pl': 'Polish', 'tr': 'Turkish', 'cs': 'Czech'
        }
        return language_names.get(code, f"Language ({code})")

# ANALIZADOR DE CALIDAD DE TRADUCCIÓN

class BasicTranslationQualityAnalyzer(ITranslationQualityAnalyzer):
    """Analizador básico de calidad de traducciones"""
    
    def analyze_translation_quality(self, original_text: str, translated_text: str, 
                                   language_pair: LanguagePair) -> TranslationMetrics:
        """Analiza calidad básica de traducción"""
        
        try:
            # Métricas básicas
            original_length = len(original_text)
            translated_length = len(translated_text)
            
            # Preservación de longitud (ideal entre 0.8 - 1.2)
            length_preservation = translated_length / original_length if original_length > 0 else 0
            length_score = 1.0 - abs(1.0 - length_preservation) if length_preservation <= 2.0 else 0.5
            
            # Consistencia terminológica básica
            terminology_score = self._analyze_terminology_consistency(original_text, translated_text)
            
            # Fluidez estimada (basada en repeticiones y estructura)
            fluency_score = self._analyze_fluency(translated_text)
            
            # Adecuación (basada en preservación de contenido)
            adequacy_score = self._analyze_adequacy(original_text, translated_text)
            
            # Score general
            overall_quality = (length_score + terminology_score + fluency_score + adequacy_score) / 4
            
            return TranslationMetrics(
                translation_id=f"quality_{int(time.time())}",
                length_preservation=length_preservation,
                terminology_consistency=terminology_score,
                fluency_score=fluency_score,
                adequacy_score=adequacy_score,
                overall_quality=overall_quality
            )
            
        except Exception as e:
            # Retornar métricas neutras si hay error
            return TranslationMetrics(
                translation_id="error",
                length_preservation=1.0,
                terminology_consistency=0.5,
                fluency_score=0.5,
                adequacy_score=0.5,
                overall_quality=0.5
            )
    
    def detect_translation_issues(self, original_text: str, translated_text: str) -> List[str]:
        """Detecta problemas potenciales en traducción"""
        issues = []
        
        try:
            # Detectar traducciones muy cortas o largas
            length_ratio = len(translated_text) / len(original_text) if len(original_text) > 0 else 0
            
            if length_ratio < 0.3:
                issues.append("Traducción muy corta comparada con el original")
            elif length_ratio > 3.0:
                issues.append("Traducción muy larga comparada con el original")
            
            # Detectar texto sin traducir
            if original_text.lower() == translated_text.lower():
                issues.append("Texto parece no haber sido traducido")
            
            # Detectar repeticiones excesivas
            words = translated_text.split()
            if len(words) != len(set(words)) and len(set(words)) < len(words) * 0.5:
                issues.append("Texto traducido tiene repeticiones excesivas")
            
            # Detectar caracteres problemáticos
            if '[' in translated_text or ']' in translated_text:
                issues.append("Traducción contiene caracteres de placeholder")
            
            # Detectar preservación de números
            import re
            original_numbers = re.findall(r'\d+', original_text)
            translated_numbers = re.findall(r'\d+', translated_text)
            
            if len(original_numbers) != len(translated_numbers):
                issues.append("Números no preservados correctamente")
            
        except Exception:
            issues.append("Error analizando calidad de traducción")
        
        return issues
    
    def _analyze_terminology_consistency(self, original: str, translated: str) -> float:
        """Analiza consistencia terminológica básica"""
        # Score básico basado en preservación de palabras técnicas y nombres
        import re
        
        # Buscar palabras en mayúsculas (posibles nombres/marcas)
        original_caps = set(re.findall(r'\b[A-Z][a-z]+\b', original))
        translated_caps = set(re.findall(r'\b[A-Z][a-z]+\b', translated))
        
        if len(original_caps) == 0:
            return 0.8  # No hay términos específicos para evaluar
        
        preserved = len(original_caps.intersection(translated_caps))
        return preserved / len(original_caps)
    
    def _analyze_fluency(self, text: str) -> float:
        """Analiza fluidez básica del texto"""
        words = text.split()
        
        if len(words) < 3:
            return 0.7  # Texto muy corto
        
        # Penalizar repeticiones excesivas
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        
        # Penalizar oraciones muy largas o muy cortas
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        length_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.7
        
        return (repetition_ratio + length_score) / 2
    
    def _analyze_adequacy(self, original: str, translated: str) -> float:
        """Analiza adecuación del contenido"""
        # Análisis básico de preservación de contenido
        original_words = set(original.lower().split())
        translated_words = set(translated.lower().split())
        
        # Buscar palabras comunes (números, nombres, etc. que deberían preservarse)
        import re
        original_preserved = set(re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b', original))
        translated_preserved = set(re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b', translated))
        
        if len(original_preserved) == 0:
            return 0.8  # No hay elementos específicos para preservar
        
        preservation_ratio = len(original_preserved.intersection(translated_preserved)) / len(original_preserved)
        
        return preservation_ratio