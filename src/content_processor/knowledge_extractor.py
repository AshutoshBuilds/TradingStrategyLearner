import spacy
from typing import List, Dict, Any, Optional
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class KnowledgeExtractor:
    def __init__(self, model_path: str = "en_core_web_sm",
                 sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the knowledge extractor.
        
        Args:
            model_path (str): Path to spaCy model
            sentiment_model (str): HuggingFace sentiment model name
            embedding_model (str): SentenceTransformer model name
        """
        # Load spaCy model for NLP tasks
        self.nlp = spacy.load(model_path)
        
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer(embedding_model)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, Any]]: Extracted entities with type and position
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'start_token': ent.start,
                'end_token': ent.end
            })
            
        return entities

    def extract_key_phrases(self, text: str, min_length: int = 3) -> List[Dict[str, Any]]:
        """Extract key phrases from text.
        
        Args:
            text (str): Input text
            min_length (int): Minimum phrase length in tokens
            
        Returns:
            List[Dict[str, Any]]: Extracted key phrases
        """
        doc = self.nlp(text)
        phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk) >= min_length:
                phrases.append({
                    'text': chunk.text,
                    'root': chunk.root.text,
                    'root_dep': chunk.root.dep_,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                })
        
        # Extract verb phrases
        for token in doc:
            if token.pos_ == "VERB":
                phrase = ' '.join([t.text for t in token.subtree])
                if len(phrase.split()) >= min_length:
                    phrases.append({
                        'text': phrase,
                        'root': token.text,
                        'root_dep': token.dep_,
                        'start_char': token.idx,
                        'end_char': token.idx + len(token.text)
                    })
                    
        return phrases

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        result = self.sentiment_analyzer(text)[0]
        return {
            'label': result['label'],
            'score': result['score']
        }

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings for semantic analysis.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[List[float]]: Text embeddings
        """
        return self.embedding_model.encode(texts).tolist()

    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, Any]]: Extracted relationships
        """
        doc = self.nlp(text)
        relations = []
        
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                # Find the subject and its related verb
                subject = token.text
                verb = token.head.text
                
                # Find objects related to the verb
                objects = []
                for child in token.head.children:
                    if child.dep_ in ("dobj", "pobj"):
                        objects.append(child.text)
                
                if objects:
                    relations.append({
                        'subject': subject,
                        'verb': verb,
                        'objects': objects,
                        'sentence': token.sent.text
                    })
                    
        return relations

    def summarize_content(self, text: str, ratio: float = 0.3) -> Dict[str, Any]:
        """Generate a summary of the text content.
        
        Args:
            text (str): Input text
            ratio (float): Summary length ratio
            
        Returns:
            Dict[str, Any]: Text summary and metadata
        """
        doc = self.nlp(text)
        
        # Calculate sentence scores based on word importance
        scores = {}
        for sent in doc.sents:
            words = [token.text for token in sent if not token.is_stop and token.is_alpha]
            score = len(words) / len(sent)
            scores[sent] = score
            
        # Sort sentences by score
        sorted_sents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top sentences
        n_sents = max(1, int(len(sorted_sents) * ratio))
        summary_sents = sorted_sents[:n_sents]
        
        # Sort selected sentences by original order
        summary_sents = sorted(summary_sents, key=lambda x: text.find(x[0].text))
        
        return {
            'summary': ' '.join(sent.text for sent, _ in summary_sents),
            'sentence_count': len(summary_sents),
            'compression_ratio': n_sents / len(list(doc.sents))
        }

    def extract_topics(self, text: str, n_topics: int = 3) -> List[Dict[str, Any]]:
        """Extract main topics from text.
        
        Args:
            text (str): Input text
            n_topics (int): Number of topics to extract
            
        Returns:
            List[Dict[str, Any]]: Extracted topics with related terms
        """
        doc = self.nlp(text)
        
        # Count term frequencies
        term_freq = {}
        for token in doc:
            if not token.is_stop and token.is_alpha:
                term = token.lemma_.lower()
                term_freq[term] = term_freq.get(term, 0) + 1
                
        # Sort terms by frequency
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Group related terms
        topics = []
        used_terms = set()
        
        for term, freq in sorted_terms:
            if len(topics) >= n_topics:
                break
                
            if term in used_terms:
                continue
                
            # Find related terms
            related = []
            for other_term, other_freq in sorted_terms:
                if other_term != term and other_term not in used_terms:
                    # Check similarity using word vectors
                    similarity = doc.vocab[term].similarity(doc.vocab[other_term])
                    if similarity > 0.5:
                        related.append({
                            'term': other_term,
                            'frequency': other_freq,
                            'similarity': similarity
                        })
                        used_terms.add(other_term)
                        
            topics.append({
                'main_term': term,
                'frequency': freq,
                'related_terms': sorted(related, key=lambda x: x['similarity'], reverse=True)
            })
            used_terms.add(term)
            
        return topics 