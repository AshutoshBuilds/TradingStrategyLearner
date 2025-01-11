from typing import List, Dict, Any, Optional
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

class StrategyGenerator:
    def __init__(self, 
                 llm_model: str = "gpt2",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 strategy_templates_path: Optional[str] = None):
        """Initialize the strategy generator.
        
        Args:
            llm_model (str): HuggingFace model for text generation
            embedding_model (str): SentenceTransformer model name
            strategy_templates_path (Optional[str]): Path to strategy templates file
        """
        # Initialize text generation pipeline
        self.generator = pipeline("text-generation", model=llm_model)
        
        # Initialize sentence transformer for semantic similarity
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load strategy templates
        self.strategy_templates = self._load_templates(strategy_templates_path)
        
        # Initialize strategy evaluation metrics
        self._setup_evaluation_metrics()

    def _load_templates(self, templates_path: Optional[str]) -> List[Dict[str, Any]]:
        """Load strategy templates from file or use defaults.
        
        Args:
            templates_path (Optional[str]): Path to templates file
            
        Returns:
            List[Dict[str, Any]]: Strategy templates
        """
        if templates_path:
            # TODO: Implement template loading from file
            pass
            
        # Default templates
        return [
            {
                'name': 'trend_following',
                'description': 'Follow market trends based on momentum',
                'parameters': ['timeframe', 'threshold'],
                'conditions': ['price_trend', 'volume_trend']
            },
            {
                'name': 'mean_reversion',
                'description': 'Trade price reversions to the mean',
                'parameters': ['window_size', 'std_dev'],
                'conditions': ['price_deviation', 'volatility']
            },
            {
                'name': 'breakout',
                'description': 'Trade breakouts from price ranges',
                'parameters': ['range_period', 'breakout_threshold'],
                'conditions': ['price_range', 'volume_surge']
            }
        ]

    def _setup_evaluation_metrics(self):
        """Set up metrics for strategy evaluation."""
        self.evaluation_metrics = {
            'risk_reward': {
                'weight': 0.3,
                'threshold': 2.0
            },
            'win_rate': {
                'weight': 0.2,
                'threshold': 0.55
            },
            'sharpe_ratio': {
                'weight': 0.3,
                'threshold': 1.5
            },
            'max_drawdown': {
                'weight': 0.2,
                'threshold': -0.2
            }
        }

    def generate_strategy(self, market_data: Dict[str, Any],
                         extracted_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading strategy based on market data and extracted knowledge.
        
        Args:
            market_data (Dict[str, Any]): Market data and indicators
            extracted_knowledge (Dict[str, Any]): Extracted market insights
            
        Returns:
            Dict[str, Any]: Generated trading strategy
        """
        # Analyze market conditions
        market_conditions = self._analyze_market_conditions(market_data)
        
        # Find most relevant template
        template = self._select_strategy_template(market_conditions, extracted_knowledge)
        
        # Generate strategy parameters
        parameters = self._generate_parameters(template, market_conditions)
        
        # Generate entry/exit conditions
        conditions = self._generate_conditions(template, extracted_knowledge)
        
        # Evaluate strategy
        evaluation = self._evaluate_strategy(parameters, conditions, market_data)
        
        return {
            'template_name': template['name'],
            'description': template['description'],
            'parameters': parameters,
            'conditions': conditions,
            'evaluation': evaluation,
            'confidence_score': self._calculate_confidence(evaluation)
        }

    def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market conditions.
        
        Args:
            market_data (Dict[str, Any]): Market data and indicators
            
        Returns:
            Dict[str, Any]: Market condition analysis
        """
        # Extract key market features
        price_data = np.array(market_data.get('prices', []))
        volume_data = np.array(market_data.get('volumes', []))
        
        if len(price_data) < 2:
            return {}
            
        # Calculate basic metrics
        returns = np.diff(price_data) / price_data[:-1]
        volatility = np.std(returns)
        trend = np.polyfit(np.arange(len(price_data)), price_data, 1)[0]
        
        return {
            'trend_direction': 'up' if trend > 0 else 'down',
            'trend_strength': abs(trend),
            'volatility': volatility,
            'volume_profile': np.mean(volume_data) if len(volume_data) > 0 else 0,
            'market_regime': self._identify_market_regime(returns, volatility)
        }

    def _identify_market_regime(self, returns: np.ndarray, volatility: float) -> str:
        """Identify the current market regime.
        
        Args:
            returns (np.ndarray): Price returns
            volatility (float): Market volatility
            
        Returns:
            str: Market regime identifier
        """
        if volatility > 0.02:  # High volatility threshold
            if np.mean(returns) > 0:
                return 'volatile_bullish'
            else:
                return 'volatile_bearish'
        else:
            if np.mean(returns) > 0:
                return 'stable_bullish'
            else:
                return 'stable_bearish'

    def _select_strategy_template(self, market_conditions: Dict[str, Any],
                                extracted_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Select most appropriate strategy template.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            extracted_knowledge (Dict[str, Any]): Extracted market insights
            
        Returns:
            Dict[str, Any]: Selected strategy template
        """
        # Create market context embedding
        context = f"{market_conditions['market_regime']} {market_conditions['trend_direction']} "
        context += f"volatility: {market_conditions['volatility']}"
        
        if 'summary' in extracted_knowledge:
            context += f" {extracted_knowledge['summary']}"
            
        context_embedding = self.embedding_model.encode(context)
        
        # Calculate similarity with each template
        best_score = -1
        selected_template = self.strategy_templates[0]
        
        for template in self.strategy_templates:
            template_text = f"{template['name']} {template['description']}"
            template_embedding = self.embedding_model.encode(template_text)
            
            score = util.pytorch_cos_sim(context_embedding, template_embedding).item()
            
            if score > best_score:
                best_score = score
                selected_template = template
                
        return selected_template

    def _generate_parameters(self, template: Dict[str, Any],
                           market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy parameters based on template and market conditions.
        
        Args:
            template (Dict[str, Any]): Strategy template
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            Dict[str, Any]: Generated parameters
        """
        parameters = {}
        
        # Generate parameters based on template type
        if template['name'] == 'trend_following':
            parameters['timeframe'] = self._optimize_timeframe(market_conditions)
            parameters['threshold'] = self._optimize_threshold(market_conditions)
            
        elif template['name'] == 'mean_reversion':
            parameters['window_size'] = self._optimize_window_size(market_conditions)
            parameters['std_dev'] = self._optimize_std_dev(market_conditions)
            
        elif template['name'] == 'breakout':
            parameters['range_period'] = self._optimize_range_period(market_conditions)
            parameters['breakout_threshold'] = self._optimize_breakout_threshold(market_conditions)
            
        return parameters

    def _optimize_timeframe(self, market_conditions: Dict[str, Any]) -> int:
        """Optimize timeframe parameter based on market conditions.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            int: Optimized timeframe
        """
        volatility = market_conditions.get('volatility', 0.02)
        
        # Higher volatility -> shorter timeframe
        if volatility > 0.03:
            return 5  # 5-minute timeframe
        elif volatility > 0.02:
            return 15  # 15-minute timeframe
        else:
            return 30  # 30-minute timeframe

    def _optimize_threshold(self, market_conditions: Dict[str, Any]) -> float:
        """Optimize threshold parameter based on market conditions.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            float: Optimized threshold
        """
        volatility = market_conditions.get('volatility', 0.02)
        
        # Higher volatility -> higher threshold
        return min(0.02, volatility * 1.5)

    def _optimize_window_size(self, market_conditions: Dict[str, Any]) -> int:
        """Optimize window size parameter based on market conditions.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            int: Optimized window size
        """
        volatility = market_conditions.get('volatility', 0.02)
        
        # Higher volatility -> smaller window
        if volatility > 0.03:
            return 10
        elif volatility > 0.02:
            return 20
        else:
            return 30

    def _optimize_std_dev(self, market_conditions: Dict[str, Any]) -> float:
        """Optimize standard deviation parameter based on market conditions.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            float: Optimized standard deviation
        """
        volatility = market_conditions.get('volatility', 0.02)
        
        # Higher volatility -> higher std dev threshold
        return max(1.5, 2.0 * volatility / 0.02)

    def _optimize_range_period(self, market_conditions: Dict[str, Any]) -> int:
        """Optimize range period parameter based on market conditions.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            int: Optimized range period
        """
        volatility = market_conditions.get('volatility', 0.02)
        
        # Higher volatility -> shorter range period
        if volatility > 0.03:
            return 20
        elif volatility > 0.02:
            return 30
        else:
            return 40

    def _optimize_breakout_threshold(self, market_conditions: Dict[str, Any]) -> float:
        """Optimize breakout threshold parameter based on market conditions.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            float: Optimized breakout threshold
        """
        volatility = market_conditions.get('volatility', 0.02)
        
        # Higher volatility -> higher breakout threshold
        return max(0.01, volatility * 1.2)

    def _generate_conditions(self, template: Dict[str, Any],
                           extracted_knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate entry/exit conditions based on template and market insights.
        
        Args:
            template (Dict[str, Any]): Strategy template
            extracted_knowledge (Dict[str, Any]): Extracted market insights
            
        Returns:
            List[Dict[str, Any]]: Generated conditions
        """
        conditions = []
        
        # Generate base conditions from template
        for condition_type in template['conditions']:
            condition = self._generate_base_condition(condition_type)
            conditions.append(condition)
            
        # Add conditions from extracted knowledge
        if 'sentiment' in extracted_knowledge:
            conditions.append(self._generate_sentiment_condition(extracted_knowledge['sentiment']))
            
        if 'topics' in extracted_knowledge:
            conditions.append(self._generate_topic_condition(extracted_knowledge['topics']))
            
        return conditions

    def _generate_base_condition(self, condition_type: str) -> Dict[str, Any]:
        """Generate base condition from template condition type.
        
        Args:
            condition_type (str): Type of condition
            
        Returns:
            Dict[str, Any]: Generated condition
        """
        if condition_type == 'price_trend':
            return {
                'type': 'trend',
                'indicator': 'price',
                'operator': 'greater_than',
                'lookback': 20,
                'threshold': 0.02
            }
        elif condition_type == 'volume_trend':
            return {
                'type': 'trend',
                'indicator': 'volume',
                'operator': 'greater_than',
                'lookback': 10,
                'threshold': 1.5
            }
        elif condition_type == 'price_deviation':
            return {
                'type': 'deviation',
                'indicator': 'price',
                'operator': 'outside',
                'lookback': 30,
                'threshold': 2.0
            }
        elif condition_type == 'volatility':
            return {
                'type': 'volatility',
                'indicator': 'price',
                'operator': 'less_than',
                'lookback': 20,
                'threshold': 0.03
            }
        elif condition_type == 'price_range':
            return {
                'type': 'range',
                'indicator': 'price',
                'operator': 'inside',
                'lookback': 30,
                'threshold': 0.01
            }
        elif condition_type == 'volume_surge':
            return {
                'type': 'surge',
                'indicator': 'volume',
                'operator': 'greater_than',
                'lookback': 5,
                'threshold': 2.0
            }
        else:
            return {}

    def _generate_sentiment_condition(self, sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate condition based on sentiment analysis.
        
        Args:
            sentiment (Dict[str, Any]): Sentiment analysis results
            
        Returns:
            Dict[str, Any]: Generated condition
        """
        return {
            'type': 'sentiment',
            'indicator': 'market_sentiment',
            'operator': 'equals',
            'value': sentiment['label'],
            'confidence': sentiment['score']
        }

    def _generate_topic_condition(self, topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate condition based on extracted topics.
        
        Args:
            topics (List[Dict[str, Any]]): Extracted topics
            
        Returns:
            Dict[str, Any]: Generated condition
        """
        return {
            'type': 'topic',
            'indicator': 'market_topics',
            'operator': 'contains',
            'values': [topic['main_term'] for topic in topics[:3]],
            'confidence': sum(topic['frequency'] for topic in topics[:3]) / 100
        }

    def _evaluate_strategy(self, parameters: Dict[str, Any],
                         conditions: List[Dict[str, Any]],
                         market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate generated strategy using historical data.
        
        Args:
            parameters (Dict[str, Any]): Strategy parameters
            conditions (List[Dict[str, Any]]): Strategy conditions
            market_data (Dict[str, Any]): Market data for backtesting
            
        Returns:
            Dict[str, Any]: Strategy evaluation metrics
        """
        # TODO: Implement proper backtesting
        # For now, return estimated metrics based on market conditions
        
        volatility = market_data.get('volatility', 0.02)
        trend_strength = market_data.get('trend_strength', 0.01)
        
        return {
            'risk_reward': 2.0 + trend_strength / volatility,
            'win_rate': 0.55 + trend_strength * 10,
            'sharpe_ratio': 1.5 + trend_strength / volatility,
            'max_drawdown': -0.1 - volatility * 5
        }

    def _calculate_confidence(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the strategy.
        
        Args:
            evaluation (Dict[str, Any]): Strategy evaluation metrics
            
        Returns:
            float: Confidence score between 0 and 1
        """
        score = 0
        total_weight = 0
        
        for metric, value in evaluation.items():
            if metric in self.evaluation_metrics:
                weight = self.evaluation_metrics[metric]['weight']
                threshold = self.evaluation_metrics[metric]['threshold']
                
                # Calculate normalized score for the metric
                if metric == 'max_drawdown':
                    # Lower is better for drawdown
                    metric_score = min(1.0, abs(threshold / value)) if value < 0 else 0
                else:
                    # Higher is better for other metrics
                    metric_score = min(1.0, value / threshold)
                    
                score += weight * metric_score
                total_weight += weight
                
        return score / total_weight if total_weight > 0 else 0 